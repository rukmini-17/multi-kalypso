#!/usr/bin/env bash
# Stop on errors and resolve all relative paths from this checkout's root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Describe the engine-free comparison and the options needed to reproduce it.
usage() {
    cat <<'EOF'
Usage: skip_inference_old_vs_new.sh [options]
  --old-ref REF   Old seeded ref checked out into OUT_DIR/src_old
                  (default: samuel/old-seeded-baseline, also tried as REMOTE/REF)
  --old DIR       Use this existing old checkout instead of --old-ref
  -n N            Worker count (default: 2)
  -p POLICY       Routing policy (default: round_robin)
  -m MODEL        Model tokenizer/config (default: Qwen/Qwen2.5-0.5B-Instruct)
  -o OUT_DIR      Results directory (default: ./kalypso_skip_old_vs_new)
  --payload FILE  Payload (default: vllm/kalypso/benchmark/sample_data/filter_map_20.json)
  --port BASE     Old port; coordinator BASE+1, workers BASE+2 onward (default: 8300)
  -h, --help      Show this help

Relative paths resolve from this checkout's root. Uses python from PATH unless
this checkout has .venv/bin/python. The new side includes uncommitted edits.
Runs with KALYPSO_COIN_FLIP_SEED (default: 42) and KALYPSO_SKIP_INFERENCE=1.
Requires tokenizer/config downloads, but no engine, weights, or compiled vLLM.
Saves responses, input CSVs, requests.jsonl, and logs in OUT_DIR.
Exits with the harness check's status; cleans up child processes on exit.
EOF
}

# Supply defaults and reject malformed options before starting any servers.
die() { echo "$*" >&2; exit 1; }
old='' old_ref=samuel/old-seeded-baseline n=2 policy=round_robin model=Qwen/Qwen2.5-0.5B-Instruct
out=./kalypso_skip_old_vs_new payload=vllm/kalypso/benchmark/sample_data/filter_map_20.json base=8300
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --old|--old-ref|-n|-p|-m|-o|--payload|--port)
            (($# >= 2)) || die "Missing value for $1"
            [[ -n "$2" ]] || die "Empty value for $1"
            case "$1" in
                --old) old=$2 ;; --old-ref) old_ref=$2 ;; -n) n=$2 ;; -p) policy=$2 ;;
                -m) model=$2 ;; -o) out=$2 ;; --payload) payload=$2 ;; --port) base=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# Enable identical seeded replies on both sides and reserve 2 GiB per worker.
[[ "${KALYPSO_COIN_FLIP_SEED-unset}" != '' ]] || die 'KALYPSO_COIN_FLIP_SEED must not be empty'
export KALYPSO_COIN_FLIP_SEED="${KALYPSO_COIN_FLIP_SEED-42}"
export KALYPSO_SKIP_INFERENCE=1
worker_budget=$((2 * 1024 * 1024 * 1024))

# Validate inputs and dependencies now so failures do not leave child processes.
[[ -f "$payload" ]] || die "Payload is missing: $payload"
[[ "$n" =~ ^[1-9][0-9]*$ && "$base" =~ ^[1-9][0-9]*$ ]] || die 'N and base port must be positive integers'
((base <= 65533 && n <= 65534 - base)) || die 'Ports must be at most 65535'
[[ "${WORKER_READY_TIMEOUT:-900}" =~ ^[1-9][0-9]*$ ]] || die 'WORKER_READY_TIMEOUT must be a positive integer'
command -v curl >/dev/null || die 'curl is required'
python=python
[[ ! -x .venv/bin/python ]] || python="$PWD/.venv/bin/python"
command -v "$python" >/dev/null || die "Python is required: $python"
root=$PWD
standins="$root/vllm/kalypso/benchmark/skip_inference_servers.py"
harness="$root/vllm/kalypso/benchmark/correctness_harness.py"

# Use the supplied old checkout, or resolve its ref locally and then on each remote.
if [[ -n "$old" ]]; then
    [[ -d "$old" ]] || die "Old checkout directory is missing: $old"
    old=$(cd "$old" && pwd)
else
    old_sha=''
    for candidate in "$old_ref" $(git remote | sed "s|\$|/$old_ref|"); do
        old_sha=$(git rev-parse --verify --quiet "$candidate^{commit}") && break
    done
    [[ -n "$old_sha" ]] || die "Cannot resolve old ref: $old_ref (git fetch the remote that has it, or pass --old-ref)"
fi

# Keep the resolved oracle in a reusable detached worktree without copying libraries.
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -z "$old" ]]; then
    old="$out/src_old"
    if [[ -e "$old" ]]; then
        [[ -f "$old/.git" && "$(git -C "$old" rev-parse HEAD)" == "$old_sha" ]] || die "Existing $old must be a worktree at $old_sha (remove it or choose another -o)"
        [[ -z "$(git -C "$old" status --porcelain --untracked-files=no)" ]] || die "Worktree has tracked edits: $old"
    else
        git worktree add --detach "$old" "$old_sha"
    fi
fi

# Print the source revisions so saved logs identify exactly what was compared.
[[ -f "$old/vllm/kalypso/execution/determinism.py" ]] || die "Old checkout lacks seeded replies: $old"
echo "Old commit: $(git -C "$old" rev-parse HEAD)"
echo "New commit: $(git -C "$root" rev-parse HEAD)"
echo "Skip inference on (KALYPSO_COIN_FLIP_SEED=$KALYPSO_COIN_FLIP_SEED); results: $out"

# Track every child and wait for shutdown before releasing its port or checking records.
pids=''
stop_all() {
    local pid deadline

    # Ask every child to stop so graceful shutdown can happen in parallel.
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done

    # Allow each child up to ten seconds, then force any survivor to exit.
    for pid in $pids; do
        deadline=$((SECONDS + 10))
        while kill -0 "$pid" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Reap every child before forgetting its PID or reusing its port.
    for pid in $pids; do wait "$pid" 2>/dev/null || true; done
    pids=''
}
cleanup() {
    local status=$?
    trap '' INT TERM
    stop_all
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Refuse any HTTP responder on a port before starting a server there.
ensure_port_free() {
    local port=$1 code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/") || true
    [[ "$code" == 000 ]] || die "Port $port is already in use (HTTP $code); choose another --port BASE"
}

# Wait for an HTTP 200 while the child is alive; show its log if startup fails.
ready() {
    local url=$1 log=$2 pid=$3 timeout=${WORKER_READY_TIMEOUT:-900} code
    local deadline=$((SECONDS + timeout))
    while ((SECONDS < deadline)); do
        kill -0 "$pid" 2>/dev/null || break
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url") || code=000
        if [[ "$code" == 200 ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    tail -n 30 "$log" >&2
    die "Readiness failed (timeout ${timeout}s or process exited): $url"
}

# Run only the old query processor, with the same total budget as the worker pool.
ensure_port_free "$base"
echo "Running old stand-in on port $base"
(
    cd "$old"
    export PYTHONPATH="$old"
    exec "$python" "$standins" old --port "$base" --model "$model" \
        --budget "$((n * worker_budget))" --expect-root "$old"
) > "$out/old_server.log" 2>&1 < /dev/null &
pids=$!
ready "http://127.0.0.1:$base/v1/semantic/healthz" "$out/old_server.log" "$pids"
"$python" "$harness" run --payload "$payload" \
    --endpoint "http://127.0.0.1:$base/v1/semantic/query" --out "$out/old.json"
stop_all

# Start discovery-only workers; any real completion request must fail with HTTP 500.
export PYTHONPATH="$root"
urls=''
for ((i=0; i<n; i++)); do
    port=$((base + 2 + i))
    ensure_port_free "$port"
    "$python" "$standins" worker --port "$port" --model "$model" --budget "$worker_budget" \
        --expect-root "$root" \
        > "$out/worker_$i.log" 2>&1 < /dev/null &
    pid=$!; pids="$pids $pid"
    urls="${urls:+$urls,}http://127.0.0.1:$port"
    ready "http://127.0.0.1:$port/v1/models" "$out/worker_$i.log" "$pid"
done

# Start this checkout's real coordinator with a fresh request record for this query.
ensure_port_free "$((base + 1))"
: > "$out/requests.jsonl"
echo "Running new coordinator with $n workers and policy $policy"
"$python" -m vllm.kalypso.server --host 127.0.0.1 --port "$((base + 1))" \
    --workers "$urls" --policy "$policy" --record-path "$out/requests.jsonl" \
    > "$out/coordinator.log" 2>&1 < /dev/null &
pid=$!; pids="$pids $pid"
ready "http://127.0.0.1:$((base + 1))/v1/semantic/healthz" "$out/coordinator.log" "$pid"
"$python" "$harness" run --payload "$payload" \
    --endpoint "http://127.0.0.1:$((base + 1))/v1/semantic/query" --out "$out/new.json"
stop_all

# Compare outputs and recorded calls with the existing harness, preserving its exit status.
"$python" "$harness" check --payload "$payload" --baseline "$out/old.json" \
    --candidate "$out/new.json" --records "$out/requests.jsonl"
