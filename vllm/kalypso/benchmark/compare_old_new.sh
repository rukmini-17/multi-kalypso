#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
usage() {
    cat <<'EOF'
Usage: compare_old_new.sh [options]
  --old-ref REF   Old seeded commit, branch, or tag, checked out into OUT_DIR/src_old
                  (default: samuel/old-seeded-baseline, also tried as REMOTE/REF)
  --old DIR       Use this existing old checkout instead of --old-ref
  -n N            Worker count (default: 2)
  -p POLICY       Routing policy (default: round_robin)
  -m MODEL        Model (default: Qwen/Qwen2.5-0.5B-Instruct)
  -o OUT_DIR      Results directory (default: ./kalypso_old_vs_new)
  --payload FILE  Payload (default: vllm/kalypso/benchmark/sample_data/filter_map_20.json)
  --old-port P    Old server port (default: 8200)
  -h, --help      Show this help

Relative paths are resolved from this checkout's root.
The new side is this checkout as it is, including uncommitted edits.
Runs old then new sequentially with KALYPSO_COIN_FLIP_SEED (default: 42).
The coordinator uses port 8100; workers use 8101 through 8100+N.
Saves old.json, old.json.input.csv, old_server.log, new.json,
new.json.input.csv, and new_run/; exits with the harness check's status.
EOF
}
die() { echo "$*" >&2; exit 1; }
old='' old_ref=samuel/old-seeded-baseline n=2 policy=round_robin model=Qwen/Qwen2.5-0.5B-Instruct
out=./kalypso_old_vs_new payload=vllm/kalypso/benchmark/sample_data/filter_map_20.json old_port=8200
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --old|--old-ref|-n|-p|-m|-o|--payload|--old-port)
            (($# >= 2)) || die "Missing value for $1"
            [[ -n "$2" ]] || die "Empty value for $1"
            case "$1" in
                --old) old=$2 ;; --old-ref) old_ref=$2 ;; -n) n=$2 ;; -p) policy=$2 ;;
                -m) model=$2 ;; -o) out=$2 ;; --payload) payload=$2 ;;
                --old-port) old_port=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
[[ "${KALYPSO_COIN_FLIP_SEED-unset}" != '' ]] || die 'KALYPSO_COIN_FLIP_SEED must not be empty'
export KALYPSO_COIN_FLIP_SEED="${KALYPSO_COIN_FLIP_SEED-42}"
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
[[ -f "$payload" ]] || die "Payload is missing: $payload"
[[ "$n" =~ ^[1-9][0-9]*$ && "$old_port" =~ ^[1-9][0-9]*$ ]] || die 'N and old port must be positive integers'
((n <= 65535 - 8100 && old_port <= 65535)) || die 'Ports must be at most 65535'
[[ "${WORKER_READY_TIMEOUT:-900}" =~ ^[1-9][0-9]*$ ]] || die 'WORKER_READY_TIMEOUT must be a positive integer'
command -v curl >/dev/null || die 'curl is required'
python=python
[[ ! -x .venv/bin/python ]] || python="$PWD/.venv/bin/python"
command -v "$python" >/dev/null || die "Python is required: $python"
launcher="$PWD/vllm/kalypso/benchmark/launch_multi_worker.sh"
harness="$PWD/vllm/kalypso/benchmark/correctness_harness.py"
[[ ! -e "$out/new_run/pids" ]] || die "Run the launcher --stop first: $out/new_run/pids already exists"
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
    # The old source tree needs the compiled vllm libraries from this checkout.
    (cd vllm && find . -name '*.so' -type f) | while read -r lib; do
        mkdir -p "$old/vllm/$(dirname "$lib")"
        cp -p "vllm/$lib" "$old/vllm/$lib"
    done
fi
[[ -f "$old/vllm/kalypso/execution/determinism.py" ]] || die "Old checkout lacks seeded replies: $old/vllm/kalypso/execution/determinism.py"
echo "Old commit: $(git -C "$old" rev-parse --short HEAD)"
echo "New commit: $(git -C "$PWD" rev-parse --short HEAD)"
echo "Seeded replies on (KALYPSO_COIN_FLIP_SEED=$KALYPSO_COIN_FLIP_SEED)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
old_pid='' launcher_pid=''
stop_old() {
    if [[ -n "$old_pid" ]]; then
        kill "$old_pid" 2>/dev/null || true
        wait "$old_pid" 2>/dev/null || true
        old_pid=''
    fi
}
cleanup() {
    local status=$?
    trap '' INT TERM
    stop_old
    if [[ -n "$launcher_pid" ]]; then
        kill "$launcher_pid" 2>/dev/null || true
        wait "$launcher_pid" 2>/dev/null || true
    fi
    if [[ -f "$out/new_run/pids" ]]; then
        "$launcher" --stop -o "$out/new_run" || true
    fi
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready() {
    local url=$1 timeout=$2 log=$3 pid=$4 deadline=$((SECONDS + $2)) code
    while ((SECONDS < deadline)); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url") || code=000
        [[ "$code" != 200 ]] || return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    tail -n 30 "$log" >&2
    die "Readiness failed (timeout ${timeout}s or process exited): $url"
}
echo "Running old server on port $old_port; results: $out"
(
    cd "$old"
    export PYTHONPATH="$old"
    export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_KVCACHE_SPACE="${KV_GIB:-1}"
    exec "$python" -m vllm.entrypoints.openai.api_server --model "$model" \
        --enable-prefix-caching --max-model-len 4096 --port "$old_port"
) > "$out/old_server.log" 2>&1 < /dev/null &
old_pid=$!
ready "http://localhost:$old_port/v1/models" "${WORKER_READY_TIMEOUT:-900}" "$out/old_server.log" "$old_pid"
"$python" "$harness" run --payload "$payload" \
    --endpoint "http://localhost:$old_port/v1/semantic/query" --out "$out/old.json"
stop_old
echo "Running new coordinator with $n workers"
"$launcher" -n "$n" -p "$policy" -m "$model" --detach -o "$out/new_run" &
launcher_pid=$!
wait "$launcher_pid"
launcher_pid=''
"$python" "$harness" run --payload "$payload" \
    --endpoint http://localhost:8100/v1/semantic/query --out "$out/new.json"
"$launcher" --stop -o "$out/new_run"
"$python" "$harness" check --payload "$payload" --baseline "$out/old.json" \
    --candidate "$out/new.json" --records "$out/new_run/requests.jsonl"
