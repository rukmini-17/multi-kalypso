#!/usr/bin/env bash
# Initialize the runner and stop on failed commands or unset variables.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
# Describe command options, server ports, and saved artifacts.
usage() {
    cat <<'EOF'
Usage: gpu_old_vs_new.sh --payload FILE [options]
  --repo DIR           Repository containing both refs (default: this checkout)
  --old-ref REF        Old single-server ref (default: 05024f027)
  --new-ref REF        New coordinator ref (default: HEAD of --repo)
  -n N                 Worker count (default: 2)
  -m MODEL             Served model (default: Qwen/Qwen2.5-7B-Instruct)
  -p POLICY            Coordinator policy (default: round_robin)
  --payload FILE       Input payload (required; model_name is replaced by MODEL)
  -o OUT_DIR           Output directory (default: ./kalypso_gpu_run)
  --so-from DIR        Copy all vllm/**/*.so from DIR (default: --repo)
  --device gpu|cpu     GPU inference or local CPU dry run (default: gpu)
  --gpu-util FLOAT     GPU memory utilization (default: 0.85)
  --max-model-len N    Maximum model length (default: 4096)
  -h, --help           Show this help

Paths resolve from the invoking directory. Activate the vLLM environment first;
--repo/.venv/bin/python is preferred when present. Seeded replies are disabled.
Ports: old 8200, coordinator 8100, workers 8101 through 8100+N.
WORKER_READY_TIMEOUT defaults to 1800 seconds for every server.
Saves responses, input CSVs, logs, requests.jsonl, commits.txt and row_diff.md.
Exit status comes from the harness check. Existing worktrees must match the refs.
EOF
}
# Report fatal setup or runtime errors with a failing exit status.
die() { echo "$*" >&2; exit 1; }
# Parse args: start with defaults for refs, workers, model, and output.
repo=$root old_ref=05024f027 new_ref=HEAD n=2 model=Qwen/Qwen2.5-7B-Instruct
policy=round_robin payload='' out=./kalypso_gpu_run so_from='' device=gpu
gpu_util=0.85 max_model_len=4096 timeout=${WORKER_READY_TIMEOUT:-1800}
# Read options and reject missing values or unsupported flags.
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --repo|--old-ref|--new-ref|-n|-m|-p|--payload|-o|--so-from|--device|--gpu-util|--max-model-len)
            # Require a nonempty value before assigning the selected option.
            (($# >= 2)) || die "Missing value for $1"
            [[ -n "$2" ]] || die "Empty value for $1"
            case "$1" in
                --repo) repo=$2 ;; --old-ref) old_ref=$2 ;; --new-ref) new_ref=$2 ;;
                -n) n=$2 ;; -m) model=$2 ;; -p) policy=$2 ;; --payload) payload=$2 ;;
                -o) out=$2 ;; --so-from) so_from=$2 ;; --device) device=$2 ;;
                --gpu-util) gpu_util=$2 ;; --max-model-len) max_model_len=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done
# Validate inputs and keep worker ports below the old server port.
[[ -f "$payload" ]] || die 'A readable --payload FILE is required'
[[ "$n" =~ ^[1-9][0-9]*$ && "$n" -lt 100 ]] || die 'N must be 1..99 (worker ports must not overlap old port 8200)'
[[ "$max_model_len" =~ ^[1-9][0-9]*$ ]] || die '--max-model-len must be a positive integer'
[[ "$timeout" =~ ^[1-9][0-9]*$ ]] || die 'WORKER_READY_TIMEOUT must be a positive integer'
[[ "$device" == gpu || "$device" == cpu ]] || die '--device must be gpu or cpu'
# Check tools: require enough visible GPUs for the requested workers.
if [[ "$device" == gpu ]]; then
    command -v nvidia-smi >/dev/null || die 'nvidia-smi is required in GPU mode'
    gpu_list=$(nvidia-smi -L) || die 'Cannot list GPUs with nvidia-smi -L'
    gpu_count=$(awk '/^GPU [0-9]+:/ {n++} END {print n+0}' <<< "$gpu_list")
    ((gpu_count >= n)) || die "Need $n GPUs, found $gpu_count"
fi
# Find curl, resolve source paths, and prefer the checkout Python environment.
command -v curl >/dev/null || die 'curl is required'
repo=$(cd "$repo" && pwd)
so_from=$(cd "${so_from:-$repo}" && pwd)
python=python
[[ ! -x "$repo/.venv/bin/python" ]] || python="$repo/.venv/bin/python"
python=$(command -v "$python") || die 'Activate a working vLLM Python environment'
# Resolve both refs to exact commits and show which versions will run.
old_sha=$(git -C "$repo" rev-parse --verify "$old_ref^{commit}")
new_sha=$(git -C "$repo" rev-parse --verify "$new_ref^{commit}")
printf 'Old commit: %s\nNew commit: %s\n' "$old_sha" "$new_sha"
# Disable seeded replies and skip-inference mode for this comparison.
unset KALYPSO_COIN_FLIP_SEED KALYPSO_SKIP_INFERENCE
if [[ "$device" == cpu ]]; then unset CUDA_VISIBLE_DEVICES; fi
echo 'Seeded replies off; running real inference'
# Prepare the output directory and normalize its path for child processes.
mkdir -p "$out"
out=$(cd "$out" && pwd)
# Read the input before writing payload.json, including when it is the input.
"$python" - "$payload" "$out" "$model" "$gpu_util" "$so_from" "$n" <<'PY'
import json
import socket
import sys
from pathlib import Path

# Check the GPU memory fraction and that --so-from has compiled vllm/*.so files.
payload, out, model, util, so_from, n = sys.argv[1:]
if not 0 < float(util) <= 1:
    raise SystemExit('--gpu-util must be in (0, 1]')
if not any((Path(so_from) / 'vllm').rglob('*.so')):
    raise SystemExit(f'No compiled vllm/*.so files under {so_from}')
# Load the input and select the same served model for both versions.
data = json.loads(Path(payload).read_text())
data['model_name'] = model
# Verify that all old, coordinator, and worker ports are available.
for port in [8200, *range(8100, 8101 + int(n))]:
    with socket.socket() as probe:
        try:
            probe.bind(('', port))
        except OSError as exc:
            raise SystemExit(f'Port {port} is unavailable: {exc}')
# Save the shared payload only after reading and validating the input.
Path(out, 'payload.json').write_text(json.dumps(data, indent=2) + '\n')
PY
# Prepare checkouts: reuse clean matching worktrees or create detached ones.
worktree() {
    local target=$1 sha=$2
    # Require existing worktrees to match the commit, repository, and clean state.
    if [[ -e "$target" ]]; then
        [[ -f "$target/.git" && "$(git -C "$target" rev-parse HEAD)" == "$sha" ]] || die "Existing worktree must be at $sha: $target (choose a fresh -o directory)"
        [[ "$(git -C "$target" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)" ]] || die "Worktree belongs to another repository: $target"
        [[ -z "$(git -C "$target" status --porcelain --untracked-files=no)" ]] || die "Worktree has tracked edits: $target"
    # Create a detached checkout when this output directory has none yet.
    else
        git -C "$repo" worktree add --detach --force "$target" "$sha"
    fi
}
# Prepare both source versions and copy compiled libraries into each.
worktree "$out/src_old" "$old_sha"
worktree "$out/src_new" "$new_sha"
"$python" - "$so_from" "$out/src_old" "$out/src_new" <<'PY'
import shutil
import sys
from pathlib import Path

# Find shared libraries while preserving their paths inside vLLM.
source = Path(sys.argv[1]) / 'vllm'
for library in list(source.rglob('*.so')):
    if library.is_file():
        # Create destination folders and copy each library into both checkouts.
        for checkout in sys.argv[2:]:
            target = Path(checkout) / 'vllm' / library.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)
PY
# Record commit identities and clear artifacts left by a previous run.
printf 'Old commit: %s\nNew commit: %s\n' "$old_sha" "$new_sha" > "$out/commits.txt"
rm -f "$out/requests.jsonl" "$out/old.json" "$out/new.json" \
    "$out/old.json.input.csv" "$out/new.json.input.csv" \
    "$out/row_diff.md"
# Manage server lifetimes: track process groups for reliable cleanup.
pids=()
stop_servers() {
    local pid deadline=$((SECONDS + 30)) alive
    ((${#pids[@]})) || return 0
    # Ask every tracked server group to stop, with a single-process fallback.
    for pid in "${pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    # Allow up to 30 seconds for server groups to exit gracefully.
    while ((SECONDS < deadline)); do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
        done
        ((alive)) || break
        sleep 1
    done
    # Force remaining groups to exit, reap children, and clear the list.
    for pid in "${pids[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    pids=()
}
# Preserve the exit status while shutting down servers on exit or signals.
cleanup() {
    local status=$?
    trap '' INT TERM
    stop_servers
    return "$status"
}
# Ensure normal exits and interrupt signals all trigger cleanup.
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Wait for readiness: poll HTTP until success, process exit, or timeout.
ready() {
    local url=$1 log=$2 pid=$3 deadline=$((SECONDS + timeout)) code
    while ((SECONDS < deadline)); do
        # Stop waiting if the server dies; accept only an HTTP 200 response.
        kill -0 "$pid" 2>/dev/null || break
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$url") || code=000
        [[ "$code" != 200 ]] || return 0
        sleep 1
    done
    # Show the server log tail before reporting readiness failure.
    tail -n 30 "$log" >&2
    die "Readiness failed (timeout ${timeout}s or process exited): $url"
}
# Launch servers: use the selected checkout and capture each process group.
start_process() {
    local checkout=$1 log=$2
    shift 2
    (
        # Import sources from this checkout and stream logs without buffering.
        cd "$checkout"
        export PYTHONPATH="$checkout" PYTHONUNBUFFERED=1
        # A separate session lets cleanup terminate vLLM engine children too.
        exec "$python" -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"
    ) > "$log" 2>&1 < /dev/null &
    # Save the background process ID so cleanup can stop its whole session.
    pids+=("$!")
}
# Configure a model server with prefix caching and the shared model limits.
start_worker() {
    local checkout=$1 log=$2 port=$3 gpu=$4
    local env_args=(env)
    local args=("$python" -m vllm.entrypoints.openai.api_server --model "$model"
                --enable-prefix-caching --max-model-len "$max_model_len" --port "$port")
    # Bind GPU workers to one device or configure the CPU dry-run environment.
    if [[ "$device" == gpu ]]; then
        env_args+=("CUDA_VISIBLE_DEVICES=$gpu")
        args+=(--gpu-memory-utilization "$gpu_util")
    else
        env_args+=(-u CUDA_VISIBLE_DEVICES VLLM_ENABLE_V1_MULTIPROCESSING=0
                   "VLLM_CPU_KVCACHE_SPACE=${KV_GIB:-1}" "OMP_NUM_THREADS=$threads")
    fi
    # Start the worker and wait until its model endpoint is ready.
    start_process "$checkout" "$log" "${env_args[@]}" "${args[@]}"
    ready "http://localhost:$port/v1/models" "$log" "${pids[${#pids[@]}-1]}"
}
# Divide CPU threads among workers and locate the shared request harness.
threads=$("$python" -c 'import os,sys; print(max(1, (os.cpu_count() or 1) // int(sys.argv[1])))' "$n")
harness="$root/vllm/kalypso/benchmark/correctness_harness.py"
# Start old server: run the payload once and save its response and input CSV.
echo "Running old single server on $device (port 8200)"
start_worker "$out/src_old" "$out/old_server.log" 8200 0
"$python" "$harness" run --payload "$out/payload.json" \
    --endpoint http://localhost:8200/v1/semantic/query --out "$out/old.json"
# Release the old server before giving its GPU to the new worker pool.
stop_servers
# Run new side: start each worker on its own port and GPU.
urls=''
for ((i=1; i<=n; i++)); do
    start_worker "$out/src_new" "$out/worker_$i.log" "$((8100+i))" "$((i-1))"
    urls+="${urls:+,}http://localhost:$((8100+i))"
    echo "Worker $i ready"
done
# Start the coordinator with worker URLs, routing policy, and request logging.
start_process "$out/src_new" "$out/coordinator.log" "$python" \
    -m vllm.kalypso.server --port 8100 --workers "$urls" \
    --policy "$policy" --record-path "$out/requests.jsonl"
ready http://localhost:8100/v1/semantic/healthz "$out/coordinator.log" "${pids[${#pids[@]}-1]}"
# Send the same payload through the coordinator and save its response.
"$python" "$harness" run --payload "$out/payload.json" \
    --endpoint http://localhost:8100/v1/semantic/query --out "$out/new.json"
# Stop the new servers before comparing the saved results.
stop_servers
# Check correctness and retain its status even when the check fails.
check_status=0
"$python" "$harness" check --payload "$out/payload.json" \
    --baseline "$out/old.json" --candidate "$out/new.json" \
    --records "$out/requests.jsonl" || check_status=$?
# Always write the differing-rows report without overriding the check status.
"$python" "$root/vllm/kalypso/benchmark/row_diff.py" --payload "$out/payload.json" \
    --old "$out/old.json" --new "$out/new.json" --out-dir "$out" || true
# Let the harness check determine the comparison's exit status.
exit "$check_status"
