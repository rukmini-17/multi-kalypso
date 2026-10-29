# CPU testing environment and correctness harness

The launcher starts N CPU vLLM workers sequentially on BASE+1 through BASE+N, then the standalone coordinator on BASE (default 8100).
End-to-end runs need the rest of the team's pieces and a working CPU-compatible vLLM build; this harness is unit-tested with fakes.
Activate that environment first; the launcher prefers the checkout's `.venv/bin/python` when present.

```bash
KV_GIB=1 \
  vllm/kalypso/benchmark/launch_multi_worker.sh -n 2 -p round_robin --detach -o ./kalypso_run
python vllm/kalypso/benchmark/correctness_harness.py compare \
  --payload vllm/kalypso/benchmark/sample_data/filter_map_20.json \
  --baseline-endpoint http://localhost:8101/v1/semantic/query \
  --candidate-endpoint http://localhost:8100/v1/semantic/query \
  --out-dir kalypso_run --records kalypso_run/requests.jsonl
vllm/kalypso/benchmark/launch_multi_worker.sh --stop -o ./kalypso_run
```

Worker 1's in-process semantic endpoint is the single-worker baseline; the candidate is `python -m vllm.kalypso.server`.
Use `-n N`, `-m MODEL`, `-b BASE_PORT`, `-p POLICY`, and `-o OUT_DIR` to choose workers, model, coordinator port, policy, and output directory.
The policy defaults to `round_robin` and is passed through to the coordinator.
Worker readiness uses `WORKER_READY_TIMEOUT` (default 900 seconds); coordinator readiness uses `/v1/semantic/healthz` and `COORD_READY_TIMEOUT` (default 600 seconds).
Each worker uses `KV_GIB` GiB of KV cache (default 1), plus model weights and runtime memory.
The launcher sets `KALYPSO_COIN_FLIP_SEED` to 42 unless it is already set; every worker and the coordinator inherit it, and real inference still runs before seeded reply replacement.
Pass `--no-seed` to unset it so operators return real model output; use that mode to try the system, not for `compare`, since real replies can differ between endpoints.
Use the same payload, model, checkout, and seed for both endpoints.

The harness saves `baseline.json`, `candidate.json`, and their input CSVs in `--out-dir`.
Client and servers must share a filesystem because requests submit the CSV's absolute `data_path`.
It checks unique valid output indices, row mapping, and exact operator-value agreement; filtered-out rows are allowed when the baseline agrees.
It also checks non-pin calls exactly once with successful status and full input coverage and prints per-worker call counts.
The launcher clears `requests.jsonl` in the output directory at start; send exactly one candidate query per launch because the recorder appends.
Exit status is 0 on success and 1 on failure; `run` and `check` remain available separately.
Omit `--detach` to keep the launcher attached; Ctrl-C stops its child processes.

## Compare against the old checkout

Run `vllm/kalypso/benchmark/compare_old_new.sh -n 2 -p round_robin -o ./kalypso_old_vs_new` to compare old single-server Kalypso against N coordinator workers on this checkout, with the same payload, model, and seed (default 42).
The old side is `samuel/old-seeded-baseline` (team `main` plus seeded replies); run `git fetch` first, since the script also tries `REMOTE/samuel/old-seeded-baseline` for each remote.
The script checks that ref out as a detached worktree in `OUT_DIR/src_old` and copies this checkout's compiled `vllm/**/*.so` into it; pass `--old-ref REF` for another seeded commit, or `--old DIR` to use an existing checkout.
An existing `src_old` must be a clean worktree at the same commit; an explicitly empty `KALYPSO_COIN_FLIP_SEED` is rejected.
The script prints both commits, stops and waits for the old server before starting the new side, and cleans up servers on exit.
Results are saved as `old.json`, `new.json`, their input CSVs, `old_server.log`, and `new_run/` with request records and logs; the exit status comes from harness `check` using old as baseline.
Use `--help` for payload, model, port, and output options; relative paths resolve from this checkout's root.

## Compare with real inference on Modal

Run `modal run vllm/kalypso/benchmark/modal_old_vs_new.py --payload vllm/kalypso/benchmark/sample_data/fever_people_1000.json` to run `gpu_old_vs_new.sh` in one Modal container with two A10G GPUs.
Install the client with `pip install modal` and log in once with `modal token new` first.
The old side defaults to `05024f027` and the new side to `HEAD`; pass `--old-ref` and `--new-ref` for other commits, which are sent as `git archive` snapshots, so uncommitted edits are not included.
`gpu_old_vs_new.sh`, `row_diff.py`, and `correctness_harness.py` are taken from the working tree.
The image installs the precompiled vLLM wheel for the fork's upstream base, `2c0ee0fde`, and adds its compiled files to both checkouts in place of a source build.
Model weights are cached in the `kalypso-hf-cache` Modal volume, so only the first run downloads them.
Results, logs, and `row_diff.md` are saved to `kalypso_modal_run/<timestamp>/` (or `--out DIR`), and the exit status comes from `correctness_harness.py check`.
