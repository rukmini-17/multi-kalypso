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
