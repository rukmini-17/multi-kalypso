# M3 multi-worker experiment sweep

This package runs the same experiment on a local multi-GPU machine or through an optional thin Modal backend.
The full design is 3 workloads x 4 policies x 3 worker counts = 36 configurations, repeated 3 times = 108 timed runs.
Each N group starts workers once, resets their prefix caches before every run, and starts a fresh coordinator with its own logs and recorder.
There is no warm-up.

## Layout

- `plans.py`, `data/`: FEVER 500 claims, MEDEC 300 notes, ContractNLI 200 contracts joined with 17 hypotheses.
- `dataset.py`, `prepare_contract_nli.py`: deterministic subsets and offline official-data conversion.
- `config.py`, `configs/`: YAML settings and ordered N, policy, workload, rep expansion.
- `servers.py`, `runner.py`: subprocess lifecycle and host execution, with no Modal dependency.
- `results.py`: `results.csv` and `results.jsonl` hand-off.
- `remote.py`, `modal_app.py`: source/output archives, merging, and optional Modal execution.
- `tests/`: CPU tests using fixtures and fake servers, never real GPU servers.

## Dry-run

Run commands from the repository root with Python 3.12 and `pyyaml` and `requests` installed.
Actual inference also requires this fork's vLLM dependencies, matching compiled kernels, CUDA and sufficient GPU memory.

```bash
python -m benchmark_multi_kalypso.runner --config benchmark_multi_kalypso/configs/m3_sweep.yaml --n 2 --gpus 0,1 --out runs/preview --dry-run
```

This prints 36 planned runs and subprocess commands without creating outputs or starting servers.
Each workload reports its data row sufficiency separately from policy availability, including the 17 ContractNLI hypotheses.
Unavailable policies are marked pending when the local registry can be imported.
If that import fails, execution tries the coordinator and records any failure.

## Own multi-GPU machine

```bash
python -m benchmark_multi_kalypso.runner --config benchmark_multi_kalypso/configs/smoke.yaml --n 2 --gpus 0,1 --out runs/smoke-001
```

For the full experiment, prepare the datasets below, set their `data_root` values in the config, and use `m3_sweep.yaml` with the desired N and exactly N GPU identifiers.
Run the command once per configured N, using a fresh output directory each time.
GPU identifiers can be CUDA indices or UUIDs; each worker receives one through `CUDA_VISIBLE_DEVICES`.
Ports 8000 and 8100 through 8100+N-1 must be free.
Workers and the coordinator bind to localhost.
Data preflight runs before worker startup and rejects undersized datasets unless `allow_fewer_rows` is true.
SIGTERM and SIGHUP interrupt the runner so its cleanup blocks stop the server process groups.
After the first interrupt, further SIGTERM, SIGHUP and SIGINT signals are ignored until cleanup completes.
The command exits nonzero for failed runs and retains their logs and result rows; unavailable policies have status `skipped_policy`.
Use a fresh output directory for retries because recorder files append.

Outputs include host worker logs, a config snapshot, source commit, CSV/JSONL results and per-run request, response, coordinator log, recorder and result files when available.
`completion_s` measures wall time around the query POST, excluding startup, data preparation and prefix reset.
`rows_per_s` uses actual input rows, and is left empty for failed or skipped runs.
ContractNLI also records the actual hypothesis count.
Local results record the Git SHA with a `-dirty` suffix when tracked files have uncommitted changes.
The recorder can be inspected further with `vllm/kalypso/benchmark/summarize_requests.py`.

## Prepare official data offline

The included FEVER CSV contains 1000 claims extracted from `samuel/skip-inference-ci` in record ID order.
The repository's MEDEC and ContractNLI samples contain only 10 inputs, and ContractNLI has only 10 hypotheses.
Download the official datasets yourself; the ContractNLI converter performs no network requests.

```bash
python -m benchmark_multi_kalypso.prepare_contract_nli --input /path/to/train.json --out benchmark_multi_kalypso/data/contract_nli
```

Set MEDEC `data_root: benchmark_multi_kalypso/data/medec` and ContractNLI `data_root: benchmark_multi_kalypso/data/contract_nli` in your config.
Each `data_root` is a folder with the same layout as that workload's default root.
FEVER reads `<root>/fever_claims_1000.csv`, MEDEC reads `<root>/MEDEC-TrainingSet-1000.csv`, and ContractNLI reads `<root>/contracts/` and `<root>/hypotheses/`.
Relative data roots resolve from the repository root.
ContractNLI conversion preserves all labels as integer-named hypothesis files and emits documents in source order.
Use a fresh ContractNLI output directory to avoid stale documents from earlier splits.
Place the official MEDEC CSV directly at `<root>/MEDEC-TrainingSet-1000.csv` and point `data_root` at that folder.
Kalypso reads its `Text ID`, `Text` and `Sentences` columns directly without conversion.
CSV materialization preserves headers and multiline cells; text files are selected by integer stem.
Each workload's inputs are copied once per host into `out/data/<workload>/`, with right tables in `table_<op index>` folders.
All runs reuse those copies and retain their own small `request.json` files.
Too few rows or hypotheses fail clearly by default.
`allow_fewer_rows: true` warns and records actual counts for sample exploration; those runs do not represent the full M3 workload.

## Modal backend

Install a Modal SDK supporting [Function.with_options](https://modal.com/docs/guide/dynamic-function-config) in your launch environment and configure your own Modal account before using this optional command.
Pure helpers have CPU test coverage; the Modal module is syntax-checked only.
No Modal execution or GPU validation was performed.

```bash
modal run benchmark_multi_kalypso/modal_app.py --config benchmark_multi_kalypso/configs/m3_sweep.yaml --ref HEAD
# Limit to one host group when needed.
modal run benchmark_multi_kalypso/modal_app.py --config benchmark_multi_kalypso/configs/smoke.yaml --only-n 2
```

The entrypoint spawns one container per N concurrently, requesting `A10:2`, `A10:3`, and `A10:4` from `hardware.gpu_type` and `n_workers`.
Commit package changes before a Modal run.
Uncommitted changes under `benchmark_multi_kalypso/` block archiving with a commit-first error.
It archives the engine and experiment package exactly at the selected Git ref and records that ref's commit.
Only configured data root folders are overlaid from the working tree.
For Modal, custom data roots must be inside the repository and use relative paths.
The image installs the pinned precompiled wheel for upstream base `2c0ee0fde8db447bd6e7d3ea0d49070661a243fa`, then overlays its compiled artifacts into the uploaded source tree.
Each container invokes the same runner CLI used for a local host in a separate process.
The runner inherits stdout and stderr so progress appears live in Modal logs.
HF downloads use volume `kalypso-hf-cache` mounted at `/cache`, with `HF_HOME=/cache/hf`.
Downloaded results go to `kalypso_modal_run/<timestamp>/n<N>/`, with merged CSV/JSONL at the timestamp root.
Each container has a 24-hour timeout; unusually slow groups may require a smaller matrix.

## Configuration and current limits

- `model`, `policies`, `workloads`, `n_workers`, `reps`: define the matrix; no automatic warm-up is added.
- `vllm.max_model_len`, `gpu_memory_utilization`, `extra_args`: worker settings and additional worker CLI arguments.
- `timeouts.worker_ready_s`, `coordinator_ready_s`, `query_s`: default to 1800, 120 and 3600 seconds.
- `reset_prefix_cache`: defaults true and calls every worker's `/reset_prefix_cache` outside the timer; workers always get `VLLM_SERVER_DEV_MODE=1`.
- `instrumentation.record_requests`: passes `--record-path`; disabled or absent recorder metrics remain empty.
- `instrumentation.poll_interval`: passes coordinator `--poll-interval`; null keeps its default.
- `instrumentation.kv_cache_metrics_sample`: null omits worker KV flags; otherwise enables `--kv-cache-metrics` and sets the sample fraction after checking CLI support.
- `instrumentation.capture_kv_stats`: reads the last valid `QUERY_KV_STATS` line for aggregate prefix hit fraction and preemptions; it does not disable coordinator logging.

Child environments always remove `KALYPSO_COIN_FLIP_SEED` and `KALYPSO_SKIP_INFERENCE` for timed inference.
The M3 sweep defaults `kv_cache_metrics_sample` to null to avoid sampling overhead in timed runs.
Raise it to a fraction up to 1.0 for separate eviction analysis; sampling every block can perturb timing.
Workload requests never add ignored `max_tokens` arguments, and request table paths are absolute.
MEDEC and ContractNLI prompts use upstream `scenarios.py` constants.
FEVER uses blocking execution because aggregation input order differs when pipelined.
The existing `sem_agg` default of 8192 output tokens risks exceeding the configured 4096 context, especially with many mapped claims; this remains unresolved engine work.
Full MEDEC/ContractNLI datasets and teammate policy `cost_aware` are pending inputs.
`imbalance`, `stall_s`, and `per_worker_prefix_hit` remain empty until the later instrumentation work lands.
The skip-inference correctness test belongs to the separate CPU CI branch and is outside this package.

## CPU checks

```bash
PYTHONPATH=. ../main/.venv/bin/python -m pytest --noconftest benchmark_multi_kalypso/tests -q -p no:cacheprovider
python -m py_compile benchmark_multi_kalypso/modal_app.py
```

Tests never import `modal_app.py` and do not need the Modal SDK.
