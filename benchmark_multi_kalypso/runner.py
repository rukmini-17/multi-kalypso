import argparse
import json
import shlex
import signal
import subprocess
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from . import servers
from .config import Experiment, Run, expand, load_config
from .dataset import count_rows, materialize, materialize_table
from .plans import REPO_ROOT, Join, build_plan, to_request
from .results import Result, append_result, derive


def registered_policies() -> set[str] | None:
    try:
        from vllm.kalypso.workers.policy import POLICIES

        return set(POLICIES)
    except Exception as exc:
        warnings.warn(
            f"Cannot import policy registry; coordinator will validate policies: {exc}",
            stacklevel=2,
        )
        return None


def source_commit(repo_root: Path) -> str:
    provenance = repo_root / "benchmark_multi_kalypso/source_commit.txt"
    if provenance.exists():
        return provenance.read_text().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def preflight(
    config: Experiment, repo_root: str | Path
) -> dict[str, dict[str, tuple[int, int]]]:
    counts = {}
    for workload in config.workloads:
        root = workload.data_root
        data_root = (Path(repo_root) / root).resolve() if root is not None else None
        plan = build_plan(workload.name, workload.rows, data_root)
        tables = [("rows", plan.data_path, plan.rows), *plan.tables]
        counts[workload.name] = {}
        for label, path, needed in tables:
            try:
                available = count_rows(path)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Data preflight failed for {workload.name} {label}: {path}: {exc}"
                ) from exc
            counts[workload.name][label] = (available, needed)
    return counts


def run_one(
    config: Experiment,
    run: Run,
    workers: list[servers.Server],
    run_dir: Path,
    repo_root: Path,
    result: Result,
) -> None:
    coordinator = None
    try:
        root = run.workload.data_root
        data_root = (repo_root / root).resolve() if root is not None else None
        plan = build_plan(run.workload.name, run.workload.rows, data_root)
        data_path = materialize(
            plan, plan.rows, run_dir / "data", allow_fewer_rows=config.allow_fewer_rows
        )
        result.rows = count_rows(data_path)
        right_tables = {
            path: materialize_table(
                path, rows, run_dir / label, allow_fewer_rows=config.allow_fewer_rows
            )
            for label, path, rows in plan.tables
        }
        if right_tables:
            result.hypotheses = sum(count_rows(path) for path in right_tables.values())
        plan = replace(
            plan,
            ops=[
                replace(op, table=right_tables[op.table])
                if isinstance(op, Join)
                else op
                for op in plan.ops
            ],
        )
        body = to_request(plan, data_path, config.model)
        (run_dir / "request.json").write_text(json.dumps(body, indent=2) + "\n")
        if config.reset_prefix_cache:
            servers.reset_prefix_caches(workers)
        coordinator = servers.start_coordinator(
            config, run.policy, workers, run_dir, repo_root
        )
        started = time.monotonic()
        try:
            response = servers.post_query(
                coordinator.url, body, config.timeouts.query_s
            )
        finally:
            result.completion_s = time.monotonic() - started
        (run_dir / "response.json").write_text(json.dumps(response, indent=2) + "\n")
        result.num_output_rows = response["num_output_rows"]
        result.status = "ok"
    except Exception as exc:
        result.status, result.error = "error", str(exc)
    finally:
        if coordinator is not None:
            servers.stop(coordinator)
    try:
        for key, value in derive(
            run_dir,
            record_requests=config.instrumentation.record_requests,
            capture_kv_stats=config.instrumentation.capture_kv_stats,
        ).items():
            setattr(result, key, value)
        if result.errors:
            result.status = "error"
            result.error = f"{result.errors} failed LLM requests"
    except Exception as exc:
        result.status, result.error = "error", f"Result derivation: {exc}"
    if result.status == "ok" and result.completion_s and result.rows is not None:
        result.rows_per_s = result.rows / result.completion_s


def run_host(
    config: Experiment,
    n_workers: int,
    gpus: list[str],
    out_dir: str | Path,
    repo_root: str | Path,
) -> list[Result]:
    runs = expand(config, n_workers)
    if (
        len(gpus) != n_workers
        or len(set(gpus)) != n_workers
        or any(not str(g) for g in gpus)
    ):
        raise ValueError("Supply exactly N distinct GPU identifiers")
    out, repo = Path(out_dir).resolve(), Path(repo_root).resolve()
    counts = preflight(config, repo)
    shortages = [
        f"{name} {label}: have {have}, need {need}"
        for name, tables in counts.items()
        for label, (have, need) in tables.items()
        if have < need
    ]
    if shortages and not config.allow_fewer_rows:
        raise ValueError("Data preflight failed: " + "; ".join(shortages))
    # Fresh directories prevent stale recorder rows and responses contaminating runs.
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    commit = source_commit(repo)
    (out / "source_commit.txt").write_text(commit + "\n")
    policies = registered_policies()
    results, workers = [], []
    startup_error = None
    try:
        if any(policies is None or r.policy in policies for r in runs):
            try:
                workers = servers.start_workers(
                    config, [str(g) for g in gpus], out, repo
                )
            except Exception as exc:
                startup_error = f"Worker startup: {exc}"
        for run in runs:
            run_dir = out / run.name
            run_dir.mkdir(parents=True)
            result = Result(
                commit,
                run.workload.name,
                run.workload.rows,
                None,
                n_workers,
                run.policy,
                run.rep,
            )
            if policies is not None and run.policy not in policies:
                result.status = "skipped_policy"
                result.error = (
                    f"Policy {run.policy!r} is pending/unavailable in POLICIES"
                )
            elif startup_error:
                result.error = startup_error
            else:
                run_one(config, run, workers, run_dir, repo, result)
            (run_dir / "result.json").write_text(
                json.dumps(asdict(result), indent=2) + "\n"
            )
            append_result(out, result)
            results.append(result)
            print(
                f"N={n_workers} {run.name}: {result.status} {result.error or ''}",
                flush=True,
            )
    finally:
        for worker in reversed(workers):
            servers.stop(worker)
    return results


def dry_run(config: Experiment, n: int, gpus: list[str], out: Path) -> None:
    runs = expand(config, n)
    if len(gpus) != n or len(set(gpus)) != n:
        raise ValueError("Supply exactly N distinct GPU identifiers")
    counts = preflight(config, REPO_ROOT)
    policies = registered_policies()
    print(f"N={n}: {len(runs)} runs; no processes will start")
    for i, gpu in enumerate(gpus):
        print(
            f"CUDA_VISIBLE_DEVICES={gpu} VLLM_SERVER_DEV_MODE=1 "
            + shlex.join(servers.worker_command(config, 8100 + i, REPO_ROOT))
        )
    print("Child environments unset KALYPSO_COIN_FLIP_SEED and KALYPSO_SKIP_INFERENCE")
    for run in runs:
        status = (
            "pending/unavailable"
            if policies is not None and run.policy not in policies
            else "registry unknown; try coordinator"
            if policies is None
            else "available"
        )
        tags = []
        for label, (have, need) in counts[run.workload.name].items():
            data_status = (
                "ok" if have >= need else f"insufficient: have {have}, need {need}"
            )
            prefix = "" if label == "rows" else f"{label}: "
            tags.append(f"[{prefix}{data_status}]")
        print(
            f"N={n} {run.name}: {run.workload.rows} rows "
            f"{' '.join(tags)} [policy: {status}]"
        )
        print(
            "  "
            + shlex.join(
                servers.coordinator_command(
                    config,
                    run.policy,
                    [f"http://127.0.0.1:{8100 + i}" for i in range(n)],
                    8000,
                    out.resolve() / run.name,
                )
            )
        )


def main() -> int:
    def interrupt(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    parser = argparse.ArgumentParser(
        description="Run one N-worker M3 group on this host"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    gpus = [g.strip() for g in args.gpus.split(",")]
    if args.dry_run:
        dry_run(config, args.n, gpus, args.out)
        return 0
    results = run_host(config, args.n, gpus, args.out, REPO_ROOT)
    return int(any(r.status == "error" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
