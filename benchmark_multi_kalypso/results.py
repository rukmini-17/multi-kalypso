import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Result:
    commit: str
    workload: str
    requested_rows: int
    rows: int | None
    n_workers: int
    policy: str
    rep: int
    completion_s: float | None = None
    rows_per_s: float | None = None
    num_output_rows: int | None = None
    status: str = "error"
    error: str | None = None
    hypotheses: int | None = None
    llm_calls: int | None = None
    errors: int | None = None
    per_worker_request_counts: dict | None = None
    prefix_hit_fraction: float | None = None
    preemptions: int | None = None
    # Pending instrumentation: request counts cannot establish busy-time imbalance,
    # dependency/capacity stalls, or per-worker prefix-cache hit rates.
    imbalance: float | None = None
    stall_s: float | None = None
    per_worker_prefix_hit: dict | None = None


def derive(
    run_dir: Path, *, record_requests: bool = True, capture_kv_stats: bool = True
) -> dict:
    metrics = {}
    records = run_dir / "requests.jsonl"
    if record_requests and records.exists():
        rows = [
            json.loads(line)
            for line in records.read_text().splitlines()
            if line.strip()
        ]
        requests = [r for r in rows if r.get("type", "request") == "request"]
        metrics.update(
            llm_calls=len(requests),
            errors=sum(
                bool(r.get("error")) or r.get("status", "ok") != "ok" for r in requests
            ),
            per_worker_request_counts=dict(Counter(str(r["worker"]) for r in requests)),
        )
    log = run_dir / "coordinator.log"
    if capture_kv_stats and log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "QUERY_KV_STATS " in line:
                try:
                    stats = json.loads(line.split("QUERY_KV_STATS ", 1)[1])
                except json.JSONDecodeError:
                    continue
                metrics.update(
                    prefix_hit_fraction=stats.get("prefix_hit_fraction"),
                    preemptions=stats.get("preemptions"),
                )
    return metrics


def append_result(out_dir: Path, result: Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    row = asdict(result)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    path = out_dir / "results.csv"
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                for k, v in row.items()
            }
        )
