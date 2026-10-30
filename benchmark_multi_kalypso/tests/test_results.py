import csv
import json

from benchmark_multi_kalypso.results import Result, append_result, derive


def test_run_metrics_ignore_headers_and_queue_samples_use_last_kv_stats(tmp_path):
    (tmp_path / "requests.jsonl").write_text(
        '{"type":"header"}\n'
        '{"worker":"w0","status":"ok","error":null}\n'
        '{"type":"request","worker":"w1","status":"error","error":"failed"}\n'
        '{"type":"queue_sample","worker":"w0","waiting":2,"running":1}\n'
        '{"type":"request","worker":"w0","status":"ok"}\n'
    )
    (tmp_path / "coordinator.log").write_text(
        '[QueryProcessor] QUERY_KV_STATS {"prefix_hit_fraction":0.1,"preemptions":0}\n'
        "unrelated logging\n"
        '[QueryProcessor] QUERY_KV_STATS {"prefix_hit_fraction":0.75,"preemptions":2}\n'
    )
    assert derive(tmp_path) == {
        "llm_calls": 3,
        "errors": 1,
        "per_worker_request_counts": {"w0": 2, "w1": 1},
        "prefix_hit_fraction": 0.75,
        "preemptions": 2,
    }
    assert derive(tmp_path, record_requests=False, capture_kv_stats=False) == {}


def test_appended_jsonl_and_csv_keep_unavailable_metrics_empty(tmp_path):
    result = Result(
        "abc",
        "fever",
        10,
        10,
        2,
        "round_robin",
        1,
        completion_s=2.0,
        rows_per_s=5.0,
        status="ok",
        per_worker_request_counts={"w0": 3},
    )
    append_result(tmp_path, result)
    append_result(tmp_path, result)
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["rows_per_s"] == 5.0
    assert rows[0]["imbalance"] is None
    with (tmp_path / "results.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 2
    assert (
        csv_rows[0]["imbalance"]
        == csv_rows[0]["stall_s"]
        == csv_rows[0]["per_worker_prefix_hit"]
        == ""
    )
    assert json.loads(csv_rows[0]["per_worker_request_counts"]) == {"w0": 3}


def test_missing_instrumentation_is_not_reported_as_zero(tmp_path):
    assert derive(tmp_path) == {}
