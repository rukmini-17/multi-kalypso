import json
import signal
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark_multi_kalypso import runner
from benchmark_multi_kalypso.config import WorkloadConfig, load_config

ROOT = Path(__file__).resolve().parents[2]


def fake_host(monkeypatch):
    workers = [SimpleNamespace(url="http://w0"), SimpleNamespace(url="http://w1")]
    start = Mock(return_value=workers)
    stop, reset = Mock(), Mock()
    monkeypatch.setattr(runner.servers, "start_workers", start)
    monkeypatch.setattr(runner.servers, "stop", stop)
    monkeypatch.setattr(runner.servers, "reset_prefix_caches", reset)
    return start, stop, reset


def test_success_and_unknown_policy_write_complete_result_rows(tmp_path, monkeypatch):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, policies=["round_robin", "cost_aware"], reps=2)
    start, stop, reset = fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    coordinator = SimpleNamespace(url="http://coordinator")
    coord_start = Mock(return_value=coordinator)
    monkeypatch.setattr(runner.servers, "start_coordinator", coord_start)
    post = Mock(return_value={"num_output_rows": 1, "latency_sec": 1.5, "results": []})
    monkeypatch.setattr(runner.servers, "post_query", post)
    monkeypatch.setattr(
        runner.time, "monotonic", Mock(side_effect=[10.0, 12.0, 20.0, 22.0])
    )
    out = tmp_path / "run"
    results = runner.run_host(config, 2, ["0", "1"], out, ROOT)
    assert [
        (r.policy, r.rep, r.rows, r.completion_s, r.rows_per_s, r.status)
        for r in results
    ] == [
        ("round_robin", 1, 20, 2.0, 10.0, "ok"),
        ("round_robin", 2, 20, 2.0, 10.0, "ok"),
        ("cost_aware", 1, None, None, None, "skipped_policy"),
        ("cost_aware", 2, None, None, None, "skipped_policy"),
    ]
    start.assert_called_once()
    assert reset.call_count == 2
    assert coord_start.call_count == 2
    assert stop.call_count == 4  # Two fresh coordinators and two persistent workers.
    body = post.call_args.args[1]
    assert Path(body["data_path"]) == out / "data/fever/input/input.csv"
    assert Path(body["data_path"]).is_absolute()
    assert body["blocking"] is False
    assert len((out / "results.csv").read_text().splitlines()) == 5
    assert (
        json.loads((out / "round_robin/fever/rep_1/response.json").read_text())[
            "num_output_rows"
        ]
        == 1
    )
    with pytest.raises(FileExistsError):
        runner.run_host(config, 2, ["0", "1"], out, ROOT)


def test_coordinator_failure_is_recorded_and_workers_stopped(tmp_path, monkeypatch):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    _, stop, _ = fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    start = Mock(side_effect=RuntimeError("unknown policy"))
    monkeypatch.setattr(runner.servers, "start_coordinator", start)
    (result,) = runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    assert result.status == "error"
    assert result.error == "unknown policy"
    start.assert_called_once()
    assert stop.call_count == 2


def test_post_failure_stops_coordinator_and_retains_failed_duration(
    tmp_path, monkeypatch
):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    _, stop, _ = fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    monkeypatch.setattr(
        runner.servers,
        "start_coordinator",
        Mock(return_value=SimpleNamespace(url="http://c")),
    )
    monkeypatch.setattr(
        runner.servers, "post_query", Mock(side_effect=TimeoutError("timed out"))
    )
    (result,) = runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    assert result.status == "error"
    assert result.error == "timed out"
    assert result.completion_s >= 0
    assert result.rows_per_s is None
    assert stop.call_count == 3


def test_preflight_rejects_short_data_before_starting_workers(tmp_path, monkeypatch):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[WorkloadConfig("medec", 300)])
    start, _, _ = fake_host(monkeypatch)
    with pytest.raises(ValueError, match="medec rows: have 10, need 300"):
        runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    start.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_main_signal_interrupt_cleans_coordinator_and_workers(tmp_path, monkeypatch):
    _, stop, _ = fake_host(monkeypatch)
    coordinator = SimpleNamespace(url="http://c")
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    monkeypatch.setattr(
        runner.servers, "start_coordinator", Mock(return_value=coordinator)
    )
    handlers = {}
    monkeypatch.setattr(runner.signal, "signal", handlers.__setitem__)

    def repeated_signal_during_cleanup(server):
        assert all(
            handlers[sig] == signal.SIG_IGN
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
        )

    stop.side_effect = repeated_signal_during_cleanup

    def interrupted_post(*args):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(runner.servers, "post_query", interrupted_post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--config",
            str(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml"),
            "--n",
            "2",
            "--gpus",
            "0,1",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        runner.main()
    assert stop.call_args_list[0].args == (coordinator,)
    assert [call.args[0].url for call in stop.call_args_list[1:]] == [
        "http://w1",
        "http://w0",
    ]


def test_source_commit_uses_dirty_description_or_archived_sha(tmp_path, monkeypatch):
    describe = Mock(return_value="abc123-dirty\n")
    monkeypatch.setattr(runner.subprocess, "check_output", describe)
    assert runner.source_commit(tmp_path) == "abc123-dirty"
    provenance = tmp_path / "benchmark_multi_kalypso/source_commit.txt"
    provenance.parent.mkdir()
    provenance.write_text("archived-sha\n")
    assert runner.source_commit(tmp_path) == "archived-sha"
    assert describe.call_count == 1
