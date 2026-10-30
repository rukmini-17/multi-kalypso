import json
import signal
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark_multi_kalypso import plans, runner
from benchmark_multi_kalypso.config import Workload, load_config

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
    assert Path(body["data_path"]).is_absolute()
    assert body["blocking"] is True
    assert post.call_args.args[2] == 3600
    assert len((out / "results.jsonl").read_text().splitlines()) == 4
    assert (
        json.loads((out / "round_robin/fever/rep_1/response.json").read_text())[
            "num_output_rows"
        ]
        == 1
    )
    with pytest.raises(FileExistsError):
        runner.run_host(config, 2, ["0", "1"], out, ROOT)


def test_registry_import_failure_attempts_policy_and_records_coordinator_failure(
    tmp_path, monkeypatch
):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, policies=["future_policy"])
    _, stop, _ = fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: None)
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


def test_allow_fewer_rows_records_actual_rows_and_hypotheses(tmp_path, monkeypatch):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(
        config, workloads=[Workload("contract_nli", 200)], allow_fewer_rows=True
    )
    fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    monkeypatch.setattr(
        runner.servers,
        "start_coordinator",
        Mock(return_value=SimpleNamespace(url="http://c")),
    )
    post = Mock(return_value={"num_output_rows": 5, "results": []})
    monkeypatch.setattr(runner.servers, "post_query", post)
    with pytest.warns(UserWarning):
        (result,) = runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    assert (result.rows, result.hypotheses, result.status) == (10, 10, "ok")
    right = Path(post.call_args.args[1]["ops"][1]["args"]["right_table"])
    assert right == tmp_path / "out/round_robin/contract_nli/rep_1/hypotheses"
    assert runner.count_rows(right) == 10


def test_runner_uses_declared_join_position_and_rows(tmp_path, monkeypatch):
    workload = plans.WORKLOADS["contract_nli"]
    join = replace(workload.ops[1], rows=3)
    monkeypatch.setitem(
        plans.WORKLOADS,
        "contract_nli",
        replace(workload, ops=[join, workload.ops[0], workload.ops[2]]),
    )
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[Workload("contract_nli", 2)])
    assert runner.preflight(config, ROOT) == {
        "contract_nli": {"rows": (10, 2), "hypotheses": (10, 3)}
    }
    fake_host(monkeypatch)
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    monkeypatch.setattr(
        runner.servers,
        "start_coordinator",
        Mock(return_value=SimpleNamespace(url="http://c")),
    )
    post = Mock(return_value={"num_output_rows": 1, "results": []})
    monkeypatch.setattr(runner.servers, "post_query", post)
    (result,) = runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    assert (result.rows, result.hypotheses, result.status) == (2, 3, "ok")
    right = Path(post.call_args.args[1]["ops"][0]["args"]["right_table"])
    assert right == tmp_path / "out/round_robin/contract_nli/rep_1/hypotheses"
    assert runner.count_rows(right) == 3


def test_dry_run_starts_nothing_and_prints_36_runs(tmp_path, monkeypatch, capsys):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/m3_sweep.yaml")
    start, _, _ = fake_host(monkeypatch)
    monkeypatch.setattr(
        runner,
        "registered_policies",
        lambda: {"round_robin", "least_outstanding", "row_affinity"},
    )
    runner.dry_run(config, 2, ["0", "1"], tmp_path / "out")
    output = capsys.readouterr().out
    assert len([line for line in output.splitlines() if line.startswith("N=2 ")]) == 36
    assert (
        "cost_aware/fever/rep_1: 500 rows [ok] [policy: pending/unavailable]" in output
    )
    assert "medec/rep_1: 300 rows [insufficient: have 10, need 300]" in output
    assert "contract_nli/rep_1: 200 rows [insufficient: have 10, need 200]" in output
    assert "[hypotheses: insufficient: have 10, need 17]" in output
    assert "--kv-cache-metrics" not in output
    assert not (tmp_path / "out").exists()
    start.assert_not_called()


def test_preflight_counts_repository_relative_data_root(tmp_path):
    root = tmp_path / "claims"
    root.mkdir()
    (root / "fever_claims_1000.csv").write_text('data\n"first claim"\n"second claim"\n')
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[Workload("fever", 2, "claims")])
    assert runner.preflight(config, tmp_path) == {"fever": {"rows": (2, 2)}}


@pytest.mark.parametrize(
    "workload,message",
    [
        (Workload("medec", 300), "medec rows: have 10, need 300"),
        (Workload("contract_nli", 10), "contract_nli hypotheses: have 10, need 17"),
    ],
)
def test_preflight_rejects_short_data_before_starting_workers(
    tmp_path, monkeypatch, workload, message
):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[workload])
    start, _, _ = fake_host(monkeypatch)
    with pytest.raises(ValueError, match=message):
        runner.run_host(config, 2, ["0", "1"], tmp_path / "out", ROOT)
    start.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_main_signal_interrupt_cleans_coordinator_and_workers(
    tmp_path, monkeypatch, signum
):
    _, stop, _ = fake_host(monkeypatch)
    coordinator = SimpleNamespace(url="http://c")
    monkeypatch.setattr(runner, "registered_policies", lambda: {"round_robin"})
    monkeypatch.setattr(
        runner.servers, "start_coordinator", Mock(return_value=coordinator)
    )
    handlers = {}
    monkeypatch.setattr(runner.signal, "signal", handlers.__setitem__)

    def interrupted_post(*args):
        handlers[signum](signum, None)

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
    with pytest.raises(KeyboardInterrupt, match=signal.Signals(signum).name):
        runner.main()
    assert stop.call_args_list[0].args == (coordinator,)
    assert [call.args[0].url for call in stop.call_args_list[1:]] == [
        "http://w1",
        "http://w0",
    ]
