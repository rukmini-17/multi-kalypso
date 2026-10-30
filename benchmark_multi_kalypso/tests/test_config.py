from pathlib import Path

import pytest

from benchmark_multi_kalypso.config import expand, load_config, parse_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_smoke_expands_to_one_twenty_row_run():
    config = load_config(CONFIGS / "smoke.yaml")
    assert [
        (r.n_workers, r.policy, r.workload.name, r.workload.rows, r.rep)
        for r in expand(config)
    ] == [(2, "round_robin", "fever", 20, 1)]
    assert config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.timeouts.query_s == 3600


def test_m3_has_36_configurations_108_runs_in_requested_order():
    config = load_config(CONFIGS / "m3_sweep.yaml")
    runs = expand(config)
    assert len(runs) == 108
    assert len({(r.n_workers, r.policy, r.workload.name) for r in runs}) == 36
    assert [r.name for r in runs[:4]] == [
        "round_robin/fever/rep_1",
        "round_robin/fever/rep_2",
        "round_robin/fever/rep_3",
        "round_robin/medec/rep_1",
    ]
    assert [r.n_workers for r in runs] == [2] * 36 + [3] * 36 + [4] * 36
    assert [runs[i].policy for i in [0, 9, 18, 27]] == [
        "round_robin",
        "least_outstanding",
        "cost_aware",
        "row_affinity",
    ]
    assert len(expand(config, 2)) == 36
    with pytest.raises(ValueError, match="not in configured"):
        expand(config, 5)


@pytest.mark.parametrize(
    "before,after",
    [
        ("reps: 1", "reps: 0"),
        ("n_workers: [2]", "n_workers: [2, 2]"),
        ("rows: 20", "rows: 0"),
        ("poll_interval: null", "poll_interval: -1"),
    ],
)
def test_invalid_configs_rejected(before, after):
    with pytest.raises(ValueError):
        parse_config((CONFIGS / "smoke.yaml").read_text().replace(before, after))


def test_extra_worker_arguments_are_passed_through_without_reserved_arg_guards():
    text = (
        (CONFIGS / "smoke.yaml")
        .read_text()
        .replace(
            "extra_args: []",
            "extra_args: [--port, '9000', --tensor-parallel-size, '1']",
        )
    )
    assert parse_config(text).vllm.extra_args == [
        "--port",
        "9000",
        "--tensor-parallel-size",
        "1",
    ]
