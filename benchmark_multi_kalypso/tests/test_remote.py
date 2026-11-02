import csv
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_multi_kalypso.config import load_config
from benchmark_multi_kalypso.remote import jobs, merge_results
from benchmark_multi_kalypso.results import Result, append_result

ROOT = Path(__file__).resolve().parents[2]


def test_jobs_are_one_group_per_n_with_gpu_count():
    config = replace(
        load_config(ROOT / "benchmark_multi_kalypso/configs/m3_sweep.yaml"),
        n_workers=[2, 3, 4],
    )
    assert jobs(config) == [(2, "A10:2"), (3, "A10:3"), (4, "A10:4")]
    assert jobs(config, 3) == [(3, "A10:3")]
    with pytest.raises(ValueError, match="not in configured"):
        jobs(config, 7)


def test_merge_appends_each_n_in_group_order(tmp_path):
    groups = [tmp_path / "n2", tmp_path / "n3"]
    for n, group in zip([2, 3], groups):
        append_result(
            group, Result("abc", "fever", 20, 20, n, "round_robin", 1, status="ok")
        )
    out = tmp_path / "merged"
    merge_results(groups, out)
    with (out / "results.csv").open(newline="") as stream:
        assert [row["n_workers"] for row in csv.DictReader(stream)] == ["2", "3"]
