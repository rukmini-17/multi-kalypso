import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmark_multi_kalypso.config import load_config
from benchmark_multi_kalypso.container import (
    overlay_compiled,
    pack_directory,
    run_uploaded,
    unpack,
)
from benchmark_multi_kalypso.remote import archive_repo, jobs, merge_results
from benchmark_multi_kalypso.results import Result, append_result

ROOT = Path(__file__).resolve().parents[2]


def test_jobs_are_one_group_per_n_with_gpu_count():
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/m3_sweep.yaml")
    assert jobs(config) == [(2, "A10:2"), (3, "A10:3"), (4, "A10:4")]
    assert jobs(config, 3) == [(3, "A10:3")]
    with pytest.raises(ValueError, match="not in configured"):
        jobs(config, 7)


def test_tar_round_trip_keeps_nested_output(tmp_path):
    root = tmp_path / "out"
    (root / "run").mkdir(parents=True)
    (root / "run/response.json").write_text('{"num_output_rows":2}\n')
    (root / "results.csv").write_text("rows,status\n2,ok\n")
    unpack(pack_directory(root), tmp_path / "restored")
    assert (
        tmp_path / "restored/run/response.json"
    ).read_text() == '{"num_output_rows":2}\n'
    assert (tmp_path / "restored/results.csv").read_text() == "rows,status\n2,ok\n"


def test_unpack_rejects_path_traversal(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = 3
        tar.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(tarfile.FilterError):
        unpack(buffer.getvalue(), tmp_path / "safe")
    assert not (tmp_path / "escaped.txt").exists()


def test_archive_keeps_committed_package_and_overlays_local_dataset(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    package = repo / "benchmark_multi_kalypso"
    package.mkdir(parents=True)
    (package / "runner.py").write_text("working package")
    (package / "__pycache__").mkdir()
    (package / "__pycache__/runner.pyc").write_bytes(b"skip")
    (repo / "datasets").mkdir()
    (repo / "datasets/fever.csv").write_text("data\ncustom claim\n")
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    from dataclasses import replace

    from benchmark_multi_kalypso.config import WorkloadConfig

    config = replace(config, workloads=[WorkloadConfig("fever", 1, "datasets")])
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in [
            ("vllm/source.py", b"committed source"),
            ("benchmark_multi_kalypso/runner.py", b"old package"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    calls = []

    def git(args, **kwargs):
        calls.append(args)
        if "status" in args:
            return ""
        return "abc123\n" if "rev-parse" in args else buffer.getvalue()

    monkeypatch.setattr(subprocess, "check_output", git)
    unpack(archive_repo(repo, "HEAD", config), tmp_path / "unpacked")
    assert (tmp_path / "unpacked/vllm/source.py").read_text() == "committed source"
    assert (
        tmp_path / "unpacked/benchmark_multi_kalypso/runner.py"
    ).read_text() == "old package"
    assert (
        tmp_path / "unpacked/benchmark_multi_kalypso/source_commit.txt"
    ).read_text() == "abc123\n"
    assert (
        tmp_path / "unpacked/datasets/fever.csv"
    ).read_text() == "data\ncustom claim\n"
    assert not (tmp_path / "unpacked/benchmark_multi_kalypso/__pycache__").exists()
    assert calls[1][-1] == "HEAD^{commit}"


@pytest.mark.parametrize(
    "status",
    [" M benchmark_multi_kalypso/runner.py\n", "?? benchmark_multi_kalypso/new.py\n"],
)
def test_archive_rejects_uncommitted_package_changes(tmp_path, monkeypatch, status):
    git = Mock(return_value=status)
    monkeypatch.setattr(subprocess, "check_output", git)
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    with pytest.raises(ValueError, match="Commit package changes before a Modal run"):
        archive_repo(tmp_path, "HEAD", config)
    git.assert_called_once_with(
        [
            "git",
            "-C",
            str(tmp_path),
            "status",
            "--porcelain",
            "--",
            "benchmark_multi_kalypso",
        ],
        text=True,
    )


def test_overlay_replaces_kernels_but_preserves_source_python(tmp_path):
    site, repo = tmp_path / "site", tmp_path / "repo"
    for root in (site, repo):
        (root / "vllm/vllm_flash_attn").mkdir(parents=True)
    (site / "vllm/kernel.so").write_text("compiled")
    (site / "vllm/vllm_flash_attn/missing.py").write_text("wheel support")
    (site / "vllm/vllm_flash_attn/existing.py").write_text("wheel version")
    (repo / "vllm/vllm_flash_attn/existing.py").write_text("fork version")
    overlay_compiled(site, repo)
    assert (repo / "vllm/kernel.so").read_text() == "compiled"
    assert (repo / "vllm/vllm_flash_attn/missing.py").read_text() == "wheel support"
    assert (repo / "vllm/vllm_flash_attn/existing.py").read_text() == "fork version"


def test_merge_appends_each_n_in_group_order(tmp_path):
    groups = [tmp_path / "n2", tmp_path / "n3"]
    for n, group in zip([2, 3], groups):
        append_result(
            group, Result("abc", "fever", 20, 20, n, "round_robin", 1, status="ok")
        )
    out = tmp_path / "merged"
    merge_results(groups, out)
    assert [
        json.loads(line)["n_workers"]
        for line in (out / "results.jsonl").read_text().splitlines()
    ] == [2, 3]
    assert len((out / "results.csv").read_text().splitlines()) == 3


@pytest.mark.parametrize("config_text,code", [("config text", 0), ("fail", 3)])
def test_uploaded_job_runs_archived_cli_and_packs_failures(
    tmp_path, capfd, config_text, code
):
    source = tmp_path / "source/benchmark_multi_kalypso"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "runner.py").write_text(
        "import argparse, json\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "for arg in ('config', 'n', 'gpus', 'out'): p.add_argument('--' + arg)\n"
        "a = p.parse_args()\n"
        "config = Path(a.config).read_text()\n"
        "print('archived runner')\n"
        "if config == 'fail': raise SystemExit(3)\n"
        "out = Path(a.out)\nout.mkdir()\n"
        "(out / 'called.json').write_text(json.dumps([config, a.n, a.gpus]))\n"
    )
    old_path = list(sys.path)
    status, output = run_uploaded(
        config_text,
        2,
        pack_directory(source.parent),
        tmp_path / "site",
        tmp_path / "job",
    )
    assert status == code
    unpack(output, tmp_path / "result")
    assert capfd.readouterr().out == "archived runner\n"
    assert not (tmp_path / "result/host.log").exists()
    if code == 0:
        assert json.loads((tmp_path / "result/called.json").read_text()) == [
            "config text",
            "2",
            "0,1",
        ]
    assert sys.path == old_path
