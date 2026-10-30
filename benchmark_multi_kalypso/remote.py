import io
import subprocess
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Experiment


def jobs(config: "Experiment", only_n: int | None = None) -> list[tuple[int, str]]:
    if only_n is not None and only_n not in config.n_workers:
        raise ValueError(f"N={only_n} is not in configured n_workers")
    return [
        (n, f"{config.hardware.gpu_type}:{n}")
        for n in config.n_workers
        if only_n is None or n == only_n
    ]


def archive_repo(repo: Path, ref: str, config: "Experiment") -> bytes:
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--",
            "benchmark_multi_kalypso",
        ],
        text=True,
    ).strip():
        raise ValueError(
            "Uncommitted changes in benchmark_multi_kalypso/. "
            "Commit package changes before a Modal run."
        )
    sha = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True,
    ).strip()
    archive = subprocess.check_output(
        ["git", "-C", str(repo), "archive", "--format=tar", sha]
    )
    buffer = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archive)) as source,
        tarfile.open(fileobj=buffer, mode="w:gz", dereference=True) as target,
    ):
        roots = []
        for workload in config.workloads:
            if workload.data_root:
                root = Path(workload.data_root)
                roots.append(root)
        roots = [
            r
            for r in dict.fromkeys(roots)
            if not any(r != p and r.is_relative_to(p) for p in roots)
        ]
        for member in source:
            if any(Path(member.name).is_relative_to(root) for root in roots):
                continue
            target.addfile(
                member, source.extractfile(member) if member.isfile() else None
            )
        for root in roots:
            target.add(repo / root, arcname=str(root), filter=_source_filter)
        data = (sha + "\n").encode()
        member = tarfile.TarInfo("benchmark_multi_kalypso/source_commit.txt")
        member.size = len(data)
        target.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def _source_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return (
        None
        if "__pycache__" in Path(member.name).parts or member.name.endswith(".pyc")
        else member
    )


def merge_results(groups: list[Path], out: Path) -> None:
    import json

    from .results import Result, append_result

    out.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "results.csv"):
        if (out / name).exists():
            raise ValueError(f"Merged results already exist: {out / name}")
    for group in groups:
        path = group / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                append_result(out, Result(**json.loads(line)))
