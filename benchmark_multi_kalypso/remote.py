import io
import re
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Experiment


COMPILED = re.compile(
    r"vllm/(.*\.so|vllm_flash_attn/.*\.py|third_party/triton_kernels/.*\.py)"
)


def jobs(config: "Experiment", only_n: int | None = None) -> list[tuple[int, str]]:
    if only_n is not None and only_n not in config.n_workers:
        raise ValueError(f"N={only_n} is not in configured n_workers")
    return [
        (n, f"{config.hardware.gpu_type}:{n}")
        for n in config.n_workers
        if only_n is None or n == only_n
    ]


def pack_directory(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", dereference=True) as tar:
        for path in sorted(root.iterdir()):
            tar.add(path, arcname=path.name)
    return buffer.getvalue()


def unpack(data: bytes, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(target, filter="data")


def archive_repo(repo: Path, ref: str, config: "Experiment") -> bytes:
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
        # Overlay the entire working experiment package even before its first commit.
        roots = [Path("benchmark_multi_kalypso")]
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


def overlay_compiled(site: Path, repo: Path) -> None:
    for path in (site / "vllm").rglob("*"):
        rel = path.relative_to(site).as_posix()
        target = repo / rel
        if (
            path.is_file()
            and COMPILED.fullmatch(rel)
            and (rel.endswith(".so") or not target.exists())
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def run_uploaded(
    config_text: str, n: int, repo_tar: bytes, site: Path, work: Path
) -> tuple[int, bytes]:
    repo, out = work / "repo", work / "out"
    original_path = list(sys.path)
    original_modules = {
        name: module for name, module in sys.modules.items() if _source_module(name)
    }
    try:
        unpack(repo_tar, repo)
        overlay_compiled(site, repo)
        # Load uploaded sources instead of Modal's potentially different auto-mount.
        sys.path.insert(0, str(repo))
        for name in list(sys.modules):
            if _source_module(name):
                del sys.modules[name]
        from benchmark_multi_kalypso.config import parse_config
        from benchmark_multi_kalypso.runner import run_host

        results = run_host(
            parse_config(config_text), n, [str(i) for i in range(n)], out, repo
        )
        status = int(any(r.status == "error" for r in results))
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "host_error.txt").write_text(traceback.format_exc())
        status = 1
    finally:
        # A reused container must not retain imports from a deleted temporary tree.
        sys.path[:] = original_path
        for name in list(sys.modules):
            if _source_module(name):
                del sys.modules[name]
        sys.modules.update(original_modules)
    return status, pack_directory(out)


def _source_module(name: str) -> bool:
    return name in {"benchmark_multi_kalypso", "vllm"} or name.startswith(
        ("benchmark_multi_kalypso.", "vllm.")
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
