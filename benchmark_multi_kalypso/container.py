"""Standard-library helpers mounted into the container before unpacking the repo."""

import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

COMPILED = re.compile(
    r"vllm/(.*\.so|vllm_flash_attn/.*\.py|third_party/triton_kernels/.*\.py)"
)


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
    repo, out, config = work / "repo", work / "out", work / "config.yaml"
    unpack(repo_tar, repo)
    overlay_compiled(site, repo)
    config.write_text(config_text, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark_multi_kalypso.runner",
            "--config",
            str(config),
            "--n",
            str(n),
            "--gpus",
            ",".join(str(i) for i in range(n)),
            "--out",
            str(out),
        ],
        cwd=repo,
    )
    # Preflight failures can exit before the runner creates its output folder.
    out.mkdir(parents=True, exist_ok=True)
    return completed.returncode, pack_directory(out)
