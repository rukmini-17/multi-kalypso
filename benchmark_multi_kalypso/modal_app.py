"""Thin optional backend. Importing this module requires Modal."""

import sysconfig
import tempfile
import time
from pathlib import Path

import modal

VLLM_BASE = "2c0ee0fde8db447bd6e7d3ea0d49070661a243fa"
WHEEL = (
    f"https://wheels.vllm.ai/{VLLM_BASE}/"
    "vllm-0.13.0rc5.dev1%2Bg2c0ee0fde-cp38-abi3-manylinux_2_31_x86_64.whl"
)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .uv_pip_install(WHEEL, "faiss-cpu", "pyyaml", "requests")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("benchmark_multi_kalypso.container")
)
hf_cache = modal.Volume.from_name("kalypso-hf-cache", create_if_missing=True)
app = modal.App("kalypso-m3-sweep", image=image)


@app.function(timeout=24 * 3600, volumes={"/cache": hf_cache})
def run_n(config_text: str, n: int, repo_tar: bytes) -> tuple[int, bytes]:
    from benchmark_multi_kalypso.container import run_uploaded

    with tempfile.TemporaryDirectory(prefix="kalypso-") as work:
        return run_uploaded(
            config_text, n, repo_tar, Path(sysconfig.get_paths()["purelib"]), Path(work)
        )


@app.local_entrypoint()
def main(config: str, ref: str = "HEAD", only_n: int = 0):
    from benchmark_multi_kalypso.config import parse_config
    from benchmark_multi_kalypso.container import unpack
    from benchmark_multi_kalypso.remote import archive_repo, jobs, merge_results

    root = Path(__file__).resolve().parents[1]
    config_text = Path(config).read_text(encoding="utf-8")
    experiment = parse_config(config_text)
    planned = jobs(experiment, only_n or None)
    archive = archive_repo(root, ref, experiment)
    out = root / "kalypso_modal_run" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    # Spawn all N groups before collecting any outputs.
    calls = [
        (n, run_n.with_options(gpu=gpu).spawn(config_text, n, archive))
        for n, gpu in planned
    ]
    status, groups = 0, []
    for n, call in calls:
        group = out / f"n{n}"
        groups.append(group)
        try:
            code, data = call.get()
            unpack(data, group)
            status = max(status, code)
        except Exception as exc:
            group.mkdir(parents=True, exist_ok=True)
            (group / "host_error.txt").write_text(str(exc) + "\n")
            status = 1
    merge_results(groups, out)
    print(f"Saved results to {out} (status {status})")
    if status:
        raise SystemExit(status)
