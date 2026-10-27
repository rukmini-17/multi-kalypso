"""Run gpu_old_vs_new.sh in one Modal container with two L4 GPUs.

    modal run vllm/kalypso/benchmark/modal_old_vs_new.py \
        --payload vllm/kalypso/benchmark/sample_data/fever_people_1000.json

Both refs are sent as git archives of this checkout. The container rebuilds
them as two commits in a scratch repo, adds the compiled files from the
precompiled vLLM wheel for the fork's upstream base, and runs the script.
Reports, logs and responses land in kalypso_modal_run/<run id>/.
"""

import io
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

# Pin the upstream base and its matching precompiled vLLM wheel.
VLLM_BASE = "2c0ee0fde8db447bd6e7d3ea0d49070661a243fa"
WHEEL = (
    f"https://wheels.vllm.ai/{VLLM_BASE}/"
    "vllm-0.13.0rc5.dev1%2Bg2c0ee0fde-cp38-abi3-manylinux_2_31_x86_64.whl"
)
# Files a precompiled vLLM install adds to a source tree (see setup.py).
COMPILED = re.compile(
    r"vllm/(.*\.so|vllm_flash_attn/.*\.py"
    r"|third_party/triton_kernels/.*\.py)"
)
# Untracked or edited runner files are taken from the working tree.
RUNNER_FILES = ["gpu_old_vs_new.sh", "row_diff.py", "correctness_harness.py"]
BENCH = "vllm/kalypso/benchmark"

# Build the GPU environment with benchmark tools and cached model downloads.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl")
    .uv_pip_install(WHEEL, "faiss-cpu")
    .env({"HF_HOME": "/cache/hf"})
)
# Share the model cache across runs and attach the image to the Modal app.
hf_cache = modal.Volume.from_name("kalypso-hf-cache", create_if_missing=True)
app = modal.App("kalypso-old-vs-new", image=image)


def git(repo, *args):
    # Run Git with a scratch identity and stop if the command fails.
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=kalypso",
            "-c",
            "user.email=kalypso@localhost",
            *args,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def build_repo(sources, site, repo):
    """Commit each (tag, tar.gz bytes) in order, plus compiled vLLM files."""
    # Initialize a scratch repository to hold both archived source versions.
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    # For each version (old, then new): clear the tree and unpack its source.
    for tag, data in sources:
        git(repo, "rm", "-rqf", "--ignore-unmatch", ".")
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(repo, filter="data")
        # Walk the installed wheel's vllm/ files that match COMPILED.
        for path in (site / "vllm").rglob("*"):
            rel = path.relative_to(site).as_posix()
            target = repo / rel
            # Always copy .so files; copy other matches only if the source lacks them.
            if (
                path.is_file()
                and COMPILED.fullmatch(rel)
                and (rel.endswith(".so") or not target.exists())
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        # Commit and tag the reconstructed version for the shell runner.
        git(repo, "add", "-Af", ".")
        git(repo, "commit", "-qm", tag)
        git(repo, "tag", tag)


@app.function(gpu="L4:2", timeout=3 * 3600, volumes={"/cache": hf_cache})
def run(
    old_tar, new_tar, runner, payload, commits, model, policy, max_model_len, gpu_util
):
    from importlib.util import find_spec

    # Locate the installed wheel and rebuild old and new sources beside it.
    site = Path(find_spec("vllm").origin).parents[1]
    repo, out = Path("/work/repo"), Path("/work/out")
    build_repo([("old", old_tar), ("new", new_tar)], site, repo)
    # Overlay local runner files and save the supplied benchmark payload.
    for name, data in runner.items():
        (repo / BENCH / name).write_bytes(data)
    out.mkdir(parents=True)
    (out / "input_payload.json").write_bytes(payload)
    # Run both versions with two workers and retain the harness check exit status.
    status = subprocess.run(
        [
            "bash",
            str(repo / BENCH / "gpu_old_vs_new.sh"),
            "--repo",
            str(repo),
            "--old-ref",
            "old",
            "--new-ref",
            "new",
            "-n",
            "2",
            "-m",
            model,
            "-p",
            policy,
            "--payload",
            str(out / "input_payload.json"),
            "-o",
            str(out),
            "--so-from",
            str(site),
            "--max-model-len",
            str(max_model_len),
            "--gpu-util",
            str(gpu_util),
        ]
    ).returncode
    # commits.txt names the scratch commits; record the real ones too.
    (out / "source_commits.txt").write_text(commits)
    # Package reports and logs for download, excluding the source worktrees.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in out.iterdir():
            if path.name not in ("src_old", "src_new"):
                tar.add(path, arcname=path.name)
    return status, buffer.getvalue()


@app.local_entrypoint()
def main(
    payload: str,
    old_ref: str = "05024f027",
    new_ref: str = "HEAD",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    policy: str = "round_robin",
    max_model_len: int = 4096,
    gpu_util: float = 0.85,
    out: str = "",
):
    # Resolve the local checkout used to archive refs and read runner files.
    root = Path(__file__).resolve().parents[3]

    def archive(ref):
        # Resolve the requested ref to its exact commit before archiving it.
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--verify", f"{ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # Archive committed sources as compressed bytes for the container.
        data = subprocess.run(
            ["git", "-C", root, "archive", "--format=tar.gz", sha],
            check=True,
            capture_output=True,
        ).stdout
        return sha, data

    # Snapshot both refs and record the original commit identities.
    old_sha, old_tar = archive(old_ref)
    new_sha, new_tar = archive(new_ref)
    commits = f"Old commit: {old_sha}\nNew commit: {new_sha}\n"
    print(commits, end="")
    # Send current runner files and the payload to the remote GPU benchmark.
    runner = {name: (root / BENCH / name).read_bytes() for name in RUNNER_FILES}
    status, report = run.remote(
        old_tar,
        new_tar,
        runner,
        Path(payload).read_bytes(),
        commits,
        model,
        policy,
        max_model_len,
        gpu_util,
    )
    # Extract returned artifacts into the chosen or timestamped local directory.
    target = Path(out or f"kalypso_modal_run/{time.strftime('%Y%m%d-%H%M%S')}")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(report)) as tar:
        tar.extractall(target, filter="data")
    # Show the artifact location and propagate any remote benchmark failure.
    print(f"Saved results to {target} (check exit {status})")
    if status:
        sys.exit(status)
