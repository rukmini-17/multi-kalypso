import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import requests

from .config import Experiment

WORKER_BASE_PORT = 8100
COORDINATOR_PORT = 8000


def worker_url(index: int) -> str:
    return f"http://127.0.0.1:{WORKER_BASE_PORT + index}"


@dataclass
class Server:
    process: subprocess.Popen
    log: IO
    url: str


def child_env(repo_root: Path, gpu: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("KALYPSO_COIN_FLIP_SEED", "KALYPSO_SKIP_INFERENCE"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def worker_command(config: Experiment, port: int, repo_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-prefix-caching",
        "--max-model-len",
        str(config.vllm.max_model_len),
        "--gpu-memory-utilization",
        str(config.vllm.gpu_memory_utilization),
    ]
    sample = config.instrumentation.kv_cache_metrics_sample
    if sample is not None:
        parser = (repo_root / "vllm/engine/arg_utils.py").read_text(encoding="utf-8")
        if '"--kv-cache-metrics-sample"' not in parser:
            raise ValueError(
                "This vLLM does not support kv_cache_metrics_sample; set it to null"
            )
        cmd += ["--kv-cache-metrics", "--kv-cache-metrics-sample", str(sample)]
    return cmd + config.vllm.extra_args


def coordinator_command(
    config: Experiment, policy: str, worker_urls: list[str], port: int, run_dir: Path
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.kalypso.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        ",".join(worker_urls),
        "--policy",
        policy,
    ]
    if config.instrumentation.record_requests:
        cmd += ["--record-path", str((run_dir / "requests.jsonl").resolve())]
    if config.instrumentation.poll_interval is not None:
        cmd += ["--poll-interval", str(config.instrumentation.poll_interval)]
    return cmd


def start(
    cmd: list[str], env: dict, repo_root: Path, log_path: Path, port: int
) -> Server:
    # Refuse an occupied port rather than accepting another server's health check.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    log = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    return Server(proc, log, f"http://127.0.0.1:{port}")


def stop(server: Server) -> None:
    # The group may still contain engine children even if the parent has exited.
    try:
        os.killpg(server.process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            server.process.wait(timeout=10)
        with suppress(ProcessLookupError):
            os.killpg(server.process.pid, signal.SIGKILL)
        server.process.wait(timeout=10)
    except ProcessLookupError:
        server.process.wait(timeout=10)
    finally:
        server.log.close()


def wait_ready(server: Server, endpoint: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"Server exited ({server.url}); see {server.log.name}")
        try:
            remaining = max(0.01, deadline - time.monotonic())
            if (
                requests.get(
                    server.url + endpoint, timeout=min(5, remaining)
                ).status_code
                == 200
            ):
                return
        except requests.RequestException:
            pass
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Server readiness timed out: {server.url}; see {server.log.name}"
    )


def start_workers(
    config: Experiment, gpus: list[str], out: Path, repo_root: Path
) -> list[Server]:
    workers = []
    try:
        for i, gpu in enumerate(gpus):
            port = WORKER_BASE_PORT + i
            workers.append(
                start(
                    worker_command(config, port, repo_root),
                    child_env(repo_root, gpu),
                    repo_root,
                    out / f"worker_{i}.log",
                    port,
                )
            )
        for worker in workers:
            wait_ready(worker, "/v1/models", config.timeouts.worker_ready_s)
        return workers
    except BaseException:
        for worker in reversed(workers):
            stop(worker)
        raise


def start_coordinator(
    config: Experiment,
    policy: str,
    workers: list[Server],
    run_dir: Path,
    repo_root: Path,
) -> Server:
    port = COORDINATOR_PORT
    cmd = coordinator_command(config, policy, [w.url for w in workers], port, run_dir)
    server = start(
        cmd, child_env(repo_root), repo_root, run_dir / "coordinator.log", port
    )
    try:
        wait_ready(server, "/v1/semantic/healthz", config.timeouts.coordinator_ready_s)
        return server
    except BaseException:
        stop(server)
        raise


def reset_prefix_caches(workers: list[Server]) -> None:
    for worker in workers:
        response = requests.post(worker.url + "/reset_prefix_cache", timeout=60)
        response.raise_for_status()


def post_query(url: str, body: dict, timeout: float) -> dict:
    response = requests.post(url + "/v1/semantic/query", json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()
