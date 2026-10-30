import signal
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark_multi_kalypso import servers
from benchmark_multi_kalypso.config import Instrumentation, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_worker_and_coordinator_argv_and_scrubbed_environment(tmp_path, monkeypatch):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/m3_sweep.yaml")
    worker = servers.worker_command(config, 8100, ROOT)
    assert worker[1:] == [
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--host",
        "127.0.0.1",
        "--port",
        "8100",
        "--enable-prefix-caching",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.95",
    ]
    coordinator = servers.coordinator_command(
        config, "round_robin", ["http://w0", "http://w1"], 8000, tmp_path
    )
    assert coordinator[1:] == [
        "-m",
        "vllm.kalypso.server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--workers",
        "http://w0,http://w1",
        "--policy",
        "round_robin",
        "--record-path",
        str(tmp_path / "requests.jsonl"),
        "--poll-interval",
        "0.5",
    ]
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "123")
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", "1")
    for gpu in (None, "2"):
        env = servers.child_env(ROOT, gpu)
        assert "KALYPSO_COIN_FLIP_SEED" not in env
        assert "KALYPSO_SKIP_INFERENCE" not in env
        assert env["PYTHONPATH"].startswith(str(ROOT))
        if gpu is not None:
            assert env["CUDA_VISIBLE_DEVICES"] == "2"
            assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_optional_instrumentation_flags_are_omitted(tmp_path):
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(
        config,
        instrumentation=Instrumentation(record_requests=False, capture_kv_stats=False),
    )
    assert "--kv-cache-metrics-sample" not in servers.worker_command(config, 8100, ROOT)
    argv = servers.coordinator_command(
        config, "round_robin", ["http://w"], 8000, tmp_path
    )
    assert "--record-path" not in argv
    assert "--poll-interval" not in argv


def test_explicit_kv_sampling_adds_worker_metrics_flags():
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/m3_sweep.yaml")
    config = replace(
        config, instrumentation=Instrumentation(kv_cache_metrics_sample=1.0)
    )
    assert servers.worker_command(config, 8100, ROOT)[-3:] == [
        "--kv-cache-metrics",
        "--kv-cache-metrics-sample",
        "1.0",
    ]


def test_subprocess_uses_own_session_and_readiness_endpoint(tmp_path, monkeypatch):
    popen = Mock(return_value=SimpleNamespace(pid=123, poll=lambda: None))
    monkeypatch.setattr(servers.subprocess, "Popen", popen)
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(servers.socket, "socket", Mock(return_value=probe))
    server = servers.start(
        ["python", "server"], {"env": "value"}, tmp_path, tmp_path / "server.log", 8000
    )
    assert popen.call_args.kwargs["start_new_session"] is True
    assert popen.call_args.kwargs["stderr"] == subprocess.STDOUT
    get = Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(servers.requests, "get", get)
    servers.wait_ready(server, "/v1/models", 10)
    assert get.call_args.args == ("http://127.0.0.1:8000/v1/models",)
    server.log.close()


def test_stop_kills_group_even_if_parent_has_exited(monkeypatch):
    killpg = Mock()
    monkeypatch.setattr(servers.os, "killpg", killpg)
    server = servers.Server(Mock(pid=42), Mock(), "http://w")
    servers.stop(server)
    assert killpg.call_args_list[0].args == (42, signal.SIGTERM)
    assert killpg.call_args_list[1].args == (42, signal.SIGKILL)
    server.log.close.assert_called_once()


def test_partial_worker_startup_failure_cleans_started_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "WORKER_BASE_PORT", 9100)
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    worker = object()
    monkeypatch.setattr(
        servers, "start", Mock(side_effect=[worker, RuntimeError("port busy")])
    )
    stop = Mock()
    monkeypatch.setattr(servers, "stop", stop)
    with pytest.raises(RuntimeError, match="port busy"):
        servers.start_workers(config, ["0", "1"], tmp_path, ROOT)
    stop.assert_called_once_with(worker)
    assert servers.start.call_args_list[0].args[-1] == 9100
    assert servers.start.call_args_list[1].args[-1] == 9101
    assert servers.worker_url(1) == "http://127.0.0.1:9101"


def test_coordinator_uses_shared_port_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "COORDINATOR_PORT", 9000)
    start, ready = Mock(), Mock()
    monkeypatch.setattr(servers, "start", start)
    monkeypatch.setattr(servers, "wait_ready", ready)
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    server = servers.start_coordinator(config, "round_robin", [], tmp_path, ROOT)
    assert start.call_args.args[-1] == 9000
    assert start.call_args.args[0][5:7] == ["--port", "9000"]
    ready.assert_called_once_with(server, "/v1/semantic/healthz", 120)
