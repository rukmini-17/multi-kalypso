"""Tests for HttpVLLMExecutor + instrumentation against a fake vLLM worker.

Runs without a GPU or the vllm package: the few vllm.kalypso modules the
executor imports are loaded from their files under stub package names.
"""
import asyncio
import importlib.util
import json
import pathlib
import re
import sys
import types
from types import SimpleNamespace as NS

import pytest
from aiohttp import ClientConnectionError, web

_kal = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_under_stubs():
    names = [
        "vllm", "vllm.kalypso", "vllm.kalypso.execution",
        "vllm.kalypso.execution.executor", "vllm.kalypso.execution.determinism",
        "vllm.kalypso.instrumentation", "vllm.kalypso.workers",
        "vllm.kalypso.workers.http_executor",
    ]
    saved = {n: sys.modules.get(n) for n in names}
    for n in ("vllm", "vllm.kalypso", "vllm.kalypso.execution", "vllm.kalypso.workers"):
        sys.modules[n] = types.ModuleType(n)
    try:
        _load("vllm.kalypso.execution.executor", _kal / "execution/executor.py")
        det = _load("vllm.kalypso.execution.determinism", _kal / "execution/determinism.py")
        instr = _load("vllm.kalypso.instrumentation", _kal / "instrumentation.py")
        http = _load("vllm.kalypso.workers.http_executor", _kal / "workers/http_executor.py")
        return instr, http, det
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


instr, http, det = _import_under_stubs()

METRICS = """\
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 3.0
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 40.0
"""


class FakeWorker:
    def __init__(self):
        self.posts: list[dict] = []
        self.delay_s = 0
        self.n = 0
        app = web.Application()
        app.router.add_post("/v1/completions", self.completions)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_get("/metrics", self.metrics)
        self.app = app

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"

    async def stop(self):
        await self.runner.cleanup()

    def _next_id(self):
        self.n += 1
        return f"cmpl-{id(self)}-{self.n}"

    async def completions(self, req):
        body = await req.json()
        self.posts.append(body)
        await asyncio.sleep(self.delay_s)
        if body["prompt"] == "boom":
            return web.Response(status=500, text="engine exploded")
        return web.json_response({
            "id": self._next_id(),
            "choices": [{"text": "yes", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        })

    async def chat(self, req):
        body = await req.json()
        self.posts.append(body)
        await asyncio.sleep(self.delay_s)
        return web.json_response({
            "id": self._next_id(),
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    async def metrics(self, req):
        return web.Response(text=METRICS)


def run(coro):
    return asyncio.run(coro)


def test_completion_and_chat_roundtrip_with_records():
    async def go():
        fw = FakeWorker()
        await fw.start()
        rec = instr.RequestRecorder()
        ex = http.HttpVLLMExecutor(fw.url, "m", name="w1", recorder=rec)
        try:
            with instr.request_meta(stage=2, operator="SemFilter", position=3, row_idx=5):
                r1 = await ex.execute(raw_request=NS(), prompt="is it true?", max_tokens=8)
            r2 = await ex.execute(
                raw_request=NS(),
                prompt=[{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
            assert ex._session.timeout.total == 600
        finally:
            await ex.close()
            await fw.stop()
        assert (r1.text, r1.finish_reason) == ("yes", "stop")
        assert r2.text == "hello"
        # same sampling settings as VLLMExecutor
        assert fw.posts[0]["temperature"] == 0.0 and fw.posts[0]["seed"] == 42
        assert all("vllm_xargs" not in post for post in fw.posts)
        rows = rec.rows("request")
        assert len(rows) == 2
        a, b = rows
        assert (a["worker"], a["operator"], a["stage"], a["row_idx"]) == ("w1", "SemFilter", 2, 5)
        assert a["position"] == 3 and b["position"] is None
        assert (a["prompt_tokens"], a["completion_tokens"], a["status"]) == (7, 2, "ok")
        assert a["kind"] == "completion" and b["kind"] == "chat"
        assert b["operator"] is None  # no metadata set for the second call
        assert a["submit_t"] <= a["finish_t"] and a["start_t"] is None

    run(go())


def test_error_is_raised_and_recorded():
    async def go():
        fw = FakeWorker()
        await fw.start()
        rec = instr.RequestRecorder()
        ex = http.HttpVLLMExecutor(fw.url, "m", name="w0", recorder=rec)
        try:
            try:
                await ex.execute(raw_request=NS(), prompt="boom", max_tokens=4)
                raised = False
            except http.WorkerRequestError:
                raised = True
        finally:
            await ex.close()
            await fw.stop()
        assert raised
        (row,) = rec.rows("request")
        assert row["status"] == "error" and "500" in row["error"]

    run(go())


@pytest.mark.parametrize("chat", [False, True])
def test_timeout_reports_worker_elapsed_and_bounded_prompt(chat):
    async def go():
        fw = FakeWorker()
        fw.delay_s = 0.1
        await fw.start()
        rec = instr.RequestRecorder()
        ex = http.HttpVLLMExecutor(fw.url, "m", recorder=rec, timeout_s=0.01)
        text = "slow prompt " + "x" * 250 + "DO_NOT_INCLUDE"
        prompt = [{"role": "user", "content": text}] if chat else text
        try:
            with instr.request_meta(operator="SemFilter", row_idx=5):
                with pytest.raises(http.WorkerRequestError) as caught:
                    await ex.execute(NS(), prompt, 4)
        finally:
            await ex.close()
            await fw.stop()
        message = str(caught.value)
        path = "/v1/chat/completions" if chat else "/v1/completions"
        assert fw.url + path in message
        elapsed = re.search(r"after ([0-9.]+)s", message)
        assert elapsed and float(elapsed.group(1)) >= 0.01
        excerpt = message.split("prompt=", 1)[1]
        expected = (json.dumps(prompt) if chat else prompt)[:200]
        assert excerpt == repr(expected)
        assert "DO_NOT_INCLUDE" not in message
        assert isinstance(caught.value.__cause__, asyncio.TimeoutError)
        (row,) = rec.rows("request")
        assert (row["status"], row["operator"], row["row_idx"]) == ("error", "SemFilter", 5)
        assert row["error"] == repr(caught.value)

    run(go())


def test_connection_failure_reports_context_and_original_cause():
    async def go():
        cause = ClientConnectionError("connection reset")

        class BrokenSession:
            def post(self, *args, **kwargs):
                raise cause

        ex = http.HttpVLLMExecutor("http://worker:8000", "m", session=BrokenSession())
        with pytest.raises(http.WorkerRequestError) as caught:
            await ex.execute(NS(), "hello worker", 4)
        message = str(caught.value)
        assert "http://worker:8000/v1/completions" in message
        assert re.search(r"after [0-9.]+s", message)
        assert "prompt='hello worker'" in message
        assert caught.value.__cause__ is cause

    run(go())


@pytest.mark.parametrize("method", ["execute", "complete", "chatcomplete", "unpin", "abort"])
def test_pinning_and_abort_are_not_supported(method):
    async def go():
        ex = http.HttpVLLMExecutor("http://worker:8000", "m")
        with pytest.raises(NotImplementedError):
            if method in ("unpin", "abort"):
                await getattr(ex, method)(NS(), "cmpl-1")
            else:
                await getattr(ex, method)(raw_request=NS(), prompt="hello", max_tokens=4, pin=True)
        assert ex._session is None

    run(go())


def test_concurrent_requests_keep_their_own_metadata():
    async def go():
        fw = FakeWorker()
        await fw.start()
        rec = instr.RequestRecorder()
        ex = http.HttpVLLMExecutor(fw.url, "m", name="w0", recorder=rec)

        async def one(i):
            with instr.request_meta(stage=i % 2, operator=f"Op{i}", row_idx=i):
                await asyncio.sleep(0.01 * (5 - i))  # finish out of order
                await ex.execute(raw_request=NS(), prompt=f"p{i}", max_tokens=4)

        try:
            await asyncio.gather(*(asyncio.create_task(one(i)) for i in range(5)))
        finally:
            await ex.close()
            await fw.stop()
        got = {r["row_idx"]: (r["operator"], r["stage"]) for r in rec.rows("request")}
        assert got == {i: (f"Op{i}", i % 2) for i in range(5)}

    run(go())


def test_poller_samples_queue_depth_and_totals(tmp_path):
    async def go():
        fw = FakeWorker()
        await fw.start()
        path = tmp_path / "rec.jsonl"
        rec = instr.RequestRecorder(str(path))
        poller = instr.WorkerMetricsPoller(
            {"w0": fw.url, "w1": fw.url},
            rec,
            extra_metrics=["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"],
        )
        try:
            await poller.poll_once()
        finally:
            await poller.stop()
            await fw.stop()
            rec.close()
        samples = rec.rows("queue_sample")
        assert sorted(s["worker"] for s in samples) == ["w0", "w1"]
        assert all(s["waiting"] == 3.0 and s["running"] == 2.0 for s in samples)
        tot = poller.totals(["vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"])
        assert tot == {"vllm:prefix_cache_queries_total": 200.0, "vllm:prefix_cache_hits_total": 80.0}
        assert poller.has("vllm:prefix_cache_hits_total")
        assert not poller.has("vllm:kv_block_lifetime_seconds_count")
        # JSONL on disk: header + 2 samples
        assert len(path.read_text().strip().splitlines()) == 3

    run(go())
