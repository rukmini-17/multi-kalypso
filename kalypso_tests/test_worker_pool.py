"""Tests for WorkerPool: construction checks, config parsing, discovery with
retries, and that Rukmini's policies and admit() work over it.

Runs without a GPU or the vllm package: pool.py and policy.py are loaded by
file path, and discovery runs against small fake vLLM servers.
"""
import asyncio
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest
from aiohttp import web

_root = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso/workers"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, _root / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


pool = _load("kalypso_pool", "pool.py")
policy = _load("kalypso_policy_for_pool", "policy.py")


class FakeManager:
    def __init__(self, inflight=0, capacity=100):
        self.inflight = inflight
        self._cap = capacity

    def total_inflight(self):
        return self.inflight

    def capacity(self):
        return self._cap


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- construction

def test_interface():
    ms = [FakeManager(3, 100), FakeManager(1, 300)]
    p = pool.WorkerPool([pool.Worker(f"w{i}", "exec", m) for i, m in enumerate(ms)])
    w0, w1 = p.workers
    assert p.manager(w1) is ms[1]
    assert p.managers() == ms
    assert p.outstanding(w0) == 3
    assert p.total_capacity() == 400


def test_rejects_bad_pools():
    with pytest.raises(ValueError):
        pool.WorkerPool([])
    with pytest.raises(ValueError, match="names"):
        pool.WorkerPool([pool.Worker("a", None, FakeManager()),
                         pool.Worker("a", None, FakeManager())])
    with pytest.raises(ValueError, match="urls"):
        pool.WorkerPool([pool.Worker("a", None, FakeManager(), url="http://x:1"),
                         pool.Worker("b", None, FakeManager(), url="http://x:1")])
    shared = FakeManager()
    with pytest.raises(ValueError, match="own KVMemoryManager"):
        pool.WorkerPool([pool.Worker("a", None, shared), pool.Worker("b", None, shared)])


def test_single_and_simulated():
    m = FakeManager()

    p = pool.WorkerPool.simulated(3, "exec", FakeManager)
    assert [w.name for w in p.workers] == ["w0", "w1", "w2"]
    assert len({id(m) for m in p.managers()}) == 3  # each worker has its own budget


# ---------------------------------------------------------------------- config

def test_parse_urls():
    specs = pool.parse_urls(" localhost:1/, http://b:2 ,")
    assert [(s.name, s.url) for s in specs] == [("w0", "http://localhost:1"), ("w1", "http://b:2")]
    with pytest.raises(ValueError):
        pool.parse_urls("http://a:1,a:1/")  # same worker written two ways
    with pytest.raises(ValueError):
        pool.parse_urls("")


def test_load_yaml_and_json(tmp_path):
    y = tmp_path / "workers.yaml"
    y.write_text("policy: least_outstanding\n"
                 "workers:\n"
                 "  - url: http://localhost:8001\n"
                 "    name: gpu0\n"
                 "  - localhost:8002\n")
    cfg = pool.load_config(y)
    assert cfg.policy == "least_outstanding"
    assert [(s.name, s.url) for s in cfg.workers] == [
        ("gpu0", "http://localhost:8001"), ("w1", "http://localhost:8002")]

    j = tmp_path / "workers.json"
    j.write_text(json.dumps({"workers": [{"url": "http://h:1"}]}))
    cfg = pool.load_config(j)
    assert cfg.policy is None and cfg.workers[0].url == "http://h:1"


@pytest.mark.parametrize("bad", [
    {},
    {"workers": "x"},
    {"workers": []},
    {"workers": [{"name": "a"}]},
    {"workers": [{"url": "a", "name": "n"}, {"url": "b", "name": "n"}]},
])
def test_bad_configs(bad):
    with pytest.raises(ValueError):
        pool.parse_config(bad)


# ------------------------------------------------------------------- discovery

class FakeVLLM:
    """Answers /v1/semantic/kv_budget like a Kalypso worker. The first
    `down_for` calls return 503, as if the model were still loading."""

    def __init__(self, model="qwen", budget=1000, down_for=0):
        self.model, self.budget, self.down_for = model, budget, down_for
        self.calls = 0
        self.app = web.Application()
        self.app.router.add_get("/v1/semantic/kv_budget", self.kv_budget)

    async def kv_budget(self, req):
        self.calls += 1
        if self.calls <= self.down_for:
            return web.Response(status=503)
        return web.json_response({"model": self.model, "budget_bytes": self.budget})

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def stop(self):
        await self.runner.cleanup()


async def discover(fakes, **kw):
    for f in fakes:
        await f.start()
    try:
        specs = pool.parse_urls([f.url for f in fakes])
        kw.setdefault("delay", 0.01)  # keep retries fast in tests
        return await pool.WorkerPool.discover(
            specs,
            executor_factory=lambda url, model, name: f"exec:{name}",
            manager_factory=lambda model, budget: FakeManager(capacity=budget),
            **kw,
        )
    finally:
        for f in fakes:
            await f.stop()


def test_discover_builds_one_worker_per_server():
    p = run(discover([FakeVLLM(budget=1000), FakeVLLM(budget=3000)]))
    assert [w.name for w in p.workers] == ["w0", "w1"]
    assert [p.manager(w).capacity() for w in p.workers] == [1000, 3000]
    assert p.total_capacity() == 4000
    assert all(w.model == "qwen" for w in p.workers)
    assert p.workers[1].executor == "exec:w1"


def test_discover_retries_until_worker_is_up():
    slow = FakeVLLM(down_for=3)
    p = run(discover([slow], retries=5))
    assert slow.calls == 4 and len(p.workers) == 1


def test_discover_gives_up_after_retries():
    dead = FakeVLLM(down_for=100)
    with pytest.raises(Exception):
        run(discover([dead], retries=3))
    assert dead.calls == 3


def test_discover_rejects_mixed_models():
    with pytest.raises(RuntimeError, match="same model"):
        run(discover([FakeVLLM(model="a"), FakeVLLM(model="b")]))


# ------------------------------------------------- Rukmini's policies over it

class FakeStage:
    def __init__(self, full=()):
        self.full = list(full)

    async def accept(self, task, manager):
        return None if any(manager is f for f in self.full) else 100


def make_task():
    return NS(ctx=NS(state=NS(worker=None, executor=None)))


def test_round_robin_over_worker_pool():
    p = pool.WorkerPool.simulated(3, "exec", FakeManager)
    rr = policy.RoundRobinPolicy()
    got = []
    for _ in range(6):
        t = make_task()
        assert run(policy.admit(FakeStage(), t, p, rr)) == 100
        got.append(t.ctx.state.worker.name)
    assert got == ["w0", "w1", "w2", "w0", "w1", "w2"]


def test_least_outstanding_skips_full_worker():
    ms = [FakeManager(0), FakeManager(5), FakeManager(2)]
    p = pool.WorkerPool([pool.Worker(f"w{i}", f"e{i}", m) for i, m in enumerate(ms)])
    t = make_task()
    stage = FakeStage(full=[ms[0]])  # least-loaded worker has no room
    assert run(policy.admit(stage, t, p, policy.LeastOutstandingPolicy())) == 100
    assert t.ctx.state.worker.name == "w2" and t.ctx.state.executor == "e2"