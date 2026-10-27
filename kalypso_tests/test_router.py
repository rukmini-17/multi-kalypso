import asyncio
import importlib.util
import pathlib
from types import SimpleNamespace as NS

_root = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso/workers"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, _root / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


policy = _load("kalypso_policy", "policy.py")
router_mod = _load("kalypso_router", "router.py")


class FakeExec:
    def __init__(self, name, log, gate=None):
        self.name, self.log, self.gate, self.fail = name, log, gate, False
        self.determinism = []

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0, determinism=None):
        self.log.append(self.name)
        self.determinism.append(determinism)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("boom")
        return NS(text="x")


class FakeWorker:
    def __init__(self, name, executor, cap):
        self.name, self.executor = name, executor
        self.manager = NS(capacity=lambda: cap)


class FakePool:
    def __init__(self, workers):
        self.workers = workers

    def manager(self, w):
        return w.manager


def make(cap=100, pol=None, gates=None, budget=80):
    log, gates = [], gates or {}
    workers = [FakeWorker(f"w{i}", FakeExec(f"w{i}", log, gates.get(f"w{i}")), cap) for i in range(2)]
    r = router_mod.CallRouter(FakePool(workers), pol or policy.RoundRobinPolicy(),
                              lambda prompt, max_tokens: budget)
    return r, log


def test_round_robin_alternates_per_call():
    async def go():
        r, log = make()
        for _ in range(4):
            await r.execute(None, "p", 8)
        assert log == ["w0", "w1", "w0", "w1"]
    asyncio.run(go())


def test_execute_passes_determinism_to_the_chosen_worker():
    async def go():
        r, log = make()
        strategy = object()
        await r.execute(None, "p", 8, determinism=strategy)
        await r.execute(None, "p", 8)
        assert log == ["w0", "w1"]
        assert r.workers[0].executor.determinism == [strategy]
        assert r.workers[1].executor.determinism == [None]
    asyncio.run(go())


def test_skips_full_worker():
    async def go():
        gate = asyncio.Event()
        r, log = make(gates={"w0": gate})
        a = asyncio.create_task(r.execute(None, "p", 8))  # w0, held
        await asyncio.sleep(0.01)
        await r.execute(None, "p", 8)                      # w1
        await r.execute(None, "p", 8)                      # turn is w0 (full) -> w1
        assert log == ["w0", "w1", "w1"]
        gate.set()
        await a
    asyncio.run(go())


def test_waits_when_all_full_then_proceeds():
    async def go():
        gate = asyncio.Event()
        r, log = make(gates={"w0": gate, "w1": gate})
        a = asyncio.create_task(r.execute(None, "p", 8))
        b = asyncio.create_task(r.execute(None, "p", 8))
        await asyncio.sleep(0.01)
        c = asyncio.create_task(r.execute(None, "p", 8))
        await asyncio.sleep(0.01)
        assert log == ["w0", "w1"] and not c.done()
        gate.set()
        await asyncio.gather(a, b, c)
        assert len(log) == 3
    asyncio.run(go())


def test_oversized_call_runs_on_idle_worker():
    async def go():
        r, log = make(budget=1000)  # larger than any worker's capacity
        await r.execute(None, "p", 8)
        assert log == ["w0"]
    asyncio.run(go())


def test_least_outstanding_picks_idle_worker():
    async def go():
        gate = asyncio.Event()
        r, log = make(cap=1000, pol=policy.LeastOutstandingPolicy(), gates={"w0": gate})
        a = asyncio.create_task(r.execute(None, "p", 8))  # tie -> w0, held
        await asyncio.sleep(0.01)
        await r.execute(None, "p", 8)                      # w1
        await r.execute(None, "p", 8)                      # w0 busy -> w1 again
        assert log == ["w0", "w1", "w1"]
        gate.set()
        await a
    asyncio.run(go())


def test_reservation_released_on_error():
    async def go():
        r, log = make()
        r.workers[0].executor.fail = True
        try:
            await r.execute(None, "p", 8)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
        assert r.inflight == {"w0": 0, "w1": 0} and r.used == {"w0": 0, "w1": 0}
    asyncio.run(go())


def test_unpin_and_abort_not_supported():
    async def go():
        r, _ = make()
        for fn in (r.unpin, r.abort):
            try:
                await fn(None, "id")
                assert False, "expected NotImplementedError"
            except NotImplementedError:
                pass
    asyncio.run(go())


def test_row_affinity_policy_is_sticky():
    assert policy.make_policy("row_affinity").sticky is True
    assert policy.make_policy("round_robin").sticky is False
    assert policy.make_policy("least_outstanding").sticky is False


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\n{len(tests)} tests passed")