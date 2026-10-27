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


stub = _load("kalypso_stub_pool", "stub_pool.py")
policy = _load("kalypso_policy", "policy.py")


class FakeManager:
    def __init__(self, inflight=0):
        self.inflight = inflight

    def total_inflight(self):
        return self.inflight


class FakeStage:
    """Accepts unless the manager is in `full`; records calls."""
    def __init__(self, full=()):
        self.full = list(full)
        self.calls = []

    async def accept(self, task, manager):
        self.calls.append(manager)
        return None if any(manager is f for f in self.full) else 100


def make_task():
    return NS(ctx=NS(state=NS(worker=None, executor=None)))


def test_single_wraps_one_worker():
    m = FakeManager()
    pool = stub.StubPool.single("exec", m)
    assert len(pool.workers) == 1
    w = pool.workers[0]
    assert w.name == "w0" and w.executor == "exec"
    assert pool.manager(w) is m
    assert pool.managers() == [m]


def test_simulated_builds_distinct_managers():
    made = []

    def factory():
        made.append(FakeManager())
        return made[-1]

    pool = stub.StubPool.simulated(3, "exec", factory)
    assert [w.name for w in pool.workers] == ["w0", "w1", "w2"]
    assert len(made) == 3
    assert pool.managers() == made
    assert all(w.executor == "exec" for w in pool.workers)


def test_outstanding_reads_the_workers_manager():
    pool = stub.StubPool.simulated(2, "exec", FakeManager)
    w0, w1 = pool.workers
    w0.manager.inflight = 4
    assert pool.outstanding(w0) == 4
    assert pool.outstanding(w1) == 0


def test_rejects_empty_and_duplicate_names():
    for bad in ([], [stub.Worker("a", "e", None), stub.Worker("a", "e", None)]):
        try:
            stub.StubPool(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_works_with_least_outstanding_policy():
    pool = stub.StubPool.simulated(2, "exec", FakeManager)
    pool.workers[0].manager.inflight = 3
    pol = policy.LeastOutstandingPolicy()
    assert [w.name for w in pol.rank(pool)] == ["w1", "w0"]


def test_admit_with_stub_pool_sets_worker_and_per_worker_executor():
    pool = stub.StubPool.simulated(2, "shared-exec", FakeManager)
    stage, task = FakeStage(), make_task()
    budget = asyncio.run(policy.admit(stage, task, pool, policy.RoundRobinPolicy()))
    assert budget == 100
    assert task.ctx.state.worker is pool.workers[0]
    assert task.ctx.state.executor == "shared-exec"
    assert stage.calls[0] is pool.workers[0].manager


def test_admit_falls_through_to_second_worker_manager():
    pool = stub.StubPool.simulated(2, "exec", FakeManager)
    stage = FakeStage(full=[pool.workers[0].manager])
    task = make_task()
    asyncio.run(policy.admit(stage, task, pool, policy.RoundRobinPolicy()))
    assert task.ctx.state.worker is pool.workers[1]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\n{len(tests)} tests passed")