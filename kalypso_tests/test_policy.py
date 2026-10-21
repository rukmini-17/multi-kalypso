import importlib.util
import pathlib

# Load policy.py by file path so `import vllm` (and torch) never runs.
_path = pathlib.Path(__file__).resolve().parents[1] / "vllm/kalypso/workers/policy.py"
_spec = importlib.util.spec_from_file_location("kalypso_policy", _path)
policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policy)


class FakePool:
    def __init__(self, names, load=None):
        self.workers = list(names)
        self.load = dict(load or {})

    def outstanding(self, w):
        return self.load.get(w, 0)


def assign(pol, pool, full=()):
    """Mimic the admission loop: try workers in ranked order, skip full ones."""
    for w in pol.rank(pool, None):
        if w in full:
            continue
        pol.on_assigned(w, pool)
        return w
    return None


def test_round_robin_rank_rotates():
    pool = FakePool(["w0", "w1", "w2"])
    pol = policy.RoundRobinPolicy()
    assert pol.rank(pool) == ["w0", "w1", "w2"]
    pol.next = 1
    assert pol.rank(pool) == ["w1", "w2", "w0"]
    pol.next = 2
    assert pol.rank(pool) == ["w2", "w0", "w1"]


def test_round_robin_cycles():
    pool = FakePool(["w0", "w1"])
    pol = policy.RoundRobinPolicy()
    assert [assign(pol, pool) for _ in range(4)] == ["w0", "w1", "w0", "w1"]


def test_round_robin_skips_full_worker_and_resumes_after_acceptor():
    pool = FakePool(["w0", "w1", "w2"])
    pol = policy.RoundRobinPolicy()
    assert assign(pol, pool, full={"w0"}) == "w1"  # w0 full, w1 accepts
    assert assign(pol, pool) == "w2"               # next turn is after w1


def test_all_workers_full_returns_none():
    pool = FakePool(["w0", "w1"])
    pol = policy.RoundRobinPolicy()
    assert assign(pol, pool, full={"w0", "w1"}) is None


def test_empty_pool():
    assert policy.RoundRobinPolicy().rank(FakePool([])) == []
    assert policy.LeastOutstandingPolicy().rank(FakePool([])) == []


def test_least_outstanding_picks_least_loaded():
    pool = FakePool(["w0", "w1", "w2"], {"w0": 3, "w1": 1, "w2": 2})
    pol = policy.LeastOutstandingPolicy()
    assert pol.rank(pool) == ["w1", "w2", "w0"]


def test_least_outstanding_ties_break_by_index():
    pool = FakePool(["w0", "w1", "w2"])  # all zero
    pol = policy.LeastOutstandingPolicy()
    assert pol.rank(pool) == ["w0", "w1", "w2"]


def test_least_outstanding_follows_load_changes():
    pool = FakePool(["w0", "w1"], {"w0": 1, "w1": 1})
    pol = policy.LeastOutstandingPolicy()
    assert assign(pol, pool) == "w0"   # tie -> lowest index
    pool.load["w1"] = 0                # w1 finished its work
    assert assign(pol, pool) == "w1"


def test_least_outstanding_falls_back_when_best_is_full():
    pool = FakePool(["w0", "w1"], {"w0": 0, "w1": 5})
    pol = policy.LeastOutstandingPolicy()
    assert assign(pol, pool, full={"w0"}) == "w1"


def test_make_policy():
    assert isinstance(policy.make_policy("round_robin"), policy.RoundRobinPolicy)
    assert isinstance(policy.make_policy("least_outstanding"), policy.LeastOutstandingPolicy)
    try:
        policy.make_policy("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass

import asyncio
from types import SimpleNamespace as NS


class FakeWorker:
    def __init__(self, name):
        self.name = name
        self.executor = f"exec-{name}"


class FakeAdmitPool:
    def __init__(self, names, load=None):
        self.workers = [FakeWorker(n) for n in names]
        self.load = dict(load or {})

    def manager(self, w):
        return w.name  # stand-in for a per-worker KVMemoryManager

    def outstanding(self, w):
        return self.load.get(w.name, 0)


class FakeStage:
    """Accepts a task on a worker's manager unless that worker is marked full."""
    def __init__(self, full=()):
        self.full = set(full)
        self.calls = []

    async def accept(self, task, manager):
        self.calls.append(manager)
        return None if manager in self.full else 100


def make_task():
    return NS(ctx=NS(state=NS(worker=None, executor=None)))


def run(coro):
    return asyncio.run(coro)


def test_admit_assigns_worker_and_executor():
    pool = FakeAdmitPool(["w0", "w1"])
    stage, task = FakeStage(), make_task()
    assert run(policy.admit(stage, task, pool, policy.RoundRobinPolicy())) == 100
    assert task.ctx.state.worker.name == "w0"
    assert task.ctx.state.executor == "exec-w0"


def test_admit_round_robin_across_rows():
    pool, pol, stage = FakeAdmitPool(["w0", "w1"]), policy.RoundRobinPolicy(), FakeStage()
    tasks = [make_task() for _ in range(4)]
    for t in tasks:
        run(policy.admit(stage, t, pool, pol))
    assert [t.ctx.state.worker.name for t in tasks] == ["w0", "w1", "w0", "w1"]


def test_admit_falls_through_when_first_worker_full():
    pool = FakeAdmitPool(["w0", "w1"])
    stage, task = FakeStage(full={"w0"}), make_task()
    run(policy.admit(stage, task, pool, policy.RoundRobinPolicy()))
    assert task.ctx.state.worker.name == "w1"
    assert stage.calls == ["w0", "w1"]  # asked w0 first, then w1


def test_admit_returns_none_when_all_full_and_assigns_nothing():
    pool = FakeAdmitPool(["w0", "w1"])
    stage, task = FakeStage(full={"w0", "w1"}), make_task()
    assert run(policy.admit(stage, task, pool, policy.RoundRobinPolicy())) is None
    assert task.ctx.state.worker is None and task.ctx.state.executor is None


def test_admit_sticky_row_only_asks_its_own_worker():
    pool = FakeAdmitPool(["w0", "w1"])
    stage, task = FakeStage(), make_task()
    task.ctx.state.worker = pool.workers[1]  # row already lives on w1
    run(policy.admit(stage, task, pool, policy.RoundRobinPolicy()))
    assert stage.calls == ["w1"]
    assert task.ctx.state.worker.name == "w1"


def test_admit_sticky_row_waits_if_its_worker_is_full():
    pool = FakeAdmitPool(["w0", "w1"])
    stage, task = FakeStage(full={"w1"}), make_task()
    task.ctx.state.worker = pool.workers[1]
    assert run(policy.admit(stage, task, pool, policy.RoundRobinPolicy())) is None
    assert stage.calls == ["w1"]  # never moved to w0


def test_admit_least_outstanding_picks_idle_worker():
    pool = FakeAdmitPool(["w0", "w1"], {"w0": 2, "w1": 0})
    stage, task = FakeStage(), make_task()
    run(policy.admit(stage, task, pool, policy.LeastOutstandingPolicy()))
    assert task.ctx.state.worker.name == "w1"

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\n{len(tests)} tests passed")