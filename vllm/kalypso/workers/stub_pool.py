"""Temporary worker pool for developing against until the real WorkerPool exists.

No vllm imports, so it can be unit-tested without a GPU.

Interface the rest of the code relies on (a real WorkerPool should match it):
    pool.workers         ordered list of workers
    pool.manager(w)      that worker's KVMemoryManager
    pool.managers()      all managers, in worker order
    pool.outstanding(w)  admitted, unfinished tasks on worker w
    w.name, w.executor
"""


class Worker:
    def __init__(self, name, executor, manager, url=None):
        self.name = name
        self.executor = executor
        self.manager = manager
        self.url = url

    def __repr__(self):
        return f"Worker({self.name})"


class StubPool:
    def __init__(self, workers):
        workers = list(workers)
        if not workers:
            raise ValueError("a pool needs at least one worker")
        names = [w.name for w in workers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate worker names: {names}")
        self.workers = workers

    @classmethod
    def single(cls, executor, manager, name="w0"):
        """One worker wrapping the existing singleton manager and executor."""
        return cls([Worker(name, executor, manager)])

    @classmethod
    def simulated(cls, n, executor, manager_factory):
        """n logical workers sharing one real engine, each with its own budget.

        manager_factory() must return a fresh KVMemoryManager each call.
        """
        return cls([Worker(f"w{i}", executor, manager_factory()) for i in range(n)])

    def manager(self, w):
        return w.manager

    def managers(self):
        return [w.manager for w in self.workers]

    def outstanding(self, w):
        return w.manager.total_inflight()