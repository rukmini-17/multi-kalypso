"""Worker-selection policies for multi-worker Kalypso.

No vllm imports here, so these can be unit-tested without a GPU or torch.

A policy never admits anything itself. It only ranks the workers in the order
they should be tried. The caller tries each worker until one has memory, then
calls on_assigned() with the worker that actually accepted the row.

The pool is expected to provide:
    pool.workers         ordered list of workers
    pool.outstanding(w)  number of admitted, unfinished tasks on worker w
    pool.cost(w)         reserved KV bytes on worker w (only cost_aware uses it)
"""


class RoundRobinPolicy:
    name = "round_robin"
    sticky = False

    def __init__(self):
        self.next = 0  # index of the worker whose turn it is

    def rank(self, pool, task=None):
        ws = list(pool.workers)
        if not ws:
            return []
        start = self.next % len(ws)
        return ws[start:] + ws[:start]

    def on_assigned(self, worker, pool):
        # Advance from the worker that actually accepted the row, so a full
        # worker that was skipped does not cause the next row to land on the
        # worker that just took one.
        self.next = (pool.workers.index(worker) + 1) % len(pool.workers)


class LeastOutstandingPolicy:
    name = "least_outstanding"
    sticky = False

    def rank(self, pool, task=None):
        ws = list(pool.workers)
        # Ties break by worker index so the result is deterministic.
        return sorted(ws, key=lambda w: (pool.outstanding(w), ws.index(w)))

    def on_assigned(self, worker, pool):
        pass  # stateless: the load figure comes from the pool


class CostAwarePolicy:
    name = "cost_aware"
    sticky = False

    def rank(self, pool, task=None):
        ws = list(pool.workers)
        # Lowest reserved bytes first. Ties break by worker index so runs are reproducible.
        return sorted(ws, key=lambda w: (pool.cost(w), ws.index(w)))

    def on_assigned(self, worker, pool):
        pass  # stateless: the cost figure comes from the pool


class RowAffinityPolicy(RoundRobinPolicy):
    """Round robin for a row's first placement; every later call of the row stays there."""
    name = "row_affinity"
    sticky = True


POLICIES = {
    RoundRobinPolicy.name: RoundRobinPolicy,
    LeastOutstandingPolicy.name: LeastOutstandingPolicy,
    CostAwarePolicy.name: CostAwarePolicy,
    RowAffinityPolicy.name: RowAffinityPolicy,
}


def make_policy(name):
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")


async def admit(stage, task, pool, policy):
    """Pick a worker for a task and reserve memory on it.

    Returns the reserved budget, or None if no worker has room.
    `stage` only needs an async accept(task, manager) method that returns a
    budget on success and None (with no side effects) on failure.
    """
    state = task.ctx.state

    # Follow-up (retry, later operator): stay on the assigned worker.
    if state.worker is not None:
        return await stage.accept(task, pool.manager(state.worker))

    # Fresh row: try workers in the policy's order until one has room.
    for w in policy.rank(pool, task):
        budget = await stage.accept(task, pool.manager(w))
        if budget is not None:
            state.worker = w
            state.executor = w.executor  # operators call state.executor, so they need no edits
            policy.on_assigned(w, pool)
            return budget
    return None