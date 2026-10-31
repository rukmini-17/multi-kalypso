"""Per-call worker routing: each LLM call goes to a worker chosen at call time.

No vllm imports, so it can be unit-tested without a GPU.

CallRouter is an executor (same execute/complete/chatcomplete interface as
VLLMExecutor). For every call it asks the policy for the worker order, reserves
the call's estimated KV bytes on the first worker with room, and waits if none
has room. An idle worker always accepts, so an oversized call still progresses.
"""

import asyncio
import contextvars

# Estimated KV bytes the router reserved for the call being made right now.
# The executor reads it so the request record can log it. It lives here, with
# no vllm imports, so the router stays unit-testable.
_CALL_COST = contextvars.ContextVar("kalypso_call_cost", default=None)


def current_call_cost():
    return _CALL_COST.get()


class CallRouter:
    def __init__(self, pool, policy, budget_fn):
        self.policy = policy
        self.budget_fn = budget_fn  # (prompt, max_tokens) -> bytes
        self.workers = list(pool.workers)
        self.capacity = {w.name: pool.manager(w).capacity() for w in self.workers}
        self.used = {w.name: 0 for w in self.workers}      # reserved bytes
        self.inflight = {w.name: 0 for w in self.workers}  # calls in flight
        self.calls = {w.name: 0 for w in self.workers}     # calls routed so far
        self._cond = asyncio.Condition()

    # The policy sees the router as its "pool".
    def outstanding(self, w):
        return self.inflight[w.name]
    
    def cost(self, w):
        return self.used[w.name]  # reserved bytes, which is what cost_aware ranks by

    def _try_reserve(self, budget):
        for w in self.policy.rank(self, None):
            n = w.name
            if self.inflight[n] == 0 or self.used[n] + budget <= self.capacity[n]:
                self.used[n] += budget
                self.inflight[n] += 1
                self.calls[n] += 1
                self.policy.on_assigned(w, self)
                return w
        return None

    async def _route(self, method, raw_request, prompt, max_tokens, pin, priority, **extra):
        budget = self.budget_fn(prompt, max_tokens)
        async with self._cond:
            w = self._try_reserve(budget)
            while w is None:
                await self._cond.wait()
                w = self._try_reserve(budget)
        token = _CALL_COST.set(budget)
        try:
            return await getattr(w.executor, method)(
                raw_request=raw_request, prompt=prompt, max_tokens=max_tokens,
                pin=pin, priority=priority, **extra,
            )
        finally:
            _CALL_COST.reset(token)
            async with self._cond:
                self.used[w.name] -= budget
                self.inflight[w.name] -= 1
                self._cond.notify_all()

    async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0, determinism=None):
        # The chosen worker's executor applies determinism to its reply.
        return await self._route(
            "execute", raw_request, prompt, max_tokens, pin, priority, determinism=determinism
        )

    async def complete(self, *, raw_request, prompt, max_tokens, pin=False, priority=0):
        return await self._route("complete", raw_request, prompt, max_tokens, pin, priority)

    async def chatcomplete(self, raw_request, prompt, max_tokens, pin=False, priority=0):
        return await self._route("chatcomplete", raw_request, prompt, max_tokens, pin, priority)

    async def unpin(self, raw_request, request_id):
        raise NotImplementedError("pinning is off; per-call routing has nothing to unpin")

    async def abort(self, raw_request, request_id):
        raise NotImplementedError("abort is not supported by per-call routing")

    def stats(self):
        return {"workers": [
            {"name": w.name, "inflight": self.inflight[w.name], "used": self.used[w.name],
             "capacity": self.capacity[w.name], "calls": self.calls[w.name]}
            for w in self.workers
        ]}