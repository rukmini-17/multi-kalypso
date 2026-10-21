"""Standalone Kalypso coordinator.

Runs the semantic query processor in its own HTTP server and sends every LLM
request to one of N separate `vllm serve` processes.

    # workers (each its own process / GPU), started with the Kalypso vLLM fork:
    VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve <model> --port 8001 --enable-prefix-caching
    VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve <model> --port 8002 --enable-prefix-caching

    # coordinator:
    python -m vllm.kalypso.server --port 8000 \
        --workers http://localhost:8001,http://localhost:8002 \
        --policy round_robin --record-path runs/requests.jsonl

Clients keep calling POST /v1/semantic/query on the coordinator exactly as
they called it on the single vLLM server.
"""

import argparse
import asyncio
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request

from vllm.kalypso import interface
from vllm.kalypso import query as query_mod
from vllm.kalypso.instrumentation import RequestRecorder, WorkerMetricsPoller
from vllm.kalypso.query_processor import QueryProcessor
from vllm.kalypso.workers.http_executor import HttpVLLMExecutor, unpin_owner
from vllm.kalypso.workers.policy import make_policy
from vllm.kalypso.workers.pool import WorkerPool, load_config, parse_urls
from vllm.kalypso.workers.router import CallRouter


class CoordinatorQueryProcessor(QueryProcessor):
    """QueryProcessor with the in-process vLLM engine calls replaced by HTTP.

    Three places in QueryProcessor reach into a local engine; each is
    overridden here:
      * _kv_metric_snapshot: read the local Prometheus registry. Now sums the
        counters most recently scraped from every worker's /metrics.
      * _cleanup_query_pins: called engine_core.unpin_requests locally. Now
        routes each pin to the worker that owns it.
      * start_stuck_monitor: polled the local scheduler. Disabled for now; the
        per-worker stuck monitor (Milestone 3) replaces it.
    """

    def __init__(self, *, poller: WorkerMetricsPoller, **kwargs):
        super().__init__(**kwargs)
        self.poller = poller

    # -- metrics -------------------------------------------------------------
    def _kv_metric_snapshot(self) -> dict[str, float]:
        names = set(self._KV_METRICS.values()) | set(self._LLM_WORK_METRICS.values())
        totals = self.poller.totals(names)
        snap = {key: totals[name] for key, name in self._KV_METRICS.items()}
        snap["eviction_metric_available"] = float(
            self.poller.has(self._KV_METRICS["evicted_blocks"])
        )
        snap.update({key: totals[name] for key, name in self._LLM_WORK_METRICS.items()})
        snap["llm_work_metrics_available"] = float(
            all(self.poller.has(n) for n in self._LLM_WORK_METRICS.values())
        )
        return snap

    async def execute(self, raw_request, query, blocking=None):
        await self.poller.poll_once()  # fresh "before" snapshot
        return await super().execute(raw_request, query, blocking=blocking)

    # -- pins ----------------------------------------------------------------
    async def _cleanup_query_pins(self, raw_request, owner_key: str):
        count = await unpin_owner(owner_key)
        if count:
            print(f"[Coordinator] query finished; unpinned {count} pinned requests")
        # Called right before the "after" KV snapshot, so refresh here.
        await self.poller.poll_once()

    def start_stuck_monitor(self, engine_client):
        pass  # TODO(Milestone 3): per-worker stuck monitor over HTTP





def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import aiohttp

        recorder = RequestRecorder(args.record_path)
        from vllm.kalypso.budget import KVMemoryManager

        cfg = load_config(args.config) if args.config else None
        specs = cfg.workers if cfg else parse_urls(args.workers)
        pool = await WorkerPool.discover(
            specs,
            executor_factory=lambda url, model, name: HttpVLLMExecutor(url, model, name=name, recorder=recorder),
            manager_factory=lambda model, budget: KVMemoryManager(model, budget, torch.float16),
        )
        workers = pool.workers
        model_name = workers[0].model

        poller = WorkerMetricsPoller(
            {w.name: w.url for w in workers},
            recorder,
            interval=args.poll_interval,
            extra_metrics=[
                *QueryProcessor._KV_METRICS.values(),
                *QueryProcessor._LLM_WORK_METRICS.values(),
            ],
        )

        policy = make_policy(args.policy or (cfg.policy if cfg else None) or "round_robin")
        total_budget = pool.total_capacity()
        if policy.sticky:
            # row_affinity: admit() places each row once; its later calls stay there.
            qp_pool, qp_policy, executor = pool, policy, workers[0].executor
            router = None
        else:
            # Per-call routing: tasks are admitted against the pool total,
            # and each LLM call picks its own worker.
            router = CallRouter(
                pool, policy,
                lambda prompt, max_tokens: (
                    KVMemoryManager.get_instance().token_length(prompt) + max_tokens
                ) * KVMemoryManager.get_instance().bytes_per_token,
            )
            qp_pool, qp_policy, executor = None, None, router

        qp = CoordinatorQueryProcessor(
            model_name=model_name,
            budget=total_budget,
            pool=qp_pool,
            policy=qp_policy,
            poller=poller,
        )
        qp.executor = executor
        app.state.router = router
        app.state.query_processor = qp
        app.state.pool = pool
        app.state.recorder = recorder
        app.state.poller = poller
        await poller.poll_once()
        poller.start()
        print(
            f"[Coordinator] {len(workers)} workers, model={model_name}, "
            f"policy={policy.name}, records={args.record_path}"
        )
        try:
            yield
        finally:
            await poller.stop()
            for w in workers:
                await w.executor.close()
            recorder.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/semantic/healthz")
    async def healthz(raw_request: Request):
        return {
            "status": "ok",
            "component": "kalypso_coordinator",
            "query_processor_initialized": hasattr(raw_request.app.state, "query_processor"),
        }

    @app.get("/v1/semantic/workers")
    async def workers_state(raw_request: Request):
        router = getattr(raw_request.app.state, "router", None)
        if router is not None:
            return router.stats()
        pool = raw_request.app.state.pool
        poller = raw_request.app.state.poller
        return {
            "workers": [
                {
                    "name": w.name,
                    "url": w.url,
                    "inflight": pool.manager(w).total_inflight(),
                    "capacity": pool.manager(w).capacity(),
                    "engine_waiting": poller.latest.get(w.name, {}).get(poller.WAITING),
                    "engine_running": poller.latest.get(w.name, {}).get(poller.RUNNING),
                    "scrape_failures": poller.failures.get(w.name, 0),
                }
                for w in pool.workers
            ]
        }

    @app.post("/v1/semantic/query")
    async def semantic_query(sem_request: interface.SemanticQueryRequest, raw_request: Request):
        started = asyncio.get_running_loop().time()
        _query = query_mod.Query(sem_request.ops, sem_request.data_path)
        processor = raw_request.app.state.query_processor
        out_ctxs = await processor.execute(
            raw_request, _query, blocking=sem_request.blocking
        )

        results = []
        for ctx in out_ctxs:
            outputs = []
            for item in ctx.output:
                if isinstance(item, dict) and len(item) == 1:
                    op_name, value = next(iter(item.items()))
                    outputs.append({"op": op_name, "value": value})
                else:
                    outputs.append({"op": "unknown", "value": item})
            results.append(
                {
                    "idx": getattr(ctx.state, "idx", None),
                    "worker": getattr(getattr(ctx.state, "worker", None), "name", None),
                    "input_context": getattr(ctx.input, "data", None),
                    "right_input_context": getattr(ctx.input, "right_data", None),
                    "outputs": outputs,
                }
            )

        elapsed = asyncio.get_running_loop().time() - started
        return {
            "request_id": uuid.uuid4().hex,
            "model_name": processor.model_name,
            "data_path": sem_request.data_path,
            "ops": sem_request.ops,
            "blocking": sem_request.blocking,
            "num_output_rows": len(out_ctxs),
            "latency_sec": round(elapsed, 3),
            "results": results,
        }

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(description="Kalypso coordinator (multi-worker)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--workers", help="comma-separated worker base URLs")
    src.add_argument("--config", help="YAML/JSON worker pool config")
    p.add_argument("--policy", default=None)
    p.add_argument("--record-path", default=None, help="JSONL file for instrumentation rows")
    p.add_argument("--poll-interval", type=float, default=0.5)
    args = p.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
