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
from vllm.kalypso.workers.stub_pool import StubPool, Worker


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


async def _fetch_worker_info(session, url: str) -> dict:
    async with session.get(f"{url}/v1/semantic/kv_budget") as resp:
        resp.raise_for_status()
        return await resp.json()


def build_app(args) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import aiohttp

        urls = [u.strip().rstrip("/") for u in args.workers.split(",") if u.strip()]
        if not urls:
            raise ValueError("--workers needs at least one URL")

        recorder = RequestRecorder(args.record_path)

        # Worker discovery / health check: each worker must answer kv_budget.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            infos = await asyncio.gather(*(_fetch_worker_info(session, u) for u in urls))

        models = {i["model"] for i in infos}
        if len(models) != 1:
            raise RuntimeError(f"workers must serve the same model, got {sorted(models)}")
        model_name = models.pop()

        from vllm.kalypso.budget import KVMemoryManager

        workers = []
        for i, (url, info) in enumerate(zip(urls, infos)):
            name = f"w{i}"
            executor = HttpVLLMExecutor(url, model_name, name=name, recorder=recorder)
            manager = KVMemoryManager(model_name, info["budget_bytes"], torch.float16)
            workers.append(Worker(name, executor, manager, url=url))
        pool = StubPool(workers)

        poller = WorkerMetricsPoller(
            {w.name: w.url for w in workers},
            recorder,
            interval=args.poll_interval,
            extra_metrics=[
                *QueryProcessor._KV_METRICS.values(),
                *QueryProcessor._LLM_WORK_METRICS.values(),
            ],
        )

        total_budget = sum(i["budget_bytes"] for i in infos)
        qp = CoordinatorQueryProcessor(
            model_name=model_name,
            budget=total_budget,
            pool=pool,
            policy=make_policy(args.policy),
            poller=poller,
        )
        qp.executor = workers[0].executor  # only used to seed rows before admit()

        app.state.query_processor = qp
        app.state.pool = pool
        app.state.recorder = recorder
        app.state.poller = poller
        await poller.poll_once()
        poller.start()
        print(
            f"[Coordinator] {len(workers)} workers, model={model_name}, "
            f"policy={args.policy}, records={args.record_path}"
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
    p.add_argument("--workers", required=True, help="comma-separated worker base URLs")
    p.add_argument("--policy", default="round_robin")
    p.add_argument("--record-path", default=None, help="JSONL file for instrumentation rows")
    p.add_argument("--poll-interval", type=float, default=0.5)
    args = p.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
