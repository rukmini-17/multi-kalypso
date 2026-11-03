"""Engine-free old Kalypso route and discovery-only workers for CI."""

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse


def trace_calls(executor_class, path):
    """Observe the real executor without changing prompts, replies, or routing."""
    execute = executor_class.execute
    Path(path).write_text("")

    async def traced(
        self, raw_request, prompt, max_tokens, pin=False, priority=0, determinism=None
    ):
        canonical = (
            prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        )
        result = await execute(
            self,
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=max_tokens,
            pin=pin,
            priority=priority,
            determinism=determinism,
        )
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "request_id": result.request_id,
                        "prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "reply": result.text,
                        "pin": pin,
                    }
                )
                + "\n"
            )
        return result

    executor_class.execute = traced


def build_app(args) -> FastAPI:
    import vllm.kalypso
    from vllm.kalypso import interface, query
    from vllm.kalypso.execution.determinism import skip_inference

    @asynccontextmanager
    async def lifespan(app):
        print(f"vllm.kalypso: {vllm.kalypso.__file__}", flush=True)
        loaded_path = Path(vllm.kalypso.__file__).resolve()
        if not loaded_path.is_relative_to(Path(args.expect_root).resolve()):
            raise RuntimeError(
                f"Loaded {loaded_path} outside expected checkout {args.expect_root}"
            )
        if not skip_inference():
            raise RuntimeError("Stand-ins require KALYPSO_SKIP_INFERENCE=1 and a seed")
        if args.command == "old":
            from vllm.kalypso.execution.vllm_executor import VLLMExecutor
            from vllm.kalypso.query_processor import QueryProcessor

            if args.call_record_path:
                trace_calls(VLLMExecutor, args.call_record_path)
            app.state.query_processor = QueryProcessor(args.model, args.budget)
        yield

    app = FastAPI(lifespan=lifespan)
    if args.command == "worker":

        @app.get("/v1/semantic/kv_budget")
        async def kv_budget():
            return {"model": args.model, "budget_bytes": args.budget}

        @app.get("/v1/models")
        async def models():
            return {"data": [{"id": args.model}]}

        @app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            return ""

        @app.post("/v1/completions")
        @app.post("/v1/chat/completions")
        async def inference():
            raise HTTPException(500, "Inference was called during a skip-inference run")

        return app

    @app.get("/v1/semantic/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/semantic/query")
    async def semantic_query(req: interface.SemanticQueryRequest, request: Request):
        if any(op.get("pin") for op in req.ops):
            raise HTTPException(
                400,
                "Pinned ops need the engine and are not supported "
                "in skip-inference runs",
            )
        started = asyncio.get_running_loop().time()
        _query = query.Query(req.ops, req.data_path)
        processor = request.app.state.query_processor
        try:
            out_ctxs = await processor.execute(request, _query, blocking=req.blocking)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
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
                    "input_context": getattr(ctx.input, "data", None),
                    "right_input_context": getattr(ctx.input, "right_data", None),
                    "outputs": outputs,
                }
            )
        elapsed = asyncio.get_running_loop().time() - started
        return {
            "request_id": uuid.uuid4().hex,
            "model_name": processor.model_name,
            "data_path": req.data_path,
            "ops": req.ops,
            "blocking": req.blocking,
            "num_output_rows": len(out_ctxs),
            "latency_sec": round(elapsed, 3),
            "results": results,
        }

    return app


if __name__ == "__main__":
    # File execution must not let benchmark modules shadow standard imports.
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)
    if sys.argv[1:2] == ["coordinator"]:
        from vllm.kalypso import server
        from vllm.kalypso.workers.http_executor import HttpVLLMExecutor

        # Consume only the observer option; the real coordinator parses the rest.
        observer = argparse.ArgumentParser(add_help=False)
        observer.add_argument("--call-record-path", required=True)
        args, remaining = observer.parse_known_args(sys.argv[2:])
        trace_calls(HttpVLLMExecutor, args.call_record_path)
        sys.argv[1:] = remaining
        server.main()
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("old", "worker"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--expect-root", required=True)
    parser.add_argument("--call-record-path")
    args = parser.parse_args()
    uvicorn.run(build_app(args), host="127.0.0.1", port=args.port)
