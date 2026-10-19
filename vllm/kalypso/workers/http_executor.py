"""Executor that talks to one vLLM worker over HTTP.

Drop-in replacement for VLLMExecutor: same execute / complete / chatcomplete /
unpin / abort signatures, so operators call it exactly as before. One instance
is created per worker and stored in Worker.executor; admit() copies it into
ctx.state.executor.

Worker-side requirements (see the api_server.py patch):
  GET  /v1/semantic/kv_budget      used by the coordinator at startup
  POST /v1/semantic/kv/unpin       {"request_ids": [...]}
  POST /v1/semantic/kv/abort       {"request_id": "..."}
plus the standard /v1/completions and /v1/chat/completions.
"""

import asyncio
import json
import time

from vllm.kalypso.execution.determinism import Determinism, apply_determinism
from vllm.kalypso.execution.executor import LLMExecutor, CompletionResult
from vllm.kalypso.instrumentation import RequestRecorder, current_request_meta
from vllm.kalypso.pin_registry import PinnedRequestRegistry


class WorkerRequestError(RuntimeError):
    """A worker returned a non-2xx response or could not be reached."""


# request_id -> executor that owns the pin, so cleanup can reach the right
# worker without the registry having to know about workers.
_PIN_OWNERS: dict[str, "HttpVLLMExecutor"] = {}


class HttpVLLMExecutor(LLMExecutor):
    MAX_TOKENS_CAP = 16384

    def __init__(
        self,
        base_url: str,
        model: str,
        name: str = "w0",
        recorder: RequestRecorder | None = None,
        session=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.recorder = recorder
        self._session = session  # injectable for tests
        self._owns_session = session is None

    # -- plumbing ----------------------------------------------------------
    @staticmethod
    def _owner_key(raw_request) -> str:
        return str(id(raw_request))

    async def _get_session(self):
        if self._session is None:
            import aiohttp

            # No overall timeout (LLM calls can be long) and no connection cap:
            # concurrency is already bounded by Kalypso's KV admission.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None),
                connector=aiohttp.TCPConnector(limit=0),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None

    async def _post(self, path: str, payload: dict) -> dict:
        session = await self._get_session()
        try:
            async with session.post(self.base_url + path, json=payload) as resp:
                body = await resp.text()
                if resp.status >= 300:
                    raise WorkerRequestError(
                        f"{self.name} {path} -> HTTP {resp.status}: {body[:300]}"
                    )
                return json.loads(body)
        except WorkerRequestError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # connection refused, reset, bad JSON, ...
            raise WorkerRequestError(f"{self.name} {path} failed: {exc!r}") from exc

    # -- payloads (same sampling settings as VLLMExecutor) -------------------
    def _payload(self, key: str, prompt, max_tokens: int, pin: bool, priority: int):
        return {
            "model": self.model,
            key: prompt,
            "max_tokens": min(max_tokens, self.MAX_TOKENS_CAP),
            "temperature": 0.0,
            "seed": 42,
            "top_p": 1.0,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.3,
            "priority": priority,
            "vllm_xargs": {"pinned": pin},
            "stream": False,
        }

    # -- the one call path, with instrumentation -----------------------------
    async def _call(
        self, kind: str, raw_request, prompt, max_tokens: int, pin: bool, priority: int
    ) -> CompletionResult:
        path, key = (
            ("/v1/chat/completions", "messages")
            if kind == "chat"
            else ("/v1/completions", "prompt")
        )
        payload = self._payload(key, prompt, max_tokens, pin, priority)
        rec = self.recorder
        clock = rec.now if rec is not None else time.monotonic
        submit_t = clock()
        status, error, data, finish_reason = "ok", None, None, None
        try:
            data = await self._post(path, payload)
            choice = data["choices"][0]
            request_id = data["id"]
            text = choice["message"]["content"] if kind == "chat" else choice["text"]
            finish_reason = choice.get("finish_reason", "")
            if pin:
                PinnedRequestRegistry.instance().add(
                    request_id,
                    kind="chat" if kind == "chat" else "completion",
                    max_tokens=max_tokens,
                    owner_key=self._owner_key(raw_request),
                )
                _PIN_OWNERS[request_id] = self
            return CompletionResult(
                text=text, request_id=request_id, finish_reason=finish_reason
            )
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status, error = "error", repr(exc)
            raise
        finally:
            if rec is not None:
                usage = (data or {}).get("usage") or {}
                meta = current_request_meta()
                rec.record_request(
                    request_id=(data or {}).get("id"),
                    worker=self.name,
                    operator=meta.get("operator"),
                    stage=meta.get("stage"),
                    position=meta.get("position"),
                    row_idx=meta.get("row_idx"),
                    kind=kind,
                    pin=pin,
                    priority=priority,
                    submit_t=submit_t,
                    start_t=None,  # see instrumentation.py docstring
                    finish_t=clock(),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=finish_reason,
                    status=status,
                    error=error,
                )

    # -- LLMExecutor interface ------------------------------------------------
    async def execute(
        self,
        raw_request,
        prompt,
        max_tokens: int,
        pin: bool = False,
        priority: int = 0,
        determinism: Determinism | None = None,
    ) -> CompletionResult:
        kind = "completion" if type(prompt) is str else "chat"
        result = await self._call(kind, raw_request, prompt, max_tokens, pin, priority)
        if pin:
            return result
        return apply_determinism(prompt, result, determinism)

    async def complete(
        self, *, raw_request, prompt, max_tokens: int, pin: bool = False, priority: int = 0
    ) -> CompletionResult:
        return await self._call("completion", raw_request, prompt, max_tokens, pin, priority)

    async def chatcomplete(
        self, raw_request, prompt, max_tokens: int, pin: bool = False, priority: int = 0
    ) -> CompletionResult:
        return await self._call("chat", raw_request, prompt, max_tokens, pin, priority)

    async def unpin_many(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        await self._post("/v1/semantic/kv/unpin", {"request_ids": list(request_ids)})
        registry = PinnedRequestRegistry.instance()
        for rid in request_ids:
            registry.remove(rid)
            _PIN_OWNERS.pop(rid, None)

    async def unpin(self, raw_request, request_id: str) -> None:
        await self.unpin_many([request_id])

    async def abort(self, raw_request, request_id: str) -> None:
        await self._post("/v1/semantic/kv/abort", {"request_id": request_id})


async def unpin_owner(owner_key: str) -> int:
    """Unpin every request still pinned for one query, on whichever worker
    holds it. Returns how many were unpinned. Used for query-exit cleanup."""
    items = PinnedRequestRegistry.instance().list_by_owner(owner_key)
    by_exec: dict[HttpVLLMExecutor, list[str]] = {}
    for item in items:
        ex = _PIN_OWNERS.get(item["request_id"])
        if ex is not None:
            by_exec.setdefault(ex, []).append(item["request_id"])
    results = await asyncio.gather(
        *(ex.unpin_many(ids) for ex, ids in by_exec.items()), return_exceptions=True
    )
    done = 0
    for (ex, ids), res in zip(by_exec.items(), results):
        if not isinstance(res, Exception):
            done += len(ids)
    return done
