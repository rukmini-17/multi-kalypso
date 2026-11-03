# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.execution.determinism import (
    BOOLEAN,
    Bucket,
    Choice,
    Text,
    apply_determinism,
    check_determinism,
)
from vllm.kalypso.execution.executor import CompletionResult
from vllm.kalypso.execution.vllm_executor import VLLMExecutor


def test_literal_seeded_replies(monkeypatch):
    result = CompletionResult("real reply", "req-1", "length")
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    assert apply_determinism("hello", result, BOOLEAN).text == "true"
    assert apply_determinism("hello", result, Bucket()).text == "bucket 4"
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    assert apply_determinism("hello", result, BOOLEAN).text == "false"
    assert apply_determinism("hello", result, Text()).text == "79057cc0"
    assert result.text == "real reply"


def test_chat_order_and_call_order_do_not_change_reply(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "7")
    result = CompletionResult("x", "id", "stop")
    first = apply_determinism([{"role": "user", "content": "hi"}], result, Text())
    apply_determinism("other call", result, BOOLEAN)
    second = apply_determinism([{"content": "hi", "role": "user"}], result, Text())
    assert first.text == second.text


def test_choice_returns_every_option(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "7")
    result = CompletionResult("x", "id", "stop")
    choices = Choice(["red", "blue"])
    assert {apply_determinism(str(i), result, choices).text for i in range(100)} == {
        "red",
        "blue",
    }


def test_seeded_call_without_strategy_raises(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "7")
    result = CompletionResult("x", "id", "stop")
    with pytest.raises(RuntimeError, match="no determinism strategy"):
        apply_determinism("hi", result, None)


@pytest.mark.parametrize("strategy", [lambda: Choice([]), lambda: Bucket(0)])
def test_strategy_rejects_empty_space(strategy):
    with pytest.raises(ValueError):
        strategy()


@pytest.mark.parametrize("skip", ["0", "1"])
def test_pin_call_reply_is_left_alone(monkeypatch, skip):
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", skip)
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "3")
    executor = VLLMExecutor("model")
    original = CompletionResult("x", "pin-1", "length")
    monkeypatch.setattr(executor, "complete", AsyncMock(return_value=original))
    result = asyncio.run(executor.execute(object(), "row data", 1, pin=True))
    assert result is original


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
@pytest.mark.parametrize("strategy", [BOOLEAN, Bucket(), Text(), Choice(["red", "blue"])])
def test_skip_matches_inference_without_calling_model(monkeypatch, prompt, strategy):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    monkeypatch.delenv("KALYPSO_SKIP_INFERENCE", raising=False)
    executor = VLLMExecutor("model")
    backend = AsyncMock(return_value=CompletionResult("real reply", "req-1", "length"))
    monkeypatch.setattr(executor, "complete", backend)
    monkeypatch.setattr(executor, "chatcomplete", backend)
    expected = asyncio.run(executor.execute(None, prompt, 8, determinism=strategy))
    backend.reset_mock()
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", "1")
    result = asyncio.run(executor.execute(None, prompt, 8, determinism=strategy))
    backend.assert_not_called()
    assert (result.text, result.finish_reason) == (expected.text, "stop")
    assert result.request_id.startswith("seeded-")
    assert result.request_id != asyncio.run(
        executor.execute(None, prompt, 8, determinism=strategy)
    ).request_id
    if prompt == "hello" and isinstance(strategy, Text):
        assert result.text == "79057cc0"


@pytest.mark.parametrize("seed", [None, ""])
def test_skip_without_seed_raises_before_model_call(monkeypatch, seed):
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", "1")
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    if seed is not None:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    executor = VLLMExecutor("model")
    backend = AsyncMock()
    monkeypatch.setattr(executor, "complete", backend)
    with pytest.raises(RuntimeError, match="KALYPSO_COIN_FLIP_SEED"):
        asyncio.run(executor.execute(None, "hello", 8, determinism=BOOLEAN))
    backend.assert_not_called()


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
@pytest.mark.parametrize("seed", [None, "", "3"])
@pytest.mark.parametrize("skip", [None, "0", "true"])
def test_execute_still_calls_backend(monkeypatch, prompt, seed, skip):
    monkeypatch.delenv("KALYPSO_SKIP_INFERENCE", raising=False)
    if skip is not None:
        monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", skip)
    if seed is None:
        monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    else:
        monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", seed)
    executor = VLLMExecutor("model")
    original = CompletionResult("real reply", "req-1", "length")
    backend = AsyncMock(return_value=original)
    method = "complete" if isinstance(prompt, str) else "chatcomplete"
    monkeypatch.setattr(executor, method, backend)
    request = object()
    result = asyncio.run(executor.execute(request, prompt, 32, False, 7, BOOLEAN))
    backend.assert_awaited_once_with(
        raw_request=request, prompt=prompt, max_tokens=32, pin=False, priority=7
    )
    if not seed:
        assert result is original
        assert apply_determinism(prompt, original, None) is original
    else:
        assert result.text in ("true", "false")
        assert result.request_id == "req-1"
        assert result.finish_reason == "stop"


ALL_OPS = [
    {"op": "sem_filter", "args": {"prompt": "keep?"}},
    {"op": "sem_map", "args": {"prompt": "topic"}},
    {"op": "sem_classify", "args": {"classes": ["red", "blue"]}},
    {"op": "sem_agg", "args": {"instruction": "group"}},
    {"op": "join", "args": {"instruction": "match"}},
]
UNSUPPORTED = [
    (
        [{"op": "sem_filter", "args": {"prompt": "x", "cascade": True}}],
        "CascadeOperator",
    ),
    ([{"op": "sem_topk", "args": {"instruction": "rank", "k": 2}}], "SemTopK"),
    (
        [
            {
                "op": "join",
                "args": {
                    "instruction": "match",
                    "icp": True,
                    "icp_low_threshold": 0.2,
                    "icp_high_threshold": 0.8,
                },
            }
        ],
        "ICPFilter",
    ),
]


def make_plan(monkeypatch):
    from unittest.mock import Mock

    from vllm.kalypso.budget import KVMemoryManager
    from vllm.kalypso.controller.plan import SemanticPlan

    monkeypatch.setattr(
        KVMemoryManager, "get_instance", lambda: Mock(token_length=Mock(return_value=1))
    )
    return SemanticPlan(AsyncMock())


def test_every_llm_operator_declares_determinism(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    physical = make_plan(monkeypatch).parse_ops(ALL_OPS)
    check_determinism(ALL_OPS, physical)
    assert all(op.determinism is not None for op in physical if op.CALLS_LLM)


def test_filter_and_classify_replies_match_their_parsers(monkeypatch):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    physical = make_plan(monkeypatch).parse_ops(ALL_OPS)
    result = CompletionResult("x", "id", "stop")
    assert apply_determinism("hello", result, physical[0].determinism).text == "true"
    assert apply_determinism("hello", result, physical[2].determinism).text == "red"


@pytest.mark.parametrize("ops,_", UNSUPPORTED)
def test_unsupported_operator_allowed_without_seed(monkeypatch, ops, _):
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    check_determinism(ops, make_plan(monkeypatch).parse_ops(ops))


@pytest.mark.parametrize("ops,name", UNSUPPORTED)
def test_unsupported_operator_rejected_before_loading_data(monkeypatch, ops, name):
    from vllm.kalypso.execution.determinism import DeterminismUnsupportedError
    from vllm.kalypso.query import Query

    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    plan = make_plan(monkeypatch)
    with pytest.raises(
        DeterminismUnsupportedError, match=f"{ops[0]['op']} at position 0.*{name}"
    ):
        asyncio.run(plan.execute(None, Query(ops, "/must-not-be-read.csv")))


def test_semantic_query_returns_400_for_invalid_query():
    from types import SimpleNamespace

    from fastapi import HTTPException

    from vllm.entrypoints.openai import api_server

    (route,) = [r for r in api_server.router.routes if r.path == "/v1/semantic/query"]
    processor = SimpleNamespace(execute=AsyncMock(side_effect=ValueError("bad op")))
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(query_processor=processor))
    )
    sem_request = SimpleNamespace(ops=[], data_path="x.csv", blocking=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(route.endpoint(sem_request, raw_request))
    assert (exc.value.status_code, exc.value.detail) == (400, "bad op")
