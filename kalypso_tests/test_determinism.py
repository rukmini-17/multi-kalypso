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


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
@pytest.mark.parametrize("seed", [None, "", "3"])
def test_execute_still_calls_backend(monkeypatch, prompt, seed):
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
