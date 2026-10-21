# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from test_http_executor import FakeWorker, http, instr, run

from vllm.kalypso.execution.determinism import BOOLEAN


@pytest.fixture
def worker_call():
    async def call(*, pin=False):
        worker = FakeWorker()
        await worker.start()
        recorder = instr.RequestRecorder()
        executor = http.HttpVLLMExecutor(worker.url, "m", recorder=recorder)
        request = SimpleNamespace()
        try:
            result = await executor.execute(
                request, "hello", 8, pin=pin, determinism=BOOLEAN
            )
            return result, recorder.rows("request")
        finally:
            if pin:
                await http.unpin_owner(str(id(request)))
            await executor.close()
            await worker.stop()

    return call


def test_seeded_http_reply_preserves_real_request_record(monkeypatch, worker_call):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    result, (row,) = run(worker_call())
    assert (result.text, result.finish_reason) == ("true", "stop")
    assert result.request_id == row["request_id"]
    assert (row["prompt_tokens"], row["completion_tokens"], row["status"]) == (
        7,
        2,
        "ok",
    )


def test_seeded_http_pin_reply_is_untouched(monkeypatch, worker_call):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "2")
    result, (row,) = run(worker_call(pin=True))
    assert (result.text, result.finish_reason) == ("yes", "stop")
    assert result.request_id == row["request_id"]
    assert row["pin"] is True


def test_unseeded_http_reply_is_untouched(monkeypatch, worker_call):
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    result, (row,) = run(worker_call())
    assert (result.text, result.finish_reason) == ("yes", "stop")
    assert result.request_id == row["request_id"]
