# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_http_executor import FakeWorker, det, http, instr, run


@pytest.mark.parametrize("prompt", ["hello", [{"role": "user", "content": "hello"}]])
def test_skip_does_not_post_and_records_one_ok_call(monkeypatch, prompt):
    monkeypatch.setenv("KALYPSO_COIN_FLIP_SEED", "1")
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", "1")
    recorder = instr.RequestRecorder()
    executor = http.HttpVLLMExecutor("http://unused", "m", recorder=recorder)
    post = AsyncMock(side_effect=AssertionError("must not post"))
    monkeypatch.setattr(executor, "_post", post)
    with instr.request_meta(operator="SemMap", stage=2, position=3, row_idx=5):
        result = run(executor.execute(None, prompt, 8, determinism=det.Text()))
    post.assert_not_called()
    (row,) = recorder.rows("request")
    assert (row["status"], row["finish_reason"], row["error"]) == ("ok", "stop", None)
    assert (row["worker"], row["operator"], row["stage"], row["position"], row["row_idx"]) == (
        "w0", "SemMap", 2, 3, 5
    )
    assert result.request_id == row["request_id"]
    assert result.request_id.startswith("seeded-")
    if prompt == "hello":
        assert (result.text, result.finish_reason) == ("79057cc0", "stop")


def test_http_skip_without_seed_raises_and_records_error(monkeypatch):
    monkeypatch.setenv("KALYPSO_SKIP_INFERENCE", "1")
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    recorder = instr.RequestRecorder()
    executor = http.HttpVLLMExecutor("http://unused", "m", recorder=recorder)
    post = AsyncMock()
    monkeypatch.setattr(executor, "_post", post)
    with pytest.raises(RuntimeError, match="KALYPSO_COIN_FLIP_SEED"):
        run(executor.execute(None, "hello", 8, determinism=det.BOOLEAN))
    post.assert_not_called()
    (row,) = recorder.rows("request")
    assert row["status"] == "error"


@pytest.fixture
def worker_call():
    async def call():
        worker = FakeWorker()
        await worker.start()
        recorder = instr.RequestRecorder()
        executor = http.HttpVLLMExecutor(worker.url, "m", recorder=recorder)
        request = SimpleNamespace()
        try:
            result = await executor.execute(
                request, "hello", 8, determinism=det.BOOLEAN
            )
            return result, recorder.rows("request")
        finally:
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


def test_unseeded_http_reply_is_untouched(monkeypatch, worker_call):
    monkeypatch.delenv("KALYPSO_COIN_FLIP_SEED", raising=False)
    result, (row,) = run(worker_call())
    assert (result.text, result.finish_reason) == ("yes", "stop")
    assert result.request_id == row["request_id"]
