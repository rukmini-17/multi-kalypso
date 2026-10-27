import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import ExecutionState, SemContext, SemanticInput
from vllm.kalypso.controller.plan import PlanExecutor
from vllm.kalypso.instrumentation import current_request_meta, request_meta
from vllm.kalypso.sem_ops.base import BaseOp, OpBehavior
from vllm.kalypso.sem_ops.reducers import SemAgg, SemTopK


@pytest.mark.parametrize("op_type,reply,expected_calls", [
    (SemAgg, "summary", 1), (SemTopK, "A", 2),
])
def test_blocking_llm_calls_inherit_operator_labels(monkeypatch, op_type, reply, expected_calls):
    manager = NS(
        token_length=lambda text: 1, apply_chat_template=str, bytes_per_token=1,
        can_admit=AsyncMock(return_value=True), allocate=AsyncMock(), release=AsyncMock(),
    )
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: manager)
    labels = []

    async def llm_call(**kwargs):
        labels.append(current_request_meta())
        return NS(text=reply)

    async def go():
        ctxs = [SemContext(
            input=SemanticInput(data=text),
            state=ExecutionState(executor=NS(execute=llm_call), idx=i),
        ) for i, text in enumerate(["alpha", "beta", "gamma"])]
        op = op_type("summarize or rank", position=2, concurrency=2)
        if isinstance(op, SemTopK):
            op.k = 1
        with request_meta(operator="outer", stage=99, row_idx=7):
            out, stats = await PlanExecutor().execute(ctxs, [op])
            assert current_request_meta() == {"operator": "outer", "stage": 99, "row_idx": 7}
        assert len(out) == 1 and stats == []
        assert labels == [{
            "operator": op_type.__name__, "stage": 2, "position": 2, "row_idx": None,
        }] * expected_calls
        assert current_request_meta() == {}

    asyncio.run(go())


def test_blocking_metadata_restored_after_exception(monkeypatch):
    monkeypatch.setattr(KVMemoryManager, "get_instance", lambda: object())

    class FailingOp(BaseOp):
        async def __call__(self, ctxs):
            assert current_request_meta() == {
                "operator": "FailingOp", "stage": 4, "position": 4, "row_idx": None,
            }
            raise ValueError("boom")

    async def go():
        with pytest.raises(ValueError, match="boom"):
            await PlanExecutor().execute([], [FailingOp(OpBehavior.BLOCKING, position=4)])
        assert current_request_meta() == {}

    asyncio.run(go())
