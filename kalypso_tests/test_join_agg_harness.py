"""Worked example and loss/duplication checks for unindexed join/reduce calls."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm.kalypso.benchmark.correctness_harness import check, read, run
from vllm.kalypso.benchmark.skip_inference_servers import trace_calls


@pytest.fixture
def example():
    payload = {
        "text_column": "text",
        "records": [{"text": "apple"}, {"text": "pear"}],
        "ops": [{"op": op} for op in ("sem_filter", "join", "sem_filter", "sem_agg")],
    }
    output = {
        "results": [
            {
                "idx": -1,
                "input_context": "bucket 3",
                "right_input_context": [],
                "outputs": [
                    {
                        "op": "<class 'vllm.kalypso.sem_ops.reducers.SemAgg'>",
                        "value": "bucket 3",
                    }
                ],
            }
        ]
    }
    calls = [
        {
            "request_id": "a",
            "prompt_sha256": "apple filter",
            "reply": "true",
            "pin": False,
        },
        {
            "request_id": "b",
            "prompt_sha256": "pear filter",
            "reply": "false",
            "pin": False,
        },
        {
            "request_id": "c",
            "prompt_sha256": "apple topic A",
            "reply": "true",
            "pin": False,
        },
        {
            "request_id": "d",
            "prompt_sha256": "apple topic B",
            "reply": "false",
            "pin": False,
        },
        {
            "request_id": "e",
            "prompt_sha256": "apple topic A filter",
            "reply": "true",
            "pin": False,
        },
        {
            "request_id": "f",
            "prompt_sha256": "reduce apple topic A",
            "reply": "bucket 3",
            "pin": False,
        },
    ]
    records = [
        dict(
            request_id=rid,
            position=pos,
            row_idx=idx,
            stage=pos,
            operator=op,
            worker=worker,
            pin=False,
            status="ok",
            submit_t=t,
        )
        for t, (rid, pos, idx, op, worker) in enumerate(
            [
                ("a", 0, 0, "SemFilter", "w0"),
                ("b", 0, 1, "SemFilter", "w1"),
                ("c", 1, -1, "SemFilter", "w0"),
                ("d", 1, -1, "SemFilter", "w1"),
                ("e", 2, -1, "SemFilter", "w0"),
                ("f", 3, None, "SemAgg", "w1"),
            ]
        )
    ]
    return payload, output, deepcopy(output), records, calls, deepcopy(calls)


def test_apple_joins_two_topics_then_reduces_to_bucket_3(example, capsys):
    assert check(*example, round_robin_workers=2)
    assert capsys.readouterr().out == (
        "PASS exactly once (output)\nPASS aggregate output\n"
        "PASS whole output agreement\nPASS exactly once (calls)\n"
        "PASS oracle prompt/reply call agreement\nPASS input call coverage\n"
        "PASS operator call coverage\nPASS round robin routing\n"
        "Per-worker calls: {'w0': 3, 'w1': 3}\n"
        "Rows: old=1, new=1; calls: old=6, new=6\n"
        "Calls by position: {0: 2, 1: 2, 2: 1, 3: 1}\n"
    )


@pytest.mark.parametrize("position", [1, 3])
def test_dropped_join_or_aggregate_fails_even_when_final_bucket_matches(
    example, position
):
    records, calls = example[3], example[5]
    index = next(i for i, r in enumerate(records) if r["position"] == position)
    records.pop(index)
    calls.pop(index)
    assert not check(*example)


def test_duplicate_join_with_a_fresh_request_id_fails(example):
    example[3].append(dict(example[3][2], request_id="duplicate"))
    example[5].append(dict(example[5][2], request_id="duplicate"))
    assert not check(*example)


@pytest.mark.parametrize("fault", ["reply", "prompt_sha256", "request_id"])
def test_changed_join_call_fails_with_identical_final_output(example, fault):
    example[5][2][fault] = "wrong"
    assert not check(*example)


def test_wrong_worker_fails_even_with_balanced_totals(example):
    example[3][2]["worker"], example[3][3]["worker"] = "w1", "w0"
    assert not check(*example, round_robin_workers=2)


def test_wrong_join_operator_metadata_fails(example):
    example[3][2]["operator"] = "SemAgg"
    assert not check(*example)


def test_blocking_calls_with_no_row_indices_still_require_every_oracle_prompt(example):
    example[0]["blocking"] = True
    for record in example[3]:
        record["row_idx"] = None
    assert check(*example, round_robin_workers=2)
    example[3].pop(2)
    example[5].pop(2)
    assert not check(*example)


@pytest.mark.parametrize("fault", ["value", "duplicate", "missing"])
def test_aggregate_output_faults_fail(example, fault):
    rows = example[2]["results"]
    if fault == "value":
        rows[0]["outputs"][0]["value"] = "bucket 4"
    elif fault == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows.clear()
    assert not check(*example)


def test_relational_payload_requires_both_call_manifests(example):
    assert not check(*example[:4])


def test_trace_preserves_executor_reply_and_hashes_exact_prompt(tmp_path):
    result = SimpleNamespace(text="true", request_id="call-1")

    class Executor:
        async def execute(self, **kwargs):
            assert kwargs["prompt"] == "hello"
            assert kwargs["determinism"] == "boolean"
            return result

    path = tmp_path / "calls.jsonl"
    trace_calls(Executor, path)
    assert (
        asyncio.run(Executor().execute(None, "hello", 8, determinism="boolean"))
        is result
    )
    assert read(path) == {
        "request_id": "call-1",
        "reply": "true",
        "pin": False,
        "prompt_sha256": (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        ),
    }


def test_run_resolves_join_table_without_mutating_payload(tmp_path, monkeypatch):
    from unittest.mock import Mock

    payload = {
        "text_column": "text",
        "records": [{"text": "apple"}],
        "ops": [{"op": "join", "args": {"right_table": "topics.csv"}}],
    }
    post = Mock(return_value=Mock(json=Mock(return_value={"results": []})))
    monkeypatch.setattr(
        "vllm.kalypso.benchmark.correctness_harness.requests.post", post
    )
    monkeypatch.chdir(tmp_path)
    run("http://server", payload, tmp_path / "out.json")
    assert post.call_args.kwargs["json"]["ops"] == [
        {"op": "join", "args": {"right_table": str(tmp_path / "topics.csv")}}
    ]
    assert payload["ops"][0]["args"]["right_table"] == "topics.csv"


def test_manifest_checks_preserve_legacy_row_indices():
    from kalypso_tests.test_correctness_harness import BASELINE, PAYLOAD, REQUEST_ROWS

    payload = dict(PAYLOAD, ops=[{"op": "sem_filter"}])
    records = [dict(r, request_id=str(i)) for i, r in enumerate(REQUEST_ROWS)]
    calls = [
        dict(request_id=str(i), prompt_sha256=p, reply="true", pin=False)
        for i, p in enumerate(["apple", "pear"])
    ]
    assert check(payload, BASELINE, BASELINE, records, calls, calls)
    records.append(dict(records[0], request_id="bad", position=1, row_idx=-1))
    payload["ops"].append({"op": "sem_map"})
    records[-1]["operator"] = "SemMap"
    calls = calls + [dict(calls[0], request_id="bad", prompt_sha256="map apple")]
    assert not check(payload, BASELINE, BASELINE, records, calls, calls)
