# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from vllm.kalypso.benchmark.correctness_harness import check, main, read, run

PAYLOAD = {"records": [{"text": "apple"}, {"text": "pear"}], "text_column": "text"}
BASELINE = {
    "results": [
        {
            "idx": 0,
            "input_context": [{"content": {"text": "apple"}}],
            "outputs": [{"value": "fruit"}],
        },
        {"idx": 1, "input_context": "pear", "outputs": [{"value": "green"}]},
    ]
}


REQUEST_ROWS = [
    dict(
        type="request",
        row_idx=0,
        stage=0,
        position=0,
        operator="SemFilter",
        worker="w0",
        pin=False,
        status="ok",
    ),
    dict(
        type="request",
        row_idx=1,
        stage=0,
        position=0,
        operator="SemFilter",
        worker="w1",
        pin=False,
        status="ok",
    ),
]


def write_records(path, rows):
    rows = [{"type": "header"}, {"type": "queue_sample", "worker": "w9"}] + rows
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def reversed_candidate():
    candidate = deepcopy(BASELINE)
    candidate["results"].reverse()
    return candidate


def test_check_passes_when_only_row_order_differs(capsys):
    assert check(PAYLOAD, BASELINE, reversed_candidate(), REQUEST_ROWS)
    assert capsys.readouterr().out == (
        "PASS exactly once (output)\nPASS row mapping\n"
        "PASS agreement (differing idx: [])\nPASS exactly once (calls)\n"
        "Per-worker calls: {'w0': 1, 'w1': 1}\n"
    )


def duplicate(rows):
    rows.append(deepcopy(rows[0]))


def set_first(field, value):
    def fault(rows):
        rows[0][field] = value

    return fault


def change_value(rows):
    rows[0]["outputs"][0]["value"] = "wrong"


# The reversed candidate's first row is idx 1 and its last row is idx 0.
FAULTS = {
    "duplicate": (duplicate, "exactly once (output)"),
    "invalid": (set_first("idx", -1), "exactly once (output)"),
    "bool": (set_first("idx", True), "exactly once (output)"),
    "mapping": (set_first("input_context", "apple"), "row mapping"),
    "agreement": (change_value, "agreement (differing idx: [1])"),
    "missing": (lambda rows: rows.pop(), "agreement (differing idx: [0])"),
}


@pytest.mark.parametrize("fault", FAULTS)
def test_check_reports_each_fault(capsys, fault):
    inject, failed_check = FAULTS[fault]
    candidate = reversed_candidate()
    inject(candidate["results"])
    assert not check(PAYLOAD, BASELINE, candidate, REQUEST_ROWS)
    assert f"FAIL {failed_check}" in capsys.readouterr().out


def test_run_sends_records_as_shared_csv(tmp_path, monkeypatch):
    payload = {"ops": [], "text_column": "text", "records": [{"text": "apple, pear"}]}
    response = Mock()
    response.json.return_value = {"results": []}
    post = Mock(return_value=response)
    monkeypatch.setattr(
        "vllm.kalypso.benchmark.correctness_harness.requests.post", post
    )
    out = tmp_path / "baseline.json"
    result, elapsed = run("http://localhost:8100/v1/semantic/query", payload, out)
    assert result == read(out) == {"results": []}
    assert elapsed >= 0
    data_path = tmp_path / "baseline.json.input.csv"
    with data_path.open() as stream:
        assert list(csv.reader(stream)) == [["data"], ["apple, pear"]]
    assert post.call_args.kwargs["json"] == {"ops": [], "data_path": str(data_path)}
    response.raise_for_status.assert_called_once_with()


def test_main_returns_1_on_invalid_payload(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "no_records.json"
    bad.write_text('{"results": []}\n')
    records = tmp_path / "requests.jsonl"
    write_records(records, REQUEST_ROWS)
    argv = ["harness", "check", "--payload", str(bad)]
    argv += ["--baseline", str(bad), "--candidate", str(bad)]
    argv += ["--records", str(records)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 1
    assert capsys.readouterr().out == "FAIL 'records'\n"


def test_check_records_pass_and_count_only_requests(tmp_path, monkeypatch, capsys):
    payload, response, records = [tmp_path / n for n in ("p.json", "r.json", "r.jsonl")]
    payload.write_text(json.dumps(PAYLOAD))
    response.write_text(json.dumps(BASELINE))
    write_records(records, REQUEST_ROWS)
    argv = ["harness", "check", "--payload", str(payload)]
    argv += ["--baseline", str(response), "--candidate", str(response)]
    argv += ["--records", str(records)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert capsys.readouterr().out == (
        "PASS exactly once (output)\nPASS row mapping\n"
        "PASS agreement (differing idx: [])\nPASS exactly once (calls)\n"
        "Per-worker calls: {'w0': 1, 'w1': 1}\n"
    )


RECORD_FAULTS = {
    "duplicate call": (duplicate, "exactly once (calls)"),
    "error status": (set_first("status", "error"), "exactly once (calls)"),
    "missing row": (lambda rows: rows.pop(), "exactly once (calls)"),
    **{
        f"invalid {i!r}": (set_first("row_idx", i), "exactly once (calls)")
        for i in (-1, 2, "0", 0.5, True, False)
    },
}


@pytest.mark.parametrize("fault", RECORD_FAULTS)
def test_check_reports_each_records_fault(capsys, fault):
    inject, failed_check = RECORD_FAULTS[fault]
    rows = deepcopy(REQUEST_ROWS)
    inject(rows)
    assert not check(PAYLOAD, BASELINE, BASELINE, rows)
    assert f"FAIL {failed_check}" in capsys.readouterr().out


def test_pin_and_untagged_rows_are_excluded_from_call_checks(capsys):
    rows = REQUEST_ROWS + [
        dict(REQUEST_ROWS[0], pin=True, status="error"),
        dict(REQUEST_ROWS[0], pin=True, status="error"),
        dict(REQUEST_ROWS[0], row_idx=None, status="error"),
    ]
    assert check(PAYLOAD, BASELINE, BASELINE, rows)
    assert "Per-worker calls: {'w0': 4, 'w1': 1}" in capsys.readouterr().out


def test_same_operator_at_different_positions_passes():
    rows = REQUEST_ROWS + [dict(REQUEST_ROWS[0], position=1)]
    assert check(PAYLOAD, BASELINE, BASELINE, rows)


def test_calls_for_one_row_may_use_different_workers():
    rows = REQUEST_ROWS + [
        dict(REQUEST_ROWS[0], pin=True, worker="w1"),
        dict(REQUEST_ROWS[0], operator="SemMap", worker="w1"),
    ]
    assert check(PAYLOAD, BASELINE, BASELINE, rows)


def test_compare_runs_both_endpoints_and_checks_them(tmp_path, monkeypatch):
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(PAYLOAD))
    records = tmp_path / "requests.jsonl"
    write_records(records, REQUEST_ROWS)
    out = tmp_path / "new" / "run"
    post = Mock(
        side_effect=[
            Mock(json=Mock(return_value=BASELINE)),
            Mock(json=Mock(return_value=reversed_candidate())),
        ]
    )
    monkeypatch.setattr(
        "vllm.kalypso.benchmark.correctness_harness.requests.post", post
    )
    argv = ["harness", "compare", "--payload", str(payload), "--out-dir", str(out)]
    argv += ["--records", str(records)]
    argv += ["--baseline-endpoint", "http://baseline/v1/semantic/query"]
    argv += ["--candidate-endpoint", "http://candidate/v1/semantic/query"]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert read(out / "baseline.json") == BASELINE
    assert read(out / "candidate.json") == reversed_candidate()
    assert [call.args[0] for call in post.call_args_list] == [
        "http://baseline/v1/semantic/query",
        "http://candidate/v1/semantic/query",
    ]


@pytest.mark.parametrize("duplicate_call", [False, True])
def test_compare_checks_records(tmp_path, monkeypatch, duplicate_call):
    payload, records = tmp_path / "payload.json", tmp_path / "requests.jsonl"
    payload.write_text(json.dumps(PAYLOAD))
    rows = REQUEST_ROWS + [REQUEST_ROWS[0]] if duplicate_call else REQUEST_ROWS
    write_records(records, rows)
    responses = [Mock(json=Mock(return_value=BASELINE)) for _ in range(2)]
    monkeypatch.setattr(
        "vllm.kalypso.benchmark.correctness_harness.requests.post",
        Mock(side_effect=responses),
    )
    argv = ["harness", "compare", "--payload", str(payload)]
    argv += ["--out-dir", str(tmp_path), "--records", str(records)]
    argv += ["--baseline-endpoint", "http://b", "--candidate-endpoint", "http://c"]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == int(duplicate_call)


@pytest.mark.parametrize("command", ["check", "compare"])
def test_main_requires_records(monkeypatch, capsys, command):
    argv = ["harness", command, "--payload", "payload.json"]
    if command == "check":
        argv += ["--baseline", "baseline.json", "--candidate", "candidate.json"]
    else:
        argv += ["--baseline-endpoint", "http://b", "--candidate-endpoint", "http://c"]
        argv += ["--out-dir", "out"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "the following arguments are required: --records" in capsys.readouterr().err
