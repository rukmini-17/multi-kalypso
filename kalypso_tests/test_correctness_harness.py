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


def reversed_candidate():
    candidate = deepcopy(BASELINE)
    candidate["results"].reverse()
    return candidate


def test_check_passes_when_only_row_order_differs(capsys):
    assert check(PAYLOAD, BASELINE, reversed_candidate())
    assert capsys.readouterr().out == (
        "PASS exactly once (output)\nPASS row mapping\n"
        "PASS agreement (differing idx: [])\n"
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
    assert not check(PAYLOAD, BASELINE, candidate)
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


def test_main_returns_1_on_invalid_payload(tmp_path, monkeypatch):
    bad = tmp_path / "no_records.json"
    bad.write_text('{"results": []}\n')
    argv = ["harness", "check", "--payload", str(bad)]
    argv += ["--baseline", str(bad), "--candidate", str(bad)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 1


def test_compare_runs_both_endpoints_and_checks_them(tmp_path, monkeypatch):
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(PAYLOAD))
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
