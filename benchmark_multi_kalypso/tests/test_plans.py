import ast
import csv
from pathlib import Path

import pytest

from benchmark_multi_kalypso import plans


def test_fever_wire_body_is_blocking_filter_map_agg(tmp_path):
    plan = plans.build_plan("fever", rows=2)
    body = plans.to_request(plan, tmp_path / "input.csv", "small-model")
    assert body == {
        "ops": [
            {
                "op": "sem_filter",
                "args": {
                    "prompt": (
                        "Is this claim about a specific person, place or work? "
                        "Reply with only true or false."
                    )
                },
            },
            {
                "op": "sem_map",
                "args": {
                    "prompt": (
                        "Answer SUPPORTED or REFUTED from general knowledge, "
                        "with a one-sentence reason."
                    )
                },
            },
            {
                "op": "sem_agg",
                "args": {
                    "instruction": (
                        "Summarize the most common reasons claims were refuted."
                    )
                },
            },
        ],
        "data_path": str(tmp_path / "input.csv"),
        "blocking": True,
        "model_name": "small-model",
    }
    assert plan.rows == 2


def test_upstream_prompts_are_loaded_at_runtime(tmp_path, monkeypatch):
    source = tmp_path / "scenarios.py"
    source.write_text(
        'MEDEC_ERROR_FILTER = "filter from upstream"\n'
        'MEDEC_ERROR_SENTENCE_ID_MAP = "find ID"\n'
        'MEDEC_CORRECTED_SENTENCE_MAP = "correct sentence"\n'
    )
    monkeypatch.setattr(plans, "SCENARIOS", source)
    plan = plans.build_plan("medec")
    assert plan.ops == [
        {"op": "sem_filter", "args": {"prompt": "filter from upstream"}},
        {"op": "sem_map", "args": {"prompt": "find ID"}},
        {"op": "sem_map", "args": {"prompt": "correct sentence"}},
    ]
    source.write_text(source.read_text().replace("find ID", "updated ID prompt"))
    assert plans.build_plan("medec").ops[1]["args"]["prompt"] == "updated ID prompt"


@pytest.mark.parametrize(
    "name,names,constants",
    [
        (
            "medec",
            ["sem_filter", "sem_map", "sem_map"],
            [
                "MEDEC_ERROR_FILTER",
                "MEDEC_ERROR_SENTENCE_ID_MAP",
                "MEDEC_CORRECTED_SENTENCE_MAP",
            ],
        ),
        (
            "contract_nli",
            ["sem_filter", "join", "sem_map"],
            [
                "CONTRACT_NLI_VALID_CONTRACT",
                "CONTRACT_NLI_ENTAILMENT_JOIN",
                "CONTRACT_NLI_EXPLAIN_ENTAILMENT",
            ],
        ),
    ],
)
def test_plans_match_actual_scenario_constants(name, names, constants):
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(plans.SCENARIOS.read_text()).body
        if isinstance(node, ast.Assign)
    }
    plan = plans.build_plan(name)
    assert [o["op"] for o in plan.ops] == names
    for op, constant in zip(plan.ops, constants):
        assert (
            op["args"].get("prompt", op["args"].get("instruction")) == values[constant]
        )
        assert "max_tokens" not in op["args"]
    assert not plan.blocking
    assert plan.data_path.is_absolute()
    if name == "contract_nli":
        body = plans.to_request(plan, plan.data_path, None)
        assert Path(body["ops"][1]["args"]["right_table"]).is_absolute()
        body["ops"][1]["args"]["instruction"] = "changed"
        assert plan.ops[1]["args"]["instruction"] != "changed"


def test_fever_data_has_1000_claims():
    with plans.build_plan("fever").data_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["data"]
        rows = list(reader)
    assert len(rows) == 1000
    assert all(row["data"] for row in rows)


def test_data_root_and_unknown_workload(tmp_path):
    assert (
        plans.build_plan("medec", data_root=tmp_path).data_path
        == tmp_path / "medec.csv"
    )
    assert (
        plans.build_plan("contract_nli", data_root=tmp_path).data_path
        == tmp_path / "contracts"
    )
    with pytest.raises(ValueError, match="Unknown workload"):
        plans.build_plan("missing")
