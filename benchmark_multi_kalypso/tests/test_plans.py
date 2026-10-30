import csv
from pathlib import Path

import pytest

from benchmark_multi_kalypso import plans
from vllm.kalypso.benchmark import scenarios


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


@pytest.mark.parametrize(
    "name,ops,blocking",
    [
        (
            "fever",
            [
                ("sem_filter", {"prompt"}),
                ("sem_map", {"prompt"}),
                ("sem_agg", {"instruction"}),
            ],
            True,
        ),
        (
            "medec",
            [
                ("sem_filter", {"prompt"}),
                ("sem_map", {"prompt"}),
                ("sem_map", {"prompt"}),
            ],
            False,
        ),
        (
            "contract_nli",
            [
                ("sem_filter", {"prompt"}),
                ("join", {"instruction", "right_table"}),
                ("sem_map", {"prompt"}),
            ],
            False,
        ),
    ],
)
def test_request_bodies_keep_previous_wire_shapes(tmp_path, name, ops, blocking):
    body = plans.to_request(plans.build_plan(name), tmp_path / "input.csv", None)
    assert set(body) == {"ops", "data_path", "blocking"}
    assert [(op["op"], set(op["args"])) for op in body["ops"]] == ops
    assert all(set(op) == {"op", "args"} for op in body["ops"])
    assert body["data_path"] == str(tmp_path / "input.csv")
    assert body["blocking"] is blocking


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
    plan = plans.build_plan(name)
    wire_ops = [op.to_wire() for op in plan.ops]
    assert [o["op"] for o in wire_ops] == names
    for op, constant in zip(wire_ops, constants):
        assert (
            op["args"].get("prompt", op["args"].get("instruction")) == getattr(scenarios, constant)
        )
        assert "max_tokens" not in op["args"]
    assert not plan.blocking
    assert plan.data_path.is_absolute()
    if name == "contract_nli":
        body = plans.to_request(plan, plan.data_path, None)
        assert Path(body["ops"][1]["args"]["right_table"]).is_absolute()
        body["ops"][1]["args"]["instruction"] = "changed"
        assert plan.ops[1].to_wire()["args"]["instruction"] != "changed"


def test_fever_data_has_1000_claims():
    with plans.build_plan("fever").data_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["data"]
        rows = list(reader)
    assert len(rows) == 1000
    assert all(row["data"] for row in rows)


@pytest.mark.parametrize(
    "name,relative,rows",
    [
        ("fever", "fever_claims_1000.csv", 500),
        ("medec", "MEDEC-TrainingSet-1000.csv", 300),
        ("contract_nli", "contracts", 200),
    ],
)
def test_data_root_is_always_a_folder(tmp_path, name, relative, rows):
    root = tmp_path / "new.csv"
    plan = plans.build_plan(name, data_root=root)
    assert plan.data_path == root / relative
    assert plan.rows == rows
    assert plan.tables == (
        [("hypotheses", root / "hypotheses", 17)] if name == "contract_nli" else []
    )


def test_unknown_workload():
    with pytest.raises(ValueError, match="Unknown workload"):
        plans.build_plan("missing")
