import pytest

from benchmark_multi_kalypso import plans


def test_fever_wire_body_is_blocking_filter_map_agg(tmp_path):
    workload = plans.WORKLOADS["fever"]
    body = plans.to_request(workload, tmp_path / "input.csv", {}, "small-model")
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
    workload = plans.WORKLOADS[name]
    right_tables = {index: tmp_path / f"table_{index}" for index, _ in workload.joins}
    body = plans.to_request(workload, tmp_path / "input.csv", right_tables, None)
    assert set(body) == {"ops", "data_path", "blocking"}
    assert [(op["op"], set(op["args"])) for op in body["ops"]] == ops
    assert all(set(op) == {"op", "args"} for op in body["ops"])
    assert body["data_path"] == str(tmp_path / "input.csv")
    assert body["blocking"] is blocking
