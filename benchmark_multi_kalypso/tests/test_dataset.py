import csv
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmark_multi_kalypso import dataset, plans
from benchmark_multi_kalypso.config import WorkloadConfig, load_config
from benchmark_multi_kalypso.dataset import count_rows, materialize_table

ROOT = Path(__file__).resolve().parents[2]


def test_official_medec_columns_are_counted_and_preserved(tmp_path):
    source = tmp_path / "MEDEC-TrainingSet-1000.csv"
    source.write_text(
        'Text ID,Text,Sentences\nnote-1,"A note, with commas.","0 First sentence.\n'
        '1 Second sentence."\nnote-2,Other note.,0 Other note.\n'
    )
    assert count_rows(source) == 2
    target = materialize_table(source, 1, tmp_path / "subset")
    assert count_rows(target) == 1
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["Text ID", "Text", "Sentences"]
        assert list(reader) == [
            {
                "Text ID": "note-1",
                "Text": "A note, with commas.",
                "Sentences": "0 First sentence.\n1 Second sentence.",
            }
        ]


def test_csv_keeps_multiline_cells_and_header(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text('data,id\n"line one\nline two, quoted",7\nlast,8\n')
    path = materialize_table(source, 1, tmp_path / "run")
    with path.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"data": "line one\nline two, quoted", "id": "7"}
        ]
    assert path.is_absolute()
    assert count_rows(path) == 1


def test_directory_truncates_by_integer_stem(tmp_path):
    source = tmp_path / "contracts"
    source.mkdir()
    for name, text in [("10.txt", "ten"), ("2.txt", "two"), ("1.txt", "one")]:
        (source / name).write_text(text)
    path = materialize_table(source, 2, tmp_path / "run")
    assert {p.name: p.read_text() for p in path.iterdir()} == {
        "1.txt": "one",
        "2.txt": "two",
    }
    assert count_rows(path) == 2


def test_short_dataset_rejected_or_warned_and_actual_count_kept(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("data\none\n")
    with pytest.raises(ValueError, match="requested 2 rows, found 1"):
        materialize_table(source, 2, tmp_path / "strict")
    with pytest.warns(UserWarning, match="using available rows"):
        path = materialize_table(source, 2, tmp_path / "lenient", allow_fewer_rows=True)
    assert count_rows(path) == 1
    with pytest.raises(FileExistsError):
        materialize_table(source, 1, tmp_path / "lenient")


def test_missing_header_and_empty_data_fail_even_in_lenient_mode(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("")
    with pytest.raises(ValueError, match="header"):
        materialize_table(source, 1, tmp_path / "bad")
    source.write_text("data\n")
    with pytest.raises(ValueError, match="found 0"):
        materialize_table(source, 1, tmp_path / "empty", allow_fewer_rows=True)


def test_preflight_counts_repository_relative_data_root(tmp_path):
    root = tmp_path / "claims"
    root.mkdir()
    (root / "fever_claims_1000.csv").write_text('data\n"first claim"\n"second claim"\n')
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[WorkloadConfig("fever", 2, "claims")])
    assert dataset.preflight(config, tmp_path) == {"fever": {"rows": (2, 2)}}
    body, rows, right_rows = dataset.prepare_inputs(config, tmp_path, tmp_path / "out")[
        "fever"
    ]
    assert (rows, right_rows) == (2, None)
    assert Path(body["data_path"]) == tmp_path / "out/data/fever/input/input.csv"


def test_preparation_uses_declared_join_position_and_rows(tmp_path, monkeypatch):
    workload = plans.WORKLOADS["contract_nli"]
    join = replace(workload.ops[1], rows=3)
    monkeypatch.setitem(
        plans.WORKLOADS,
        "contract_nli",
        replace(workload, ops=[join, workload.ops[0], workload.ops[2]]),
    )
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[WorkloadConfig("contract_nli", 2)])
    assert dataset.preflight(config, ROOT) == {
        "contract_nli": {"rows": (10, 2), "table_0": (10, 3)}
    }
    body, rows, right_rows = dataset.prepare_inputs(config, ROOT, tmp_path / "out")[
        "contract_nli"
    ]
    assert (rows, right_rows) == (2, 3)
    right = Path(body["ops"][0]["args"]["right_table"])
    assert right == tmp_path / "out/data/contract_nli/table_0"
    assert count_rows(right) == 3


def test_multiple_joins_with_shared_paths_and_folder_names_are_copied_once(
    tmp_path, monkeypatch
):
    first, second = tmp_path / "a/hypotheses", tmp_path / "b/hypotheses"
    for folder, text in [(first, "first"), (second, "second")]:
        folder.mkdir(parents=True)
        (folder / "1.txt").write_text(text)
    workload = plans.WORKLOADS["contract_nli"]
    monkeypatch.setitem(
        plans.WORKLOADS,
        "contract_nli",
        replace(
            workload,
            ops=[
                plans.Join("one", first, 1),
                plans.Join("two", second, 1),
                plans.Join("three", first, 1),
            ],
        ),
    )
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[WorkloadConfig("contract_nli", 2)])
    materialize = Mock(wraps=dataset.materialize_table)
    monkeypatch.setattr(dataset, "materialize_table", materialize)
    body, rows, right_rows = dataset.prepare_inputs(config, ROOT, tmp_path / "out")[
        "contract_nli"
    ]
    assert (rows, right_rows) == (2, 3)
    assert materialize.call_count == 4  # Main table and three distinct Join ops.
    copies = [Path(op["args"]["right_table"]) for op in body["ops"]]
    assert [path.name for path in copies] == ["table_0", "table_1", "table_2"]
    assert [(path / "1.txt").read_text() for path in copies] == [
        "first",
        "second",
        "first",
    ]
    assert dataset.preflight(config, ROOT) == {
        "contract_nli": {
            "rows": (10, 2),
            "table_0": (1, 1),
            "table_1": (1, 1),
            "table_2": (1, 1),
        }
    }


def test_byte_order_mark_is_dropped_from_the_copied_header(tmp_path):
    source = tmp_path / "medec.csv"
    source.write_text("Text ID,Text,Sentences\nms-1,note,0 one\n", encoding="utf-8-sig")
    assert count_rows(source) == 1
    copy = materialize_table(source, 1, tmp_path / "out")
    assert copy.read_text(encoding="utf-8").startswith("Text ID,")


@pytest.mark.parametrize(
    "name,relative",
    [
        ("fever", "fever_claims_1000.csv"),
        ("medec", "MEDEC-TrainingSet-1000.csv"),
        ("contract_nli", "contracts"),
    ],
)
def test_source_paths_use_workload_layout_without_mutating_declarations(
    tmp_path, name, relative
):
    root = tmp_path / "new.csv"
    workload, data, tables = dataset.source_paths(
        WorkloadConfig(name, 2, str(root)), ROOT
    )
    assert data == root / relative
    assert tables == ({1: root / "hypotheses"} if name == "contract_nli" else {})
    assert workload is plans.WORKLOADS[name]
    assert not workload.data_path.is_absolute()
    assert all(not join.table.is_absolute() for _, join in workload.joins)


def test_unknown_workload_lists_registry_names():
    config = load_config(ROOT / "benchmark_multi_kalypso/configs/smoke.yaml")
    config = replace(config, workloads=[WorkloadConfig("missing", 1)])
    with pytest.raises(ValueError) as error:
        dataset.preflight(config, ROOT)
    assert str(error.value) == (
        "Unknown workload: missing. Valid workloads: " + ", ".join(plans.WORKLOADS)
    )
