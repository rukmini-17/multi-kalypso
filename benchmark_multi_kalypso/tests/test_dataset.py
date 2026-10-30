import csv

import pytest

from benchmark_multi_kalypso.dataset import count_rows, materialize_table


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


def test_missing_column_and_empty_data_fail_even_in_lenient_mode(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("text\none\n")
    with pytest.raises(ValueError, match="data column"):
        materialize_table(source, 1, tmp_path / "bad")
    source.write_text("data\n")
    with pytest.raises(ValueError, match="found 0"):
        materialize_table(source, 1, tmp_path / "empty", allow_fewer_rows=True)
