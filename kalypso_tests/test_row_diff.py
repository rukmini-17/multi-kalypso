# SPDX-License-Identifier: Apache-2.0
import json

from vllm.kalypso.benchmark.row_diff import main, write_report


def test_only_old_and_only_new_rows_are_sorted(tmp_path):
    payload = {
        "records": [{"text": t} for t in ("a", "b", "c", "d", "e", "f")],
        "text_column": "text",
        "ops": [],
    }
    old = {"results": [{"idx": 4, "outputs": []}, {"idx": 0, "outputs": []}]}
    new = {"results": [{"idx": 5, "outputs": []}, {"idx": 2, "outputs": []}]}
    assert write_report(payload, old, new, tmp_path) == (
        "only old kept: 2\nonly new kept: 2\nvalues differ: 0"
    )
    assert (tmp_path / "row_diff.md").read_text() == (
        "# Row diff\n\n## only old kept\n\nonly old kept: 2\n\n"
        "| idx | input text |\n| --- | --- |\n| 0 | a |\n| 4 | e |\n\n"
        "## only new kept\n\nonly new kept: 2\n\n"
        "| idx | input text |\n| --- | --- |\n| 2 | c |\n| 5 | f |\n\n"
        "## values differ\n\nvalues differ: 0\n\n"
        "| idx | input text | op | old | new |\n| --- | --- | --- | --- | --- |\n"
    )


def test_identical_responses_have_empty_sections(tmp_path):
    payload = {
        "records": [{"text": "a"}, {"text": "absent"}],
        "text_column": "text", "ops": [{"op": "sem_map"}],
    }
    response = {"results": [{"idx": 0, "outputs": [{"value": "a"}]}]}
    assert write_report(payload, response, response, tmp_path) == (
        "only old kept: 0\nonly new kept: 0\nvalues differ: 0"
    )
    markdown = (tmp_path / "row_diff.md").read_text()
    assert markdown.count("## ") == 3
    assert "| 0 |" not in markdown and "| 1 |" not in markdown


def test_value_differences_show_only_differing_ops_in_idx_order(tmp_path):
    payload = {
        "records": [{"text": "<a|b>\nc"}, {"text": "d"}],
        "text_column": "text",
        "ops": [{"op": "sem_filter"}, {"op": "sem_map"}],
    }
    old = {"results": [
        {"idx": i, "outputs": [{"value": True}, {"value": " fruit "}]}
        for i in (1, 0)
    ]}
    new = {"results": [
        {"idx": i, "outputs": [{"value": True}, {"value": "fruit"}]}
        for i in (1, 0)
    ]}
    write_report(payload, old, new, tmp_path)
    markdown = (tmp_path / "row_diff.md").read_text()
    assert "values differ: 2" in markdown
    assert "sem_filter" not in markdown
    assert markdown.endswith(
        "| 0 | &lt;a&#124;b&gt;<br>c | sem_map |  fruit  | fruit |\n"
        "| 1 | d | sem_map |  fruit  | fruit |\n"
    )


def test_main_writes_report_commits_and_json_values(tmp_path, capsys):
    payload = {
        "records": [{"text": "a"}], "text_column": "text",
        "ops": [{"op": "sem_map"}],
    }
    old = {"results": [{"idx": 0, "outputs": [{"value": {"x": 1}}]}]}
    new = {"results": [{"idx": 0, "outputs": [{"value": ["x"]}]}]}
    argv = []
    for name, data in (("payload", payload), ("old", old), ("new", new)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        argv.extend([f"--{name}", str(path)])
    (tmp_path / "commits.txt").write_text("Old commit: abc\nNew commit: def\n")
    assert main([*argv, "--out-dir", str(tmp_path)]) == 0
    markdown = (tmp_path / "row_diff.md").read_text()
    assert "Old commit: abc\nNew commit: def" in markdown
    assert "| sem_map | {&quot;x&quot;: 1} | [&quot;x&quot;] |" in markdown
    assert not (tmp_path / "row_diff.csv").exists()
    assert capsys.readouterr().out == (
        "only old kept: 0\nonly new kept: 0\nvalues differ: 1\n"
    )


def test_main_returns_one_on_unreadable_or_invalid_json(tmp_path):
    path = tmp_path / "missing.json"
    argv = [f"--{name}={path}" for name in ("payload", "old", "new")]
    assert main([*argv, "--out-dir", str(tmp_path)]) == 1
    path.write_text("{")
    assert main([*argv, "--out-dir", str(tmp_path)]) == 1
