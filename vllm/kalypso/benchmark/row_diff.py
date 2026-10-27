# SPDX-License-Identifier: Apache-2.0
"""Report differing output rows; correctness_harness.py determines pass/fail."""

import argparse
import html
import json
from pathlib import Path


def cell(value):
    # Serialize non-strings and escape Markdown cells without normalizing values.
    text = value if isinstance(value, str) else json.dumps(value)
    text = html.escape(text).replace("|", "&#124;")
    return text.replace("\r\n", "\n").replace("\n", "<br>")


def table(headers, rows):
    # Start with column labels and the Markdown separator, even for empty tables.
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    # Escape each row's cells and keep the supplied row order.
    for row in rows:
        lines.append("| " + " | ".join(map(cell, row)) + " |")
    return lines


def write_report(payload: dict, old: dict, new: dict, out_dir: Path) -> str:
    # Match the harness's exact value lists, indexed by input row.
    before, after = [
        {r["idx"]: [o["value"] for o in r["outputs"]] for r in side["results"]}
        for side in (old, new)
    ]
    differing = {i for i in before.keys() & after.keys() if before[i] != after[i]}
    # Include recorded commits before the three differing-row sections.
    out_dir.mkdir(parents=True, exist_ok=True)
    sections, counts = ["# Row diff"], []
    commits = out_dir / "commits.txt"
    if commits.exists():
        sections.append("```text\n" + commits.read_text().strip() + "\n```")

    # List rows kept only by old, in input-index order.
    old_rows = [
        (idx, payload["records"][idx][payload["text_column"]])
        for idx in sorted(before.keys() - after.keys())
    ]
    counts.append(f"only old kept: {len(old_rows)}")
    sections += [
        "## only old kept",
        counts[-1],
        "\n".join(table(["idx", "input text"], old_rows)),
    ]

    # List rows kept only by new, in input-index order.
    new_rows = [
        (idx, payload["records"][idx][payload["text_column"]])
        for idx in sorted(after.keys() - before.keys())
    ]
    counts.append(f"only new kept: {len(new_rows)}")
    sections += [
        "## only new kept",
        counts[-1],
        "\n".join(table(["idx", "input text"], new_rows)),
    ]

    # For shared rows, list only differing operations with both original values.
    value_rows = []
    for idx in sorted(differing):
        text = payload["records"][idx][payload["text_column"]]
        for op, old_value, new_value in zip(payload["ops"], before[idx], after[idx]):
            if old_value != new_value:
                value_rows.append((idx, text, op["op"], old_value, new_value))
    # Count differing input rows once, even when several operations differ.
    counts.append(f"values differ: {len(differing)}")
    sections += [
        "## values differ",
        counts[-1],
        "\n".join(table(["idx", "input text", "op", "old", "new"], value_rows)),
    ]
    # Save Markdown only and return the three count lines for stdout.
    markdown = "\n\n".join(sections) + "\n"
    (out_dir / "row_diff.md").write_text(markdown, encoding="utf-8")
    return "\n".join(counts)


def main(argv=None) -> int:
    # Read the payload and responses, then report regardless of agreement.
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("payload", "old", "new", "out-dir"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload, old, new = [
            json.loads(p.read_text()) for p in (args.payload, args.old, args.new)
        ]
        print(write_report(payload, old, new, args.out_dir))
        return 0
    # Fail only for unreadable files or unparseable report data.
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"FAIL {exc}")
        return 1


# Run the report command and propagate file or parsing failures.
if __name__ == "__main__":
    raise SystemExit(main())
