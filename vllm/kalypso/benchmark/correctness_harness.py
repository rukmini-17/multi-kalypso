# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare semantic query outputs against a single-worker baseline.

Check output uniqueness, row mapping, and operator value agreement by row index.
KALYPSO_COIN_FLIP_SEED must match on all servers; real inference still runs.
The run command writes records to a CSV beside the response for team main's
server-visible data_path API, so client and server must share a filesystem.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def contains(value: Any, text: str) -> bool:
    if isinstance(value, str):
        return text in value
    children = value.values() if isinstance(value, dict) else value
    return isinstance(value, (list, dict)) and any(
        contains(child, text) for child in children
    )


def values(response: dict) -> dict:
    return {r["idx"]: [o["value"] for o in r["outputs"]] for r in response["results"]}


def check(payload: dict, baseline: dict, candidate: dict) -> bool:
    records = payload["records"]
    rows = candidate["results"]
    indices = [r["idx"] for r in rows]
    valid = all(type(i) is int and 0 <= i < len(records) for i in indices)
    checks = {"exactly once (output)": valid and len(set(indices)) == len(indices)}
    checks["row mapping"] = valid and all(
        contains(r["input_context"], records[r["idx"]][payload["text_column"]])
        for r in rows
    )
    before = values(baseline)
    after = values(candidate) if valid else {}
    differing = sorted(
        i
        for i in before.keys() | after.keys()
        if i not in before or i not in after or before[i] != after[i]
    )
    checks[f"agreement (differing idx: {differing})"] = not differing
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return all(checks.values())


def run(endpoint: str, payload: dict, out: str | Path) -> tuple[dict, float]:
    started = time.monotonic()
    data_path = Path(str(out) + ".input.csv").resolve()
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["data"])
        writer.writerows([r[payload["text_column"]]] for r in payload["records"])
    wire = {k: v for k, v in payload.items() if k not in ("records", "text_column")}
    wire["data_path"] = str(data_path)
    response = requests.post(endpoint, json=wire, timeout=600)
    response.raise_for_status()
    result = response.json()
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return result, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "compare"):
        sub = subs.add_parser(name)
        sub.add_argument("--payload", required=True)
        required = {
            "run": ["endpoint", "out"],
            "check": ["baseline", "candidate"],
            "compare": ["baseline-endpoint", "candidate-endpoint", "out-dir"],
        }[name]
        for option in required:
            sub.add_argument(f"--{option}", required=True)
    args = parser.parse_args()
    try:
        payload = read(args.payload)
        if args.command == "run":
            run(args.endpoint, payload, args.out)
            return 0
        if args.command == "compare":
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            baseline, _ = run(
                args.baseline_endpoint, payload, out_dir / "baseline.json"
            )
            candidate, _ = run(
                args.candidate_endpoint, payload, out_dir / "candidate.json"
            )
        else:
            baseline, candidate = read(args.baseline), read(args.candidate)
        return int(not check(payload, baseline, candidate))
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
