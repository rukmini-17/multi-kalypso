# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare semantic query outputs against a single-worker baseline.

Check row mapping and values for row-preserving queries, and whole relational
outputs for joins/reductions. Optional executor manifests compare every exact
prompt/reply against the oracle and link each call to coordinator records.
Relational queries require these manifests, including calls without row indices.
KALYPSO_COIN_FLIP_SEED must match on all servers.
The run command writes records to a CSV beside the response for team main's
server-visible data_path API, so client and server must share a filesystem.
"""

import argparse
import csv
import json
import time
from collections import Counter
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


def json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def relational_checks(payload, baseline, candidate, request_rows, old_calls, new_calls):
    """Compare whole outputs and every prompt/reply, including unindexed calls."""
    aggregate = any(op["op"] == "sem_agg" for op in payload["ops"])
    rows = candidate["results"]
    identities = [
        json_key([r["idx"], r["input_context"], r.get("right_input_context")])
        for r in rows
    ]
    checks = {"exactly once (output)": len(set(identities)) == len(rows)}
    if aggregate:
        checks["aggregate output"] = len(rows) == 1 and any(
            "SemAgg" in o["op"] for o in rows[0]["outputs"]
        )

    def outputs(response):
        return Counter(
            json_key({k: v for k, v in r.items() if k != "worker"})
            for r in response["results"]
        )

    checks["whole output agreement"] = outputs(baseline) == outputs(candidate)
    if old_calls is None or new_calls is None:
        checks["oracle call manifests required"] = False
        return checks
    calls = [r for r in request_rows if not r["pin"]]
    ids = [r["request_id"] for r in calls]
    old_ids = [r["request_id"] for r in old_calls]
    new_ids = [r["request_id"] for r in new_calls]
    checks["exactly once (calls)"] = (
        bool(old_calls)
        and len(set(old_ids)) == len(old_ids)
        and len(old_ids) == len(new_ids)
        and len(set(ids)) == len(ids) == len(new_ids) == len(set(new_ids))
        and set(ids) == set(new_ids)
        and all(r["status"] == "ok" for r in calls)
        and not any(r["pin"] for r in old_calls + new_calls)
    )
    checks["oracle prompt/reply call agreement"] = Counter(
        (r["prompt_sha256"], r["reply"]) for r in old_calls
    ) == Counter((r["prompt_sha256"], r["reply"]) for r in new_calls)
    first = [r["row_idx"] for r in calls if r["position"] == 0]
    # Blocking stages deliberately have no row index. Their exact input prompts
    # are still checked against the oracle's full call multiset above.
    checks["input call coverage"] = (
        len(first) == len(payload["records"]) and all(i is None for i in first)
        if payload.get("blocking")
        else all(type(i) is int for i in first)
        and Counter(first) == Counter(range(len(payload["records"])))
    )
    operators = {
        "sem_filter": "SemFilter",
        "join": "SemFilter",
        "sem_map": "SemMap",
        "sem_agg": "SemAgg",
    }
    checks["operator call coverage"] = {r["position"] for r in calls} == set(
        range(len(payload["ops"]))
    ) and all(
        type(r["position"]) is int
        and r["operator"] == operators[payload["ops"][r["position"]]["op"]]
        for r in calls
    )
    return checks


def check(
    payload: dict,
    baseline: dict,
    candidate: dict,
    request_rows: list[dict],
    old_calls: list[dict] | None = None,
    new_calls: list[dict] | None = None,
    round_robin_workers: int | None = None,
) -> bool:
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
    calls = [r for r in request_rows if not r["pin"] and r["row_idx"] is not None]
    valid_calls = all(
        type(r["row_idx"]) is int and 0 <= r["row_idx"] < len(records) for r in calls
    )
    checks["exactly once (calls)"] = (
        valid_calls
        and all(r["status"] == "ok" for r in calls)
        and all(
            count == 1
            for count in Counter(
                (r["row_idx"], r["stage"], r["position"], r["operator"]) for r in calls
            ).values()
        )
        and {r["row_idx"] for r in calls} == set(range(len(records)))
    )
    if any(op["op"] in ("join", "sem_agg") for op in payload.get("ops", [])):
        checks = relational_checks(
            payload, baseline, candidate, request_rows, old_calls, new_calls
        )
    elif old_calls is not None or new_calls is not None:
        for name, passed in relational_checks(
            payload, baseline, candidate, request_rows, old_calls, new_calls
        ).items():
            checks[name] = checks.get(name, True) and passed
    # Exact w{i % n} order holds only under skip inference, where no call waits.
    if round_robin_workers is not None:
        ordered = sorted(request_rows, key=lambda r: r["submit_t"])
        checks["round robin routing"] = round_robin_workers > 0 and all(
            r["worker"] == f"w{i % round_robin_workers}" for i, r in enumerate(ordered)
        )
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"Per-worker calls: {dict(Counter(r['worker'] for r in request_rows))}")
    if old_calls is not None and new_calls is not None:
        print(
            f"Rows: old={len(baseline['results'])}, new={len(rows)}; "
            f"calls: old={len(old_calls)}, new={len(new_calls)}"
        )
        print(
            f"Calls by position: {dict(Counter(r['position'] for r in request_rows))}"
        )
    return all(checks.values())


def run(endpoint: str, payload: dict, out: str | Path) -> tuple[dict, float]:
    started = time.monotonic()
    data_path = Path(str(out) + ".input.csv").resolve()
    with data_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["data"])
        writer.writerows([r[payload["text_column"]]] for r in payload["records"])
    wire = {k: v for k, v in payload.items() if k not in ("records", "text_column")}
    wire = json.loads(json.dumps(wire))
    for op in wire.get("ops", []):
        args = op.get("args", {})
        if "right_table" in args:
            args["right_table"] = str(Path(args["right_table"]).resolve())
    wire["data_path"] = str(data_path)
    response = requests.post(endpoint, json=wire, timeout=600)
    response.raise_for_status()
    result = response.json()
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    elapsed = time.monotonic() - started
    print(f"Saved {out}: {len(result['results'])} rows in {elapsed:.3f}s")
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "compare"):
        sub = subs.add_parser(name)
        sub.add_argument("--payload", required=True)
        if name != "run":
            sub.add_argument("--baseline-calls")
            sub.add_argument("--candidate-calls")
            sub.add_argument("--round-robin-workers", type=int)
        required = {
            "run": ["endpoint", "out"],
            "check": ["baseline", "candidate", "records"],
            "compare": [
                "baseline-endpoint",
                "candidate-endpoint",
                "out-dir",
                "records",
            ],
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
        with Path(args.records).open() as stream:
            request_rows = [
                row for line in stream if (row := json.loads(line))["type"] == "request"
            ]
        manifests = [
            [json.loads(line) for line in Path(path).read_text().splitlines()]
            if path
            else None
            for path in (args.baseline_calls, args.candidate_calls)
        ]
        return int(
            not check(
                payload,
                baseline,
                candidate,
                request_rows,
                *manifests,
                round_robin_workers=args.round_robin_workers,
            )
        )
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
