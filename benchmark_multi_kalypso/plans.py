import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = REPO_ROOT / "vllm/kalypso/benchmark/scenarios.py"
SAMPLE_DATA = SCENARIOS.parent / "sample_data"


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    data_path: Path
    ops: list[dict]
    blocking: bool
    rows: int
    source: str


def scenario_prompts() -> dict[str, str]:
    # Reading constants avoids importing the enclosing vllm package on CPU hosts.
    return {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(SCENARIOS.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id.startswith(("MEDEC_", "CONTRACT_NLI_"))
    }


def build_plan(
    name: str, rows: int | None = None, data_root: str | Path | None = None
) -> Plan:
    root = Path(data_root).resolve() if data_root is not None else None
    if name == "fever":
        path = root or Path(__file__).parent / "data"
        path = path if path.suffix == ".csv" else path / "fever_claims_1000.csv"
        ops = [
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
        ]
        return Plan(
            name,
            "Filter, verify and summarize claims",
            path.resolve(),
            ops,
            True,
            rows if rows is not None else 500,
            "M3 FEVER plan",
        )
    prompts = scenario_prompts()
    source = "vllm/kalypso/benchmark/scenarios.py"
    if name == "medec":
        path = SAMPLE_DATA / "MEDEC-TrainingSet-1000.csv" if root is None else root
        if path.is_dir():
            path = path / "medec.csv"
        keys = [
            "MEDEC_ERROR_FILTER",
            "MEDEC_ERROR_SENTENCE_ID_MAP",
            "MEDEC_CORRECTED_SENTENCE_MAP",
        ]
        ops = [
            {"op": op, "args": {"prompt": prompts[key]}}
            for op, key in zip(["sem_filter", "sem_map", "sem_map"], keys)
        ]
        return Plan(
            name,
            "Detect and correct clinical errors",
            path.resolve(),
            ops,
            False,
            rows if rows is not None else 300,
            source,
        )
    if name == "contract_nli":
        root = root or SAMPLE_DATA / "contract-nli"
        ops = [
            {
                "op": "sem_filter",
                "args": {"prompt": prompts["CONTRACT_NLI_VALID_CONTRACT"]},
            },
            {
                "op": "join",
                "args": {
                    "instruction": prompts["CONTRACT_NLI_ENTAILMENT_JOIN"],
                    "right_table": str((root / "hypotheses").resolve()),
                },
            },
            {
                "op": "sem_map",
                "args": {"prompt": prompts["CONTRACT_NLI_EXPLAIN_ENTAILMENT"]},
            },
        ]
        return Plan(
            name,
            "Filter contracts, join 17 hypotheses and explain",
            (root / "contracts").resolve(),
            ops,
            False,
            rows if rows is not None else 200,
            source,
        )
    raise ValueError(f"Unknown workload: {name}")


def to_request(plan: Plan, data_path: str | Path, model: str | None) -> dict:
    body = {
        "ops": deepcopy(plan.ops),
        "data_path": str(Path(data_path).resolve()),
        "blocking": plan.blocking,
    }
    for op in body["ops"]:
        if "right_table" in op["args"]:
            op["args"]["right_table"] = str(Path(op["args"]["right_table"]).resolve())
    if model is not None:
        body["model_name"] = model
    return body
