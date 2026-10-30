from dataclasses import dataclass, replace
from pathlib import Path

from vllm.kalypso.benchmark import scenarios

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = REPO_ROOT / "vllm/kalypso/benchmark/sample_data"


@dataclass(frozen=True)
class Filter:
    prompt: str

    def to_wire(self) -> dict:
        return {"op": "sem_filter", "args": {"prompt": self.prompt}}


@dataclass(frozen=True)
class Map:
    prompt: str

    def to_wire(self) -> dict:
        return {"op": "sem_map", "args": {"prompt": self.prompt}}


@dataclass(frozen=True)
class Agg:
    instruction: str

    def to_wire(self) -> dict:
        return {
            "op": "sem_agg",
            "args": {"instruction": self.instruction},
        }


@dataclass(frozen=True)
class Join:
    instruction: str
    table: Path
    rows: int

    def to_wire(self) -> dict:
        return {
            "op": "join",
            "args": {
                "instruction": self.instruction,
                "right_table": str(self.table),
            },
        }


Op = Filter | Map | Agg | Join


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    data_root: Path
    data_path: Path
    ops: list[Op]
    blocking: bool
    rows: int
    source: str


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    data_path: Path
    ops: list[Op]
    blocking: bool
    rows: int
    source: str

    @property
    def tables(self) -> list[tuple[str, Path, int]]:
        return [
            (op.table.name, op.table, op.rows)
            for op in self.ops
            if isinstance(op, Join)
        ]


WORKLOADS = {
    "fever": Workload(
        name="fever",
        description="Filter, verify and summarize claims",
        data_root=Path(__file__).parent / "data",
        data_path=Path("fever_claims_1000.csv"),
        ops=[
            Filter(
                "Is this claim about a specific person, place or work? "
                "Reply with only true or false."
            ),
            Map(
                "Answer SUPPORTED or REFUTED from general knowledge, "
                "with a one-sentence reason."
            ),
            Agg("Summarize the most common reasons claims were refuted."),
        ],
        blocking=True,
        rows=500,
        source="M3 FEVER plan",
    ),
    "medec": Workload(
        name="medec",
        description="Detect and correct clinical errors",
        data_root=SAMPLE_DATA,
        data_path=Path("MEDEC-TrainingSet-1000.csv"),
        ops=[
            Filter(scenarios.MEDEC_ERROR_FILTER),
            Map(scenarios.MEDEC_ERROR_SENTENCE_ID_MAP),
            Map(scenarios.MEDEC_CORRECTED_SENTENCE_MAP),
        ],
        blocking=False,
        rows=300,
        source="vllm/kalypso/benchmark/scenarios.py",
    ),
    "contract_nli": Workload(
        name="contract_nli",
        description="Filter contracts, join 17 hypotheses and explain",
        data_root=SAMPLE_DATA / "contract-nli",
        data_path=Path("contracts"),
        ops=[
            Filter(scenarios.CONTRACT_NLI_VALID_CONTRACT),
            Join(scenarios.CONTRACT_NLI_ENTAILMENT_JOIN, Path("hypotheses"), 17),
            Map(scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT),
        ],
        blocking=False,
        rows=200,
        source="vllm/kalypso/benchmark/scenarios.py",
    ),
}


def build_plan(
    name: str, rows: int | None = None, data_root: str | Path | None = None
) -> Plan:
    try:
        workload = WORKLOADS[name]
    except KeyError:
        raise ValueError(f"Unknown workload: {name}") from None
    root = Path(data_root if data_root is not None else workload.data_root).resolve()
    return Plan(
        workload.name,
        workload.description,
        (root / workload.data_path).resolve(),
        [
            replace(op, table=(root / op.table).resolve())
            if isinstance(op, Join)
            else op
            for op in workload.ops
        ],
        workload.blocking,
        rows if rows is not None else workload.rows,
        workload.source,
    )


def to_request(plan: Plan, data_path: str | Path, model: str | None) -> dict:
    body = {
        "ops": [op.to_wire() for op in plan.ops],
        "data_path": str(Path(data_path).resolve()),
        "blocking": plan.blocking,
    }
    if model is not None:
        body["model_name"] = model
    return body
