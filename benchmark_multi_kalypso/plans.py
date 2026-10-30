from dataclasses import dataclass
from pathlib import Path

from vllm.kalypso.benchmark import scenarios

MEDEC_CSV = "MEDEC-TrainingSet-1000.csv"
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
    data_root: Path
    data_path: Path
    ops: list[Op]
    blocking: bool

    @property
    def joins(self) -> list[tuple[int, Join]]:
        return [
            (index, op) for index, op in enumerate(self.ops) if isinstance(op, Join)
        ]


WORKLOADS = {
    # FEVER is ours; MEDEC and ContractNLI are the upstream plans from scenarios.py.
    "fever": Workload(
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
    ),
    "medec": Workload(
        data_root=SAMPLE_DATA,
        data_path=Path(MEDEC_CSV),
        ops=[
            Filter(scenarios.MEDEC_ERROR_FILTER),
            Map(scenarios.MEDEC_ERROR_SENTENCE_ID_MAP),
            Map(scenarios.MEDEC_CORRECTED_SENTENCE_MAP),
        ],
        blocking=False,
    ),
    "contract_nli": Workload(
        data_root=SAMPLE_DATA / "contract-nli",
        data_path=Path("contracts"),
        ops=[
            Filter(scenarios.CONTRACT_NLI_VALID_CONTRACT),
            Join(scenarios.CONTRACT_NLI_ENTAILMENT_JOIN, Path("hypotheses"), 17),
            Map(scenarios.CONTRACT_NLI_EXPLAIN_ENTAILMENT),
        ],
        blocking=False,
    ),
}


def to_request(
    workload: Workload,
    data_path: str | Path,
    right_tables: dict[int, Path],
    model: str | None,
) -> dict:
    ops = [op.to_wire() for op in workload.ops]
    for index, _ in workload.joins:
        ops[index]["args"]["right_table"] = str(right_tables[index])
    body = {
        "ops": ops,
        "data_path": str(Path(data_path).resolve()),
        "blocking": workload.blocking,
    }
    if model is not None:
        body["model_name"] = model
    return body
