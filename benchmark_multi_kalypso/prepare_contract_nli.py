import argparse
import json
from pathlib import Path


def prepare_contract_nli(source: Path, out: Path) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
    # Refuse existing output directories so an older split cannot leave stale rows.
    contracts, hypotheses = out / "contracts", out / "hypotheses"
    if contracts.exists() or hypotheses.exists():
        raise ValueError(f"Use a fresh output directory: {out}")
    contracts.mkdir(parents=True)
    hypotheses.mkdir()
    for i, document in enumerate(data["documents"], start=1):
        (contracts / f"{i}.txt").write_text(document["text"], encoding="utf-8")
    for key, label in sorted(
        data["labels"].items(), key=lambda item: int(item[0].split("-")[-1])
    ):
        number = int(key.split("-")[-1])
        text = (
            f"LABEL_ID: {key}\nSHORT_DESCRIPTION: {label['short_description']}"
            f"\n\nHYPOTHESIS:\n{label['hypothesis']}\n"
        )
        (hypotheses / f"{number}.txt").write_text(text, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded ContractNLI JSON; no network access"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Workload data root")
    args = parser.parse_args()
    print(prepare_contract_nli(args.input, args.out))


if __name__ == "__main__":
    main()
