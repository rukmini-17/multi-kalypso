import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_multi_kalypso.prepare_data import prepare_contract_nli, prepare_medec


@pytest.mark.parametrize("module", ["prepare_data", "config", "container"])
def test_offline_import_does_not_import_vllm_or_torch(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import benchmark_multi_kalypso.{module}; "
            "assert 'torch' not in sys.modules; "
            "print([m for m in sys.modules if m == 'vllm' or m.startswith('vllm.')])",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "[]\n"


def test_official_medec_becomes_sample_formatted_data_column(tmp_path):
    source = tmp_path / "official.csv"
    source.write_text(
        'Text ID,Text,Sentences,Error Flag\nnote-1,"A note, with commas.",0 A note.,1\n'
    )
    target = prepare_medec(source, tmp_path / "medec")
    assert target == tmp_path / "medec/MEDEC-TrainingSet-1000.csv"
    with target.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {
                "data": "Text ID: note-1\n\nClinical note:\nA note, with commas."
                "\n\nNumbered sentences:\n0 A note."
            }
        ]


def test_official_contracts_keep_all_labels_and_sample_hypothesis_format(tmp_path):
    source = tmp_path / "official.json"
    source.write_text(
        json.dumps(
            {
                "documents": [{"text": "First contract"}, {"text": "Second contract"}],
                "labels": {
                    "nda-11": {
                        "short_description": "No reverse engineering",
                        "hypothesis": "Do not reverse engineer.",
                    },
                    "nda-2": {
                        "short_description": "Confidential",
                        "hypothesis": "Keep it secret.",
                    },
                },
            }
        )
    )
    out = prepare_contract_nli(source, tmp_path / "converted")
    assert (out / "contracts/1.txt").read_text() == "First contract"
    assert (out / "contracts/2.txt").read_text() == "Second contract"
    assert (out / "hypotheses/11.txt").read_text() == (
        "LABEL_ID: nda-11\nSHORT_DESCRIPTION: No reverse engineering\n\n"
        "HYPOTHESIS:\nDo not reverse engineer.\n"
    )
    assert sorted(p.name for p in (out / "hypotheses").iterdir()) == ["11.txt", "2.txt"]
    with pytest.raises(ValueError, match="fresh output"):
        prepare_contract_nli(source, out)


def test_medec_conversion_matches_repo_sample_exactly(tmp_path):
    from benchmark_multi_kalypso.plans import SAMPLE_DATA

    source = SAMPLE_DATA / "MEDEC-TrainingSet-1000.csv"
    target = prepare_medec(source, tmp_path / "medec")
    with source.open(newline="") as src, target.open(newline="") as dst:
        assert [r["data"] for r in csv.DictReader(dst)] == [
            r["data"] for r in csv.DictReader(src)
        ]
