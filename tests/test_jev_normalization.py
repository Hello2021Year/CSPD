import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "m2rl" / "normalize_jev.py"
SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl")
USER_CONTENT = '{"state_messages":[{"role":"system","content":"历史记录: NO"}]}'


def make_archive(path, labels):
    with ZipFile(path, "w") as archive:
        for split, label in zip(SPLITS, labels, strict=True):
            record = {
                "messages": [
                    {"role": "system", "content": "Judge the candidate. Answer only YES or NO."},
                    {"role": "user", "content": USER_CONTENT},
                    {"role": "assistant", "content": label},
                ]
            }
            archive.writestr(split, json.dumps(record) + "\n")


def test_normalizes_all_splits_and_preserves_quoted_evidence(tmp_path):
    source, output = tmp_path / "original.zip", tmp_path / "normalized.zip"
    make_archive(source, ("YES", "NO", "YES"))
    original = source.read_bytes()

    result = subprocess.run([sys.executable, str(SCRIPT), str(source), str(output)], capture_output=True, check=True)

    assert source.read_bytes() == original
    assert json.loads(result.stdout)["validation.jsonl"] == {"yes": 0, "no": 1}
    with ZipFile(output) as archive:
        for split, label in zip(SPLITS, ("yes", "no", "yes"), strict=True):
            messages = json.loads(archive.read(split))["messages"]
            assert messages[0]["content"] == (
                "Judge the candidate. Reply with exactly one lowercase word: yes or no. "
                "Do not add punctuation, explanation, or any other text."
            )
            assert messages[1]["content"] == USER_CONTENT
            assert messages[2]["content"] == label


def test_invalid_label_fails_without_publishing_a_partial_archive(tmp_path):
    source, output = tmp_path / "original.zip", tmp_path / "normalized.zip"
    make_archive(source, ("YES", "unknown", "NO"))

    result = subprocess.run([sys.executable, str(SCRIPT), str(source), str(output)], capture_output=True, text=True)

    assert result.returncode == 1
    assert "validation.jsonl:1: assistant label must be yes or no" in result.stderr
    assert not output.exists()
