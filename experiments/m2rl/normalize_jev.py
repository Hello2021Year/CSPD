"""Normalize the JEV judge dataset without changing the recorded agent states."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl")
OLD_RULE = "Answer only YES or NO."
NEW_RULE = "Reply with exactly one lowercase word: yes or no. Do not add punctuation, explanation, or any other text."


def normalize_record(record):
    messages = record.get("messages") if isinstance(record, dict) else None
    if not isinstance(messages, list) or len(messages) != 3 or not all(
        isinstance(message, dict) and isinstance(message.get("content"), str) for message in messages
    ):
        raise ValueError("expected three messages with string content")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("expected system, user, and assistant roles in that order")

    system, _, assistant = messages
    prompt = system["content"]
    if prompt.endswith(OLD_RULE):
        system["content"] = prompt.removesuffix(OLD_RULE) + NEW_RULE
    elif not prompt.endswith(NEW_RULE):
        raise ValueError("unexpected JEV system prompt")
    label = assistant["content"].strip().lower()
    if label not in {"yes", "no"}:
        raise ValueError("assistant label must be yes or no")
    assistant["content"] = label
    return label


def normalize_archive(source, output):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with ZipFile(source) as original, TemporaryDirectory(dir=output.parent) as staging:
        pending = Path(staging) / "result.zip"
        pending.touch(mode=0o600)
        with ZipFile(pending, "w", compression=ZIP_DEFLATED) as updated:
            for split in SPLITS:
                if original.namelist().count(split) != 1:
                    raise ValueError(f"expected exactly one {split} in {source}")
                counts[split] = {"yes": 0, "no": 0}
                with original.open(split) as incoming, updated.open(split, "w", force_zip64=True) as outgoing:
                    for line_number, line in enumerate(incoming, 1):
                        try:
                            record = json.loads(line)
                            label = normalize_record(record)
                        except ValueError as error:
                            raise ValueError(f"{split}:{line_number}: {error}") from error
                        counts[split][label] += 1
                        outgoing.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        pending.replace(output)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JEV ZIP with train, validation, and test JSONL")
    parser.add_argument("output", type=Path, help="new ZIP path")
    args = parser.parse_args()
    try:
        print(json.dumps(normalize_archive(args.source, args.output), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
