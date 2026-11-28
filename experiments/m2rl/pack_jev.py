"""Pack labelled JEV training examples for CSPD with a local Qwen ChatML tokenizer."""

import argparse
import array
import json
import sys
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from normalize_jev import normalize_record
from pack_data import sha256

TRAIN_SPLIT = "train.jsonl"
SYSTEM_HEADER = "<|im_start|>system\n"
TURN_START = "<|im_start|>"
TURN_END = "<|im_end|>"


def pack_train(source, output, tokenizer, *, context_length=65536, reserve_tokens=512,
               limit=None, skip_oversize=False, supervise_turn_end=True):
    if not 0 <= reserve_tokens < context_length:
        raise ValueError("require 0 <= reserve_tokens < context_length")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    start_id = tokenizer.convert_tokens_to_ids(TURN_START)
    end_id = tokenizer.convert_tokens_to_ids(TURN_END)
    if (start_id is None or end_id is None or start_id == end_id
            or tokenizer.encode(TURN_START, add_special_tokens=False) != [start_id]
            or tokenizer.encode(TURN_END, add_special_tokens=False) != [end_id]):
        raise ValueError("this packer requires Qwen ChatML start/end tokens")
    header = tokenizer.encode(SYSTEM_HEADER, add_special_tokens=False)
    if not header or header[0] != start_id:
        raise ValueError("tokenizer does not encode the Qwen system header as expected")
    suffix = [end_id] if supervise_turn_end else []
    targets = {
        label: tokenizer.encode(label, add_special_tokens=False) + suffix
        for label in ("yes", "no")
    }
    entries, dropped = [], []
    labels = {"yes": 0, "no": 0}
    position = 0
    usable = context_length - reserve_tokens
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(source) as archive, TemporaryDirectory(dir=output.parent) as staging:
        if archive.namelist().count(TRAIN_SPLIT) != 1:
            raise ValueError(f"expected exactly one {TRAIN_SPLIT} in {source}")
        packed = Path(staging) / "packed"
        packed.mkdir(mode=0o700)
        token_path, index_path = packed / "tokens.uint32", packed / "index.json"
        with archive.open(TRAIN_SPLIT) as incoming, token_path.open("wb") as token_file:
            for line_number, line in enumerate(islice(incoming, limit), 1):
                location = f"{TRAIN_SPLIT}:{line_number}"
                try:
                    record = json.loads(line)
                    label = normalize_record(record)
                except ValueError as error:
                    raise ValueError(f"{location}: {error}") from error
                prefix = tokenizer.apply_chat_template(
                    record["messages"][:2], tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_dict=False,
                )
                if prefix[:len(header)] != header:
                    raise ValueError(f"{location}: chat template does not start with the Qwen system header")
                target = targets[label]
                length = len(prefix) + len(target)
                if length > usable:
                    if not skip_oversize:
                        raise ValueError(f"{location}: {length} tokens exceeds usable limit {usable}; "
                                         "adjust the context budget or pass --skip-oversize")
                    dropped.append({"line": line_number, "tokens": length, "label": label})
                    continue
                values = array.array("I", prefix + target)
                if sys.byteorder != "little":
                    values.byteswap()
                token_file.write(values.tobytes())
                sample_id = f"jev-train-{line_number:06d}"
                entries.append(dict(question_id=sample_id, request_id=sample_id, offset=position,
                                    length=length, prefix_tokens=len(prefix), supervised_tokens=len(target),
                                    insert_at=len(header)))
                labels[label] += 1
                position += length
        if not entries:
            raise ValueError("no JEV training samples were packed")
        index_path.write_text(json.dumps(entries, indent=2) + "\n")
        manifest = dict(
            split=TRAIN_SPLIT, tokenizer=tokenizer.name_or_path, records=len(entries), tokens=position,
            context_length=context_length, reserve_tokens=reserve_tokens, label_counts=labels,
            supervise_turn_end=supervise_turn_end,
            skipped_oversize=dropped, limit=limit, source_sha256=sha256(source),
            tokens_sha256=sha256(token_path), index_sha256=sha256(index_path),
        )
        (packed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        packed.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original or normalized JEV ZIP")
    parser.add_argument("output", type=Path, help="new directory for packed train records")
    parser.add_argument("--tokenizer", type=Path, required=True, help="local student model or tokenizer directory")
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--reserve-tokens", type=int, default=512, help="budget for soft prompt and inserted prefix")
    parser.add_argument("--limit", type=int, help="only consider the first N training records")
    parser.add_argument("--skip-oversize", action="store_true", help="record and skip over-budget samples instead of failing")
    parser.add_argument("--label-only", action="store_true", help="do not add an end-of-turn target")
    args = parser.parse_args()
    if not args.tokenizer.is_dir():
        parser.error("--tokenizer must be a local directory")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
        report = pack_train(args.source, args.output, tokenizer, context_length=args.context_length,
                            reserve_tokens=args.reserve_tokens, limit=args.limit, skip_oversize=args.skip_oversize,
                            supervise_turn_end=not args.label_only)
    except (BadZipFile, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
