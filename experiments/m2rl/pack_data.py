"""Pack verified teacher responses into the token stream read by train_sp.py and baselines/.

  python experiments/m2rl/pack_data.py CANDIDATES_DIR OUT_DIR [--max-per-question 1]

``CANDIDATES_DIR`` is the output directory of ``teacher/sample_teacher.py``. Its
``candidates/shard-*.jsonl`` holds one correct teacher response per line with ``input_ids`` (prompt + response, Qwen3 chat template, thinking enabled), ``labels`` (-100 on the
prompt), ``prefix_tokens``, ``supervised_tokens``, ``score``, ``question_id`` and ``request_id``.
An empty system turn is added in front of inputs that have none, and ``insert_at`` records the
position right after the system header, where the soft prompt goes.

Output: ``tokens.uint32`` (little-endian), ``index.json`` (one entry per response with offset,
length, prefix_tokens, supervised_tokens, insert_at) and ``manifest.json``.
The packed domains used in the paper are released with the checkpoints.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path

USER_HEADER = [151644, 872, 198]          # <|im_start|>user\n
END_TOKENS = (151645, 151643)             # <|im_end|>, <|endoftext|>


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_scaffold(tokenizer):
    header = tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)
    close = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    example = [{"role": "user", "content": "scaffold-check"}]
    kw = dict(tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
    original = tokenizer.apply_chat_template(example, **kw)
    empty = tokenizer.apply_chat_template([{"role": "system", "content": ""}, *example], **kw)
    if empty != header + close + original:
        raise ValueError("empty system scaffold does not match the chat template")
    return header, close


def add_system(input_ids, prefix_tokens, header, close):
    ids = list(input_ids)
    if ids[:len(header)] == header:
        return ids, prefix_tokens, len(header), False
    if ids[:3] != USER_HEADER:
        raise ValueError("expected a system or user turn first")
    return header + close + ids, prefix_tokens + len(header) + len(close), len(header), True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--max-per-question", type=int, default=1, help="1 for math/science/IF, 2 for code")
    p.add_argument("--model", default="Jackwang111/M2RL-SFT")
    p.add_argument("--revision", default="aa49d49a86b15ac920cf356c359f299edd0c1eb3")
    args = p.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    header, close = system_scaffold(tokenizer)
    files = sorted((args.candidates / "candidates").glob("shard-*.jsonl"))
    if not files:
        raise SystemExit(f"no candidates/shard-*.jsonl under {args.candidates}")
    args.output.mkdir(parents=True, exist_ok=True)
    entries, position, seen, per_question = [], 0, set(), {}
    counts = {"existing_system": 0, "added_empty_system": 0}
    with (args.output / "tokens.uint32").open("wb") as stream:
        for path in files:
            for line in path.open():
                row = json.loads(line)
                prefix, original, labels = row["prefix_tokens"], row["input_ids"], row["labels"]
                if row["score"] != 1:
                    raise ValueError(f"{row['request_id']}: only verified-correct responses are packed")
                per_question[row["question_id"]] = per_question.get(row["question_id"], 0) + 1
                if per_question[row["question_id"]] > args.max_per_question:
                    raise ValueError(f"{row['question_id']}: more than {args.max_per_question} responses")
                if row["request_id"] in seen or labels != [-100] * prefix + original[prefix:]:
                    raise ValueError(f"{row['request_id']}: duplicate request or response mask mismatch")
                if not original[prefix:] or row["supervised_tokens"] != len(original) - prefix or original[-1] not in END_TOKENS:
                    raise ValueError(f"{row['request_id']}: incomplete response")
                seen.add(row["request_id"])
                ids, prefix, insert, added = add_system(original, prefix, header, close)
                counts["added_empty_system" if added else "existing_system"] += 1
                values = array.array("I", ids)
                if sys.byteorder != "little":
                    values.byteswap()
                stream.write(values.tobytes())
                entries.append(dict(request_id=row["request_id"], question_id=row["question_id"],
                                    offset=position, length=len(ids), prefix_tokens=prefix,
                                    supervised_tokens=len(ids) - prefix, insert_at=insert))
                position += len(ids)
    (args.output / "index.json").write_text(json.dumps(entries))
    manifest = dict(records=len(entries), tokens=position, system_counts=counts,
                    tokens_sha256=sha256(args.output / "tokens.uint32"), index_sha256=sha256(args.output / "index.json"))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
