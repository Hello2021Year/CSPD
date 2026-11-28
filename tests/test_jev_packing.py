import json
import struct
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest


M2RL = Path(__file__).resolve().parents[1] / "experiments" / "m2rl"


class ChatTokenizer:
    name_or_path = "test-tokenizer"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"<|im_start|>": [1], "<|im_end|>": [9], "<|im_start|>system\n": [1, 2, 3],
                "yes": [7], "no": [8]}[text]

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 9}[token]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict):
        assert tokenize and add_generation_prompt and not enable_thinking and not return_dict
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "lowercase word: yes or no." in messages[0]["content"]
        return [1, 2, 3] + [4] * len(messages[1]["content"]) + [5]


def test_jev_packer_preserves_both_labels_and_masks_the_prefix(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(M2RL))
    from pack_jev import pack_train
    from train_sp import load_packed

    source = tmp_path / "input.zip"
    rows = []
    for label, evidence in [("YES", "a"), ("NO", "long evidence"), ("NO", "b")]:
        rows.append(json.dumps({"messages": [
            {"role": "system", "content": "Judge this. Answer only YES or NO."},
            {"role": "user", "content": evidence},
            {"role": "assistant", "content": label},
        ]}))
    with ZipFile(source, "w") as archive:
        archive.writestr("train.jsonl", "\n".join(rows))
        archive.writestr("validation.jsonl", "not a training record")
        archive.writestr("test.jsonl", "not a training record")

    output = tmp_path / "packed"
    with pytest.raises(ValueError, match="train.jsonl:2: .* exceeds usable limit"):
        pack_train(source, output, ChatTokenizer(), context_length=10, reserve_tokens=2)
    assert not output.exists()

    manifest = pack_train(source, output, ChatTokenizer(), context_length=10, reserve_tokens=2, skip_oversize=True)
    entries, tokens = load_packed(output)
    assert manifest["label_counts"] == {"yes": 1, "no": 1}
    assert [report["line"] for report in manifest["skipped_oversize"]] == [2]
    assert [entry["offset"] for entry in entries] == [0, 7]
    assert [entry["question_id"] for entry in entries] == ["jev-train-000001", "jev-train-000003"]
    assert all(entry["insert_at"] == 3 and entry["prefix_tokens"] == 5 and entry["supervised_tokens"] == 2
               for entry in entries)
    assert tokens.tolist() == [1, 2, 3, 4, 5, 7, 9, 1, 2, 3, 4, 5, 8, 9]
    assert (output / "tokens.uint32").read_bytes() == struct.pack("<14I", *tokens)

    label_only = tmp_path / "label-only"
    report = pack_train(source, label_only, ChatTokenizer(), context_length=10, reserve_tokens=2,
                        skip_oversize=True, supervise_turn_end=False)
    entries, tokens = load_packed(label_only)
    assert report["supervise_turn_end"] is False
    assert [entry["supervised_tokens"] for entry in entries] == [1, 1]
    assert tokens.tolist() == [1, 2, 3, 4, 5, 7, 1, 2, 3, 4, 5, 8]


def test_prefix_pool_handles_a_dataset_smaller_than_2048(monkeypatch):
    monkeypatch.syspath_prepend(str(M2RL))
    from train_sp import prefix_pool

    tokens = np.arange(64)
    entries = [{"offset": 0, "insert_at": 3, "prefix_tokens": 40}]
    assert [span.tolist() for span in prefix_pool(entries, tokens)] == [list(range(3, 40))]
