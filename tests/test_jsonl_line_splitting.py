"""JSONL records must only ever be split on a newline.

Every writer in this package uses ensure_ascii=False, so whatever a teacher
generates reaches the file verbatim. JSON does not escape U+2028, U+0085 or the
C0 vertical tab / form feed, but Python's str.splitlines() treats all of them as
line breaks. Reading a snapshot with splitlines() therefore cuts a record in two
and both fragments fail to parse.

This is not hypothetical: prorl-1p5b (job 865) finished a 26-minute sampling
stage, wrote 23042 valid records, and then died on the first line of training
with "Unterminated string starting at: line 1 column 17195". Two of those 23042
teacher completions contained U+2028.
"""

from __future__ import annotations

import json

import pytest

# common.py imports torch at module scope, so this module cannot be collected on
# a torch-less host even though the readers under test are pure text handling.
pytest.importorskip("torch")

from soft_prompt_distillation.common import read_json_rows, write_jsonl


# json.dumps escapes every C0 control character, so a vertical tab or form feed
# can never reach the file raw. The reachable separators are exactly the
# non-ASCII ones, which ensure_ascii=False passes through untouched.
SPLITLINES_ONLY = [
    ("line_separator", "\u2028"),
    ("paragraph_separator", "\u2029"),
    ("next_line", "\u0085"),
]


@pytest.mark.parametrize(
    "separator", [s for _, s in SPLITLINES_ONLY], ids=[n for n, _ in SPLITLINES_ONLY]
)
def test_record_holding_a_splitlines_separator_survives_a_round_trip(
    tmp_path, separator
):
    path = tmp_path / "teacher_trajectories.jsonl"
    rows = [
        {"question_id": "a", "completion": "before"},
        {"question_id": "b", "completion": "x" + separator + "y"},
        {"question_id": "c", "completion": "after"},
    ]
    write_jsonl(path, rows)

    # Guard the premise. The separator has to reach the file unescaped, and
    # str.splitlines() has to see one more line than there are records --
    # otherwise this test would pass against the reader that shipped the bug.
    text = path.read_text(encoding="utf-8")
    assert separator in text
    assert len(text.strip().split("\n")) == 3
    assert len(text.strip().splitlines()) == 4

    assert read_json_rows(path) == rows


def test_writer_escapes_a_real_newline_inside_a_completion(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"completion": "first\nsecond"}]
    write_jsonl(path, rows)

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert read_json_rows(path) == rows


def test_reader_still_accepts_a_json_array(tmp_path):
    path = tmp_path / "rows.json"
    rows = [{"a": 1}, {"b": 2}]
    path.write_text(json.dumps(rows), encoding="utf-8")

    assert read_json_rows(path) == rows
