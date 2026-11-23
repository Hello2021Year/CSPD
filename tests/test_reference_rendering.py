from __future__ import annotations

import pytest

from soft_prompt_distillation.config import ConfigError, parse_config
from soft_prompt_distillation.rendering import (
    render_for_execution,
    resolve_insertion_index,
    token_ids_sha256,
)
from test_reference_config import valid_config_dict


class FakeTokenizer:
    def __init__(self, chat_template: str, marker: str, *, offset: int = 0):
        self.chat_template = chat_template
        self.marker = marker
        self.offset = offset
        self.chat_template_kwargs = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        assert tokenize is False
        self.chat_template_kwargs.append(kwargs)
        body = "|".join(f"{item['role']}={item['content']}" for item in messages)
        suffix = "|assistant=" if add_generation_prompt else ""
        return f"{self.marker}|{body}{suffix}"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.offset + ord(character) for character in text]


class BoundaryMergingTokenizer(FakeTokenizer):
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        position = 0
        while position < len(text):
            if text.startswith("=Q", position):
                ids.append(999)
                position += 2
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids


def test_marker_resolves_to_sentinel_but_not_model_token_ids() -> None:
    raw = valid_config_dict()
    config = parse_config(raw)
    tokenizer = FakeTokenizer("shared-template", "SHARED")

    rendered = render_for_execution(
        config,
        execution_model="student",
        row={"problem": "2+2"},
        tokenizer=tokenizer,
    )

    assert rendered.text.startswith("SHARED|")
    assert rendered.template_source == "teacher"
    assert rendered.execution_model == "student"
    assert rendered.token_ids[0] == ord("S")
    assert -200 not in rendered.token_ids
    assert rendered.control_token_ids is not None
    assert rendered.control_token_ids[rendered.soft_prompt_insert_at] == -200
    assert "<|soft_prompt|>" not in rendered.text
    assert all(
        "<|soft_prompt|>" not in message["content"] for message in rendered.messages
    )


def test_rendering_accepts_an_evaluation_only_template() -> None:
    raw = valid_config_dict()
    raw["evaluation"]["template"] = {
        "source": "teacher",
        "system": {"mode": "text", "text": "Evaluation system."},
        "user": {
            "field": "problem",
            "prefix": "<|soft_prompt|>",
            "suffix": "",
        },
        "add_generation_prompt": True,
    }
    config = parse_config(raw)
    tokenizer = FakeTokenizer("teacher-template", "TEACHER")

    rollout = render_for_execution(
        config,
        execution_model="teacher",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )
    evaluation = render_for_execution(
        config,
        execution_model="teacher",
        row={"problem": "x"},
        tokenizer=tokenizer,
        template=config.evaluation.template,
    )

    assert "system=Solve carefully." in rollout.text
    assert "system=Evaluation system." in evaluation.text


def test_placeholder_resolves_to_a_stable_token_boundary() -> None:
    raw = valid_config_dict()
    config = parse_config(raw)
    tokenizer = FakeTokenizer("teacher-template", "TEACHER")
    rendered = render_for_execution(
        config,
        execution_model="student",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )
    expected = len("TEACHER|system=Solve carefully.|user=")
    assert resolve_insertion_index(config, rendered) == expected
    assert rendered.text.startswith(
        "TEACHER|system=Solve carefully.|user=Question: x"
    )


def test_placeholder_in_question_is_rejected_as_second_occurrence() -> None:
    raw = valid_config_dict()
    config = parse_config(raw)
    tokenizer = FakeTokenizer("teacher-template", "TEACHER")
    with pytest.raises(ConfigError, match="exactly one"):
        render_for_execution(
            config,
            execution_model="student",
            row={"problem": "bad <|soft_prompt|> input"},
            tokenizer=tokenizer,
        )


def test_placeholder_uses_segmented_ids_if_removal_changes_token_boundary() -> None:
    raw = valid_config_dict()
    config = parse_config(raw)
    tokenizer = BoundaryMergingTokenizer("teacher-template", "TEACHER")
    rendered = render_for_execution(
        config,
        execution_model="student",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )

    assert rendered.control_token_ids is not None
    without_sentinel = (
        rendered.control_token_ids[: rendered.soft_prompt_insert_at]
        + rendered.control_token_ids[rendered.soft_prompt_insert_at + 1 :]
    )
    canonical_ids = tuple(tokenizer.encode(rendered.text, add_special_tokens=False))
    assert rendered.tokenization_strategy == "marker_segmented"
    assert rendered.token_ids == without_sentinel
    assert rendered.token_ids != canonical_ids
    assert rendered.canonical_token_count == len(canonical_ids)
    assert rendered.canonical_token_ids_sha256 == token_ids_sha256(canonical_ids)
    assert rendered.tokenization_matches_canonical is False
    assert 999 in canonical_ids
    assert 999 not in rendered.token_ids


def test_template_text_is_encoded_with_the_execution_tokenizer() -> None:
    raw = valid_config_dict()
    config = parse_config(raw)
    template_tokenizer = FakeTokenizer("teacher-template", "TEACHER")
    execution_tokenizer = FakeTokenizer(
        "student-template", "STUDENT", offset=1000
    )

    rendered = render_for_execution(
        config,
        execution_model="student",
        row={"problem": "2+2"},
        tokenizer=template_tokenizer,
        execution_tokenizer=execution_tokenizer,
    )

    assert rendered.text.startswith("TEACHER|")
    assert all(token_id >= 1000 for token_id in rendered.token_ids)
    assert rendered.control_token_ids[rendered.soft_prompt_insert_at] == -200
    assert (
        rendered.control_token_ids[: rendered.soft_prompt_insert_at]
        + rendered.control_token_ids[rendered.soft_prompt_insert_at + 1 :]
        == rendered.token_ids
    )


def test_rendered_text_and_token_ids_have_separate_hashes() -> None:
    raw = valid_config_dict()
    raw["template"]["source"] = "student"
    config = parse_config(raw)
    tokenizer = FakeTokenizer("student-template", "STUDENT")
    rendered = render_for_execution(
        config,
        execution_model="student",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )
    assert len(rendered.text_sha256) == 64
    assert len(rendered.token_ids_sha256) == 64
    assert len(rendered.canonical_token_ids_sha256) == 64
    assert rendered.tokenization_matches_canonical is True
    assert rendered.text_sha256 != rendered.token_ids_sha256


def test_rendering_passes_chat_kwargs_and_appends_assistant_prefix() -> None:
    raw = valid_config_dict()
    raw["template"]["chat_template_kwargs"] = {"enable_thinking": True}
    raw["template"]["assistant_prefix"] = "<think>\n"
    config = parse_config(raw)
    tokenizer = FakeTokenizer("teacher-template", "TEACHER")

    rendered = render_for_execution(
        config,
        execution_model="teacher",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )

    assert tokenizer.chat_template_kwargs == [{"enable_thinking": True}]
    assert rendered.text.endswith("|assistant=<think>\n")


def test_raw_rendering_bypasses_chat_template_byte_for_byte() -> None:
    raw = valid_config_dict()
    raw["template"]["render_mode"] = "raw"
    raw["template"]["system"] = {"mode": "model_default"}
    raw["template"]["add_generation_prompt"] = False
    config = parse_config(raw)
    tokenizer = FakeTokenizer("teacher-template", "TEACHER")

    rendered = render_for_execution(
        config,
        execution_model="teacher",
        row={"problem": "x"},
        tokenizer=tokenizer,
    )

    assert tokenizer.chat_template_kwargs == []
    assert rendered.render_mode == "raw"
    assert rendered.text == "Question: x"
    assert rendered.soft_prompt_insert_at == 0
