"""Single rendering path for sampling, distillation, and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import ConfigError, OffPolicyConfig, TemplateSpec


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids], separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def tokenizer_chat_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ConfigError("tokenizer has no non-empty chat_template")
    return _sha256_text(template)


def build_messages(template: TemplateSpec, row: Mapping[str, Any]) -> list[dict[str, str]]:
    value = row.get(template.user.field)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"row field {template.user.field!r} must be a non-empty string"
        )
    user = f"{template.user.prefix}{value}{template.user.suffix}"
    messages = [{"role": "user", "content": user}]
    if template.system.mode == "empty":
        messages.insert(0, {"role": "system", "content": ""})
    elif template.system.mode == "text":
        assert template.system.text is not None
        messages.insert(0, {"role": "system", "content": template.system.text})
    return messages


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    template_source: str
    render_mode: str
    execution_model: str
    messages: tuple[Mapping[str, str], ...]
    text: str
    token_ids: tuple[int, ...]
    control_token_ids: tuple[int, ...] | None
    soft_prompt_insert_at: int | None
    tokenization_strategy: str
    text_sha256: str
    token_ids_sha256: str
    canonical_token_count: int
    canonical_token_ids_sha256: str
    tokenization_matches_canonical: bool


def render_for_execution(
    config: OffPolicyConfig,
    *,
    execution_model: str,
    row: Mapping[str, Any],
    tokenizer: Any,
    execution_tokenizer: Any | None = None,
    template: TemplateSpec | None = None,
) -> RenderedPrompt:
    """Render the shared template and resolve its soft-prompt placeholder.

    ``tokenizer`` owns the canonical prompt contract: either its chat template
    or the configured raw user text.  The marker's left and right text are
    encoded independently with ``execution_tokenizer`` (or the template
    tokenizer when omitted), so the no-soft-prompt IDs are exactly the control
    IDs with the sentinel removed while each model receives IDs from its own
    vocabulary.
    """

    if execution_model not in {"student", "teacher"}:
        raise ConfigError("execution_model must be student or teacher")
    template = template or config.template
    messages = build_messages(template, row)
    if template.render_mode == "chat":
        tokenizer_chat_template_sha256(tokenizer)
        chat_template_kwargs = dict(template.chat_template_kwargs)
        marked_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=template.add_generation_prompt,
            **chat_template_kwargs,
        )
        if not isinstance(marked_text, str) or not marked_text:
            raise ConfigError("apply_chat_template returned no rendered text")
        marked_text += template.assistant_prefix
    else:
        marked_text = messages[-1]["content"]
    insertion = config.distillation.insertion
    marker_count = marked_text.count(insertion.marker)
    if marker_count != 1:
        raise ConfigError("rendered template must contain exactly one soft-prompt marker")
    left, right = marked_text.split(insertion.marker)
    text = left + right
    clean_messages = [
        {**message, "content": message["content"].replace(insertion.marker, "")}
        for message in messages
    ]
    if execution_tokenizer is None:
        execution_tokenizer = tokenizer
    left_ids = tuple(
        int(token_id)
        for token_id in execution_tokenizer.encode(left, add_special_tokens=False)
    )
    right_ids = tuple(
        int(token_id)
        for token_id in execution_tokenizer.encode(right, add_special_tokens=False)
    )
    insert_at = len(left_ids)
    control_ids = left_ids + (insertion.sentinel_id,) + right_ids
    ids = left_ids + right_ids
    if not ids:
        raise ConfigError("tokenizer produced an empty marker-segmented prompt")
    if any(token_id < 0 for token_id in ids):
        raise ConfigError("tokenizer produced a negative model token ID")
    if ids != control_ids[:insert_at] + control_ids[insert_at + 1:]:
        raise ConfigError(
            "removing the soft-prompt sentinel did not reproduce the "
            "marker-segmented prompt IDs"
        )

    canonical_token_ids = execution_tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(canonical_token_ids, Sequence) or isinstance(
        canonical_token_ids, (str, bytes)
    ):
        raise ConfigError("tokenizer did not return a token-ID sequence")
    canonical_ids = tuple(int(token_id) for token_id in canonical_token_ids)
    if not canonical_ids:
        raise ConfigError("tokenizer produced an empty canonical prompt")
    if any(token_id < 0 for token_id in canonical_ids):
        raise ConfigError("tokenizer produced a negative model token ID")
    return RenderedPrompt(
        template_source=template.source,
        render_mode=template.render_mode,
        execution_model=execution_model,
        messages=tuple(dict(message) for message in clean_messages),
        text=text,
        token_ids=ids,
        control_token_ids=control_ids,
        soft_prompt_insert_at=insert_at,
        tokenization_strategy="marker_segmented",
        text_sha256=_sha256_text(text),
        token_ids_sha256=token_ids_sha256(ids),
        canonical_token_count=len(canonical_ids),
        canonical_token_ids_sha256=token_ids_sha256(canonical_ids),
        tokenization_matches_canonical=ids == canonical_ids,
    )


def resolve_insertion_index(
    config: OffPolicyConfig,
    rendered: RenderedPrompt,
) -> int:
    """Return the placeholder position resolved during marker-segmented rendering."""

    if rendered.soft_prompt_insert_at is None or rendered.control_token_ids is None:
        raise ConfigError("rendered template has no resolved soft-prompt placeholder")
    sentinel = config.distillation.insertion.sentinel_id
    if rendered.control_token_ids.count(sentinel) != 1:
        raise ConfigError("control token IDs must contain exactly one soft-prompt sentinel")
    if rendered.control_token_ids[rendered.soft_prompt_insert_at] != sentinel:
        raise ConfigError("soft-prompt sentinel and insertion index disagree")
    return rendered.soft_prompt_insert_at


__all__ = [
    "RenderedPrompt",
    "build_messages",
    "render_for_execution",
    "resolve_insertion_index",
    "token_ids_sha256",
    "tokenizer_chat_template_sha256",
]
