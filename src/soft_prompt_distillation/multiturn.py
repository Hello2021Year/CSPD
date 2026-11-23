"""Canonical ChatML multi-turn rendering with assistant-only loss.

Explicitly supports ChatML templates (including Nemotron/Qwen tool messages).
It never infers roles from arbitrary substrings inside message content. Reserved
ChatML controls inside untrusted payloads are rejected to avoid mask injection.
Other serialization formats fail closed instead of silently labelling tools.
"""
from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from .objective import AlignmentError, ResponseSFTExample, make_response_sft_example


@dataclass(frozen=True)
class MultiTurnExample:
    example: ResponseSFTExample
    text: str
    assistant_spans: tuple[tuple[int, int], ...]
    token_offsets: tuple[tuple[int, int], ...]
    insert_at: int


def render_multiturn(tokenizer: Any, messages: Sequence[Mapping[str, Any]], *,
                     template_kwargs: Mapping[str, Any] | None = None,
                     max_length: int | None = None,
                     insertion: str = 'before_first_user') -> MultiTurnExample:
    """Label assistant bodies (including tool calls/think) plus im_end, not headers.

    All system/user/tool bodies and all inter-message separators are ignored.
    Insertion defaults to before first user for compatibility; system_body_start
    places it after the first system role header, before all rendered tools/text.
    Canonical full-chat tokenization is used, not concatenated per-message IDs.
    Overlength examples are rejected whole. Mask boundaries must match tokens.

    Some templates remove earlier reasoning after a new user turn: this function
    labels only what the canonical full rendering retains, not erased tokens.
    Training must explicitly choose whether to preserve reasoning in its template.
    """
    if insertion not in {'before_first_user', 'system_body_start'}:
        raise AlignmentError('unsupported soft prompt insertion')
    if not messages or messages[0]['role'] not in {'system', 'user'}:
        raise AlignmentError('trajectory must start with system or user')
    if any(m.get('role') not in {'system', 'user', 'assistant', 'tool'} for m in messages):
        raise AlignmentError('unsupported message role')
    kwargs = dict(template_kwargs or {})
    if set(kwargs) & {'tokenize', 'add_generation_prompt', 'return_dict'}:
        raise AlignmentError('reserved template kwargs')
    payload = json.dumps([list(messages), {k: v for k, v in kwargs.items() if k != 'chat_template'}], ensure_ascii=False)
    if any(control in payload for control in ('<|im_start|>', '<|im_end|>')):
        raise AlignmentError('reserved role delimiter inside message payload')
    text = tokenizer.apply_chat_template(list(messages), tokenize=False,
                                         add_generation_prompt=False, **kwargs)
    pattern = re.compile(r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>', re.DOTALL)
    spans = []; rendered_roles = []; previous = 0; insertion_char = None
    for match in pattern.finditer(text):
        if text[previous:match.start()].strip():
            raise AlignmentError('unsupported or malformed ChatML template')
        role = match.group(1); rendered_roles.append(role)
        if insertion_char is None:
            if insertion == 'before_first_user' and role == 'user':
                insertion_char = match.start()
            elif insertion == 'system_body_start' and role == 'system' and len(rendered_roles) == 1:
                insertion_char = match.start(2)
        if role == 'assistant':
            spans.append((match.start(2), match.end()))
        previous = match.end()
    if text[previous:].strip() or not spans or insertion_char is None:
        raise AlignmentError('missing or malformed ChatML turns')
    if rendered_roles.count('assistant') != sum(m['role'] == 'assistant' for m in messages):
        raise AlignmentError('template changed number of assistant turns')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = tuple(encoded['input_ids']); offsets = tuple(map(tuple, encoded['offset_mapping']))
    if max_length is not None and len(ids) > max_length:
        raise AlignmentError('overlength trajectory: filter whole example')
    selected = []
    for start, end in offsets:
        overlaps = [(a, b) for a, b in spans if start < b and end > a]
        if overlaps and not any(a <= start < end <= b for a, b in overlaps):
            raise AlignmentError('token crosses supervised boundary')
        selected.append(bool(overlaps))
    for a, b in spans:
        covered = [(s, e) for s, e in offsets if a <= s < e <= b]
        if not covered or covered[0][0] != a or covered[-1][1] != b:
            raise AlignmentError('assistant span not fully represented by token offsets')
    first = selected.index(True)
    if first == 0:
        raise AlignmentError('missing unlabelled context')
    if any(s < insertion_char < e for s, e in offsets):
        raise AlignmentError('soft prompt boundary crosses token')
    insert_at = sum(e <= insertion_char for s, e in offsets)
    if insert_at > first:
        raise AlignmentError('soft prompt must precede supervised tokens')
    return MultiTurnExample(make_response_sft_example(ids[:first], ids[first:], selected[first:]),
                            text, tuple(spans), offsets, insert_at)
