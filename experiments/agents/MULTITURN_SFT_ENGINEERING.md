# Multi-turn soft-prompt SFT engineering check (task #34)

## Scope

User requested testing agent trajectories with new user messages and tool observations interleaved between assistant replies. This is a mechanics check, not a SWE benchmark result, teacher-data generation, or a formal distillation run. The fixture is hand-authored and contains no SWE Verified instances. It is not an Agentless evaluation response relabelled as a multi-turn training trajectory.

Each complete fixture is **one example**, with multiple disjoint supervised spans. It is not split into separate assistant/tool calls. The batch contains one ten-message conversation (four assistant messages, three tool observations including consecutive observations, two user turns, one system message) and a shorter three-message conversation to exercise padding.

## What the existing code does and does not support

`ResponseSFTExample.response_loss_mask`, `collate_examples`, `SoftPromptModel`, and `train_update` already support disjoint target spans, soft-prompt insertion, causal shift, and ignored padding. However, the existing ordinary `distillation.py` snapshot loop calls `make_response_sft_example(prompt_ids, response_ids)` **without a mask**; it assumes every response token is an assistant target. Its shared rendering builder also constructs a single user prompt. Thus appending tool/user text into `response_ids` would incorrectly supervise it. The legacy CLI has not been silently declared multi-turn-ready.

New `soft_prompt_distillation.multiturn.render_multiturn` renders the complete conversation once using the selected ChatML template, tokenizes the entire resulting string canonically, and aligns supervised assistant spans to fast-tokenizer offsets. It feeds the existing training primitives, not a separate model or optimizer implementation.

### Explicit mask contract

- Assistant reasoning, assistant natural-language answers, serialized tool-call payloads, and assistant `<|im_end|>` are targets.
- System/user/tool content, tool-response wrappers, role headers, separators between messages, padding, and soft-prompt positions are not targets.
- Only logits immediately preceding target tokens receive direct cross-entropy gradients (the usual causal shift). A masked tool/context token can still condition a later target; **loss masking is not attention masking**.
- One soft-prompt block is inserted before the first rendered user header. It is not repeated at every turn. This position is an explicit engineering choice, not a historical agent-training reproduction claim.
- Tokens crossing a supervision or insertion boundary fail closed; oversized complete examples are rejected, not truncated mid-message. The renderer's length cap is text-only; callers must reserve soft-prompt length when applying a model's total context limit.
- Reserved ChatML control strings inside messages or tool definitions are rejected to prevent a tool/user payload from forging a supervised assistant region. Literal words such as `assistant` are not delimiters.
- Non-ChatML templates are unsupported and rejected. Tool messages may be rendered as user-role `<tool_response>` blocks by the original tokenizer; they remain ignored.

## Reasoning removal is a separate context decision

The fixed Nemotron SFT tokenizer's template conditionally removes reasoning from earlier assistant turns when a subsequent real user message appears. Applying it to a complete trajectory can therefore erase reasoning we intended to train on.

The current full-conversation numerical probe explicitly replaces exactly one training-template conditional:

```jinja
{%- if loop.index0 > ns.last_query_index %}
```

with:

```jinja
{%- if true %}
```

This preserves reasoning for all assistant turns. It **changes later replies' training context**, and is **not equivalent** to the original rollout context. Both template hashes and the modified template are retained. No evaluation/rollout template is changed.

If the intended contract is to reproduce removal of earlier reasoning, loss masks alone cannot accomplish both learning an earlier reasoning span and hiding it from later answers in a conventional causal sequence. Recommended alternative discussed with the user: one example per **real user turn**, retaining that turn's full assistant/tool interaction; earlier history is zero-loss context with old reasoning removed. This is not one example per tool call. More complex branch/attention layouts are possible but not implemented. Do not silently switch grouping or claim either choice is context-equivalent without defining the training contract.

## Validation and artifacts

The CPU checks use the exact SFT tokenizer, plus synthetic boundary tests and a direct logits-gradient check. The GPU probe uses the real fixed SFT model:

- `nvidia/Nemotron-Cascade-8B-Intermediate-ckpts`
- revision `9b379bd8f8ecb1feb19825efefe319bd30d8a223`
- subfolder `Nemotron-Cascade-8B-SFT`

It uses 16 soft tokens for a short engineering test, FP32 prompt parameters, BF16 frozen model weights, SDPA, token-normalized response CE, and AdamW at 0.01 with zero weight decay. Only the prompt parameter is passed to the optimizer. Two updates precede saving prompt/optimizer/RNG state; a third uninterrupted update is compared with reloading and continuing one update. Entire backbone parameter bytes are hashed before and after, rather than checking only a few sampled parameters.

Local artifacts are under `artifacts/agent-multiturn-20260911/`: complete synthetic messages, rendered text, original/training templates, per-token mask trace, prompt-only engineering checkpoint, training report and logs. Archive scripts assume this workspace's `ops/` and checkout layout. The short checkpoint format is for this probe; it does not claim validation of the legacy CLI's resumable multi-turn data-loader format.

### Preserved initial failure

The first CPU/GPU attempts (55941/55932) rejected the trusted template string itself in the reserved-delimiter payload check, before model training. The payload check was corrected to exclude only the explicitly supplied template while continuing to inspect messages and tool definitions. Both failure logs were retained. CPU retry55956 passed the real-tokenizer alignment check: 355 text tokens, 115 supervised targets, four assistant spans. Real-model numerical completion and regression results must be read from the final receipts, not inferred from that CPU pass.

## Completed numerical result

GPU job55961/workers-26 (1GPU/8CPU/64GiB) exited0. Two full conversation examples contained355/198 text tokens and4/1 assistant spans,146 supervised tokens in the batch. Loss was2.6925→2.4098→2.2176; all measured prompt gradients were finite and nonzero. The full frozen-backbone SHA256 matched before/after. Reloading prompt and AdamW state reproduced the continued third update **bitwise**, including optimizer tensors. Peak allocated GPU memory was21.29GB. These short-fixture loss values are not generalization results.

CPU regression job55974/workers-25 exited0:475 tests passed,1 two-GPU test skipped. Both allocations were released. Receipts, synthetic data, exact rendered chat and token-level mask trace are committed under `multiturn-sft/receipts/`.

Remaining integration: define the actual training-data grouping/context policy, connect that schema to the legacy snapshot/CLI entry, and validate authentic multi-turn trajectories. The existing CLI's single-response snapshot path is not represented as fully upgraded by this probe.
