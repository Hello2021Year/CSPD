"""Shared options, data loading, model loading, and run manifests."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from .config import OffPolicyConfig
from .rendering import tokenizer_chat_template_sha256
from .tasks import TaskAdapter


@dataclass(frozen=True, slots=True)
class RunOptions:
    config: Path
    output: Path
    cache_dir: Path
    student_model_path: Path | None = None
    teacher_model_path: Path | None = None
    soft_prompt_model_path: Path | None = None
    sglang_tp_size: int = 1
    sglang_dp_size: int = 1
    sglang_batch_size: int | str = 8192
    sglang_log_level: str = "error"
    generation_backend: str = "sglang"
    train_model_placement: str = "replicated"
    resume_training: bool = False
    reuse_sample_dir: Path | None = None
    sample_shard_index: int = 0
    sample_shard_count: int = 1
    sample_finalize_only: bool = False
    code_verifier_command: Path | None = None
    code_verifier_timeout_seconds: int = 21600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    rows = (
        json.loads(text)
        if text.startswith("[")
        # Split on "\n" and not with str.splitlines(). Every writer here uses
        # ensure_ascii=False, so a teacher completion containing U+2028, U+0085
        # or a vertical tab reaches the file verbatim -- JSON does not escape
        # them, but str.splitlines() treats them as line breaks and would cut
        # that record in half, leaving two fragments that fail to parse.
        else [json.loads(line) for line in text.split("\n") if line.strip()]
    )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path} does not contain JSON object rows")
    return rows


def read_dataset_rows(path: Path) -> list[dict[str, Any]]:
    """Read one downloaded JSON/JSONL, Parquet, or Arrow artifact."""

    suffix = path.suffix.lower()
    if suffix not in {".parquet", ".arrow"}:
        return read_json_rows(path)
    try:
        import pyarrow
        import pyarrow.ipc as ipc
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet/Arrow datasets require the project data dependencies"
        ) from exc
    if suffix == ".parquet":
        rows = parquet.read_table(path).to_pylist()
    else:
        with pyarrow.memory_map(str(path), "r") as source:
            try:
                table = ipc.open_stream(source).read_all()
            except pyarrow.ArrowInvalid:
                source.seek(0)
                table = ipc.open_file(source).read_all()
        rows = table.to_pylist()
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path} does not contain tabular object rows")
    return rows


def iter_dataset_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Stream JSONL or Parquet rows without retaining a multi-GB code corpus."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Parquet datasets require the project data dependencies"
            ) from exc
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise RuntimeError(f"{path} contains a non-object row")
                yield row
        return
    if suffix == ".arrow":
        yield from read_dataset_rows(path)
        return
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from read_json_rows(path)
            return
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number} is not a JSON object row"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number} is not a JSON object")
            yield row


def _select(
    rows: list[dict[str, Any]], count: int | str, seed: int
) -> list[dict[str, Any]]:
    if count == "all":
        count = len(rows)
    if len(rows) < count:
        raise RuntimeError(f"requested {count} rows from a dataset with {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:count]]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def snapshot_config(options: RunOptions) -> Path:
    """Atomically retain the exact config used by one completed output directory."""

    source = options.config.resolve()
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read config snapshot source {source}: {exc}"
        ) from exc
    destination = options.output / "config.yaml"
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"cannot read existing config snapshot {destination}: {exc}"
            ) from exc
        if existing != payload:
            raise RuntimeError(
                f"output contains a different config snapshot: {destination}"
            )
        return destination
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def resolved_request_batch_size(value: int | str, total: int) -> int:
    if total <= 0:
        raise RuntimeError("cannot generate an empty request set")
    if value == "all":
        return total
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError("--sglang-batch-size must be a positive integer or all")
    return min(value, total)


def model_source(
    options: RunOptions, config: OffPolicyConfig, role: str
) -> tuple[str, str | None]:
    local = getattr(options, f"{role}_model_path")
    model = getattr(config, role)
    if local:
        return str(local), None
    subfolder = getattr(model, "subfolder", None)
    revision = getattr(model, "revision", None)
    if subfolder is None and revision is None:
        return model.checkpoint, None
    snapshot_options: dict[str, Any] = {
        "repo_id": model.checkpoint,
        "cache_dir": options.cache_dir,
    }
    if revision is not None:
        snapshot_options["revision"] = revision
    if subfolder is not None:
        snapshot_options["allow_patterns"] = [f"{subfolder}/**"]
    snapshot = Path(snapshot_download(**snapshot_options))
    if subfolder is None:
        return str(snapshot), None
    source = snapshot / subfolder
    if not (source / "config.json").is_file():
        raise RuntimeError(
            f"model subfolder has no config.json: {model.checkpoint}/{subfolder}"
        )
    return str(source), None


def _object_commit(value: Any) -> str | None:
    commit = getattr(value, "init_kwargs", {}).get("_commit_hash")
    if not commit:
        commit = getattr(getattr(value, "config", None), "_commit_hash", None)
    return str(commit) if commit else None


def _snapshot_commit(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def _local_model_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    receipt = path / ".resolved_revision"
    if not receipt.is_file():
        return None
    value = receipt.read_text(encoding="utf-8").strip()
    return value or None


def tokenizer_vocab_sha256(tokenizer: Any) -> str:
    """Hash the complete token-to-ID mapping, independent of chat metadata."""

    return _sha256_json(
        sorted((token, int(index)) for token, index in tokenizer.get_vocab().items())
    )


def load_tokenizer(
    options: RunOptions,
    config: OffPolicyConfig,
    role: str,
    *,
    source_override: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if source_override is None:
        source, revision = model_source(options, config, role)
        model_name = getattr(config, role).checkpoint
        local_path = getattr(options, f"{role}_model_path")
        if (
            local_path is None
            and getattr(getattr(config, role), "subfolder", None) is not None
        ):
            local_path = Path(source)
    else:
        source, revision = str(source_override), None
        model_name = str(source_override)
        local_path = source_override
    trust_remote_code = getattr(getattr(config, role), "trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    replacements = (
        getattr(config.template, "chat_template_replacements", {})
        if role == config.template.source
        else {}
    )
    if replacements:
        if not isinstance(tokenizer.chat_template, str):
            raise RuntimeError(
                "template.chat_template_replacements requires one string chat template"
            )
        for old, new in replacements.items():
            occurrences = tokenizer.chat_template.count(old)
            if occurrences != 1:
                raise RuntimeError(
                    "template.chat_template_replacements expected exactly one "
                    f"occurrence of {old!r}, found {occurrences}"
                )
            tokenizer.chat_template = tokenizer.chat_template.replace(old, new)
    vocab_hash = tokenizer_vocab_sha256(tokenizer)
    identity = {
        "models": {
            role: {
                "name": model_name,
                "configured_revision": getattr(
                    getattr(config, role), "revision", None
                ),
                "subfolder": getattr(getattr(config, role), "subfolder", None),
                "resolved_revision": _object_commit(tokenizer)
                or _local_model_commit(local_path)
                or (_snapshot_commit(Path(source)) if Path(source).exists() else None),
                "trust_remote_code": trust_remote_code,
                "context_length": getattr(
                    getattr(config, role), "context_length", None
                ),
                "config_overrides": dict(
                    getattr(getattr(config, role), "config_overrides", {})
                ),
            }
        },
        "tokenizer_vocab_sha256": vocab_hash,
        "template_chat_sha256": tokenizer_chat_template_sha256(tokenizer),
        "template_chat_replacements": dict(replacements),
        "template_render_mode": getattr(config.template, "render_mode", "chat"),
    }
    return tokenizer, identity


def load_template_tokenizer(
    options: RunOptions, config: OffPolicyConfig
) -> tuple[Any, dict[str, Any]]:
    """Load the one tokenizer that owns prompt rendering for every stage."""

    role = config.template.source
    tokenizer, identity = load_tokenizer(options, config, role)
    identity["template_source"] = role
    return tokenizer, identity


def require_matching_tokenizer_vocabulary(
    template_identity: dict[str, Any], execution_identity: dict[str, Any]
) -> None:
    """Require canonical token IDs to have identical execution-model meanings."""

    if (
        template_identity["tokenizer_vocab_sha256"]
        != execution_identity["tokenizer_vocab_sha256"]
    ):
        raise RuntimeError(
            "execution-model tokenizer vocabulary differs from canonical "
            "template tokenizer"
        )


def merge_model_identities(
    template_identity: dict[str, Any], execution_identity: dict[str, Any]
) -> None:
    """Record an execution model alongside the canonical template model."""

    template_identity["models"].update(execution_identity["models"])


def snapshot_template_identity(snapshot: Path) -> dict[str, str | None]:
    """Read the canonical tokenizer contract stored beside a snapshot."""

    manifest_path = snapshot.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        vocab_hash = source["tokenizer_vocab_sha256"]
        chat_hash = source["template_chat_sha256"]
        template_source = source.get("template_source")
        render_mode = source.get("template_render_mode")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"snapshot manifest lacks template tokenizer identity: {manifest_path}"
        ) from exc
    for label, value in (
        ("tokenizer vocabulary", vocab_hash),
        ("chat template", chat_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(
                f"snapshot manifest has invalid {label} identity: {manifest_path}"
            )
    if template_source is not None and template_source not in {"student", "teacher"}:
        raise RuntimeError(
            f"snapshot manifest has invalid template source: {manifest_path}"
        )
    if render_mode is not None and render_mode not in {"chat", "raw"}:
        raise RuntimeError(
            f"snapshot manifest has invalid template render mode: {manifest_path}"
        )
    return {
        "tokenizer_vocab_sha256": vocab_hash,
        "template_chat_sha256": chat_hash,
        "template_source": template_source,
        "template_render_mode": render_mode,
    }


def load_model(
    source: str,
    revision: str | None,
    *,
    placement: str = "replicated",
    attention_implementation: str = "sdpa",
    trust_remote_code: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> Any:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "soft-prompt training requires a CUDA GPU with BF16 support; "
            "FP16 training is intentionally unsupported"
        )
    if placement not in {"replicated", "balanced"}:
        raise RuntimeError(f"unsupported train model placement: {placement!r}")
    if attention_implementation not in {
        "eager",
        "sdpa",
        "flash_attention_2",
        "flex_attention",
    }:
        raise RuntimeError(
            f"unsupported train attention implementation: {attention_implementation!r}"
        )
    if (
        attention_implementation == "flash_attention_2"
        and not is_flash_attn_2_available()
    ):
        raise RuntimeError(
            "training requested flash_attention_2, but Transformers cannot load "
            "the FlashAttention-2 CUDA extension; install flash-attn in the "
            "selected SOFT_PROMPT_PYTHON runtime"
        )
    if attention_implementation == "flex_attention":
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "training requested flex_attention, but the selected "
                "SOFT_PROMPT_PYTHON runtime does not provide PyTorch FlexAttention"
            ) from exc
    model_config = AutoConfig.from_pretrained(
        source,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    if (
        getattr(model_config, "model_type", None) == "gpt_oss"
        and attention_implementation == "flash_attention_2"
    ):
        raise RuntimeError(
            "GPT-OSS learned attention sinks are not supported by the generic "
            "FlashAttention-2 backend. Use flex_attention (recommended on "
            "Blackwell) or eager; refusing to train with silently dropped sinks."
        )
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention_implementation,
        "trust_remote_code": trust_remote_code,
    }
    overrides = dict(config_overrides or {})
    collisions = set(load_kwargs) & set(overrides)
    if collisions:
        raise RuntimeError(
            "model config overrides contain loader-owned fields: "
            f"{', '.join(sorted(collisions))}"
        )
    load_kwargs.update(overrides)
    if placement == "balanced":
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "balanced train model placement requires at least two visible GPUs"
            )
        load_kwargs["device_map"] = "balanced"
    model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    if placement == "replicated":
        model = model.cuda()
    model = model.eval()
    if model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise RuntimeError("student model did not load with BF16 embeddings")
    return model


def _dataset_identity(
    source: Any,
    selection: Any,
    path: Path,
    resolved_count: int,
    *,
    available_count: int | None = None,
) -> dict[str, Any]:
    identity = {
        "name": source.repo,
        "split": source.split,
        "domain": source.domain,
        "requested_num_prompts": selection.num_prompts,
        "configured_revision": source.revision,
        "record_count": resolved_count,
        "selection_seed": selection.selection_seed,
        "resolved_revision": _snapshot_commit(path),
        "filename": path.name,
        "content_sha256": sha256_file(path),
    }
    if available_count is not None:
        identity["available_record_count"] = available_count
    return identity


def download_rows(
    options: RunOptions,
    config: OffPolicyConfig,
    task: TaskAdapter,
    kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if kind == "sampling":
        selection = config.data.sampling
        sources = task.get_sampling_datasets()
        tagged_rows = []
        source_metadata = []
        stream_selection = bool(
            getattr(task, "stream_sampling_selection", False)
            and isinstance(selection.num_prompts, int)
        )
        stream_rng = random.Random(selection.selection_seed)
        stream_seen = 0
        for source_index, source in enumerate(sources):
            path = Path(
                hf_hub_download(
                    source.repo,
                    source.filename,
                    repo_type="dataset",
                    revision=source.revision,
                    cache_dir=options.cache_dir,
                )
            )
            if stream_selection:
                available_count = 0
                assert isinstance(selection.num_prompts, int)
                for row in iter_dataset_rows(path):
                    if not task.include_sampling_row(row):
                        continue
                    available_count += 1
                    stream_seen += 1
                    tagged = {"source_index": source_index, "row": row}
                    if len(tagged_rows) < selection.num_prompts:
                        tagged_rows.append(tagged)
                    else:
                        replacement = stream_rng.randrange(stream_seen)
                        if replacement < selection.num_prompts:
                            tagged_rows[replacement] = tagged
            else:
                source_rows = [
                    row
                    for row in read_dataset_rows(path)
                    if task.include_sampling_row(row)
                ]
                available_count = len(source_rows)
                tagged_rows.extend(
                    {"source_index": source_index, "row": row}
                    for row in source_rows
                )
            source_metadata.append((source, path, available_count))
        if stream_selection:
            assert isinstance(selection.num_prompts, int)
            if stream_seen < selection.num_prompts:
                raise RuntimeError(
                    f"requested {selection.num_prompts} rows from a dataset with "
                    f"{stream_seen}"
                )
            selected = tagged_rows
        else:
            selected = _select(
                tagged_rows,
                selection.num_prompts,
                selection.selection_seed,
            )
        selected_counts = [0] * len(sources)
        rows = []
        for tagged in selected:
            selected_counts[tagged["source_index"]] += 1
            rows.append(task.normalize_sampling_row(tagged["row"]))
        identities = [
            _dataset_identity(
                source,
                selection,
                path,
                selected_counts[source_index],
                available_count=available_count,
            )
            for source_index, (source, path, available_count) in enumerate(
                source_metadata
            )
        ]
        return rows, identities
    rows = []
    identities = []
    for label, selection in config.data.evaluation.items():
        try:
            sources = task.get_evaluation_datasets(label)
        except KeyError as exc:
            raise RuntimeError(
                f"task {config.task} has no evaluation dataset {label!r}; "
                f"available: {sorted(task.evaluation_datasets)}"
            ) from exc
        tagged_rows = []
        source_metadata = []
        for source_index, source in enumerate(sources):
            path = Path(
                hf_hub_download(
                    source.repo,
                    source.filename,
                    repo_type="dataset",
                    revision=source.revision,
                    cache_dir=options.cache_dir,
                )
            )
            source_rows = [
                row
                for row in read_dataset_rows(path)
                if not hasattr(task, "include_evaluation_row")
                or task.include_evaluation_row(row, label)
            ]
            tagged_rows.extend(
                {"source_index": source_index, "row": row} for row in source_rows
            )
            source_metadata.append((source, path, len(source_rows)))
        selected = _select(
            tagged_rows,
            selection.num_prompts,
            selection.selection_seed,
        )
        selected_counts = [0] * len(sources)
        for tagged in selected:
            selected_counts[tagged["source_index"]] += 1
            rows.append(task.normalize_evaluation_row(tagged["row"], label))
        identities.extend(
            _dataset_identity(
                source,
                selection,
                path,
                selected_counts[source_index],
                available_count=available_count,
            )
            for source_index, (source, path, available_count) in enumerate(
                source_metadata
            )
        )
    return rows, identities


def write_manifest(
    options: RunOptions,
    config: OffPolicyConfig,
    stage: str,
    counts: dict[str, int],
    files: list[Path],
    runtime_identity: dict[str, Any],
    *,
    stage_source: Path,
) -> None:
    config_snapshot = snapshot_config(options)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"])
    body = {
        "stage": stage,
        "source": {"config": str(options.config), **runtime_identity},
        "record_counts": counts,
        "content_sha256": {
            **{path.name: sha256_file(path) for path in files},
            config_snapshot.name: sha256_file(config_snapshot),
        },
        "code": {
            "commit": commit,
            "dirty": bool(diff),
            "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        },
        "source_sha256": {
            "stage": sha256_file(stage_source),
            "common": sha256_file(Path(__file__)),
            "config": sha256_file(options.config.resolve()),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (options.output / "manifest.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
