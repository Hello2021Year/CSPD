"""Train one CSPD vector on JEV using a local, unquantized Qwen3.8 checkpoint."""

import argparse
import json
import os
from contextlib import nullcontext
from functools import partial, wraps
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from safetensors.torch import save_file

import train_sp
from pack_jev import pack_train


class QwenTextStudent(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model.language_model
        self.lm_head = model.get_output_embeddings()
        self.config = self.model.config
        self.supports_gradient_checkpointing = self.model.supports_gradient_checkpointing

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        hidden = self.model(**kwargs).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def stage_forward(forward, device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    @wraps(forward)
    def staged(*args, **kwargs):
        guard = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with guard:
            return forward(*move(args), **move(kwargs))

    return staged


def place_student(student, input_device, *, devices):
    if len(devices) < 2 or devices[0] != input_device:
        raise ValueError("model-parallel stages must begin on the training input device")
    decoder = student.model
    layers = decoder.layers
    if len(layers) < len(devices):
        raise ValueError("model parallelism needs at least one decoder layer per GPU")
    embedding = student.get_input_embeddings().to(devices[0])
    decoder.rotary_emb.to(devices[0])
    decoder.norm.to(devices[-1])
    head = student.get_output_embeddings()
    head.to(devices[0] if head.weight is embedding.weight else devices[-1])
    for index, device in enumerate(devices):
        start = index * len(layers) // len(devices)
        stop = (index + 1) * len(layers) // len(devices)
        for layer in layers[start:stop]:
            layer.to(device)
            layer.forward = stage_forward(layer.forward, device)
        print(f"Model shard {device}: decoder layers {start}–{stop - 1}", flush=True)
    return student


def resolve_checkpoint(path):
    path = path.expanduser().resolve()
    candidates = (path, path / "Qwen3.8-27B", path / "Qwen" / "Qwen3.8-27B")
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    raise ValueError(f"no config.json found; pass the actual local Qwen3.8 checkpoint directory under {path}")


def model_class(model_type):
    names = {"qwen3_5": "Qwen3_5ForConditionalGeneration", "qwen3_5_text": "Qwen3_5ForCausalLM"}
    try:
        return getattr(transformers, names[model_type])
    except (AttributeError, ImportError) as error:
        raise RuntimeError(f"Transformers {transformers.__version__} cannot load {names[model_type]}; "
                           "use a Qwen3.8-compatible training environment") from error


def load_student(path, dtype, *, model_type, attention):
    student, info = model_class(model_type).from_pretrained(
        path, dtype=dtype, attn_implementation=attention,
        local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    text_prefix = "model.language_model." if model_type == "qwen3_5" else "model."
    missing = [name for name in info.get("missing_keys", ()) if name.startswith((text_prefix, "lm_head."))]
    if missing:
        raise RuntimeError(f"student weights did not load completely; first missing tensor: {missing[0]}")
    return QwenTextStudent(student) if model_type == "qwen3_5" else student


def run(args):
    checkpoint = resolve_checkpoint(args.model)
    config = json.loads((checkpoint / "config.json").read_text())
    kind = config.get("model_type")
    if kind not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError(f"expected Qwen3.8 (qwen3_5), found {kind!r} in {checkpoint}")
    if config.get("quantization_config") or config.get("text_config", {}).get("quantization_config"):
        raise ValueError("use the original BF16 checkpoint, not a quantized inference checkpoint")
    if args.updates < 1 or args.batch_size < 1:
        raise ValueError("--updates and --batch-size must be positive")
    batch_size = 1 if args.smoke else args.batch_size
    placement = None
    if not args.prepare_only:
        gpu_count = torch.cuda.device_count()
        if "RANK" in os.environ or not torch.cuda.is_available():
            raise RuntimeError("run directly with Python on CUDA; do not use torchrun")
        if (args.model_parallel and gpu_count < 2) or (not args.model_parallel and gpu_count != 1):
            raise RuntimeError("expose one CUDA GPU, or expose multiple CUDA GPUs with --model-parallel")
        model_class(kind)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        devices = tuple(torch.device("cuda", index) for index in range(gpu_count))
        for device in devices:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"{device} lacks BF16 support")
                free, _ = torch.cuda.mem_get_info()
                print(f"GPU {device}: {torch.cuda.get_device_name(device)}, {free / 2**30:.1f} GiB free", flush=True)
        if args.model_parallel:
            placement = partial(place_student, devices=devices)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    packed = args.output / "packed"
    report = pack_train(args.data, packed, tokenizer, context_length=args.context_length, reserve_tokens=1,
                        limit=1 if args.smoke else None, skip_oversize=args.skip_oversize,
                        supervise_turn_end=args.with_turn_end)
    print(f"Packed {report['records']} rows ({report['label_counts']}); "
          f"skipped {len(report['skipped_oversize'])} oversize rows", flush=True)
    if args.prepare_only:
        return
    if report["records"] < batch_size:
        raise ValueError("not enough packed records for the requested batch size")
    if not args.smoke and not all(report["label_counts"].values()):
        raise ValueError("full training needs at least one retained example of each label")

    output = args.output / "training"
    options = [
        "--model", str(checkpoint), "--data", str(packed), "--output", str(output),
        "--prompt-length", "1", "--no-prefix-robust", "--batch-size", str(batch_size),
        "--updates", str(args.updates),
    ]
    if args.smoke:
        options.extend(["--stop-after", "1"])
    train_sp.main(options, student_loader=partial(load_student, model_type=kind, attention=args.attention),
                  student_placement=placement)
    states = sorted((output / "checkpoints").glob("update-*/state.pt"))
    if not states:
        raise RuntimeError("training produced no checkpoint")
    prompt = torch.load(states[-1], map_location="cpu", weights_only=True)["prompt"].float().contiguous()
    width = config.get("text_config", config)["hidden_size"]
    if prompt.shape != (1, width) or not torch.isfinite(prompt).all():
        raise RuntimeError(f"invalid trained soft prompt; expected shape (1, {width})")
    artifact = args.output / "jev_soft_prompt.safetensors"
    save_file({"soft_prompt": prompt}, artifact, metadata={
        "model": str(checkpoint),
        "task": "jev_yes_no",
        "supervise_turn_end": str(args.with_turn_end).lower(),
        "smoke": str(args.smoke).lower(),
    })
    print(f"Saved {artifact} with shape {list(prompt.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="original or normalized JEV ZIP")
    parser.add_argument("--model", type=Path, default=Path("/mnt_upfs/qwen3"))
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--model-parallel", action="store_true", help="split decoder layers over the visible GPUs")
    parser.add_argument("--smoke", action="store_true", help="train on one row for one update")
    parser.add_argument("--prepare-only", action="store_true", help="pack data without loading weights or using CUDA")
    parser.add_argument("--with-turn-end", action="store_true", help="also supervise the end-of-turn token")
    parser.add_argument("--skip-oversize", action="store_true", help="report over-budget rows in the packed manifest")
    args = parser.parse_args()
    try:
        run(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
