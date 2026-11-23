"""Concatenate M2RL domain prompts (M, C, S, I) into one composed soft prompt.

  python experiments/m2rl/compose.py --paper-order MCSI --output artifacts/m2rl/MCSI.safetensors
  python experiments/m2rl/compose.py --module I=runs/if/checkpoints/update-0200/state.pt \
      --module M=runs/math/checkpoints/update-0200/state.pt --output artifacts/m2rl/MI.safetensors

``--paper-order`` downloads the released step-200 prompts listed in paper-prompts.json and
checks their SHA-256. Prompts are concatenated left to right without separators, so the
last module sits closest to the question. Use I, SI, CSI, MCSI for the continual-addition
stages and drop one letter for the leave-one-out conditions.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import torch
from safetensors.torch import save_file



def concatenate_ordered(order, prompts):
    for name in order:
        prompt = prompts[name]
        if prompt.shape != (256, 2560) or not torch.isfinite(prompt).all():
            raise ValueError(f"module {name} is not a finite 256 x 2560 prompt")
    return torch.cat([prompts[name].float() for name in order], dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--module", action="append", metavar="M=STATE.PT")
    inputs.add_argument("--paper-order", metavar="MCSI", help="Fetch verified step-200 checkpoints in this order")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    prompts = {}
    for entry in args.module or []:
        label, sep, checkpoint = entry.partition("=")
        if not sep or label not in {"M", "C", "S", "I"} or label in prompts:
            parser.error(f"expected a unique domain and checkpoint path, got {entry!r}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        prompts[label] = state["prompt"].float()

    if args.paper_order:
        manifest = json.loads(Path(__file__).with_name("paper-prompts.json").read_text())
        order = args.paper_order
        if len(set(order)) != len(order) or any(label not in manifest for label in order):
            parser.error("--paper-order must contain unique domain letters from MCSI")
        for label in order:
            artifact = manifest[label]
            with urlopen(artifact["url"], timeout=60) as response:
                checkpoint = response.read()
            if hashlib.sha256(checkpoint).hexdigest() != artifact["sha256"]:
                raise ValueError(f"Checkpoint failed checksum verification for domain {label}")
            state = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
            prompts[label] = state["prompt"].float()

    order = tuple(prompts)
    composed = concatenate_ordered(order, prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"soft_prompt": composed.contiguous()}, args.output,
              metadata={"order": "".join(order), "insertion": "after_system_header"})
    print(json.dumps({"path": str(args.output), "order": "".join(order), "shape": list(composed.shape)}))


if __name__ == "__main__":
    main()
