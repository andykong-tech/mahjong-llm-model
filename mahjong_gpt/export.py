"""Write a 16-bit copy of a checkpoint.

The copy is half the size. Draw hands from it with --precision fp16, and keep
the 32-bit original for any further training.
"""

import argparse
from pathlib import Path

import torch

from mahjong_gpt.model import GPT, GPTConfig


def export(source: str | Path, out: str | Path) -> Path:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    # Halving the model, not each tensor, keeps the tied embedding stored once.
    checkpoint["model"] = model.half().state_dict()
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a 16-bit copy of a checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/small.pt")
    parser.add_argument("--out", default="", help="Defaults to the checkpoint name with -fp16 added")
    args = parser.parse_args()
    source = Path(args.checkpoint)
    out = Path(args.out) if args.out else source.with_name(f"{source.stem}-fp16{source.suffix}")
    if out.resolve() == source.resolve():
        parser.error("--out would overwrite the 32-bit checkpoint")
    export(source, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, from {source.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
