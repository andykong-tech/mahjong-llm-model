"""Train the Mahjong transformer on this Mac.

The loss ignores the prompt, the padding, and only supervises the hand and its
fan names. Pass --overfit to memorize a few thousand examples; the loss should
fall close to zero, which checks the data and the model before a real run.

--precision fp16 runs the matrix math in 16-bit and keeps layer norms and the
loss in 32-bit. The weights, the optimizer, and the saved checkpoint stay
32-bit, and the loss is scaled up before backward so small gradients do not
round to zero.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mahjong_gpt.dataset import load_split
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.tokenizer import Tokenizer

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}


def supervised_targets(ids: torch.Tensor, sep_id: int, pad_id: int) -> torch.Tensor:
    """Label every token after <sep>, and ignore the prompt and the padding."""
    targets = ids.clone()
    for row, target in zip(ids, targets):
        matches = (row == sep_id).nonzero(as_tuple=False)
        if len(matches) != 1:
            raise ValueError("each sequence needs exactly one <sep>")
        target[: int(matches[0]) + 1] = -100
        target[row == pad_id] = -100
    return targets


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def learning_rate(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    data, meta = load_split(args.data, "train")
    if args.overfit:
        data = data[: args.overfit]
    val, _ = load_split(args.data, "val")
    tokenizer = Tokenizer()
    if len(tokenizer) != meta["vocab_size"]:
        raise ValueError("saved vocab size does not match the tokenizer")

    raw = json.loads(Path(args.config).read_text())
    raw["vocab_size"] = meta["vocab_size"]
    if args.overfit:
        raw["dropout"] = 0.0
    config = GPTConfig(**raw)
    if meta["block_size"] > config.block_size:
        raise ValueError("data sequences are longer than the model block size")
    model = GPT(config).to(device)
    weight_decay = 0.0 if args.overfit else args.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    warmup = args.warmup if args.warmup < args.steps else max(1, args.steps // 10)
    dtype = PRECISIONS[args.precision]
    mixed = dtype != torch.float32
    device_type = torch.device(device).type
    scaler = torch.amp.GradScaler(device_type, enabled=mixed)

    print(f"device {device}, precision {args.precision}, parameters {model.parameter_count():,}, rows {len(data):,}")
    model.train()
    started = time.time()
    tokens = 0
    last_loss = None
    for step in range(args.steps):
        # A decaying schedule stops before a small set can be memorized.
        rate = args.lr if args.overfit else learning_rate(step, warmup, args.steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = rate
        choice = np.random.randint(0, len(data), size=args.batch_size)
        batch = torch.tensor(data[choice], dtype=torch.long, device=device)
        targets = supervised_targets(batch, meta["sep_id"], meta["pad_id"])
        with torch.autocast(device_type, dtype=dtype, enabled=mixed):
            _logits, loss = model(batch, targets)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Clipping has to see the true gradients, not the scaled ones.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach())
        tokens += int(batch.numel())
        if step % args.log_every == 0 or step + 1 == args.steps:
            elapsed = max(time.time() - started, 1e-6)
            message = f"step {step:5d}  loss {last_loss:.4f}  lr {rate:.2e}  tok/s {tokens / elapsed:,.0f}"
            if mixed:
                message += f"  scale {scaler.get_scale():.0f}"
            if len(val) and not args.overfit and step % args.eval_every == 0:
                message += f"  val {evaluate(model, val, meta, device, args.batch_size, dtype):.4f}"
            print(message, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "model_config": config.__dict__, "vocab_size": len(tokenizer)},
        out,
    )
    print(f"saved {out}")
    return out


@torch.no_grad()
def evaluate(
    model: GPT, rows: np.ndarray, meta: dict, device: str, batch_size: int, dtype: torch.dtype = torch.float32
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    seen = 0
    for start in range(0, len(rows), batch_size):
        batch = torch.tensor(rows[start : start + batch_size], dtype=torch.long, device=device)
        targets = supervised_targets(batch, meta["sep_id"], meta["pad_id"])
        with torch.autocast(torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32):
            _logits, loss = model(batch, targets)
        total += float(loss) * len(batch)
        seen += len(batch)
    if was_training:
        model.train()
    return total / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Mahjong transformer")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=sorted(PRECISIONS), default="fp32", help="fp16 trains in mixed precision")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
