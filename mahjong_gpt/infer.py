"""Sample a winning hand for a suit and tile-count request.

Tokens that would break the hand grammar are masked. Each finished sample is
checked with the rules, and a sample that is not an 8-fan win is drawn again.
"""

import argparse

import torch
import torch.nn.functional as F

from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.tokenizer import MAX_SEQUENCE, Tokenizer
from mahjong_gpt.train import pick_device


def load_model(path: str, device: str) -> GPT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def complete(model, tokenizer, ids, *, temperature: float, device: str, greedy_fans: bool = False):
    """Extend `ids` one token at a time under the hand grammar.

    Returns the ids and why decoding stopped: "done", "dead end" (no legal next
    token), or "too long". With `greedy_fans`, the fan names are taken greedily
    even when the hand itself is sampled.
    """
    ids = list(ids)
    fans_id = tokenizer.stoi["<fans>"]
    in_fans = fans_id in ids
    with torch.no_grad():
        while len(ids) < MAX_SEQUENCE:
            allowed = allowed_next(tokenizer, ids)
            if allowed is None:
                return ids, "done"
            if not allowed:
                return ids, "dead end"
            logits, _loss = model(torch.tensor([ids], dtype=torch.long, device=device))
            choices = sorted(allowed)
            scores = logits[0, -1, choices]
            if temperature <= 0 or (greedy_fans and in_fans):
                pick = int(torch.argmax(scores))
            else:
                pick = int(torch.multinomial(F.softmax(scores / temperature, dim=-1), num_samples=1))
            token = choices[pick]
            ids.append(token)
            if token == tokenizer.eos_id:
                return ids, "done"
            in_fans = in_fans or token == fans_id
    return ids, "too long"


def sample_hand(model, tokenizer, suits, tile_count, *, temperature: float, device: str):
    model.eval()
    ids, status = complete(model, tokenizer, tokenizer.prompt(suits, tile_count), temperature=temperature, device=device)
    if status != "done":
        return None
    parsed = parse_ids(tokenizer, ids)
    if parsed is None or parsed.suits != frozenset(suits) or parsed.tile_count != tile_count:
        return None
    scored = score(
        parsed.pack,
        parsed.concealed,
        parsed.win_tile,
        self_drawn=parsed.self_drawn,
        seat_wind=parsed.seat_wind,
        round_wind=parsed.round_wind,
    )
    if not is_legal_win(scored):
        return None
    return parsed, scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Mahjong hand")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suits", required=True, help="Comma-separated, e.g. bamboo,dragons")
    parser.add_argument("--tiles", type=int, required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attempts", type=int, default=0)
    args = parser.parse_args()
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = Tokenizer()
    suits = [part.strip() for part in args.suits.split(",") if part.strip()]
    tries = args.attempts or args.num * 30
    found = 0
    for _ in range(tries):
        if found == args.num:
            break
        result = sample_hand(model, tokenizer, suits, args.tiles, temperature=args.temperature, device=device)
        if result is None:
            continue
        parsed, scored = result
        fans, total = scored
        claimed = ", ".join(parsed.fan_names)
        actual = ", ".join(f"{fan.name} {fan.points * fan.count}" for fan in fans)
        print(f"total {total}  win {parsed.win_tile}  model [{claimed}]  scorer [{actual}]")
        found += 1
    if found < args.num:
        print(f"stopped after {tries} draws with {found} legal hands")


if __name__ == "__main__":
    main()
