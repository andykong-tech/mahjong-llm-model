"""Build the training shards from the rules and the hand generator.

Hands are spread across every suit and tile-count request that can produce a
legal win. Exact duplicate sequences are kept once. Rare patterns are added
on purpose, up to a fan floor when the pattern has enough distinct hands.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from mahjong_gpt.hands import hand_suits, physical_tiles, within_copies
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.sampler import (
    Request,
    all_green,
    all_honors,
    all_terminals,
    big_four_winds,
    big_three_dragons,
    four_kongs,
    greater_honors_and_knitted,
    knitted_straight,
    lesser_honors_and_knitted,
    nine_gates,
    pure_straight,
    quadruple_chow,
    sample_hand,
    sample_rare,
    seven_pairs,
    seven_shifted_pairs,
    thirteen_orphans,
)
from mahjong_gpt.tiles import SUIT_NAMES, suit_capacity
from mahjong_gpt.tokenizer import MAX_SEQUENCE, Tokenizer

_SEEDS = (
    (thirteen_orphans, frozenset(SUIT_NAMES), 14),
    (seven_pairs, frozenset({"characters", "bamboo"}), 14),
    (seven_shifted_pairs, frozenset({"dots"}), 14),
    (nine_gates, frozenset({"bamboo"}), 14),
    (greater_honors_and_knitted, frozenset(SUIT_NAMES), 14),
    (lesser_honors_and_knitted, frozenset(SUIT_NAMES), 14),
    (knitted_straight, frozenset({"characters", "bamboo", "dots", "dragons"}), 14),
    (all_green, frozenset({"bamboo", "dragons"}), 14),
    (big_three_dragons, frozenset({"dragons", "characters", "dots"}), 14),
    (big_four_winds, frozenset({"winds", "bamboo"}), 14),
    (four_kongs, frozenset({"characters"}), 18),
    (all_honors, frozenset({"winds"}), 14),
    (quadruple_chow, frozenset({"bamboo"}), 14),
    (pure_straight, frozenset({"characters", "bamboo", "dots"}), 14),
    (all_terminals, frozenset({"characters", "bamboo", "dots"}), 14),
)


def every_request() -> list[Request]:
    requests = []
    for mask in range(1, 1 << len(SUIT_NAMES)):
        suits = frozenset(SUIT_NAMES[i] for i in range(len(SUIT_NAMES)) if mask & (1 << i))
        for tile_count in range(14, 19):
            requests.append(Request(suits, tile_count))
    return requests


def build_dataset(
    out_dir: str | Path,
    train_size: int,
    *,
    fan_floor: int = 0,
    seed: int = 0,
    probe_tries: int = 80,
    requests: list[Request] | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tokenizer = Tokenizer()
    candidates = list(requests) if requests is not None else every_request()
    feasible = [request for request in candidates if _feasible(request, rng, probe_tries)]
    if not feasible:
        raise RuntimeError("no suit and tile-count request produced a legal hand")

    quotas = _quotas(train_size, len(feasible))
    buckets = {"train": [], "val": [], "test": []}
    seen: set[bytes] = set()
    fan_counts: Counter[str] = Counter()
    per_request = []

    def accept(ids: list[int], fans) -> str:
        key = hashlib.sha1(bytes(ids)).digest()
        if key in seen:
            return "dup"
        seen.add(key)
        dest = _split_of(ids)
        buckets[dest].append(bytes(ids))
        if dest == "train":
            for fan in fans:
                fan_counts[fan.name] += fan.count
        return dest

    for builder, suits, tile_count in _SEEDS:
        if requests is not None and Request(suits, tile_count) not in feasible:
            continue
        hand = builder(Request(suits, tile_count), rng)
        found = _score_if_legal(hand, Request(suits, tile_count)) if hand is not None else None
        if found is None:
            continue
        scored_hand, fans = found
        accept(tokenizer.encode(suits, tile_count, scored_hand, fans), fans)

    for request, quota in zip(feasible, quotas):
        got = 0
        tries = 0
        stall = 0
        while got < quota and tries < quota * 40:
            tries += 1
            found = _draw(request, rng, rare_only=False)
            if found is None:
                stall += 1
                if stall > max(80, quota):
                    break
                continue
            hand, fans = found
            stall = 0
            kind = accept(tokenizer.encode(request.suits, request.tile_count, hand, fans), fans)
            if kind == "train":
                got += 1
        per_request.append({"suits": sorted(request.suits), "tile_count": request.tile_count, "train": got})
        print(
            f"{len(per_request):3d}/{len(feasible)} {' '.join(sorted(request.suits)):<42} {request.tile_count}  train {got}",
            flush=True,
        )

    if fan_floor > 0:
        print(f"topping up fans below {fan_floor}", flush=True)
        _top_up_fans(feasible, rng, tokenizer, accept, fan_counts, fan_floor)

    width = MAX_SEQUENCE
    for name, rows in buckets.items():
        array = np.zeros((len(rows), width), dtype=np.uint8)
        for index, raw in enumerate(rows):
            array[index, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        np.save(out / f"{name}.npy", array)

    meta = {
        "block_size": MAX_SEQUENCE,
        "vocab_size": len(tokenizer),
        "sep_id": tokenizer.sep_id,
        "pad_id": tokenizer.pad_id,
        "counts": {name: len(rows) for name, rows in buckets.items()},
    }
    report = {
        "feasible": len(feasible),
        "requested": len(candidates),
        "per_request": per_request,
        "fans": dict(sorted(fan_counts.items())),
        "fan_floor": fan_floor,
        "below_floor": sorted(name for name, count in fan_counts.items() if count < fan_floor),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "vocab.json").write_text(json.dumps(list(tokenizer.tokens), indent=2))
    (out / "feasibility.json").write_text(json.dumps([
        {"suits": sorted(request.suits), "tile_count": request.tile_count} for request in feasible
    ], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return {"meta": meta, "report": report}


def load_split(data_dir: str | Path, name: str) -> tuple[np.ndarray, dict]:
    root = Path(data_dir)
    meta = json.loads((root / "meta.json").read_text())
    return np.load(root / f"{name}.npy"), meta


def _quotas(total: int, cells: int) -> list[int]:
    base, extra = divmod(total, cells)
    return [base + (1 if index < extra else 0) for index in range(cells)]


def _split_of(ids: list[int]) -> str:
    bucket = int.from_bytes(hashlib.sha1(bytes(ids)).digest()[:4], "little") % 102
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def _feasible(request: Request, rng: random.Random, tries: int) -> bool:
    if request.tile_count > suit_capacity(request.suits):
        return False
    for _ in range(tries):
        if _draw(request, rng) is not None:
            return True
    return False


def _draw(request: Request, rng: random.Random, rare_only: bool = False):
    hand = sample_rare(request, rng) if rare_only else sample_hand(request, rng)
    return _score_if_legal(hand, request)


def _score_if_legal(hand, request: Request):
    if hand is None:
        return None
    tiles = physical_tiles(hand)
    if len(tiles) != request.tile_count or hand_suits(hand) != request.suits or not within_copies(hand):
        return None
    scored = score(
        hand.pack,
        hand.concealed,
        hand.win_tile,
        self_drawn=hand.self_drawn,
        seat_wind=hand.seat_wind,
        round_wind=hand.round_wind,
    )
    if not is_legal_win(scored):
        return None
    return hand, scored[0]


def _top_up_fans(feasible, rng, tokenizer, accept, fan_counts, fan_floor: int) -> None:
    stall = 0
    steps = 0
    limit = max(fan_floor, 1) * 30
    while stall < len(feasible) and steps < limit and any(count < fan_floor for count in fan_counts.values()):
        request = feasible[steps % len(feasible)]
        steps += 1
        found = _draw(request, rng, rare_only=True)
        if found is None:
            stall += 1
            continue
        hand, fans = found
        if not any(fan_counts[fan.name] < fan_floor for fan in fans):
            stall += 1
            continue
        kind = accept(tokenizer.encode(request.suits, request.tile_count, hand, fans), fans)
        stall = 0 if kind != "dup" else stall + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mahjong training shards")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-size", type=int, default=2_000_000)
    parser.add_argument("--fan-floor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-tries", type=int, default=80)
    args = parser.parse_args()
    floor = args.fan_floor
    if floor is None:
        floor = min(5000, max(1, args.train_size // 400))
    result = build_dataset(
        args.out,
        args.train_size,
        fan_floor=floor,
        seed=args.seed,
        probe_tries=args.probe_tries,
    )
    counts = result["meta"]["counts"]
    print(
        f"wrote {args.out}: train {counts['train']}, val {counts['val']}, test {counts['test']}"
        f" from {result['report']['feasible']} feasible requests"
    )


if __name__ == "__main__":
    main()
