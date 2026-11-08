"""Harder checks for a trained checkpoint, with the scorer as referee.

Sections:
  exam        name the fans of every held-out test hand
  tricks      name the fans of hand-built hands aimed at single rules
  stress      generate for every possible request at rising temperatures
  impossible  requests that have no legal hand at all
  profile     compare generated hands with the training data
"""

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from mahjong_gpt.dataset import every_request, load_split
from mahjong_gpt.grammar import parse_ids
from mahjong_gpt.hands import hand_suits, physical_tiles
from mahjong_gpt.infer import complete, load_model
from mahjong_gpt.rules import MIN_FAN, score
from mahjong_gpt.sampler import Request
from mahjong_gpt.tiles import SUIT_NAMES, chow_of, suit_of
from mahjong_gpt.tokenizer import MAX_SEQUENCE, Tokenizer
from mahjong_gpt.train import pick_device
from mahjong_gpt.trick_hands import TRICK_HANDS

SECTIONS = ("exam", "tricks", "stress", "impossible", "profile")
TOTAL_BANDS = ((8, 15), (16, 31), (32, 63), (64, None))
_MELD_PREFIXES = ("CHOW", "PUNG", "KONG")


def fan_counter(fans) -> Counter:
    return Counter({fan.name: fan.count for fan in fans})


def scored_ids(tokenizer: Tokenizer, ids):
    """Parse and score a finished sequence. Either result may be None."""
    parsed = parse_ids(tokenizer, ids)
    if parsed is None:
        return None, None
    scored = score(
        parsed.pack,
        parsed.concealed,
        parsed.win_tile,
        self_drawn=parsed.self_drawn,
        seat_wind=parsed.seat_wind,
        round_wind=parsed.round_wind,
    )
    return parsed, scored


def grade(tokenizer: Tokenizer, ids, request: Request):
    """Referee one generated sequence. Returns (verdict, parsed, scored)."""
    parsed, scored = scored_ids(tokenizer, ids)
    if parsed is None:
        return "incomplete", None, None
    if scored is None:
        return "not a win", parsed, None
    if scored[1] < MIN_FAN:
        return "under 8 fan", parsed, scored
    tiles = _tiles(parsed)
    if len(tiles) != request.tile_count:
        return "wrong tile count", parsed, scored
    if {suit_of(tile) for tile in tiles} != request.suits:
        return "wrong suits", parsed, scored
    return "legal", parsed, scored


@torch.no_grad()
def name_fans(model, tokenizer: Tokenizer, prefixes, device: str, batch_size: int = 256) -> list[list[str]]:
    """Greedy fan names for each prefix that ends in <fans>."""
    fan_ids = sorted(tokenizer.fan_ids)
    eos = tokenizer.eos_id
    named: list[list[str]] = []
    for start in range(0, len(prefixes), batch_size):
        chunk = prefixes[start : start + batch_size]
        block = torch.full((len(chunk), MAX_SEQUENCE), tokenizer.pad_id, dtype=torch.long)
        for row, ids in enumerate(chunk):
            block[row, : len(ids)] = torch.tensor(ids)
        block = block.to(device)
        lengths = [len(ids) for ids in chunk]
        names: list[list[str]] = [[] for _ in chunk]
        open_rows = set(range(len(chunk)))
        while open_rows:
            rows = sorted(open_rows)
            logits, _loss = model(block[:, : max(lengths[row] for row in rows)])
            last = logits[rows, [lengths[row] - 1 for row in rows]]
            allowed = torch.full_like(last, float("-inf"))
            allowed[:, fan_ids] = last[:, fan_ids]
            for index, row in enumerate(rows):
                if names[row]:
                    allowed[index, eos] = last[index, eos]
            for row, token in zip(rows, allowed.argmax(dim=-1).tolist()):
                if token == eos or lengths[row] >= MAX_SEQUENCE:
                    open_rows.discard(row)
                    continue
                names[row].append(tokenizer.token(token))
                block[row, lengths[row]] = token
                lengths[row] += 1
        named.extend(names)
    return named


def exam(model, tokenizer: Tokenizer, rows, device: str) -> dict:
    fans_id = tokenizer.stoi["<fans>"]
    prefixes, truths, groups = [], [], []
    label_mismatches = 0
    for row in rows:
        ids = [int(token) for token in row if token != tokenizer.pad_id]
        cut = ids.index(fans_id) + 1
        truth = Counter(tokenizer.token(token) for token in ids[cut:-1])
        _parsed, scored = scored_ids(tokenizer, ids)
        if scored is None or fan_counter(scored[0]) != truth:
            label_mismatches += 1
        prefixes.append(ids[:cut])
        truths.append(truth)
        groups.append(_groups(tokenizer, ids, truth, scored[1] if scored else 0))
    guesses = [Counter(names) for names in name_fans(model, tokenizer, prefixes, device)]
    by_group: dict[str, list[int]] = {}
    hit, wanted, claimed = Counter(), Counter(), Counter()
    for truth, guess, labels in zip(truths, guesses, groups):
        for label in labels:
            cell = by_group.setdefault(label, [0, 0])
            cell[0] += truth == guess
            cell[1] += 1
        hit.update(truth & guess)
        wanted.update(truth)
        claimed.update(guess)
    return {
        "rows": len(rows),
        "label_mismatches": label_mismatches,
        "by_group": by_group,
        "missed": {name: (wanted[name] - hit[name], wanted[name]) for name in wanted},
        "added": {name: claimed[name] - hit[name] for name in claimed if claimed[name] > hit[name]},
    }


def tricks(model, tokenizer: Tokenizer, device: str) -> list[tuple]:
    fans_id = tokenizer.stoi["<fans>"]
    prefixes, truths = [], []
    for trick in TRICK_HANDS:
        hand = trick.hand
        scored = score(
            hand.pack,
            hand.concealed,
            hand.win_tile,
            self_drawn=hand.self_drawn,
            seat_wind=hand.seat_wind,
            round_wind=hand.round_wind,
        )
        if scored is None or scored[1] < MIN_FAN:
            raise ValueError(f"trick hand {trick.name!r} is not a legal win")
        suits = sorted(hand_suits(hand), key=SUIT_NAMES.index)
        ids = tokenizer.encode(suits, len(physical_tiles(hand)), hand, scored[0])
        prefixes.append(ids[: ids.index(fans_id) + 1])
        truths.append(scored)
    guesses = name_fans(model, tokenizer, prefixes, device)
    return [(trick, scored, Counter(guess)) for trick, scored, guess in zip(TRICK_HANDS, truths, guesses)]


def stress(model, tokenizer: Tokenizer, requests, temperatures, per_request: int, device: str) -> dict:
    win_id = tokenizer.stoi["<win>"]
    results = {}
    for temperature in temperatures:
        verdicts: Counter = Counter()
        exact = 0
        distinct = set()
        legal = []
        started = time.time()
        for request in requests:
            for _ in range(per_request):
                ids, status = complete(
                    model,
                    tokenizer,
                    tokenizer.prompt(request.suits, request.tile_count),
                    temperature=temperature,
                    device=device,
                    greedy_fans=True,
                )
                if status != "done":
                    verdicts[status] += 1
                    continue
                verdict, parsed, scored = grade(tokenizer, ids, request)
                verdicts[verdict] += 1
                if verdict != "legal":
                    continue
                exact += Counter(parsed.fan_names) == fan_counter(scored[0])
                distinct.add(tuple(ids[: ids.index(win_id)]))
                legal.append(scored)
        results[temperature] = {
            "draws": sum(verdicts.values()),
            "verdicts": verdicts,
            "exact": exact,
            "distinct": len(distinct),
            "legal": legal,
            "seconds": time.time() - started,
        }
    return results


def impossible(model, tokenizer: Tokenizer, requests, draws: int, device: str) -> list[tuple]:
    rows = []
    for request in requests:
        verdicts: Counter = Counter()
        started = time.time()
        for _ in range(draws):
            ids, status = complete(
                model,
                tokenizer,
                tokenizer.prompt(request.suits, request.tile_count),
                temperature=0.8,
                device=device,
            )
            verdicts[status if status != "done" else grade(tokenizer, ids, request)[0]] += 1
        rows.append((request, verdicts, time.time() - started))
    return rows


def training_profile(tokenizer: Tokenizer, rows, sample: int, seed: int) -> list:
    chosen = random.Random(seed).sample(range(len(rows)), min(sample, len(rows)))
    scoreds = []
    for index in chosen:
        ids = [int(token) for token in rows[index] if token != tokenizer.pad_id]
        _parsed, scored = scored_ids(tokenizer, ids)
        if scored is not None:
            scoreds.append(scored)
    return scoreds


def summarize(scoreds) -> dict:
    totals = [total for _fans, total in scoreds]
    present = Counter(name for fans, _total in scoreds for name in {fan.name for fan in fans})
    return {
        "hands": len(scoreds),
        "median": statistics.median(totals),
        "64 fan or more": sum(total >= 64 for total in totals) / len(totals),
        "under 16 fan": sum(total < 16 for total in totals) / len(totals),
        "top": [(name, count / len(scoreds)) for name, count in present.most_common(8)],
    }


def feasible_requests(data_dir: str) -> list[Request]:
    raw = json.loads((Path(data_dir) / "feasibility.json").read_text())
    return [Request(frozenset(item["suits"]), item["tile_count"]) for item in raw]


def _tiles(parsed) -> list[str]:
    tiles = list(parsed.concealed) + [parsed.win_tile]
    for kind, tile, _offer in parsed.pack:
        if kind == "CHI":
            tiles.extend(chow_of(tile))
        else:
            tiles.extend([tile] * (4 if kind == "GANG" else 3))
    return tiles


def _groups(tokenizer: Tokenizer, ids, truth: Counter, total: int) -> list[str]:
    names = [tokenizer.token(token) for token in ids]
    labels = ["all hands"]
    index = names.index("<sep>") + 1
    if names[index].startswith("FORM_"):
        labels.append("special forms")
    else:
        wind_pungs = 0
        while names[index].split("_")[0] in _MELD_PREFIXES:
            size = 4 if names[index].startswith("KONG") else 3
            if not names[index].startswith("CHOW") and names[index + 1].startswith("F"):
                wind_pungs += 1
            index += 1 + size
        if wind_pungs == 0:
            labels.append("no wind pungs")
        elif wind_pungs <= 2:
            labels.append("one or two wind pungs")
        else:
            labels.append("three or four wind pungs")
        if any(name.startswith("KONG") for name in names):
            labels.append("has a kong")
    if any(count > 1 for count in truth.values()):
        labels.append("a fan counted twice or more")
    for low, high in TOTAL_BANDS:
        if total >= low and (high is None or total <= high):
            labels.append(f"worth {low}+ fan" if high is None else f"worth {low}-{high} fan")
    return labels


def _fans_text(fans) -> str:
    return ", ".join(f"{fan.name}{f' x{fan.count}' if fan.count > 1 else ''} {fan.points * fan.count}" for fan in fans)


def _names_text(counter: Counter) -> str:
    return ", ".join(f"{name}{f' x{count}' if count > 1 else ''}" for name, count in counter.items()) or "(none)"


def _print_exam(report: dict) -> None:
    print(f"\nEXAM: name the fans of {report['rows']:,} unseen test hands")
    print(f"  training labels that disagree with the scorer: {report['label_mismatches']}")
    print("  fan list exactly right:")
    for label, (right, total) in report["by_group"].items():
        print(f"    {label:30s} {right:6,}/{total:<6,} {right / total:6.1%}")
    missed = sorted(
        ((count / total, name, count, total) for name, (count, total) in report["missed"].items() if total >= 30),
        reverse=True,
    )
    print("  fans most often missed (at least 30 cases):")
    for share, name, count, total in missed[:6]:
        print(f"    {name:34s} missed {count}/{total} ({share:.0%})")
    print("  fans most often added when they don't apply:")
    for name, count in sorted(report["added"].items(), key=lambda item: -item[1])[:6]:
        print(f"    {name:34s} {count}")


def _print_tricks(results: list) -> None:
    passed = sum(fan_counter(scored[0]) == guess for _trick, scored, guess in results)
    print(f"\nTRICK HANDS: {passed}/{len(results)} named exactly")
    for trick, scored, guess in results:
        truth = fan_counter(scored[0])
        verdict = "PASS" if truth == guess else "MISS"
        print(f"  {verdict}  {trick.name}: {trick.tests}")
        if verdict == "MISS":
            print(f"        scorer: {_fans_text(scored[0])} = {scored[1]}")
            print(f"        model:  {_names_text(guess)}")
            print(f"        missed: {_names_text(truth - guess)}   added: {_names_text(guess - truth)}")


def _print_stress(results: dict, requests: int, per_request: int) -> None:
    print(f"\nSTRESS: {requests} possible requests x {per_request} draw(s) per temperature")
    print("  (the hand is sampled at the temperature; fan names are taken greedily)")
    for temperature, report in results.items():
        verdicts = report["verdicts"]
        legal = verdicts["legal"]
        failures = ", ".join(f"{name} {count}" for name, count in verdicts.items() if name != "legal") or "none"
        exact = f"{report['exact']}/{legal} ({report['exact'] / legal:.1%})" if legal else "n/a"
        print(
            f"  temperature {temperature}: legal {legal}/{report['draws']} ({legal / report['draws']:.1%})"
            f"  | failures: {failures}"
            f"  | fans exact {exact}"
            f"  | distinct {report['distinct']}  | {report['seconds']:.0f}s"
        )


def _print_impossible(rows: list, draws: int) -> None:
    print(f"\nIMPOSSIBLE REQUESTS: {len(rows)} requests with no legal hand, {draws} draws each")
    for request, verdicts, seconds in rows:
        suits = ", ".join(sorted(request.suits, key=SUIT_NAMES.index))
        outcome = ", ".join(f"{name} {count}" for name, count in verdicts.items())
        print(f"  {suits}, {request.tile_count} tiles: legal {verdicts['legal']}  ({outcome})  {seconds:.1f}s")


def _print_profile(train: dict, generated: dict, temperature: float) -> None:
    print(f"\nPROFILE: generated hands (temperature {temperature}) vs a sample of the training data")
    print(f"  {'':22s} {'generated':>10s} {'training':>10s}")
    print(f"  {'hands':22s} {generated['hands']:>10,} {train['hands']:>10,}")
    print(f"  {'median total fan':22s} {generated['median']:>10} {train['median']:>10}")
    for key in ("64 fan or more", "under 16 fan"):
        print(f"  {key:22s} {generated[key]:>10.1%} {train[key]:>10.1%}")
    for label, report in (("generated", generated), ("training", train)):
        top = ", ".join(f"{name} {share:.0%}" for name, share in report["top"])
        print(f"  most common fans, {label}: {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harder checks for a trained checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/small.pt")
    parser.add_argument("--data", default="data/hands")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--exam-rows", type=int, default=0, help="0 means the whole test split")
    parser.add_argument("--temperatures", default="0.8,1.2,1.6")
    parser.add_argument("--per-request", type=int, default=1)
    parser.add_argument("--impossible-draws", type=int, default=10)
    parser.add_argument("--profile-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sections = [name.strip() for name in args.sections.split(",") if name.strip()]
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown sections: {', '.join(sorted(unknown))}")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = Tokenizer()
    feasible = feasible_requests(args.data)

    if "exam" in sections:
        rows, _meta = load_split(args.data, "test")
        if args.exam_rows:
            rows = rows[: args.exam_rows]
        _print_exam(exam(model, tokenizer, rows, device))
    if "tricks" in sections:
        _print_tricks(tricks(model, tokenizer, device))
    generated = None
    if "stress" in sections or "profile" in sections:
        temperatures = [float(value) for value in args.temperatures.split(",")] if "stress" in sections else [0.8]
        generated = stress(model, tokenizer, feasible, temperatures, args.per_request, device)
        if "stress" in sections:
            _print_stress(generated, len(feasible), args.per_request)
    if "impossible" in sections:
        possible = set(feasible)
        infeasible = [request for request in every_request() if request not in possible]
        _print_impossible(impossible(model, tokenizer, infeasible, args.impossible_draws, device), args.impossible_draws)
    if "profile" in sections:
        train_rows, _meta = load_split(args.data, "train")
        first = next(iter(generated))
        training = summarize(training_profile(tokenizer, train_rows, args.profile_rows, args.seed))
        _print_profile(training, summarize(generated[first]["legal"]), first)


if __name__ == "__main__":
    main()
