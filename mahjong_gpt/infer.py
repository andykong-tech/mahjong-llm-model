"""Sample winning hands for a suit and tile-count request.

Tokens that would break the hand grammar are masked, and with --with every
given tile is kept placeable until the hand is finished. Each sample is checked
with the rules; one that is not an 8-fan win is drawn again.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.rules import fan_counter, is_legal_win, score
from mahjong_gpt.tiles import PREFIX, parse_suits, suit_of
from mahjong_gpt.tokenizer import CLOSED_FORMS, MAX_SEQUENCE, Tokenizer
from mahjong_gpt.train import pick_device

_WINDS = {"E": "East", "S": "South", "W": "West", "N": "North"}
_FORM_NAMES = {f"FORM_{form.upper()}": form.replace("_", " ") for form in CLOSED_FORMS}
_SUIT_WORDS = {
    "character": "characters",
    "characters": "characters",
    "bamboo": "bamboo",
    "bamboos": "bamboo",
    "dot": "dots",
    "dots": "dots",
    "wind": "winds",
    "winds": "winds",
    "dragon": "dragons",
    "dragons": "dragons",
}
_HONOR_RANKS = {
    "winds": ("east", "south", "west", "north"),
    "dragons": ("red", "green", "white"),
}


def load_model(path: str, device: str) -> GPT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def complete(
    model,
    tokenizer,
    ids,
    *,
    temperature: float,
    device: str,
    greedy_fans: bool = False,
    required: Counter | None = None,
):
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
            allowed = allowed_next(tokenizer, ids, required)
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


def sample_hand(model, tokenizer, suits, tile_count, *, temperature: float, device: str, required=None):
    """Return (ids, parsed, scored) for one legal winning hand, or None."""
    ids, status = complete(
        model,
        tokenizer,
        tokenizer.prompt(suits, tile_count),
        temperature=temperature,
        device=device,
        greedy_fans=True,
        required=required,
    )
    if status != "done":
        return None
    parsed = parse_ids(tokenizer, ids)
    if parsed is None:
        return None
    tiles = parsed.tiles()
    if len(tiles) != tile_count or {suit_of(tile) for tile in tiles} != set(suits):
        return None
    if required and required - Counter(tiles):
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
    return ids, parsed, scored


def parse_tiles(text: str) -> Counter:
    """Read tiles written as codes (T5) or as a suit and a number (bamboo 5, red dragon)."""
    tiles = Counter(tile_code(part) for part in text.split(",") if part.strip())
    for tile, count in tiles.items():
        if count > 4:
            raise ValueError(f"{tile} is given {count} times, but a set only has 4")
    return tiles


def tile_code(text: str) -> str:
    words = text.replace(":", " ").lower().split()
    if len(words) == 1:
        code = words[0].upper()
        suit_of(code)
        return code
    if len(words) == 2 and words[0] not in _SUIT_WORDS and words[1] in _SUIT_WORDS:
        words.reverse()
    if len(words) != 2 or words[0] not in _SUIT_WORDS:
        raise ValueError(f"cannot read tile {text.strip()!r}; write it like T5 or bamboo 5")
    suit = _SUIT_WORDS[words[0]]
    names = _HONOR_RANKS.get(suit, ())
    if words[1] in names:
        rank = str(names.index(words[1]) + 1)
    else:
        rank = words[1]
    code = f"{PREFIX[suit]}{rank}"
    try:
        suit_of(code)
    except ValueError:
        raise ValueError(f"there is no tile {text.strip()!r}") from None
    return code


def tile_name(code: str) -> str:
    suit = suit_of(code)
    rank = int(code[1])
    if suit == "winds":
        return f"{_HONOR_RANKS['winds'][rank - 1]} wind"
    if suit == "dragons":
        return f"{_HONOR_RANKS['dragons'][rank - 1]} dragon"
    return f"{suit} {rank}"


def request_suits(text: str, given: Counter) -> frozenset[str]:
    """The suits named on the command line, or else the suits of the given tiles."""
    if text:
        suits = parse_suits(part.strip().lower() for part in text.split(",") if part.strip())
    elif given:
        suits = frozenset(suit_of(tile) for tile in given)
    else:
        raise ValueError("pass --suits, --with, or both")
    outside = sorted(tile for tile in given if suit_of(tile) not in suits)
    if outside:
        raise ValueError(f"{', '.join(outside)} is not in the requested suits")
    return suits


def check_request(suits: frozenset[str], tile_count: int, given: Counter, feasibility: Path) -> None:
    if tile_count not in range(14, 19):
        raise ValueError("--tiles must be from 14 to 18")
    if sum(given.values()) > tile_count:
        raise ValueError(f"{sum(given.values())} given tiles do not fit in a {tile_count}-tile hand")
    if feasibility.exists():
        possible = {(frozenset(item["suits"]), item["tile_count"]) for item in json.loads(feasibility.read_text())}
        if (suits, tile_count) not in possible:
            raise ValueError(f"no legal {tile_count}-tile hand uses exactly: {', '.join(sorted(suits))}")


def render_hand(tokenizer, ids, given: Counter) -> str:
    """The hand's groups in the order the model wrote them; given tiles end in *."""
    names = [tokenizer.token(index) for index in ids]
    body = names[names.index("<sep>") + 1 : names.index("<win>")]
    left = given.copy()

    def mark(tile: str) -> str:
        if left[tile] > 0:
            left[tile] -= 1
            return f"{tile}*"
        return tile

    if body[0].startswith("FORM_"):
        return f"{_FORM_NAMES[body[0]]}: " + " ".join(mark(tile) for tile in body[1:])
    groups = []
    index = 0
    while index < len(body):
        token = body[index]
        if token == "<pair>":
            groups.append("pair " + " ".join(mark(tile) for tile in body[index + 1 : index + 3]))
            index += 3
            continue
        kind, side = token.split("_")
        size = 4 if kind == "KONG" else 3
        tiles = " ".join(mark(tile) for tile in body[index + 1 : index + 1 + size])
        if kind == "KONG":
            tiles += " (open kong)" if side == "OPEN" else " (concealed kong)"
        elif side == "OPEN":
            tiles += " (open)"
        groups.append(tiles)
        index += 1 + size
    return " | ".join(groups)


def render_win(tokenizer, ids) -> str:
    names = [tokenizer.token(index) for index in ids]
    at = names.index("<win>")
    win_tile, how, seat, round_wind = names[at + 1 : at + 5]
    drawn = "self-drawn" if how == "SELF" else "by discard"
    return f"won on {win_tile} {drawn}, seat {_WINDS[seat[-1]]}, round {_WINDS[round_wind[-1]]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mahjong winning hands")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suits", default="", help="Comma-separated, e.g. bamboo,dragons; defaults to the suits of --with")
    parser.add_argument("--tiles", type=int, default=14, help="14 to 18; each tile over 14 is one kong")
    parser.add_argument(
        "--with",
        dest="given",
        default="",
        help='Tiles the hand must contain, e.g. W2,T5,J1 or "characters 2, bamboo 5, red dragon"',
    )
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attempts", type=int, default=0)
    parser.add_argument("--data", default="data/hands", help="Folder holding feasibility.json")
    args = parser.parse_args()
    try:
        given = parse_tiles(args.given)
        suits = request_suits(args.suits, given)
        check_request(suits, args.tiles, given, Path(args.data) / "feasibility.json")
    except ValueError as error:
        parser.error(str(error))

    device = pick_device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = Tokenizer()
    win_id = tokenizer.stoi["<win>"]
    if given:
        listed = ", ".join(f"{tile_name(tile)} ({tile})" for tile, count in sorted(given.items()) for _ in range(count))
        print(f"your tiles: {listed}")
        print(f"marked * below; the model adds the other {args.tiles - sum(given.values())}\n")
    tries = args.attempts or args.num * 30
    found = 0
    seen = set()
    for _ in range(tries):
        if found == args.num:
            break
        result = sample_hand(
            model, tokenizer, suits, args.tiles, temperature=args.temperature, device=device, required=given
        )
        if result is None:
            continue
        ids, parsed, scored = result
        key = tuple(ids[: ids.index(win_id)])
        if key in seen:
            continue
        seen.add(key)
        fans, total = scored
        if Counter(parsed.fan_names) == fan_counter(fans):
            verdict = "the model named these exactly"
        else:
            verdict = "the model said: " + ", ".join(parsed.fan_names)
        print(f"hand   {render_hand(tokenizer, ids, given)}")
        print(f"       {render_win(tokenizer, ids)}")
        print(f"score  {total}: " + ", ".join(f"{fan.name} {fan.points * fan.count}" for fan in fans))
        print(f"       {verdict}\n")
        found += 1
    if found < args.num:
        print(f"stopped after {tries} draws with {found} legal hands")


if __name__ == "__main__":
    main()
