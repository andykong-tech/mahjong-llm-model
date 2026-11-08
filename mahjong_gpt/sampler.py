"""Generate finished Chinese Official Mahjong hands for a suit and tile-count request.

The request lists the suits that must appear, and no other suit may appear.
Tile count is 14 plus one tile for every kong. Random hands are four melds and
a pair. Rare builders cover patterns that random melds almost never hit.
Only hands the rules score at 8 fan or more are returned.
"""

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mahjong_gpt.hands import (
    Hand,
    Meld,
    closed_hand,
    hand_suits,
    physical_tiles,
    standard_hand,
    within_copies,
)
from mahjong_gpt.rules import Fan, is_legal_win, score
from mahjong_gpt.tiles import (
    DRAGONS,
    NUMBERED_SUITS,
    PREFIX,
    SUITS,
    WINDS,
    parse_suits,
    suit_capacity,
)

GREEN_BAMBOO = ("T2", "T3", "T4", "T6", "T8")
TERMINALS = ("W1", "W9", "T1", "T9", "B1", "B9")
ORPHANS = ("W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3")


@dataclass(frozen=True)
class Request:
    suits: frozenset[str]
    tile_count: int

    @property
    def kongs(self) -> int:
        return self.tile_count - 14


@dataclass(frozen=True)
class ScoredHand:
    hand: Hand
    fans: tuple[Fan, ...]
    total: int


def generate(
    suits: Iterable[str],
    tile_count: int,
    n: int = 1,
    *,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> list[ScoredHand]:
    """Return up to `n` legal winning hands that use exactly these suits."""
    chosen = parse_suits(suits)
    if tile_count not in range(14, 19):
        raise ValueError("tile_count must be from 14 to 18")
    if n < 0:
        raise ValueError("n must be non-negative")
    request = Request(chosen, tile_count)
    if tile_count > suit_capacity(chosen):
        return []
    rng = random.Random(seed)
    limit = n * 100 if max_attempts is None else max_attempts
    found: list[ScoredHand] = []
    for _ in range(limit):
        if len(found) == n:
            break
        hand = sample_hand(request, rng)
        if hand is None or not _matches_request(hand, request):
            continue
        scored = score(
            hand.pack,
            hand.concealed,
            hand.win_tile,
            self_drawn=hand.self_drawn,
            seat_wind=hand.seat_wind,
            round_wind=hand.round_wind,
        )
        if not is_legal_win(scored):
            continue
        fans, total = scored
        found.append(ScoredHand(hand, fans, total))
    return found


def sample_hand(request: Request, rng: random.Random) -> Hand | None:
    if rng.random() < 0.25:
        rare = sample_rare(request, rng)
        if rare is not None:
            return rare
    standard = sample_standard(request, rng)
    if standard is not None:
        return standard
    return sample_rare(request, rng)


def sample_standard(request: Request, rng: random.Random) -> Hand | None:
    if request.kongs not in range(5) or len(request.suits) > 5:
        return None
    if request.tile_count > suit_capacity(request.suits):
        return None
    for _ in range(60):
        hand = _attempt_standard(request, rng)
        if hand is not None:
            return hand
    return None


def sample_rare(request: Request, rng: random.Random) -> Hand | None:
    builders = list(RARE_BUILDERS)
    rng.shuffle(builders)
    for builder in builders:
        hand = builder(request, rng)
        if hand is not None:
            return hand
    return None


def thirteen_orphans(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or request.suits != frozenset(SUITS):
        return None
    extra = rng.choice(ORPHANS)
    return closed_hand("thirteen_orphans", list(ORPHANS) + [extra], extra, rng)


def seven_pairs(request: Request, rng: random.Random) -> Hand | None:
    tiles = _seven_pair_tiles(request, rng)
    if tiles is None:
        return None
    return closed_hand("seven_pairs", tiles, rng.choice(tiles), rng)


def seven_shifted_pairs(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or len(request.suits) != 1:
        return None
    suit = next(iter(request.suits))
    if suit not in NUMBERED_SUITS:
        return None
    start = rng.choice((1, 2, 3))
    prefix = PREFIX[suit]
    ranks = range(start, start + 7)
    tiles = [f"{prefix}{rank}" for rank in ranks for _ in range(2)]
    return closed_hand("seven_shifted_pairs", tiles, rng.choice(tiles), rng)


def nine_gates(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or len(request.suits) != 1:
        return None
    suit = next(iter(request.suits))
    if suit not in NUMBERED_SUITS:
        return None
    prefix = PREFIX[suit]
    base = [f"{prefix}1"] * 3 + [f"{prefix}{rank}" for rank in range(2, 9)] + [f"{prefix}9"] * 3
    extra = f"{prefix}{rng.randint(1, 9)}"
    return closed_hand("nine_gates", base + [extra], extra, rng)


def greater_honors_and_knitted(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or request.suits != frozenset(SUITS):
        return None
    numbered = _knitted_tiles(rng)
    chosen = rng.sample(numbered, 7)
    tiles = list(WINDS + DRAGONS) + chosen
    return closed_hand("greater_honors_and_knitted", tiles, rng.choice(tiles), rng)


def lesser_honors_and_knitted(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or request.suits != frozenset(SUITS):
        return None
    tiles = _knitted_tiles(rng) + ["F1", "F2", "F3", "F4", "J1"]
    return closed_hand("lesser_honors_and_knitted", tiles, "J1", rng)


def knitted_straight(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or request.suits != NUMBERED_SUITS | {"dragons"}:
        return None
    tiles = _knitted_tiles(rng) + ["J1", "J1", "J1", "J2", "J2"]
    return closed_hand("knitted_straight", tiles, rng.choice(("J1", "J2")), rng)


def all_green(request: Request, rng: random.Random) -> Hand | None:
    if request.kongs not in range(5):
        return None
    if request.suits == frozenset({"bamboo", "dragons"}):
        pool = list(GREEN_BAMBOO)
        rng.shuffle(pool)
        meld_tiles = pool[:4]
        pair = "J2"
    elif request.suits == frozenset({"bamboo"}):
        pool = list(GREEN_BAMBOO)
        rng.shuffle(pool)
        meld_tiles, pair = pool[:4], pool[4]
    else:
        return None
    kong_at = set(rng.sample(range(4), request.kongs))
    melds = [("kong" if i in kong_at else "pung", tile) for i, tile in enumerate(meld_tiles)]
    return _finish(melds, pair, rng, "all_green")


def big_three_dragons(request: Request, rng: random.Random) -> Hand | None:
    if request.kongs not in range(5) or "dragons" not in request.suits:
        return None
    others = list(request.suits - {"dragons"})
    if not others or len(others) > 2:
        return None
    dragons = list(DRAGONS)
    rng.shuffle(dragons)
    slots = ["pung", "pung", "pung", "pung"]
    for index in range(request.kongs):
        slots[index] = "kong"
    counts: Counter[str] = Counter()
    melds = []
    for kind, tile in zip(slots, dragons):
        copies = 4 if kind == "kong" else 3
        counts[tile] += copies
        melds.append((kind, tile))
    extra_suit, pair_suit = _cover_two(others, rng)
    extra = _pick_tile(extra_suit, counts, 4 if slots[3] == "kong" else 3, rng)
    if extra is None:
        return None
    counts[extra] += 4 if slots[3] == "kong" else 3
    pair = _pick_tile(pair_suit, counts, 2, rng)
    if pair is None:
        return None
    melds.append((slots[3], extra))
    return _finish(melds, pair, rng, "big_three_dragons")


def big_four_winds(request: Request, rng: random.Random) -> Hand | None:
    if request.kongs not in range(5) or "winds" not in request.suits:
        return None
    others = list(request.suits - {"winds"})
    if len(others) != 1:
        return None
    winds = list(WINDS)
    rng.shuffle(winds)
    kong_at = set(rng.sample(range(4), request.kongs))
    melds = [("kong" if i in kong_at else "pung", wind) for i, wind in enumerate(winds)]
    pair = rng.choice(SUITS[others[0]])
    return _finish(melds, pair, rng, "big_four_winds")


def four_kongs(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 18 or len(request.suits) > 5:
        return None
    if request.tile_count > suit_capacity(request.suits):
        return None
    group_suits = _cover_groups(request.suits, 5, rng)
    counts: Counter[str] = Counter()
    melds = []
    for suit in group_suits[:4]:
        tile = _pick_tile(suit, counts, 4, rng)
        if tile is None:
            return None
        counts[tile] += 4
        melds.append(("kong", tile))
    pair = _pick_tile(group_suits[4], counts, 2, rng)
    if pair is None:
        return None
    return _finish(melds, pair, rng, "four_kongs")


def all_honors(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or not request.suits <= {"winds", "dragons"}:
        return None
    tiles = _seven_pair_tiles(request, rng)
    if tiles is None:
        return None
    return closed_hand("all_honors", tiles, rng.choice(tiles), rng)


def quadruple_chow(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14 or len(request.suits) != 1:
        return None
    suit = next(iter(request.suits))
    if suit not in NUMBERED_SUITS:
        return None
    middle = rng.randint(2, 8)
    used = {middle - 1, middle, middle + 1}
    pair_rank = rng.choice([rank for rank in range(1, 10) if rank not in used])
    prefix = PREFIX[suit]
    melds = [("chow", f"{prefix}{middle}") for _ in range(4)]
    return _finish(melds, f"{prefix}{pair_rank}", rng, "quadruple_chow")


def pure_straight(request: Request, rng: random.Random) -> Hand | None:
    if request.tile_count != 14:
        return None
    straight_suit = _straight_suit(request)
    if straight_suit is None:
        return None
    prefix = PREFIX[straight_suit]
    counts: Counter[str] = Counter({f"{prefix}{rank}": 1 for rank in range(1, 10)})
    melds = [("chow", f"{prefix}{middle}") for middle in (2, 5, 8)]
    others = list(request.suits - {straight_suit})
    extra_suit, pair_suit = _cover_two(others or [straight_suit], rng)
    extra = _pick_tile(extra_suit, counts, 3, rng)
    if extra is None:
        return None
    counts[extra] += 3
    pair = _pick_tile(pair_suit, counts, 2, rng)
    if pair is None:
        return None
    melds.append(("pung", extra))
    return _finish(melds, pair, rng, "pure_straight")


def all_terminals(request: Request, rng: random.Random) -> Hand | None:
    if request.kongs not in range(5) or request.suits != NUMBERED_SUITS:
        return None
    pool = list(TERMINALS)
    rng.shuffle(pool)
    kong_at = set(rng.sample(range(4), request.kongs))
    melds = [("kong" if i in kong_at else "pung", pool[i]) for i in range(4)]
    return _finish(melds, pool[4], rng, "all_terminals")


RARE_BUILDERS = (
    thirteen_orphans,
    seven_pairs,
    seven_shifted_pairs,
    nine_gates,
    greater_honors_and_knitted,
    lesser_honors_and_knitted,
    knitted_straight,
    all_green,
    big_three_dragons,
    big_four_winds,
    four_kongs,
    all_honors,
    quadruple_chow,
    pure_straight,
    all_terminals,
)


def _attempt_standard(request: Request, rng: random.Random) -> Hand | None:
    group_suits = _cover_groups(request.suits, 5, rng)
    kong_at = set(rng.sample(range(4), request.kongs))
    counts: Counter[str] = Counter()
    melds = []
    for index, suit in enumerate(group_suits[:4]):
        if index in kong_at:
            meld = _take_pung("kong", suit, counts, rng)
        elif suit in NUMBERED_SUITS and rng.randrange(2):
            meld = _take_chow(suit, counts, rng) or _take_pung("pung", suit, counts, rng)
        else:
            meld = _take_pung("pung", suit, counts, rng)
            if meld is None and suit in NUMBERED_SUITS:
                meld = _take_chow(suit, counts, rng)
        if meld is None:
            return None
        melds.append(meld)
    pair = _pick_tile(group_suits[4], counts, 2, rng)
    if pair is None:
        return None
    return _finish(melds, pair, rng, "standard")


def _finish(melds: Sequence[tuple[str, str]], pair: str, rng: random.Random, form: str) -> Hand:
    candidates = [index for index, (kind, _tile) in enumerate(melds) if kind != "kong"]
    candidates.append(len(melds))
    chosen = rng.choice(candidates)
    built: list[Meld] = []
    win_tile = pair
    for index, (kind, tile) in enumerate(melds):
        contains_win = index == chosen
        if kind == "kong":
            concealed = bool(rng.randrange(2))
        elif contains_win:
            concealed = True
        else:
            concealed = bool(rng.randrange(2))
        meld = Meld(kind, tile, concealed)
        built.append(meld)
        if contains_win:
            win_tile = tile if kind == "pung" else rng.choice(meld.tiles())
    return standard_hand(built, pair, win_tile, rng, form)


def _seven_pair_tiles(request: Request, rng: random.Random) -> list[str] | None:
    if request.tile_count != 14 or len(request.suits) > 7:
        return None
    usage: Counter[str] = Counter()
    chosen: list[str] = []
    for suit in request.suits:
        options = [tile for tile in SUITS[suit] if usage[tile] < 2]
        if not options:
            return None
        tile = rng.choice(options)
        usage[tile] += 1
        chosen.append(tile)
    pool = [tile for suit in request.suits for tile in SUITS[suit]]
    while len(chosen) < 7:
        options = [tile for tile in pool if usage[tile] < 2]
        if not options:
            return None
        tile = rng.choice(options)
        usage[tile] += 1
        chosen.append(tile)
    return [tile for tile in chosen for _ in range(2)]


def _knitted_tiles(rng: random.Random) -> list[str]:
    """One 147 / 258 / 369 knitting spread across the three numbered suits."""
    suits = ["characters", "bamboo", "dots"]
    rng.shuffle(suits)
    groups = ((1, 4, 7), (2, 5, 8), (3, 6, 9))
    tiles = []
    for suit, ranks in zip(suits, groups):
        prefix = PREFIX[suit]
        tiles.extend(f"{prefix}{rank}" for rank in ranks)
    return tiles


def _straight_suit(request: Request) -> str | None:
    for suit in sorted(request.suits & NUMBERED_SUITS):
        if len(request.suits - {suit}) <= 2:
            return suit
    return None


def _cover_two(suits: list[str], rng: random.Random) -> tuple[str, str]:
    if len(suits) == 1:
        return suits[0], suits[0]
    shuffled = list(suits)
    rng.shuffle(shuffled)
    return shuffled[0], shuffled[1]


def _cover_groups(suits: frozenset[str], count: int, rng: random.Random) -> list[str]:
    groups = list(suits)
    while len(groups) < count:
        groups.append(rng.choice(tuple(suits)))
    rng.shuffle(groups)
    return groups


def _pick_tile(suit: str, counts: Counter[str], copies: int, rng: random.Random) -> str | None:
    options = [tile for tile in SUITS[suit] if counts[tile] + copies <= 4]
    if not options:
        return None
    return rng.choice(options)


def _take_pung(kind: str, suit: str, counts: Counter[str], rng: random.Random):
    copies = 4 if kind == "kong" else 3
    tile = _pick_tile(suit, counts, copies, rng)
    if tile is None:
        return None
    counts[tile] += copies
    return kind, tile


def _take_chow(suit: str, counts: Counter[str], rng: random.Random):
    prefix = PREFIX[suit]
    options = []
    for middle in range(2, 9):
        tiles = (f"{prefix}{middle - 1}", f"{prefix}{middle}", f"{prefix}{middle + 1}")
        if all(counts[tile] + 1 <= 4 for tile in tiles):
            options.append(tiles)
    if not options:
        return None
    tiles = rng.choice(options)
    counts.update(tiles)
    return "chow", tiles[1]


def _matches_request(hand: Hand, request: Request) -> bool:
    tiles = physical_tiles(hand)
    return (
        len(tiles) == request.tile_count
        and hand_suits(hand) == request.suits
        and within_copies(hand)
    )
