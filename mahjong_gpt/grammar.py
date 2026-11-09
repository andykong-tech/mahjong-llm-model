"""The hand grammar the model has to follow while it generates.

Replaying a sequence with these rules both checks training examples and masks
illegal next tokens at inference. A finished sequence decodes to the same
fields the scorer expects.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, permutations

from mahjong_gpt.tiles import (
    ALL_TILES,
    DRAGONS,
    NUMBERED_SUITS,
    PREFIX,
    SUIT_NAMES,
    WINDS,
    chow_of,
    rank_of,
    suit_of,
)
from mahjong_gpt.tokenizer import FAN_NAMES, MELD_TOKENS, Tokenizer

_MELD_LENGTH = {
    "CHOW_OPEN": 3,
    "CHOW_CLOSED": 3,
    "PUNG_OPEN": 3,
    "PUNG_CLOSED": 3,
    "KONG_OPEN": 4,
    "KONG_CLOSED": 4,
}
_DRAW = ("SELF", "DISCARD")
_SEATS = ("SEAT_E", "SEAT_S", "SEAT_W", "SEAT_N")
_ROUNDS = ("ROUND_E", "ROUND_S", "ROUND_W", "ROUND_N")
_HAND_FINISHED = {"win_mark", "win", "draw", "seat", "round", "fans_mark", "fan", "done"}


@dataclass
class ParsedHand:
    suits: frozenset[str]
    tile_count: int
    pack: tuple[tuple[str, str, int], ...]
    concealed: tuple[str, ...]
    win_tile: str
    self_drawn: bool
    seat_wind: int
    round_wind: int
    fan_names: tuple[str, ...]

    def tiles(self) -> list[str]:
        tiles = list(self.concealed) + [self.win_tile]
        for kind, tile, _offer in self.pack:
            if kind == "CHI":
                tiles.extend(chow_of(tile))
            else:
                tiles.extend([tile] * (4 if kind == "GANG" else 3))
        return tiles


@dataclass
class State:
    mode: str
    suits: frozenset[str]
    tile_count: int
    counts: Counter = field(default_factory=Counter)
    closed: Counter = field(default_factory=Counter)
    melds_done: int = 0
    kongs_done: int = 0
    group: tuple[str, ...] = ()
    group_kind: str = ""
    pack: list = field(default_factory=list)
    standing: list = field(default_factory=list)
    win_tile: str = ""
    self_drawn: bool = False
    seat_wind: int = 0
    round_wind: int = 0
    fans: list = field(default_factory=list)

    def copy(self) -> "State":
        twin = State(mode=self.mode, suits=self.suits, tile_count=self.tile_count)
        twin.counts = self.counts.copy()
        twin.closed = self.closed.copy()
        twin.melds_done = self.melds_done
        twin.kongs_done = self.kongs_done
        twin.group = self.group
        twin.group_kind = self.group_kind
        twin.pack = list(self.pack)
        twin.standing = list(self.standing)
        twin.win_tile = self.win_tile
        twin.self_drawn = self.self_drawn
        twin.seat_wind = self.seat_wind
        twin.round_wind = self.round_wind
        twin.fans = list(self.fans)
        return twin


def parse_ids(tokenizer: Tokenizer, ids: Sequence[int]) -> ParsedHand | None:
    state = _replay(tokenizer, ids)
    if state is None or state.mode != "done":
        return None
    return _finish(state)


def allowed_next(tokenizer: Tokenizer, ids: Sequence[int], required: Counter | None = None) -> set[int] | None:
    """Token ids that may be appended. None means the sequence is already complete.

    A token is only allowed when every requested suit, and every tile in
    `required`, can still be placed somewhere in the finished hand afterwards.
    """
    state = _replay(tokenizer, ids)
    if state is None:
        return set()
    if state.mode == "done":
        return None
    allowed = set()
    for candidate in _candidates(tokenizer, state):
        nxt = _advance(state, candidate)
        if nxt is not None and _coverable(nxt, required or Counter()):
            allowed.add(tokenizer.stoi[candidate])
    return allowed


def _coverable(state: State, required: Counter) -> bool:
    missing = Counter(
        {tile: need - state.counts[tile] for tile, need in required.items() if need > state.counts[tile]}
    )
    if any(suit_of(tile) not in state.suits for tile in missing):
        return False
    if state.mode in _HAND_FINISHED:
        return not missing and not _uncovered(state.suits, state.counts)
    if state.mode == "form":
        return _form_possible(state.group_kind, state.group, state.suits, required)
    counts = state.counts.copy()
    melds_left = 4 - state.melds_done
    kongs_left = state.tile_count - 14 - state.kongs_done
    slots: Counter = Counter()
    if state.mode == "tiles":
        melds_left -= 1
        is_kong = state.group_kind.startswith("KONG")
        if is_kong:
            kongs_left -= 1
        if state.group:
            _place(_rest_of_group(state), missing, counts)
        else:
            slots["kong" if is_kong else state.group_kind.split("_")[0].lower()] += 1
    slots["kong"] += kongs_left
    slots["any"] += melds_left - kongs_left
    if state.mode == "pair" and state.group:
        _place([state.group[0]], missing, counts)
    else:
        slots["pair"] += 1
    return _pack(+missing, slots, counts, state.suits)


def _uncovered(suits: frozenset[str], counts: Counter) -> set[str]:
    return set(suits) - {suit_of(tile) for tile, count in counts.items() if count > 0}


def _suits_fit(suits: frozenset[str], slots: Counter, counts: Counter) -> bool:
    """Each group holds one suit, and a chow can never hold winds or dragons."""
    uncovered = _uncovered(suits, counts)
    if not uncovered:
        return True
    open_slots = sum(count for count in slots.values() if count > 0)
    honors = sum(1 for suit in uncovered if suit not in NUMBERED_SUITS)
    return len(uncovered) <= open_slots and honors <= open_slots - max(slots["chow"], 0)


def _rest_of_group(state: State) -> list[str]:
    """Tiles a meld already under way will still add; its first tile fixes them."""
    first = state.group[0]
    if state.group_kind.startswith("CHOW"):
        rank = rank_of(first)
        return [f"{first[0]}{rank + offset}" for offset in range(len(state.group), 3)]
    return [first] * (_MELD_LENGTH[state.group_kind] - len(state.group))


def _place(tiles, missing: Counter, counts: Counter) -> None:
    for tile in tiles:
        counts[tile] += 1
        if missing[tile] > 0:
            missing[tile] -= 1


def _pack(missing: Counter, slots: Counter, counts: Counter, suits: frozenset[str]) -> bool:
    """Search for open melds and the pair that hold every missing tile and every missing suit."""
    if not missing:
        return _suits_fit(suits, slots, counts)
    tile = min(missing)
    for kind in ("pair", "kong", "pung", "chow", "any"):
        if slots[kind] <= 0:
            continue
        for shape in _shapes(kind, tile):
            if any(counts[t] + n > 4 for t, n in Counter(shape).items()):
                continue
            rest_missing = missing - Counter(shape)
            rest_counts = counts + Counter(shape)
            rest_slots = slots.copy()
            rest_slots[kind] -= 1
            if _pack(rest_missing, rest_slots, rest_counts, suits):
                return True
    return False


def _shapes(kind: str, tile: str) -> list[list[str]]:
    if kind == "pair":
        return [[tile] * 2]
    if kind == "kong":
        return [[tile] * 4]
    pung = [[tile] * 3]
    chows = []
    if suit_of(tile) in NUMBERED_SUITS:
        rank = rank_of(tile)
        for start in range(max(1, rank - 2), min(7, rank) + 1):
            chows.append([f"{tile[0]}{start + offset}" for offset in range(3)])
    if kind == "pung":
        return pung
    if kind == "chow":
        return chows
    return pung + chows


def _replay(tokenizer: Tokenizer, ids: Sequence[int]) -> State | None:
    prompt = _prompt_state(tokenizer, ids)
    if prompt is None:
        return None
    state, index = prompt
    for token_id in ids[index:]:
        token = tokenizer.token(int(token_id))
        state = _advance(state, token)
        if state is None:
            return None
    return state


def _prompt_state(tokenizer: Tokenizer, ids: Sequence[int]) -> tuple[State, int] | None:
    if not ids or int(ids[0]) != tokenizer.bos_id:
        return None
    suits = []
    index = 1
    while index < len(ids) and tokenizer.token(int(ids[index])).startswith("SUIT_"):
        suits.append(tokenizer.token(int(ids[index])).removeprefix("SUIT_").lower())
        index += 1
    if index >= len(ids) or not tokenizer.token(int(ids[index])).startswith("TILES_"):
        return None
    tile_count = int(tokenizer.token(int(ids[index])).removeprefix("TILES_"))
    index += 1
    if index >= len(ids) or int(ids[index]) != tokenizer.sep_id:
        return None
    if not suits:
        return None
    state = State(mode="start", suits=frozenset(suits), tile_count=tile_count)
    return state, index + 1


def _candidates(tokenizer: Tokenizer, state: State) -> tuple[str, ...]:
    if state.mode in {"start", "meld"}:
        tokens = list(MELD_TOKENS)
        if state.mode == "start":
            tokens.extend(token for token in tokenizer.tokens if token.startswith("FORM_"))
        return tuple(tokens)
    if state.mode == "need_pair":
        return ("<pair>",)
    if state.mode == "win_mark":
        return ("<win>",)
    if state.mode in {"tiles", "pair", "form", "win"}:
        return tuple(tile for tile in tokenizer.tokens if _is_tile(tile) and suit_of(tile) in state.suits)
    if state.mode == "draw":
        return _DRAW
    if state.mode == "seat":
        return _SEATS
    if state.mode == "round":
        return _ROUNDS
    if state.mode == "fans_mark":
        return ("<fans>",)
    if state.mode == "fan":
        names = list(FAN_NAMES)
        if state.fans:
            names.append("<eos>")
        return tuple(names)
    return ()


def _advance(state: State, token: str) -> State | None:
    if state.mode == "need_pair":
        if token != "<pair>" or state.kongs_done != state.tile_count - 14:
            return None
        nxt = state.copy()
        nxt.mode = "pair"
        nxt.group = ()
        return nxt
    if state.mode == "win_mark":
        if token != "<win>":
            return None
        nxt = state.copy()
        nxt.mode = "win"
        return nxt
    if state.mode in {"start", "meld"}:
        return _begin_group(state, token)
    if state.mode == "tiles":
        return _take_group_tile(state, token)
    if state.mode == "pair":
        return _take_pair_tile(state, token)
    if state.mode == "form":
        return _take_form_tile(state, token)
    if state.mode == "win":
        return _take_win(state, token)
    if state.mode == "draw" and token in _DRAW:
        nxt = state.copy()
        nxt.self_drawn = token == "SELF"
        nxt.mode = "seat"
        return nxt
    if state.mode == "seat" and token in _SEATS:
        nxt = state.copy()
        nxt.seat_wind = _SEATS.index(token)
        nxt.mode = "round"
        return nxt
    if state.mode == "round" and token in _ROUNDS:
        nxt = state.copy()
        nxt.round_wind = _ROUNDS.index(token)
        nxt.mode = "fans_mark"
        return nxt
    if state.mode == "fans_mark" and token == "<fans>":
        nxt = state.copy()
        nxt.mode = "fan"
        return nxt
    if state.mode == "fan" and token in FAN_NAMES:
        nxt = state.copy()
        nxt.fans.append(token)
        return nxt
    if state.mode == "fan" and token == "<eos>" and state.fans:
        nxt = state.copy()
        nxt.mode = "done"
        return nxt
    return None


def _begin_group(state: State, token: str) -> State | None:
    required = state.tile_count - 14
    remaining = 4 - state.melds_done
    still = required - state.kongs_done
    if token.startswith("FORM_") and state.mode == "start":
        if state.tile_count != 14:
            return None
        nxt = state.copy()
        nxt.mode = "form"
        nxt.group_kind = token
        nxt.group = ()
        return nxt
    if token not in _MELD_LENGTH or remaining <= 0 or still > remaining or still < 0:
        return None
    is_kong = token.startswith("KONG")
    if is_kong and still <= 0:
        return None
    if not is_kong and still == remaining:
        return None
    nxt = state.copy()
    nxt.mode = "tiles"
    nxt.group_kind = token
    nxt.group = ()
    return nxt


def _take_group_tile(state: State, token: str) -> State | None:
    if not _is_tile(token) or suit_of(token) not in state.suits:
        return None
    kind = state.group_kind
    group = state.group + (token,)
    if kind.startswith("CHOW"):
        if not _chow_prefix(group):
            return None
    elif any(tile != group[0] for tile in group):
        return None
    if state.counts[token] + 1 > 4:
        return None
    nxt = state.copy()
    nxt.counts[token] += 1
    nxt.group = group
    if len(group) < _MELD_LENGTH[kind]:
        return nxt
    closed = kind.endswith("CLOSED")
    if kind.startswith("CHOW"):
        if closed:
            nxt.standing.extend(group)
        else:
            nxt.pack.append(("CHI", group[1], 2))
    elif kind.startswith("PUNG"):
        if closed:
            nxt.standing.extend(group)
        else:
            nxt.pack.append(("PENG", group[0], 1))
    else:
        offer = 0 if closed else 1
        nxt.pack.append(("GANG", group[0], offer))
        nxt.kongs_done += 1
    if closed and not kind.startswith("KONG"):
        nxt.closed.update(group)
    nxt.melds_done += 1
    nxt.group = ()
    nxt.mode = "meld" if nxt.melds_done < 4 else "need_pair"
    return nxt


def _take_pair_tile(state: State, token: str) -> State | None:
    if not _is_tile(token) or suit_of(token) not in state.suits:
        return None
    if state.group and token != state.group[0]:
        return None
    if state.counts[token] + 1 > 4:
        return None
    nxt = state.copy()
    nxt.counts[token] += 1
    nxt.group = state.group + (token,)
    if len(nxt.group) < 2:
        return nxt
    nxt.standing.extend(nxt.group)
    nxt.closed.update(nxt.group)
    nxt.group = ()
    nxt.mode = "win_mark"
    return nxt


def _tile_key(tile: str) -> tuple:
    return (SUIT_NAMES.index(suit_of(tile)), tile)


def _form_possible(kind: str, placed: tuple[str, ...], suits: frozenset[str], required: Counter) -> bool:
    """True when `placed` is a sorted prefix of this special hand and the given tiles still fit."""
    if len(placed) > 14 or any(count > 4 for count in required.values()) or sum(required.values()) > 14:
        return False
    if any(_tile_key(earlier) > _tile_key(later) for earlier, later in zip(placed, placed[1:])):
        return False
    if any(suit_of(tile) not in suits for tile in placed):
        return False
    if kind in _FORM_TARGETS:
        return _matches_any(placed, suits, required, _FORM_TARGETS[kind])
    if kind == "FORM_SEVEN_PAIRS":
        return _seven_pairs_possible(placed, suits, required, honors_only=False)
    if kind == "FORM_ALL_HONORS":
        return _seven_pairs_possible(placed, suits, required, honors_only=True)
    return False


def _matches_any(placed, suits, required, targets) -> bool:
    placed_tuple = tuple(placed)
    for suit_set, ordered, counts in targets:
        if suit_set != suits or ordered[: len(placed_tuple)] != placed_tuple:
            continue
        if all(counts[tile] >= need for tile, need in required.items()):
            return True
    return False


def _seven_pairs_possible(placed, suits, required, *, honors_only: bool) -> bool:
    """Seven pairs, each tile used twice or four times. All Honors is that pattern with no numbers."""
    if honors_only and any(suit in NUMBERED_SUITS for suit in suits):
        return False
    pool = [
        tile
        for tile in ALL_TILES
        if suit_of(tile) in suits and not (honors_only and suit_of(tile) in NUMBERED_SUITS)
    ]
    pool_set = set(pool)
    if any(tile not in pool_set for tile in placed):
        return False
    last = placed[-1] if placed else None
    frozen: Counter = Counter()
    if last is not None:
        for tile in placed:
            if tile != last:
                frozen[tile] += 1
    for tile, count in frozen.items():
        if count not in (2, 4) or count < required[tile]:
            return False
    for tile, need in required.items():
        if need > frozen[tile] and last is not None and _tile_key(tile) < _tile_key(last):
            return False
        if need > 0 and tile not in pool_set and frozen[tile] < need:
            return False
    open_count = placed.count(last) if last is not None else 0
    start = [tile for tile in pool if last is None or _tile_key(tile) >= _tile_key(last)]
    if last is not None and (not start or start[0] != last):
        return False
    covered = {suit_of(tile) for tile in frozen}
    suit_index = {suit: bit for bit, suit in enumerate(sorted(suits))}
    need_mask = 0
    for suit in suits:
        if suit not in covered:
            need_mask |= 1 << suit_index[suit]
    slots_left = 7 - sum(count // 2 for count in frozen.values())

    def finals(tile: str, is_open: bool) -> tuple[int, ...]:
        floor = max(open_count if is_open else 0, required[tile])
        if floor > 4:
            return ()
        choices = []
        if not is_open and required[tile] == 0:
            choices.append(0)
        if floor <= 2:
            choices.append(2)
        if floor <= 4:
            choices.append(4)
        return tuple(choices)

    @lru_cache(maxsize=None)
    def fill(index: int, slots: int, mask: int) -> bool:
        if slots < 0:
            return False
        if index == len(start):
            return slots == 0 and mask == 0
        tile = start[index]
        for count in finals(tile, index == 0 and last is not None):
            bit = 0 if count == 0 else 1 << suit_index[suit_of(tile)]
            if fill(index + 1, slots - count // 2, mask & ~bit):
                return True
        return False

    return fill(0, slots_left, need_mask)


def _sorted_tiles(tiles) -> tuple[str, ...]:
    return tuple(sorted(tiles, key=_tile_key))


def _target(suits, tiles) -> tuple:
    ordered = _sorted_tiles(tiles)
    return (frozenset(suits), ordered, Counter(ordered))


def _knitted_nines() -> list[tuple[str, ...]]:
    """The six ways to deal 147, 258 and 369 across the three numbered suits."""
    groups = ((1, 4, 7), (2, 5, 8), (3, 6, 9))
    nines = []
    for assignment in permutations(groups):
        tiles = []
        for suit, ranks in zip(("characters", "bamboo", "dots"), assignment):
            prefix = PREFIX[suit]
            tiles.extend(f"{prefix}{rank}" for rank in ranks)
        nines.append(_sorted_tiles(tiles))
    return nines


def _form_targets() -> dict:
    """Every tile list each fixed special hand can be, sorted the way the tokenizer writes them."""
    orphans = ("W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3")
    honors = WINDS + DRAGONS
    every_suit = frozenset(SUIT_NAMES)
    nines = _knitted_nines()
    targets = {
        "FORM_THIRTEEN_ORPHANS": [
            _target(every_suit, list(orphans) + [extra]) for extra in orphans
        ],
        "FORM_GREATER_HONORS_AND_KNITTED": [
            _target(every_suit, list(chosen) + list(honors))
            for nine in nines
            for chosen in combinations(nine, 7)
        ],
        "FORM_LESSER_HONORS_AND_KNITTED": [
            _target(every_suit, list(nine) + ["F1", "F2", "F3", "F4", "J1"]) for nine in nines
        ],
        "FORM_KNITTED_STRAIGHT": [],
        "FORM_NINE_GATES": [],
        "FORM_SEVEN_SHIFTED_PAIRS": [],
    }
    for nine in nines:
        counts = Counter(nine)
        counts["J1"] += 3
        counts["J2"] += 2
        targets["FORM_KNITTED_STRAIGHT"].append(
            _target(frozenset(NUMBERED_SUITS | {"dragons"}), list(counts.elements()))
        )
    for suit in ("characters", "bamboo", "dots"):
        prefix = PREFIX[suit]
        for extra in range(1, 10):
            tiles = [f"{prefix}1"] * 3 + [f"{prefix}{rank}" for rank in range(2, 9)] + [f"{prefix}9"] * 3
            tiles.append(f"{prefix}{extra}")
            targets["FORM_NINE_GATES"].append(_target(frozenset({suit}), tiles))
        for start in (1, 2, 3):
            tiles = [f"{prefix}{rank}" for rank in range(start, start + 7) for _ in range(2)]
            targets["FORM_SEVEN_SHIFTED_PAIRS"].append(_target(frozenset({suit}), tiles))
    return targets


_FORM_TARGETS = _form_targets()


def _take_form_tile(state: State, token: str) -> State | None:
    if not _is_tile(token) or suit_of(token) not in state.suits:
        return None
    if state.counts[token] + 1 > 4:
        return None
    group = state.group + (token,)
    if not _form_possible(state.group_kind, group, state.suits, Counter()):
        return None
    nxt = state.copy()
    nxt.counts[token] += 1
    nxt.group = group
    if len(nxt.group) < state.tile_count:
        return nxt
    nxt.standing = list(nxt.group)
    nxt.closed = Counter(nxt.group)
    nxt.group = ()
    nxt.mode = "win_mark"
    return nxt


def _take_win(state: State, token: str) -> State | None:
    if state.closed[token] <= 0:
        return None
    nxt = state.copy()
    nxt.win_tile = token
    nxt.mode = "draw"
    return nxt


def _chow_prefix(group: tuple[str, ...]) -> bool:
    if suit_of(group[0]) not in NUMBERED_SUITS:
        return False
    ranks = [rank_of(tile) for tile in group]
    if any(suit_of(tile) != suit_of(group[0]) for tile in group):
        return False
    if ranks[0] < 1 or ranks[0] > 7:
        return False
    return all(rank == ranks[0] + offset for offset, rank in enumerate(ranks))


def _finish(state: State) -> ParsedHand | None:
    standing = list(state.standing)
    try:
        standing.remove(state.win_tile)
    except ValueError:
        return None
    if len(standing) + 3 * len(state.pack) != 13:
        return None
    return ParsedHand(
        suits=state.suits,
        tile_count=state.tile_count,
        pack=tuple(state.pack),
        concealed=tuple(standing),
        win_tile=state.win_tile,
        self_drawn=state.self_drawn,
        seat_wind=state.seat_wind,
        round_wind=state.round_wind,
        fan_names=tuple(state.fans),
    )


def _is_tile(token: str) -> bool:
    if len(token) != 2 or token[0] not in "WTBFJ":
        return False
    return token[1].isdigit()
