"""The hand grammar the model has to follow while it generates.

Replaying a sequence with these rules both checks training examples and masks
illegal next tokens at inference. A finished sequence decodes to the same
fields the scorer expects.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from mahjong_gpt.tiles import NUMBERED_SUITS, rank_of, suit_of
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


def allowed_next(tokenizer: Tokenizer, ids: Sequence[int]) -> set[int] | None:
    """Token ids that may be appended. None means the sequence is already complete."""
    state = _replay(tokenizer, ids)
    if state is None:
        return set()
    if state.mode == "done":
        return None
    allowed = set()
    for candidate in _candidates(tokenizer, state):
        if _advance(state, candidate) is not None:
            allowed.add(tokenizer.stoi[candidate])
    return allowed


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


def _take_form_tile(state: State, token: str) -> State | None:
    if not _is_tile(token) or suit_of(token) not in state.suits:
        return None
    if state.counts[token] + 1 > 4:
        return None
    nxt = state.copy()
    nxt.counts[token] += 1
    nxt.group = state.group + (token,)
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
