"""Every winning tile set when at most three tiles are still missing.

A hand is one multiset of tiles. Group order, which sets are open, and which
tile won it do not make a second hand. The scorer is the referee: a multiset
counts when some kong layout, winning tile, and seat makes it at least 8 fan.
With four or more tiles missing the list is too large to build, and the caller
gets None.
"""

from collections import Counter
from itertools import combinations, combinations_with_replacement

from mahjong_gpt.rules import MIN_FAN, score
from mahjong_gpt.tiles import ALL_TILES, NUMBERED_SUITS, SUIT_NAMES, rank_of, suit_capacity, suit_of

_MISSING_LIMIT = 3


def multiset_key(tiles) -> tuple[str, ...]:
    """Tiles in suit order, one entry per copy."""
    if isinstance(tiles, Counter):
        tiles = list(tiles.elements())
    return tuple(sorted(tiles, key=lambda tile: (SUIT_NAMES.index(suit_of(tile)), tile)))


def winning_multisets(suits: frozenset[str], tile_count: int, given: Counter | None = None) -> list[Counter] | None:
    """Winning hands of this size that contain `given`.

    None means four or more tiles are missing, so the list is not built.
    An empty list means every candidate was checked and none wins.
    """
    given = Counter(given or {})
    missing = tile_count - sum(given.values())
    if tile_count not in range(14, 19) or missing < 0 or missing > _MISSING_LIMIT:
        return None if missing > _MISSING_LIMIT and tile_count in range(14, 19) else []
    if suit_capacity(suits) < tile_count:
        return []
    if any(suit_of(tile) not in suits or count > 4 for tile, count in given.items()):
        return []
    covered = {suit_of(tile) for tile, count in given.items() if count}
    if len(set(suits) - covered) > missing:
        return []
    pool = [tile for tile in ALL_TILES if suit_of(tile) in suits]
    found = []
    for extra in _extras(pool, missing):
        counts = given + extra
        if any(count > 4 for count in counts.values()):
            continue
        if {suit_of(tile) for tile, count in counts.items() if count} != set(suits):
            continue
        if wins(counts, tile_count):
            found.append(counts)
    return found


def other_winning_size(suits: frozenset[str], tile_count: int, given: Counter | None = None) -> int | None:
    """Another hand size, from 14 to 18, that can hold these tiles."""
    given = Counter(given or {})
    for size in range(14, 19):
        if size == tile_count or size > suit_capacity(suits):
            continue
        missing = size - sum(given.values())
        if not 0 <= missing <= _MISSING_LIMIT:
            continue
        if winning_multisets(suits, size, given):
            return size
    return None


def wins(counts: Counter, tile_count: int) -> bool:
    """True when some reading of this multiset scores at least 8 fan."""
    if sum(counts.values()) != tile_count or any(count > 4 for count in counts.values()):
        return False
    kongs = tile_count - 14
    quads = [tile for tile, count in counts.items() if count >= 4]
    if len(quads) < kongs:
        return False
    for chosen in combinations(quads, kongs):
        body = counts.copy()
        for tile in chosen:
            body[tile] -= 4
            if body[tile] == 0:
                del body[tile]
        if _body_wins(chosen, body):
            return True
    return False


def _extras(pool: list[str], missing: int):
    if missing == 0:
        yield Counter()
        return
    for combo in combinations_with_replacement(pool, missing):
        yield Counter(combo)


def _body_wins(kong_tiles, body: Counter) -> bool:
    """Concealed readings first. A hand under 8 that way can still win with every meld open."""
    shaped = False
    for win in sorted(body, key=lambda tile: -body[tile]):
        concealed = _without(body, win)
        if len(concealed) + 3 * len(kong_tiles) != 13:
            continue
        pack = tuple(("GANG", tile, 0) for tile in kong_tiles)
        probe = score(pack, concealed, win, self_drawn=True, seat_wind=0, round_wind=0)
        if probe is None:
            continue
        shaped = True
        if probe[1] >= MIN_FAN or _other_contexts(kong_tiles, concealed, win):
            return True
    if shaped and _melded_wins(kong_tiles, body):
        return True
    return False


def _melded_wins(kong_tiles, body: Counter) -> bool:
    """All four sets claimed, winning by discard on the pair. That is the Melded Hand reading."""
    for pair, melds in _decompositions(body, 4 - len(kong_tiles)):
        pack = [("GANG", tile, 1) for tile in kong_tiles]
        pack.extend((kind, tile, 2 if kind == "CHI" else 1) for kind, tile in melds)
        for seat in range(4):
            for round_wind in range(4):
                scored = score(
                    tuple(pack),
                    [pair],
                    pair,
                    self_drawn=False,
                    seat_wind=seat,
                    round_wind=round_wind,
                )
                if scored is not None and scored[1] >= MIN_FAN:
                    return True
    return False


def _decompositions(counts: Counter, melds_needed: int):
    """Each way to write the tiles as melds plus a pair. The lowest tile is used first."""
    found = []
    for pair, count in list(counts.items()):
        if count < 2:
            continue
        rest = counts.copy()
        rest[pair] -= 2
        _take_melds(rest, melds_needed, [], found, pair)
    return found


def _take_melds(rest: Counter, need: int, acc: list, found: list, pair: str) -> None:
    if need == 0:
        if not any(rest.values()):
            found.append((pair, list(acc)))
        return
    tile = min((name for name, count in rest.items() if count), default=None)
    if tile is None:
        return
    if rest[tile] >= 3:
        rest[tile] -= 3
        acc.append(("PENG", tile))
        _take_melds(rest, need - 1, acc, found, pair)
        acc.pop()
        rest[tile] += 3
    if suit_of(tile) in NUMBERED_SUITS and rank_of(tile) <= 7:
        seq = [f"{tile[0]}{rank_of(tile) + offset}" for offset in range(3)]
        if all(rest[name] >= 1 for name in seq):
            for name in seq:
                rest[name] -= 1
            acc.append(("CHI", seq[1]))
            _take_melds(rest, need - 1, acc, found, pair)
            acc.pop()
            for name in seq:
                rest[name] += 1


def _other_contexts(kong_tiles, concealed, win) -> bool:
    """Seat, round, draw, and open kongs. The all-concealed self-drawn probe already ran."""
    for mask in range(1 << len(kong_tiles)):
        pack = tuple(("GANG", tile, (mask >> index) & 1) for index, tile in enumerate(kong_tiles))
        for self_drawn in (True, False):
            for seat in range(4):
                for round_wind in range(4):
                    if mask == 0 and self_drawn and seat == 0 and round_wind == 0:
                        continue
                    scored = score(
                        pack,
                        concealed,
                        win,
                        self_drawn=self_drawn,
                        seat_wind=seat,
                        round_wind=round_wind,
                    )
                    if scored is not None and scored[1] >= MIN_FAN:
                        return True
    return False


def _without(body: Counter, win: str) -> list[str]:
    tiles = []
    for tile, count in body.items():
        tiles.extend([tile] * (count - (tile == win)))
    return tiles
