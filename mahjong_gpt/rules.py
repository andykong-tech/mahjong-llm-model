"""Score a finished hand with the Chinese Official Mahjong rules.

Two details of the underlying scorer matter. The concealed tiles exclude the
winning tile, and a concealed kong is still a declared pack: ("GANG", tile, 0).
A hand is a legal win only at 8 fan or more. Chicken Hand is the 8-fan score
the rules assign when a winning hand has no other pattern.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from MahjongGB import MahjongFanCalculator

from mahjong_gpt.tiles import chow_of, suit_of

MIN_FAN = 8
_PACK_TYPES = {"CHI": 3, "PENG": 3, "GANG": 4}


@dataclass(frozen=True)
class Fan:
    name: str
    points: int
    count: int
    name_zh: str


def score(
    pack: Sequence[tuple[str, str, int]],
    concealed: Sequence[str],
    win_tile: str,
    *,
    self_drawn: bool,
    seat_wind: int,
    round_wind: int,
) -> tuple[tuple[Fan, ...], int] | None:
    """Return (fans, total) for a winning hand, or None when it is not a win.

    `concealed` is the standing tiles with the winning tile removed, and
    `len(concealed) + 3 * len(pack)` is 13. Open melds and every kong, including
    a concealed kong, go in `pack`.
    """
    if not _well_formed(pack, concealed, win_tile, seat_wind, round_wind):
        return None
    try:
        raw = MahjongFanCalculator(
            pack=tuple(pack),
            hand=tuple(concealed),
            winTile=win_tile,
            flowerCount=0,
            isSelfDrawn=bool(self_drawn),
            is4thTile=False,
            isAboutKong=False,
            isWallLast=False,
            seatWind=seat_wind,
            prevalentWind=round_wind,
            verbose=True,
        )
    except TypeError:
        return None
    fans = tuple(
        Fan(name=name_en, points=points, count=count, name_zh=name_zh)
        for points, count, name_zh, name_en in raw
    )
    return fans, sum(fan.points * fan.count for fan in fans)


def is_legal_win(scored: tuple[tuple[Fan, ...], int] | None) -> bool:
    return scored is not None and scored[1] >= MIN_FAN


def fan_counter(fans) -> Counter:
    return Counter({fan.name: fan.count for fan in fans})


def _well_formed(pack, concealed, win_tile, seat_wind, round_wind) -> bool:
    if seat_wind not in range(4) or round_wind not in range(4):
        return False
    try:
        suit_of(win_tile)
        for tile in concealed:
            suit_of(tile)
    except ValueError:
        return False
    if len(concealed) + 3 * len(pack) != 13:
        return False
    counts = Counter(concealed)
    counts[win_tile] += 1
    for item in pack:
        if len(item) != 3:
            return False
        kind, tile, offer = item
        copies = _PACK_TYPES.get(kind)
        if copies is None or offer not in range(4):
            return False
        try:
            tiles = chow_of(tile) if kind == "CHI" else (tile,) * copies
        except ValueError:
            return False
        counts.update(tiles)
    return all(count <= 4 for count in counts.values())
