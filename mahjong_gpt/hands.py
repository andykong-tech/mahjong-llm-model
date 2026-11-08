"""A finished hand, and the conversion into scorer inputs."""

from collections import Counter
from dataclasses import dataclass

from mahjong_gpt.tiles import chow_of, suit_of

PACK_CODE = {"chow": "CHI", "pung": "PENG", "kong": "GANG"}


@dataclass(frozen=True)
class Meld:
    kind: str
    tile: str
    concealed: bool

    def tiles(self) -> tuple[str, ...]:
        if self.kind == "chow":
            return chow_of(self.tile)
        if self.kind == "pung":
            return (self.tile,) * 3
        if self.kind == "kong":
            return (self.tile,) * 4
        raise ValueError(f"unknown meld {self.kind}")


@dataclass(frozen=True)
class Hand:
    concealed: tuple[str, ...]
    win_tile: str
    pack: tuple[tuple[str, str, int], ...]
    self_drawn: bool
    seat_wind: int
    round_wind: int
    form: str
    melds: tuple[Meld, ...] = ()
    pair: str | None = None


def closed_hand(form: str, tiles: list[str], win_tile: str, rng) -> Hand:
    """A hand with no declared packs. `tiles` includes the winning tile."""
    remaining = list(tiles)
    remaining.remove(win_tile)
    if len(remaining) != 13:
        raise ValueError(f"{form} has {len(tiles)} tiles, expected 14")
    return Hand(
        concealed=tuple(remaining),
        win_tile=win_tile,
        pack=(),
        self_drawn=bool(rng.randrange(2)),
        seat_wind=rng.randrange(4),
        round_wind=rng.randrange(4),
        form=form,
    )


def standard_hand(melds: list[Meld], pair: str, win_tile: str, rng, form: str = "standard") -> Hand:
    """Build scorer inputs from 4 melds and a pair.

    The winning tile must belong to the pair or to a concealed chow or pung.
    Kongs, and melds that were declared open, become packs. A concealed kong
    uses offer 0.
    """
    pack = []
    standing: list[str] = []
    for meld in melds:
        declared = meld.kind == "kong" or not meld.concealed
        if declared:
            offer = 0 if meld.concealed else (2 if meld.kind == "chow" else 1)
            pack.append((PACK_CODE[meld.kind], meld.tile, offer))
        else:
            standing.extend(meld.tiles())
    standing.extend((pair, pair))
    standing.remove(win_tile)
    if len(standing) + 3 * len(pack) != 13:
        raise ValueError("standard hand does not satisfy the 13-tile identity")
    return Hand(
        concealed=tuple(standing),
        win_tile=win_tile,
        pack=tuple(pack),
        self_drawn=bool(rng.randrange(2)),
        seat_wind=rng.randrange(4),
        round_wind=rng.randrange(4),
        form=form,
        melds=tuple(melds),
        pair=pair,
    )


def physical_tiles(hand: Hand) -> tuple[str, ...]:
    tiles = list(hand.concealed)
    tiles.append(hand.win_tile)
    for kind, tile, _offer in hand.pack:
        if kind == "CHI":
            tiles.extend(chow_of(tile))
        elif kind == "PENG":
            tiles.extend((tile,) * 3)
        elif kind == "GANG":
            tiles.extend((tile,) * 4)
        else:
            raise ValueError(f"unknown pack {kind}")
    return tuple(tiles)


def hand_suits(hand: Hand) -> frozenset[str]:
    return frozenset(suit_of(tile) for tile in physical_tiles(hand))


def within_copies(hand: Hand) -> bool:
    return all(count <= 4 for count in Counter(physical_tiles(hand)).values())
