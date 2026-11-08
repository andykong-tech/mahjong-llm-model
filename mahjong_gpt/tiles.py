"""Tile codes used by the Chinese Official Mahjong scorer.

W1–W9 are characters, T1–T9 bamboo, B1–B9 dots, F1–F4 winds
(east, south, west, north), and J1–J3 dragons (red, green, white).
"""

from collections.abc import Iterable

SUIT_NAMES = ("characters", "bamboo", "dots", "winds", "dragons")
NUMBERED_SUITS = frozenset({"characters", "bamboo", "dots"})
COPIES = 4

CHARACTERS = tuple(f"W{rank}" for rank in range(1, 10))
BAMBOO = tuple(f"T{rank}" for rank in range(1, 10))
DOTS = tuple(f"B{rank}" for rank in range(1, 10))
WINDS = ("F1", "F2", "F3", "F4")
DRAGONS = ("J1", "J2", "J3")

SUITS = {
    "characters": CHARACTERS,
    "bamboo": BAMBOO,
    "dots": DOTS,
    "winds": WINDS,
    "dragons": DRAGONS,
}
PREFIX = {suit: tiles[0][0] for suit, tiles in SUITS.items()}
TILE_SUIT = {tile: suit for suit, tiles in SUITS.items() for tile in tiles}
ALL_TILES = tuple(TILE_SUIT)


def parse_suits(suits: Iterable[str]) -> frozenset[str]:
    chosen = frozenset(suits)
    unknown = chosen - frozenset(SUIT_NAMES)
    if not chosen or unknown:
        names = ", ".join(SUIT_NAMES)
        raise ValueError(f"suits must be a non-empty subset of {names}")
    return chosen


def suit_of(tile: str) -> str:
    try:
        return TILE_SUIT[tile]
    except KeyError:
        raise ValueError(f"unknown tile {tile}") from None


def rank_of(tile: str) -> int:
    suit_of(tile)
    return int(tile[1])


def chow_of(middle: str) -> tuple[str, str, str]:
    """Return the three tiles of a chow whose middle tile is `middle`."""
    suit = suit_of(middle)
    rank = rank_of(middle)
    if suit not in NUMBERED_SUITS or not 2 <= rank <= 8:
        raise ValueError(f"{middle} is not a chow middle")
    prefix = PREFIX[suit]
    return (f"{prefix}{rank - 1}", middle, f"{prefix}{rank + 1}")


def suit_capacity(suits: Iterable[str]) -> int:
    return COPIES * sum(len(SUITS[suit]) for suit in suits)
