"""Hand-built winning hands, each aimed at one rule the model could get wrong.

Every hand here is a legal win. The scorer decides the fans; `tests` says what
the hand is meant to probe.
"""

import random
from dataclasses import dataclass, replace

from mahjong_gpt.hands import Hand, Meld, closed_hand, standard_hand

KNITTED = ("W1", "W4", "W7", "T2", "T5", "T8", "B3", "B6", "B9")
ORPHANS = ("W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3")
# Chows 234, 345 and 456 in three suits score Mixed Shifted Chows (6 fan), which
# lifts the wind hands over the 8-fan minimum without adding other wind fans.
SHIFTED = (("chow", "W3", True), ("chow", "T4", True), ("chow", "B5", True))


@dataclass(frozen=True)
class TrickHand:
    name: str
    tests: str
    hand: Hand


def _melded(melds, pair, win, *, self_drawn=False, seat=0, round_wind=0) -> Hand:
    built = standard_hand([Meld(*meld) for meld in melds], pair, win, random.Random(0))
    return replace(built, self_drawn=self_drawn, seat_wind=seat, round_wind=round_wind)


def _closed(form, tiles, win, *, self_drawn=False, seat=0, round_wind=0) -> Hand:
    built = closed_hand(form, list(tiles), win, random.Random(0))
    return replace(built, self_drawn=self_drawn, seat_wind=seat, round_wind=round_wind)


TRICK_HANDS = (
    TrickHand(
        "seat wind pung",
        "A pung of South while seated South scores Seat Wind.",
        _melded(SHIFTED + (("pung", "F2", True),), "B9", "W4", seat=1),
    ),
    TrickHand(
        "round wind pung",
        "A pung of East in the East round scores Prevalent Wind.",
        _melded(SHIFTED + (("pung", "F1", True),), "B9", "W4", seat=1),
    ),
    TrickHand(
        "seat and round wind pung",
        "Seated West in the West round, one West pung scores both wind fans.",
        _melded(SHIFTED + (("pung", "F3", True),), "B9", "W4", seat=2, round_wind=2),
    ),
    TrickHand(
        "other wind pung",
        "A North pung that is neither seat nor round only counts as an honor pung.",
        _melded(SHIFTED + (("pung", "F4", True),), "B9", "W4", seat=1),
    ),
    TrickHand(
        "seat wind as the pair",
        "A pair of the seat wind scores nothing; only a pung would.",
        _melded(SHIFTED + (("pung", "J1", True),), "F2", "W4", seat=1),
    ),
    TrickHand(
        "big four winds",
        "Big Four Winds replaces the seat and round wind fans.",
        _melded(
            (("pung", "F1", False), ("pung", "F2", False), ("pung", "F3", True), ("pung", "F4", True)),
            "J1",
            "J1",
        ),
    ),
    TrickHand(
        "seven pairs with two sets of four",
        "Two groups of four identical tiles score Tile Hog twice.",
        _closed(
            "seven_pairs",
            ("W2",) * 4 + ("T5",) * 4 + ("B3", "B3", "B7", "B7", "W9", "W9"),
            "B7",
            round_wind=1,
        ),
    ),
    TrickHand(
        "two terminal pungs",
        "Pungs of a 1 and a 9 each score Pung of Terminals or Honors.",
        _melded(
            (("pung", "W1", False), ("pung", "T9", False), ("pung", "B5", True), ("pung", "W5", True)),
            "J2",
            "W5",
            round_wind=1,
        ),
    ),
    TrickHand(
        "four chows that are seven pairs",
        "Written as four chows, but the scorer takes the higher-scoring seven pairs reading.",
        _melded(
            (("chow", "W3", True), ("chow", "W3", True), ("chow", "T6", True), ("chow", "T6", True)),
            "B5",
            "T7",
            self_drawn=True,
        ),
    ),
    TrickHand(
        "three chows that are three pungs",
        "Written as 1-2-3 three times, but scored as pungs of 1, 2 and 3.",
        _melded(
            (("chow", "W2", True), ("chow", "W2", True), ("chow", "W2", True), ("pung", "T5", False)),
            "B8",
            "W3",
        ),
    ),
    TrickHand(
        "nine gates",
        "1112345678999 in one suit plus one more tile of that suit.",
        _closed(
            "nine_gates",
            ("W1",) * 3 + tuple(f"W{rank}" for rank in range(2, 9)) + ("W9",) * 3 + ("W5",),
            "W5",
        ),
    ),
    TrickHand(
        "thirteen orphans",
        "One of every 1, 9, wind and dragon, plus a duplicate.",
        _closed("thirteen_orphans", ORPHANS + ("J3",), "J3", self_drawn=True),
    ),
    TrickHand(
        "lesser honors and knitted tiles",
        "Fourteen unpaired tiles from knitted sequences and honors.",
        _closed("lesser_honors_and_knitted", KNITTED + ("F1", "F2", "F3", "F4", "J1"), "J1"),
    ),
    TrickHand(
        "seven shifted pairs",
        "Seven consecutive pairs in one suit.",
        _closed(
            "seven_shifted_pairs",
            tuple(f"B{rank}" for rank in range(3, 10) for _ in range(2)),
            "B6",
        ),
    ),
    TrickHand(
        "melded hand",
        "Every set claimed from other players, and the pair won on a discard.",
        _melded(
            (("pung", "W2", False), ("pung", "T7", False), ("pung", "B9", False), ("pung", "F3", False)),
            "J3",
            "J3",
        ),
    ),
    TrickHand(
        "fully concealed hand",
        "Nothing claimed, and the winning tile drawn by the player.",
        _melded(SHIFTED + (("pung", "J2", True),), "B9", "B9", self_drawn=True),
    ),
    TrickHand(
        "chicken hand",
        "A win with no scoring pattern at all is worth exactly 8.",
        _melded(
            (("pung", "B6", False), ("chow", "W2", False), ("chow", "B3", True), ("chow", "T5", True)),
            "F2",
            "T6",
        ),
    ),
    TrickHand(
        "four kongs",
        "Four kongs make an 18-tile hand.",
        _melded(
            (("kong", "W1", False), ("kong", "T1", False), ("kong", "B1", True), ("kong", "F1", False)),
            "J1",
            "J1",
            self_drawn=True,
        ),
    ),
    TrickHand(
        "one concealed and one melded kong",
        "A hidden kong and a claimed kong in the same hand.",
        _melded(
            (("kong", "W5", True), ("kong", "T8", False), ("chow", "B3", True), ("pung", "J1", True)),
            "F3",
            "B2",
            round_wind=1,
        ),
    ),
)
