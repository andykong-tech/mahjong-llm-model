"""The exact list of finishes, for hands with at most three tiles missing."""

import random
from collections import Counter

from mahjong_gpt.completions import multiset_key, winning_multisets, wins
from mahjong_gpt.hands import hand_suits, physical_tiles
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.sampler import Request, sample_hand
from mahjong_gpt.tiles import SUIT_NAMES


def _nine_gates_wait():
    return Counter({"W1": 3, "W2": 1, "W3": 1, "W4": 1, "W5": 1, "W6": 1, "W7": 1, "W8": 1, "W9": 3})


def test_characters_1112345678999_has_nine_finishes():
    given = _nine_gates_wait()
    assert sum(given.values()) == 13
    found = winning_multisets(frozenset({"characters"}), 14, given)
    extras = [list((counts - given).elements())[0] for counts in found]
    assert sorted(extras) == [f"W{rank}" for rank in range(1, 10)]


def test_an_unwinnable_thirteen_has_no_finish():
    given = Counter({"W5": 3, "W3": 2, "W2": 4, "W9": 2, "W6": 1, "W8": 1})
    assert sum(given.values()) == 13
    assert winning_multisets(frozenset({"characters"}), 14, given) == []


def test_dragons_alone_cannot_fill_fourteen_tiles():
    given = Counter({"J1": 4, "J2": 4, "J3": 4})
    assert winning_multisets(frozenset({"dragons"}), 14, given) == []


def test_four_missing_tiles_are_not_listed():
    assert winning_multisets(frozenset({"characters"}), 14, Counter({"W1": 1})) is None


def test_a_low_concealed_hand_still_counts_when_it_can_be_melded():
    # All concealed this scores 7. With every set open it scores 8, so it is a win.
    counts = Counter({"W5": 3, "B4": 2, "W7": 1, "W8": 1, "W9": 1, "T7": 1, "T8": 1, "T9": 1, "T3": 1, "T4": 1, "T5": 1})
    assert wins(counts, 14)


def test_sampled_wins_stay_in_the_list_when_tiles_are_removed():
    requests = [
        Request(frozenset({"characters", "bamboo", "dots"}), 14),
        Request(frozenset({"characters"}), 14),
        Request(frozenset({"bamboo", "dragons"}), 15),
        Request(frozenset(set(SUIT_NAMES)), 14),
    ]
    checked = 0
    for request in requests:
        for seed in range(6):
            hand = sample_hand(request, random.Random(seed + 20))
            if hand is None:
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
            full = Counter(physical_tiles(hand))
            suits = frozenset(hand_suits(hand))
            tile_count = len(physical_tiles(hand))
            assert wins(full, tile_count)
            tiles = physical_tiles(hand)
            for missing in (1, 2):
                given = full.copy()
                for tile in tiles[:missing]:
                    given[tile] -= 1
                    if given[tile] == 0:
                        del given[tile]
                found = winning_multisets(suits, tile_count, given)
                assert found is not None
                assert multiset_key(full) in {multiset_key(counts) for counts in found}
            checked += 1
    assert checked >= 8
