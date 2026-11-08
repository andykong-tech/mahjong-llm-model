import random

import pytest

from mahjong_gpt.hands import hand_suits, physical_tiles, within_copies
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.sampler import (
    Request,
    all_green,
    all_honors,
    all_terminals,
    big_four_winds,
    big_three_dragons,
    four_kongs,
    generate,
    greater_honors_and_knitted,
    knitted_straight,
    lesser_honors_and_knitted,
    nine_gates,
    pure_straight,
    quadruple_chow,
    sample_standard,
    seven_pairs,
    seven_shifted_pairs,
    thirteen_orphans,
)
from mahjong_gpt.tiles import NUMBERED_SUITS, SUITS

ALL_SUITS = frozenset(SUITS)


@pytest.mark.parametrize(
    ("builder", "suits", "tile_count", "fan"),
    [
        (thirteen_orphans, ALL_SUITS, 14, "Thirteen Orphans"),
        (seven_pairs, frozenset({"characters", "bamboo"}), 14, "Seven Pairs"),
        (seven_shifted_pairs, frozenset({"characters"}), 14, "Seven Shifted Pairs"),
        (nine_gates, frozenset({"dots"}), 14, "Nine Gates"),
        (greater_honors_and_knitted, ALL_SUITS, 14, "Greater Honors and Knitted Tiles"),
        (lesser_honors_and_knitted, ALL_SUITS, 14, "Lesser Honors and Knitted Tiles"),
        (knitted_straight, NUMBERED_SUITS | {"dragons"}, 14, "Knitted Straight"),
        (all_green, frozenset({"bamboo", "dragons"}), 14, "All Green"),
        (big_three_dragons, frozenset({"dragons", "bamboo", "characters"}), 14, "Big Three Dragons"),
        (big_four_winds, frozenset({"winds", "dragons"}), 14, "Big Four Winds"),
        (four_kongs, frozenset({"characters", "bamboo"}), 18, "Four Kongs"),
        (all_honors, frozenset({"winds", "dragons"}), 14, "All Honors"),
        (quadruple_chow, frozenset({"characters"}), 14, "Quadruple Chow"),
        (pure_straight, NUMBERED_SUITS, 14, "Pure Straight"),
        (all_terminals, NUMBERED_SUITS, 15, "All Terminals"),
    ],
)
def test_rare_template_is_a_legal_win(builder, suits, tile_count, fan):
    hand = builder(Request(suits, tile_count), random.Random(0))
    assert hand is not None
    tiles = physical_tiles(hand)
    assert len(tiles) == tile_count
    assert hand_suits(hand) == suits
    assert within_copies(hand)
    assert len(hand.concealed) + 3 * len(hand.pack) == 13
    scored = score(
        hand.pack,
        hand.concealed,
        hand.win_tile,
        self_drawn=hand.self_drawn,
        seat_wind=hand.seat_wind,
        round_wind=hand.round_wind,
    )
    assert is_legal_win(scored)
    assert fan in {item.name for item in scored[0]}


def test_standard_sampler_obeys_the_request_and_wins():
    request = Request(frozenset({"characters", "dots"}), 14)
    rng = random.Random(0)
    wins = 0
    for _ in range(30):
        hand = sample_standard(request, rng)
        assert hand is not None
        assert hand_suits(hand) == request.suits
        assert len(physical_tiles(hand)) == 14
        assert within_copies(hand)
        scored = score(
            hand.pack,
            hand.concealed,
            hand.win_tile,
            self_drawn=hand.self_drawn,
            seat_wind=hand.seat_wind,
            round_wind=hand.round_wind,
        )
        assert scored is not None
        wins += 1
    assert wins == 30


def test_generate_returns_legal_hands_for_the_exact_request():
    hands = generate(["characters"], 16, 5, seed=1)
    assert len(hands) == 5
    for scored in hands:
        assert scored.total >= 8
        assert hand_suits(scored.hand) == frozenset({"characters"})
        assert len(physical_tiles(scored.hand)) == 16
        assert within_copies(scored.hand)


def test_same_seed_reproduces_the_hands():
    first = generate(["bamboo", "dragons"], 14, 3, seed=7)
    second = generate(["bamboo", "dragons"], 14, 3, seed=7)
    assert [item.hand for item in first] == [item.hand for item in second]
    assert first


def test_impossible_requests_produce_nothing():
    assert generate(["dragons"], 14, 3, seed=0, max_attempts=20) == []
    assert generate(["winds"], 16, 1, seed=0, max_attempts=40) == []


def test_winds_only_can_win_with_seven_pairs():
    hands = generate(["winds"], 14, 2, seed=2)
    assert len(hands) == 2
    for scored in hands:
        assert hand_suits(scored.hand) == frozenset({"winds"})
        assert "Seven Pairs" in {fan.name for fan in scored.fans} or "All Honors" in {
            fan.name for fan in scored.fans
        }


def test_unknown_suit_and_tile_count_are_rejected():
    with pytest.raises(ValueError):
        generate(["flowers"], 14, 1)
    with pytest.raises(ValueError):
        generate(["dots"], 13, 1)
