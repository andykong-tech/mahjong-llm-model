import random

from mahjong_gpt.hands import Meld, standard_hand
from mahjong_gpt.rules import is_legal_win, score


def _names(scored):
    return [fan.name for fan in scored[0]]


def test_thirteen_orphans_scores_88():
    orphans = ("W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3")
    scored = score((), orphans, "W1", self_drawn=False, seat_wind=0, round_wind=0)
    assert scored is not None
    assert scored[1] == 88
    assert _names(scored) == ["Thirteen Orphans"]
    assert is_legal_win(scored)


def test_seven_pairs_and_big_three_dragons():
    pairs = ("W2", "W5", "W5", "W8", "W8", "T3", "T3", "T6", "T6", "B4", "B4", "B7", "B7")
    seven = score((), pairs, "W2", self_drawn=False, seat_wind=1, round_wind=2)
    assert seven is not None
    assert [(fan.name, fan.points) for fan in seven[0]] == [("Seven Pairs", 24), ("All Simples", 2)]
    assert is_legal_win(seven)

    dragons = (
        "J1", "J1", "J2", "J2", "J2", "J3", "J3", "J3", "T1", "T2", "T3", "W5", "W5",
    )
    big = score((), dragons, "J1", self_drawn=False, seat_wind=0, round_wind=0)
    assert big is not None
    assert "Big Three Dragons" in _names(big)
    assert big[1] == 93
    assert is_legal_win(big)


def test_chicken_hand_is_the_eight_fan_minimum():
    # Winning on T3 instead makes Mixed Shifted Chows, so the wait is T6.
    pack = (("PENG", "B6", 1), ("CHI", "W2", 2))
    concealed = ("B2", "B3", "B4", "T4", "T5", "F2", "F2")
    scored = score(pack, concealed, "T6", self_drawn=False, seat_wind=0, round_wind=0)
    assert scored is not None
    assert [(fan.name, fan.points) for fan in scored[0]] == [("Chicken Hand", 8)]
    assert is_legal_win(scored)


def test_winning_hand_under_eight_fan_is_not_legal():
    pack = (("CHI", "W3", 2), ("CHI", "W6", 2))
    concealed = ("B2", "B3", "B4", "T4", "T5", "B5", "B5")
    scored = score(pack, concealed, "T6", self_drawn=False, seat_wind=0, round_wind=0)
    assert scored is not None
    assert scored[1] == 6
    assert not is_legal_win(scored)


def test_concealed_kong_is_declared_with_offer_zero():
    rng = random.Random(0)
    hand = standard_hand(
        [
            Meld("kong", "W5", concealed=True),
            Meld("pung", "B6", concealed=False),
            Meld("chow", "T3", concealed=True),
            Meld("pung", "J1", concealed=True),
        ],
        "W9",
        "T3",
        rng,
    )
    assert ("GANG", "W5", 0) in hand.pack
    assert ("CHI", "T3", 2) not in hand.pack
    scored = score(
        hand.pack, hand.concealed, hand.win_tile,
        self_drawn=False, seat_wind=2, round_wind=2,
    )
    assert scored is not None
    assert "Concealed Kong" in _names(scored)


def test_non_win_and_a_fifth_copy_are_rejected():
    missing = score(
        (),
        ("W1", "W1", "W1", "W2", "W3", "W4", "W5", "W6", "T1", "T1", "T1", "B1", "B2"),
        "B9",
        self_drawn=False,
        seat_wind=0,
        round_wind=0,
    )
    assert missing is None
    # The pung plus two concealed copies plus the winning tile is six W1s.
    # The scorer has accepted that hand, so the wrapper counts copies itself.
    fifth = score(
        (("PENG", "W1", 1),),
        ("W1", "W1", "W2", "W3", "W4", "T5", "T5", "T5", "B2", "B2"),
        "W1",
        self_drawn=False,
        seat_wind=0,
        round_wind=0,
    )
    assert fifth is None


def test_wrong_tile_count_is_rejected():
    assert score((), ("W1", "W2", "W3"), "W4", self_drawn=True, seat_wind=0, round_wind=0) is None
