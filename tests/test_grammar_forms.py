"""Special hands have to actually be that hand, including while the tiles are unfinished."""

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.hands import hand_suits, physical_tiles
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.sampler import (
    RARE_BUILDERS,
    Request,
    sample_hand,
)
from mahjong_gpt.tiles import NUMBERED_SUITS, SUIT_NAMES
from mahjong_gpt.tokenizer import Tokenizer

_DATA = Path(__file__).resolve().parents[1] / "data" / "hands" / "train.npy"


def _ids(tokenizer, names):
    return [tokenizer.stoi[name] for name in names]


def _roundtrip(hand):
    tokenizer = Tokenizer()
    suits = hand_suits(hand)
    tile_count = len(physical_tiles(hand))
    scored = score(
        hand.pack,
        hand.concealed,
        hand.win_tile,
        self_drawn=hand.self_drawn,
        seat_wind=hand.seat_wind,
        round_wind=hand.round_wind,
    )
    assert scored is not None and scored[1] >= 8
    ids = tokenizer.encode(sorted(suits, key=SUIT_NAMES.index), tile_count, hand, scored[0])
    prompt = len(tokenizer.prompt(suits, tile_count))
    for index in range(prompt, len(ids)):
        allowed = allowed_next(tokenizer, ids[:index])
        assert ids[index] in allowed, hand.form
    assert parse_ids(tokenizer, ids) is not None


def test_sampled_special_hands_still_follow_the_grammar():
    requests = [
        Request(frozenset(SUIT_NAMES), 14),
        Request(frozenset({"characters"}), 14),
        Request(frozenset({"bamboo", "dragons"}), 14),
        Request(frozenset({"winds", "dragons"}), 14),
        Request(frozenset(NUMBERED_SUITS), 14),
        Request(frozenset(NUMBERED_SUITS | {"dragons"}), 14),
        Request(frozenset({"characters", "bamboo"}), 18),
    ]
    seen = set()
    for builder in RARE_BUILDERS:
        for seed, request in enumerate(requests):
            hand = builder(request, random.Random(seed))
            if hand is None:
                continue
            _roundtrip(hand)
            seen.add(builder.__name__)
            break
    assert seen == {builder.__name__ for builder in RARE_BUILDERS}


def test_ordinary_hands_still_follow_the_grammar():
    found = 0
    for seed in range(40):
        hand = sample_hand(Request(frozenset({"characters", "bamboo", "dots"}), 14), random.Random(seed))
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
        _roundtrip(hand)
        found += 1
        if found == 6:
            break
    assert found >= 4


def test_seven_pairs_must_finish_the_pair_it_started():
    tokenizer = Tokenizer()
    prefix = tokenizer.prompt(["characters"], 14)
    prefix += _ids(tokenizer, ["FORM_SEVEN_PAIRS", "W1", "W1", "W1"])
    assert {tokenizer.token(index) for index in allowed_next(tokenizer, prefix)} == {"W1"}


def test_three_of_four_tiles_cannot_be_seven_pairs():
    tokenizer = Tokenizer()
    required = Counter({"W1": 3, "W2": 3, "W3": 3, "W4": 3})
    allowed = {tokenizer.token(index) for index in allowed_next(tokenizer, tokenizer.prompt(["characters"], 14), required)}
    assert "FORM_SEVEN_PAIRS" not in allowed
    assert "FORM_NINE_GATES" not in allowed
    assert "PUNG_CLOSED" in allowed


def test_thirteen_orphans_starts_with_characters_1():
    tokenizer = Tokenizer()
    prefix = tokenizer.prompt(list(SUIT_NAMES), 14) + _ids(tokenizer, ["FORM_THIRTEEN_ORPHANS"])
    assert {tokenizer.token(index) for index in allowed_next(tokenizer, prefix)} == {"W1"}


def test_a_sample_of_training_hands_still_parses():
    if not _DATA.exists():
        pytest.skip("training data is not built")
    rows = np.load(_DATA, mmap_mode="r")
    tokenizer = Tokenizer()
    rng = np.random.default_rng(0)
    for index in rng.choice(len(rows), size=200, replace=False):
        ids = [int(token) for token in rows[index] if int(token) != tokenizer.pad_id]
        prompt = ids.index(tokenizer.sep_id) + 1
        for cursor in range(prompt, len(ids)):
            assert ids[cursor] in allowed_next(tokenizer, ids[:cursor])
        assert parse_ids(tokenizer, ids) is not None
