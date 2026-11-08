from collections import Counter

import torch

from mahjong_gpt.evaluate import fan_counter, name_fans
from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.hands import hand_suits, physical_tiles
from mahjong_gpt.infer import complete
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.rules import is_legal_win, score
from mahjong_gpt.tiles import SUIT_NAMES
from mahjong_gpt.tokenizer import FAN_NAMES, MAX_SEQUENCE, Tokenizer
from mahjong_gpt.trick_hands import TRICK_HANDS


def _scored(hand):
    return score(
        hand.pack,
        hand.concealed,
        hand.win_tile,
        self_drawn=hand.self_drawn,
        seat_wind=hand.seat_wind,
        round_wind=hand.round_wind,
    )


def _untrained_model(tokenizer):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=len(tokenizer), block_size=MAX_SEQUENCE, n_layer=1, n_head=2, n_embd=16)
    return GPT(config).eval()


def test_trick_hands_are_legal_and_follow_the_grammar():
    tokenizer = Tokenizer()
    names = [trick.name for trick in TRICK_HANDS]
    assert len(names) == len(set(names))
    for trick in TRICK_HANDS:
        scored = _scored(trick.hand)
        assert is_legal_win(scored), trick.name
        suits = sorted(hand_suits(trick.hand), key=SUIT_NAMES.index)
        ids = tokenizer.encode(suits, len(physical_tiles(trick.hand)), trick.hand, scored[0])
        prompt = len(tokenizer.prompt(suits, len(physical_tiles(trick.hand))))
        for index in range(prompt, len(ids)):
            assert ids[index] in allowed_next(tokenizer, ids[:index]), trick.name
        parsed = parse_ids(tokenizer, ids)
        assert parsed is not None, trick.name
        assert Counter(parsed.fan_names) == fan_counter(scored[0]), trick.name


def test_name_fans_only_returns_fan_names():
    tokenizer = Tokenizer()
    model = _untrained_model(tokenizer)
    fans_id = tokenizer.stoi["<fans>"]
    prefixes = []
    for trick in TRICK_HANDS[:3]:
        scored = _scored(trick.hand)
        suits = sorted(hand_suits(trick.hand), key=SUIT_NAMES.index)
        ids = tokenizer.encode(suits, len(physical_tiles(trick.hand)), trick.hand, scored[0])
        prefixes.append(ids[: ids.index(fans_id) + 1])
    named = name_fans(model, tokenizer, prefixes, "cpu", batch_size=2)
    assert len(named) == 3
    for names in named:
        assert names
        assert set(names) <= set(FAN_NAMES)


def test_complete_never_breaks_the_grammar():
    tokenizer = Tokenizer()
    model = _untrained_model(tokenizer)
    for seed in range(3):
        torch.manual_seed(seed)
        prompt = tokenizer.prompt(["characters", "dragons"], 15)
        ids, status = complete(model, tokenizer, prompt, temperature=1.0, device="cpu")
        assert status in {"done", "dead end", "too long"}
        assert ids[: len(prompt)] == prompt
        for index in range(len(prompt), len(ids)):
            assert ids[index] in allowed_next(tokenizer, ids[:index])
        if status == "done":
            assert parse_ids(tokenizer, ids) is not None
