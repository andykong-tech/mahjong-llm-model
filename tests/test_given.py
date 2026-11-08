import json
from collections import Counter

import pytest
import torch

from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.infer import check_request, complete, parse_tiles, render_hand, request_suits, tile_name
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.tokenizer import MAX_SEQUENCE, Tokenizer


def _ids(tokenizer, names):
    return [tokenizer.stoi[name] for name in names]


def _four_character_melds(tokenizer):
    prompt = tokenizer.prompt(["characters"], 14)
    melds = [
        "CHOW_CLOSED", "W2", "W3", "W4",
        "CHOW_CLOSED", "W5", "W6", "W7",
        "PUNG_CLOSED", "W8", "W8", "W8",
        "PUNG_CLOSED", "W9", "W9", "W9",
        "<pair>",
    ]
    return prompt + _ids(tokenizer, melds)


def test_the_last_open_group_must_take_the_missing_tile():
    tokenizer = Tokenizer()
    prefix = _four_character_melds(tokenizer)
    assert allowed_next(tokenizer, prefix, Counter({"W1": 1})) == {tokenizer.stoi["W1"]}
    assert allowed_next(tokenizer, prefix, Counter({"W1": 2})) == {tokenizer.stoi["W1"]}
    assert allowed_next(tokenizer, prefix, Counter({"W1": 1, "W5": 2})) == set()


def test_a_chow_under_way_cannot_be_bent_to_fit_a_tile():
    tokenizer = Tokenizer()
    prompt = tokenizer.prompt(["characters"], 14)
    prefix = prompt + _ids(tokenizer, ["CHOW_CLOSED", "W2", "W3", "W4", "CHOW_CLOSED", "W5"])
    # Two pungs can still hold three W1 and three W9 once this chow is done.
    assert allowed_next(tokenizer, prefix, Counter({"W1": 3, "W9": 3})) == {tokenizer.stoi["W6"]}
    # Nine more tiles cannot fit in the two melds and the pair that remain.
    assert allowed_next(tokenizer, prefix, Counter({"W1": 3, "W8": 3, "W9": 3})) == set()


def test_the_last_groups_must_bring_the_missing_suits():
    tokenizer = Tokenizer()
    prompt = tokenizer.prompt(["characters", "dragons"], 14)
    melds = ["CHOW_CLOSED", "W1", "W2", "W3", "CHOW_CLOSED", "W4", "W5", "W6", "PUNG_CLOSED", "W9", "W9", "W9"]
    prefix = prompt + _ids(tokenizer, melds)
    allowed = {tokenizer.token(index) for index in allowed_next(tokenizer, prefix)}
    # A chow cannot hold a dragon, but a pung of dragons or a dragon pair can still come later.
    assert allowed == {"PUNG_OPEN", "PUNG_CLOSED", "CHOW_OPEN", "CHOW_CLOSED"}
    prefix += _ids(tokenizer, ["CHOW_CLOSED", "W6", "W7", "W8", "<pair>"])
    assert {tokenizer.token(index) for index in allowed_next(tokenizer, prefix)} == {"J1", "J2", "J3"}


def test_completion_keeps_every_given_tile():
    tokenizer = Tokenizer()
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=len(tokenizer), block_size=MAX_SEQUENCE, n_layer=1, n_head=2, n_embd=16)
    model = GPT(config).eval()
    required = Counter({"W2": 1, "T5": 1, "J1": 1})
    prompt = tokenizer.prompt(["characters", "bamboo", "dragons"], 14)
    finished = 0
    for seed in range(6):
        torch.manual_seed(seed)
        ids, status = complete(model, tokenizer, prompt, temperature=1.0, device="cpu", required=required)
        for index in range(len(prompt), len(ids)):
            assert ids[index] in allowed_next(tokenizer, ids[:index], required)
        if status == "done":
            finished += 1
            parsed = parse_ids(tokenizer, ids)
            assert parsed is not None
            assert not required - Counter(parsed.tiles())
    assert finished > 0


def test_command_line_inputs(tmp_path):
    assert parse_tiles("w2, T5,J1") == Counter({"W2": 1, "T5": 1, "J1": 1})
    named = parse_tiles("bamboo 5, dots 3, characters 9, dragons red, east wind, 2 bamboo")
    assert named == Counter({"T5": 1, "B3": 1, "W9": 1, "J1": 1, "F1": 1, "T2": 1})
    assert [tile_name(tile) for tile in ("T5", "F4", "J3")] == ["bamboo 5", "north wind", "white dragon"]
    with pytest.raises(ValueError):
        parse_tiles("W2,W2,W2,W2,W2")
    with pytest.raises(ValueError):
        parse_tiles("X9")
    with pytest.raises(ValueError):
        parse_tiles("bamboo 10")
    with pytest.raises(ValueError):
        parse_tiles("dragons purple")
    assert request_suits("", Counter({"W2": 1, "J1": 1})) == frozenset({"characters", "dragons"})
    with pytest.raises(ValueError):
        request_suits("bamboo", Counter({"W2": 1}))
    with pytest.raises(ValueError):
        request_suits("", Counter())
    feasibility = tmp_path / "feasibility.json"
    feasibility.write_text(json.dumps([{"suits": ["characters"], "tile_count": 14}]))
    check_request(frozenset({"characters"}), 14, Counter({"W1": 1}), feasibility)
    with pytest.raises(ValueError):
        check_request(frozenset({"dragons"}), 14, Counter(), feasibility)
    with pytest.raises(ValueError):
        check_request(frozenset({"characters"}), 13, Counter(), feasibility)


def test_render_marks_the_given_tiles():
    tokenizer = Tokenizer()
    ids = _four_character_melds(tokenizer) + _ids(tokenizer, ["W1", "W1", "<win>", "W4", "DISCARD", "SEAT_E", "ROUND_E"])
    text = render_hand(tokenizer, ids, Counter({"W9": 1, "W1": 2}))
    assert text == "W2 W3 W4 | W5 W6 W7 | W8 W8 W8 | W9* W9 W9 | pair W1* W1*"
