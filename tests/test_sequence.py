import random
from collections import Counter

import numpy as np
import torch

from mahjong_gpt.grammar import allowed_next, parse_ids
from mahjong_gpt.model import GPT, GPTConfig
from mahjong_gpt.rules import score
from mahjong_gpt.sampler import Request, four_kongs, thirteen_orphans
from mahjong_gpt.tiles import SUIT_NAMES
from mahjong_gpt.tokenizer import Tokenizer
from mahjong_gpt.train import supervised_targets

from mahjong_gpt.dataset import build_dataset


def _scored(hand):
    return score(
        hand.pack,
        hand.concealed,
        hand.win_tile,
        self_drawn=hand.self_drawn,
        seat_wind=hand.seat_wind,
        round_wind=hand.round_wind,
    )


def _check_roundtrip(hand, suits, tile_count):
    tokenizer = Tokenizer()
    scored = _scored(hand)
    assert scored is not None
    ids = tokenizer.encode(suits, tile_count, hand, scored[0])
    prompt = tokenizer.prompt(suits, tile_count)
    assert ids[: len(prompt)] == prompt
    for index in range(len(prompt), len(ids)):
        allowed = allowed_next(tokenizer, ids[:index])
        assert ids[index] in allowed
    assert allowed_next(tokenizer, ids) is None
    parsed = parse_ids(tokenizer, ids)
    assert parsed is not None
    again = score(
        parsed.pack,
        parsed.concealed,
        parsed.win_tile,
        self_drawn=parsed.self_drawn,
        seat_wind=parsed.seat_wind,
        round_wind=parsed.round_wind,
    )
    assert again is not None
    assert Counter(fan.name for fan in scored[0] for _ in range(fan.count)) == Counter(parsed.fan_names)
    assert again[1] == scored[1]


def test_standard_kong_hand_and_thirteen_orphans_roundtrip():
    rng = random.Random(1)
    kongs = four_kongs(Request(frozenset({"characters"}), 18), rng)
    orphans = thirteen_orphans(Request(frozenset(SUIT_NAMES), 14), rng)
    _check_roundtrip(kongs, {"characters"}, 18)
    _check_roundtrip(orphans, set(SUIT_NAMES), 14)


def test_special_forms_only_start_at_fourteen_tiles():
    tokenizer = Tokenizer()
    forms = {token for token in tokenizer.tokens if token.startswith("FORM_")}
    at_fourteen = {tokenizer.token(index) for index in allowed_next(tokenizer, tokenizer.prompt(["characters"], 14))}
    fitting = {"FORM_SEVEN_PAIRS", "FORM_SEVEN_SHIFTED_PAIRS", "FORM_NINE_GATES"}
    assert fitting <= at_fourteen
    assert not (forms - fitting) & at_fourteen
    for tile_count in range(15, 19):
        allowed = {tokenizer.token(index) for index in allowed_next(tokenizer, tokenizer.prompt(["characters"], tile_count))}
        assert not forms & allowed


def test_prompt_suits_follow_a_fixed_order():
    tokenizer = Tokenizer()
    ids = tokenizer.prompt(["dragons", "bamboo"], 14)
    names = [tokenizer.token(index) for index in ids]
    assert names == ["<bos>", "SUIT_BAMBOO", "SUIT_DRAGONS", "TILES_14", "<sep>"]


def test_loss_ignores_the_prompt_and_the_padding():
    tokenizer = Tokenizer()
    ids = tokenizer.prompt(["dots"], 14) + [tokenizer.eos_id]
    padded = torch.tensor([tokenizer.pad(ids)])
    targets = supervised_targets(padded, tokenizer.sep_id, tokenizer.pad_id)
    sep = ids.index(tokenizer.sep_id)
    assert torch.all(targets[0, : sep + 1] == -100)
    assert int(targets[0, sep + 1]) == tokenizer.eos_id
    assert torch.all(targets[0, len(ids) :] == -100)


def test_transformer_backward():
    config = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(1, 32, (3, 10))
    targets = idx.clone()
    targets[:, :3] = -100
    _logits, loss = model(idx, targets)
    loss.backward()
    assert torch.isfinite(loss)


def test_small_dataset_roundtrips(tmp_path):
    result = build_dataset(
        tmp_path,
        train_size=4,
        fan_floor=0,
        seed=0,
        probe_tries=10,
        requests=[Request(frozenset({"characters"}), 14)],
    )
    assert result["meta"]["counts"]["train"] == 4
    train = np.load(tmp_path / "train.npy")
    tokenizer = Tokenizer()
    ids = train[0].tolist()
    while ids[-1] == tokenizer.pad_id:
        ids.pop()
    assert parse_ids(tokenizer, ids) is not None
