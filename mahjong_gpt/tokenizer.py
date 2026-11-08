"""Turn a suit request and a finished hand into one token sequence.

The prompt is the suits and the tile count. Everything after ``<sep>`` is the
hand, how it was won, and the fan names. Training predicts only that suffix.
"""

from collections.abc import Sequence

from mahjong_gpt.hands import Hand, physical_tiles
from mahjong_gpt.rules import Fan
from mahjong_gpt.tiles import ALL_TILES, SUIT_NAMES, suit_of

MAX_SEQUENCE = 96

FAN_NAMES = (
    "Big Four Winds",
    "Big Three Dragons",
    "All Green",
    "Nine Gates",
    "Four Kongs",
    "Seven Shifted Pairs",
    "Thirteen Orphans",
    "All Terminals",
    "Little Four Winds",
    "Little Three Dragons",
    "All Honors",
    "Four Concealed Pungs",
    "Pure Terminal Chows",
    "Quadruple Chow",
    "Four Pure Shifted Pungs",
    "Four Pure Shifted Chows",
    "Three Kongs",
    "All Terminals and Honors",
    "Seven Pairs",
    "Greater Honors and Knitted Tiles",
    "All Even Pungs",
    "Full Flush",
    "Pure Triple Chow",
    "Pure Shifted Pungs",
    "Upper Tiles",
    "Middle Tiles",
    "Lower Tiles",
    "Pure Straight",
    "Three-Suited Terminal Chows",
    "Pure Shifted Chows",
    "All Five",
    "Triple Pung",
    "Three Concealed Pungs",
    "Lesser Honors and Knitted Tiles",
    "Knitted Straight",
    "Upper Four",
    "Lower Four",
    "Big Three Winds",
    "Mixed Straight",
    "Reversible Tiles",
    "Mixed Triple Chow",
    "Mixed Shifted Pungs",
    "Chicken Hand",
    "Last Tile Draw",
    "Last Tile Claim",
    "Out with Replacement Tile",
    "Robbing The Kong",
    "All Pungs",
    "Half Flush",
    "Mixed Shifted Chows",
    "All Types",
    "Melded Hand",
    "Two Concealed Kongs",
    "Two Dragons Pungs",
    "Outside Hand",
    "Fully Concealed Hand",
    "Two Melded Kongs",
    "Last Tile",
    "Dragon Pung",
    "Prevalent Wind",
    "Seat Wind",
    "Concealed Hand",
    "All Chows",
    "Tile Hog",
    "Double Pung",
    "Two Concealed Pungs",
    "Concealed Kong",
    "All Simples",
    "Pure Double Chow",
    "Mixed Double Chow",
    "Short Straight",
    "Two Terminal Chows",
    "Pung of Terminals or Honors",
    "Melded Kong",
    "One Voided Suit",
    "No Honors",
    "Edge Wait",
    "Closed Wait",
    "Single Wait",
    "Self-Drawn",
    "Flower Tiles",
    "Concealed Kong and Melded Kong",
)

CLOSED_FORMS = (
    "thirteen_orphans",
    "seven_pairs",
    "seven_shifted_pairs",
    "nine_gates",
    "greater_honors_and_knitted",
    "lesser_honors_and_knitted",
    "knitted_straight",
    "all_honors",
)

WIND_TOKEN = ("E", "S", "W", "N")
MELD_TOKENS = (
    "CHOW_OPEN",
    "CHOW_CLOSED",
    "PUNG_OPEN",
    "PUNG_CLOSED",
    "KONG_OPEN",
    "KONG_CLOSED",
)


def _vocab() -> tuple[str, ...]:
    specials = ("<pad>", "<bos>", "<sep>", "<eos>", "<fans>", "<win>", "<pair>")
    suits = tuple(f"SUIT_{name.upper()}" for name in SUIT_NAMES)
    counts = tuple(f"TILES_{count}" for count in range(14, 19))
    forms = tuple(f"FORM_{name.upper()}" for name in CLOSED_FORMS)
    winds = tuple(f"SEAT_{name}" for name in WIND_TOKEN) + tuple(f"ROUND_{name}" for name in WIND_TOKEN)
    return specials + suits + counts + MELD_TOKENS + forms + ("SELF", "DISCARD") + winds + ALL_TILES + FAN_NAMES


class Tokenizer:
    def __init__(self):
        self.tokens = _vocab()
        self.stoi = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.sep_id = self.stoi["<sep>"]
        self.eos_id = self.stoi["<eos>"]
        self.fan_ids = {self.stoi[name] for name in FAN_NAMES}

    def __len__(self) -> int:
        return len(self.tokens)

    def token(self, index: int) -> str:
        return self.tokens[index]

    def prompt(self, suits: Sequence[str], tile_count: int) -> list[int]:
        chosen = set(suits)
        ids = [self.bos_id]
        for name in SUIT_NAMES:
            if name in chosen:
                ids.append(self.stoi[f"SUIT_{name.upper()}"])
        ids.append(self.stoi[f"TILES_{tile_count}"])
        ids.append(self.sep_id)
        return ids

    def encode(self, suits: Sequence[str], tile_count: int, hand: Hand, fans: Sequence[Fan]) -> list[int]:
        ids = self.prompt(suits, tile_count)
        if hand.melds:
            if hand.pair is None:
                raise ValueError("a melded hand needs a pair")
            for meld in hand.melds:
                side = "CLOSED" if meld.concealed else "OPEN"
                ids.append(self.stoi[f"{meld.kind.upper()}_{side}"])
                ids.extend(self.stoi[tile] for tile in meld.tiles())
            ids.append(self.stoi["<pair>"])
            ids.extend((self.stoi[hand.pair], self.stoi[hand.pair]))
        else:
            if hand.form not in CLOSED_FORMS:
                raise ValueError(f"no token for closed form {hand.form}")
            ids.append(self.stoi[f"FORM_{hand.form.upper()}"])
            ordered = sorted(physical_tiles(hand), key=lambda tile: (SUIT_NAMES.index(suit_of(tile)), tile))
            ids.extend(self.stoi[tile] for tile in ordered)
        ids.append(self.stoi["<win>"])
        ids.append(self.stoi[hand.win_tile])
        ids.append(self.stoi["SELF" if hand.self_drawn else "DISCARD"])
        ids.append(self.stoi[f"SEAT_{WIND_TOKEN[hand.seat_wind]}"])
        ids.append(self.stoi[f"ROUND_{WIND_TOKEN[hand.round_wind]}"])
        ids.append(self.stoi["<fans>"])
        for fan in fans:
            if fan.name not in self.stoi:
                raise ValueError(f"unknown fan {fan.name}")
            ids.extend([self.stoi[fan.name]] * fan.count)
        ids.append(self.eos_id)
        if len(ids) > MAX_SEQUENCE:
            raise ValueError(f"sequence length {len(ids)} exceeds {MAX_SEQUENCE}")
        return ids

    def pad(self, ids: Sequence[int]) -> list[int]:
        if len(ids) > MAX_SEQUENCE:
            raise ValueError(f"sequence length {len(ids)} exceeds {MAX_SEQUENCE}")
        return list(ids) + [self.pad_id] * (MAX_SEQUENCE - len(ids))
