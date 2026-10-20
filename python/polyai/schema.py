"""The one place that defines the tensor and shard layout the Python side expects.

This follows design.md sections 3 and 4. When `ai/docs/encoding.md` (owned by polyai-core) lands, change the
constants here; nothing else in the package hard-codes field names, vocabulary sizes or shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------------------------------------------
# Tile tensor: int16[H, W, F_TILE]
# ---------------------------------------------------------------------------------------------------------------

# Categorical fields: (name, vocabulary size). Id 0 is "none / unknown". Slot-valued fields store 0 = none and
# 1 + slot otherwise (slot 0 = me, 1..7 = other players in turn order, 8 = Nature), so their vocabulary is 10.
CAT_FIELDS: list[tuple[str, int]] = [
    ("terrain", 10),
    ("climate", 19),
    ("resource", 11),
    ("owner", 10),
    ("improvement", 50),
    ("capital_of", 10),
    ("unit_type", 60),
    ("unit_owner", 10),
    ("passenger_type", 60),
]

# Numeric fields: (name, divisor). The stored integer is divided by the divisor to get the network input.
NUM_FIELDS: list[tuple[str, float]] = [
    ("unexplored", 1.0),
    ("flooded", 1.0),
    ("swamped", 1.0),
    ("tentacle", 1.0),
    ("algae", 1.0),
    ("foam", 1.0),
    ("has_road", 1.0),
    ("has_route", 1.0),
    ("has_ruin", 1.0),
    ("ruling_city_mine", 1.0),
    ("in_my_border", 1.0),
    ("improvement_level", 8.0),
    ("is_city", 1.0),
    ("is_capital", 1.0),
    ("city_level", 8.0),
    ("population", 8.0),
    ("population_needed", 8.0),
    ("walls", 1.0),
    ("park", 1.0),
    ("workshop", 1.0),
    ("border_size", 3.0),
    ("connected_to_capital", 1.0),
    ("under_siege", 1.0),
    ("health", 100.0),
    ("veteran", 1.0),
    ("moved", 1.0),
    ("attacked", 1.0),
    ("frozen", 1.0),
    ("poisoned", 1.0),
    ("boosted", 1.0),
    ("invisible", 1.0),
    ("can_act", 1.0),
]

# Column layout inside the last axis of `tiles`: categorical fields first, then numeric ones.
CAT_NAMES = [n for n, _ in CAT_FIELDS]
CAT_VOCABS = [v for _, v in CAT_FIELDS]
NUM_NAMES = [n for n, _ in NUM_FIELDS]
NUM_SCALES = [s for _, s in NUM_FIELDS]
N_CAT = len(CAT_FIELDS)
N_NUM = len(NUM_FIELDS)
F_TILE = N_CAT + N_NUM
COLUMN = {name: i for i, name in enumerate(CAT_NAMES + NUM_NAMES)}

# ---------------------------------------------------------------------------------------------------------------
# Global vector: float32[G_DIM]
# ---------------------------------------------------------------------------------------------------------------

N_TRIBES = 13
N_TECHS = 49
N_OTHER_SLOTS = 7
PER_SLOT = 1 + 1 + 1 + N_TRIBES + 1 + 1 + 4 + 1  # present, alive, met, tribe if met, score, cities, relation, embassy
N_TRIGGER_KINDS = 8
G_DIM = (
    N_TRIBES  # my tribe, one-hot
    + 8  # stars, income, score, turn, turns left, mode, map size, living players
    + N_TECHS
    + N_OTHER_SLOTS * PER_SLOT
    + N_TRIGGER_KINDS
)

# ---------------------------------------------------------------------------------------------------------------
# Actions: int16[n_actions, 4] = (kind, arg, src_tile, dst_tile); tiles are y * W + x, -1 means "none".
# ---------------------------------------------------------------------------------------------------------------

ACTION_KINDS = [
    "Move", "Attack", "Capture", "Promote", "Recover", "HealOthers", "Explode", "Disband", "ExamineRuins", "Stay",
    "Swarm", "Boost", "Disembark", "Upgrade", "Train", "Build", "Destroy", "Decompose", "ClearTileEffect",
    "CityReward", "Research", "BreakPeace", "PeaceTreaty", "EstablishEmbassy", "PeaceResponse", "EndTurn",
]
KIND_PAD = 0  # padding rows in a batch; real kinds are 1..len(ACTION_KINDS)
KIND_ID = {name: i + 1 for i, name in enumerate(ACTION_KINDS)}
N_KINDS = 32  # embedding rows, with room for the engine to grow
ARG_MAX = 64  # arguments are clipped to [0, ARG_MAX - 1]
OFFSET_RADIUS = 4  # (dy, dx) between src and dst is clipped to +/- this; one more row means "no src or dst"

N_SLOTS = 9  # winner head: me, 7 others, Nature

# ---------------------------------------------------------------------------------------------------------------
# Shard (.npz). One shard holds one map size. N positions, T = sum of legal-list lengths, K candidate slots.
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardField:
    dtype: str
    shape: str  # human readable; the leading N is the position count
    note: str


SHARD_FIELDS: dict[str, ShardField] = {
    "tiles": ShardField("int16", "[N, H, W, F_TILE]", "see CAT_FIELDS and NUM_FIELDS"),
    "globals": ShardField("float32", "[N, G_DIM]", "global vector"),
    "act_off": ShardField("int32", "[N + 1]", "legal list of position i is actions[act_off[i]:act_off[i + 1]]"),
    "actions": ShardField("int16", "[T, 4]", "(kind, arg, src, dst)"),
    "chosen": ShardField("int16", "[N]", "index of the chosen command inside its own legal list"),
    "cand_idx": ShardField("int16", "[N, K]", "legal-list indices of the AI's top candidates, -1 padded"),
    "cand_score": ShardField("float32", "[N, K]", "the AI's scores for those candidates"),
    "winner": ShardField("int8", "[N]", "winning slot relative to the player to move, -1 for none (ignored)"),
    "margin": ShardField("float32", "[N]", "final score of me minus best opponent, raw points"),
    "margin5": ShardField("float32", "[N]", "the same margin 5 turns later (final if the game ended sooner)"),
    "turn": ShardField("int16", "[N]", "game turn of the position"),
}
REQUIRED_FIELDS = tuple(SHARD_FIELDS)
