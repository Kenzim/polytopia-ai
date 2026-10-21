"""The one place that defines the tensor and shard layout the Python side expects.

This follows `ai/docs/encoding.md`, which is the authority. Shards of encoding version 2 (the current one) and 1 are
read; version 1 shards are relabelled at load time (see `data.Shard`). `tests/test_schema.py` checks every
constant here against the `polyai_engine` module (built from polyai-core) whenever that is installed; nothing else in
the package hard-codes field names, vocabulary sizes or shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

ENCODING_VERSION = 2  # the version this package writes (synthetic shards, checkpoints)
SUPPORTED_VERSIONS = (1, 2)  # shard versions the loader accepts; v1 is relabelled to the v2 meaning on load

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
    ("improvement", 56),
    ("capital_of", 10),
    ("unit_type", 63),
    ("unit_owner", 10),
    ("passenger_type", 63),
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

# Tribe order (engine tribe idx) of the one-hots. The out-of-scope tribes (aquarion 3, elyrion 5, polaris 16) are zeros.
TRIBES = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]
N_TRIBES = len(TRIBES)
N_SCALARS = 12  # stars, income, score, turn, turns left, has turn limit, mode, map width, living, start players, cities, units
N_TECHS = 51  # a bit per engine tech idx 0..50 (idx 11 and 34 do not exist)
N_OTHER_SLOTS = 7
# present, alive, met, tribe if met, score, cities, relation (neutral, peace, war, cooldown), embassy mine, embassy theirs
PER_SLOT = 1 + 1 + 1 + N_TRIBES + 1 + 1 + 4 + 1 + 1
N_TRIGGER_KINDS = 3  # none, city reward, peace request

G_TRIBE = 0
G_SCALARS = G_TRIBE + N_TRIBES
G_STARS, G_INCOME, G_SCORE = G_SCALARS, G_SCALARS + 1, G_SCALARS + 2
G_TURNS_LEFT = G_SCALARS + 4  # turns left / 50; 0 for Domination from encoding v2 on
G_HAS_LIMIT = G_SCALARS + 5  # has a turn limit (0 / 1); also 0 from encoding v2 on
G_TECHS = G_SCALARS + N_SCALARS
G_SLOTS = G_TECHS + N_TECHS
G_TRIGGER = G_SLOTS + N_OTHER_SLOTS * PER_SLOT
G_DIM = G_TRIGGER + N_TRIGGER_KINDS
SLOT_SCORE = 16  # offset of "score / 1000" inside a per-slot block
# Entries that can be far above 1 (scores reach 25000 and more); the model compresses them with asinh.
G_LARGE = [G_STARS, G_INCOME, G_SCORE] + [G_SLOTS + k * PER_SLOT + SLOT_SCORE for k in range(N_OTHER_SLOTS)]

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

N_SLOTS = 9  # winner head: me, 7 others, Nature (which never wins, so slot 8 stays unused)
N_CANDIDATES = 16  # K candidate slots in a shard

# ---------------------------------------------------------------------------------------------------------------
# Shard (.npz). One shard holds one map size. N positions, T = sum of legal-list lengths, K candidate slots.
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardField:
    dtype: str
    shape: str  # human readable; the leading N is the position count
    note: str


# The v2 layout: no score margins, `capped` instead of `adjudicated`.
SHARD_FIELDS: dict[str, ShardField] = {
    "tiles": ShardField("int16", "[N, H, W, F_TILE]", "see CAT_FIELDS and NUM_FIELDS"),
    "globals": ShardField("float32", "[N, G_DIM]", "global vector"),
    "act_off": ShardField("int32", "[N + 1]", "legal list of position i is actions[act_off[i]:act_off[i + 1]]"),
    "actions": ShardField("int16", "[T, 4]", "(kind, arg, src, dst)"),
    "chosen": ShardField("int16", "[N]", "index of the played command inside its own legal list (the label)"),
    "cand_idx": ShardField("int16", "[N, K]", "legal-list indices of the AI's best candidates, best first, -1 padded"),
    "cand_score": ShardField("float32", "[N, K]", "their noise-free scores; may be +inf; 0 where padded"),
    "winner": ShardField("int8", "[N]", "winning slot relative to the player to move, -1 for none (capped game)"),
    "final_scores": ShardField("float32", "[N, 8]", "final scores by slot, zeros for absent slots"),
    "capped": ShardField("int8", "[N]", "1 when the game hit the compute cap with more than one player alive (a draw)"),
    "turn": ShardField("int16", "[N]", "game turn of the position"),
    "seed": ShardField("int32", "[N]", "game seed"),
    "cmd_index": ShardField("int32", "[N]", "index of the command in the game record"),
    "difficulty": ShardField("int8", "[N]", "mover's difficulty: 0 easy, 1 normal, 2 hard, 3 crazy"),
    "version": ShardField("int32", "[1]", "ENCODING_VERSION"),
}
REQUIRED_FIELDS = tuple(SHARD_FIELDS)

# What a version 1 shard holds instead: `adjudicated` (a top-score decision at the turn limit, now a draw) in place
# of `capped`. Its `margin` and `margin5` keys are ignored: points are not the objective.
V1_ONLY_FIELDS = ("adjudicated",)
V1_IGNORED_FIELDS = ("margin", "margin5")
V1_REQUIRED_FIELDS = tuple(k for k in REQUIRED_FIELDS if k != "capped") + V1_ONLY_FIELDS

# Optional keys (docs/interfaces.md section 2.2, encoding.md section 6). Generator shards before these existed lack
# them, and the loader fills the defaults in. `policy_ids` is per shard, not per position; `game_key` is a string
# array, whose default is "gen:<seed>" (see `data.Shard.game_key`).
SOURCE_GENERATOR, SOURCE_SELFPLAY, SOURCE_LIVE, SOURCE_HUMAN = 0, 1, 2, 3
OPTIONAL_FIELDS: dict[str, ShardField] = {
    "source": ShardField("int8", "[N]", "0 retail generator, 1 self-play, 2 live, 3 human replay (default 0)"),
    "ours": ShardField("int8", "[N]", "1 when the mover is one of our accounts (default 0)"),
    "behaviour_logp": ShardField("float32", "[N]", "logp of the policy that chose the move; NaN if none (default NaN)"),
    "policy": ShardField("int16", "[N]", "index into policy_ids, -1 for none (default -1)"),
    "policy_ids": ShardField("str", "[P]", "policy_id strings of this shard (default empty)"),
    "game_key": ShardField("str", "[N]", "game identity for train/validation splits (default gen:<seed>)"),
}
