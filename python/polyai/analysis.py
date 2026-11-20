"""The analysis win chance: a value network read from one player's own fogged view, plus a calibration layer per
game stage, and the rules that decide when it may give a number at all (docs/analysis.md).

The network is the trunk and value head of a policy network (`PolyNet`) with one more input: who plays each player
slot (a person, or a bot and its level; `polyai_core::slot_kinds`) and whether the viewing player is the one to move.
Those are public facts of the game. It is trained on value rows (`polyai-gen/src/value_shard.rs`): every player's
view at sampled positions, labelled with who won by the game's own rule. Slots without a player cannot win, so their
logits are masked before the softmax.

The calibration layer maps the network's win chance q to sigmoid(a * logit(q) + b), with (a, b) per game kind, stage
and whether the viewer is the one to move, fitted on held-out games (`polyai.train_value`). A (kind, stage) that the
gate did not pass has no map, and the service gives null with the reason there.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import schema as S
from .model import ModelConfig, PolyNet, build_model, masked_mean_max

N_PLAYER_SLOTS = 8
KIND_VOCAB = 7  # 0 no player, 1 a person, 2 + d a bot of difficulty d (0 easy .. 3 crazy, 4 frozen)
KIND_PERSON = 1
EXTRA_DIM = N_PLAYER_SLOTS * KIND_VOCAB + 1
FORMAT = "polyai-analysis-value"

# Game stages by game turn: [lo, hi] inclusive. Turns 0 to 5 are split so the opening is checked on its own.
STAGES: tuple[tuple[int, int], ...] = ((0, 2), (3, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 10**6))
STAGE_NAMES = tuple(f"{a}-{b}" if b < 10**6 else f"{a}+" for a, b in STAGES)

MODE_NAMES = {1: "perfection", 2: "domination", 3: "glory", 4: "might", 5: "custom"}


def stage_of(turn: int | np.ndarray) -> int | np.ndarray:
    """Index into STAGES of a game turn (vectorised over arrays)."""
    edges = np.array([b for _, b in STAGES[:-1]])
    out = np.searchsorted(edges, np.asarray(turn), side="left")
    return int(out) if np.ndim(out) == 0 else out


def kind_of(mode: int, n_players: int, kinds: np.ndarray) -> str:
    """The game kind a position belongs to, for the gate and the calibration layer: `human2` (two people, the human
    corpus), else `<mode>-<n>p` for games with at least one bot, by the game's rules mode and starting players
    grouped as 2, 3-4 and 5-8. `kinds` is the viewing player's int8[8] slot kinds."""
    k = np.asarray(kinds)
    people = int((k == KIND_PERSON).sum())
    bots = int((k >= 2).sum())
    if bots == 0:
        return "human2" if n_players == 2 and people == 2 else f"people{n_players}"
    group = "2p" if n_players <= 2 else ("3-4p" if n_players <= 4 else "5-8p")
    return f"{MODE_NAMES.get(int(mode), str(mode))}-{group}"


def extra_inputs(kinds: torch.Tensor, mover: torch.Tensor) -> torch.Tensor:
    """kinds int [B, 8], mover [B] -> float [B, EXTRA_DIM]: a one-hot of each slot's kind, then the mover flag."""
    one = F.one_hot(kinds.long().clamp(0, KIND_VOCAB - 1), KIND_VOCAB).flatten(1).float()
    return torch.cat([one, mover.float()[:, None]], dim=1)


class ValueNet(nn.Module):
    """Win chances from one player's view: logits [B, 9] over relative slots (0 = the viewer)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = replace(cfg, belief_hidden=0, belief_scalars=False, truth_input=False)
        self.net = build_model(self.cfg)
        self.extra_proj = nn.Linear(EXTRA_DIM, self.cfg.channels)
        nn.init.zeros_(self.extra_proj.weight)
        nn.init.zeros_(self.extra_proj.bias)
        self.mask_slots = True

    @classmethod
    def from_policy(cls, model: PolyNet) -> "ValueNet":
        """A value net that starts as `model`'s own value head (the new inputs start at zero)."""
        v = cls(model.cfg)
        sd = {k: t for k, t in model.state_dict().items() if not k.startswith(("belief_", "truth_"))}
        missing, unexpected = v.net.load_state_dict(sd, strict=False)
        if unexpected or missing:
            raise ValueError(f"cannot start a value net from this model: missing {missing}, unexpected {unexpected}")
        return v

    def forward(self, tiles: torch.Tensor, globals_: torch.Tensor, kinds: torch.Tensor, mover: torch.Tensor,
                board_mask: torch.Tensor | None = None, f_tiles: torch.Tensor | None = None,
                f_globals: torch.Tensor | None = None) -> torch.Tensor:
        feat, mask = self.net.encode(tiles, globals_, board_mask, f_tiles, f_globals)
        g = self.net.pool_fc(masked_mean_max(feat, mask))
        g = g + self.extra_proj(extra_inputs(kinds, mover).to(g.dtype))
        logits = self.net.value_mlp(F.relu(g)).float()
        return mask_empty_slots(logits, kinds) if self.mask_slots else logits


def mask_empty_slots(logits: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
    """Slots without a player (and Nature's slot 8) cannot win: their logits become very negative."""
    empty = torch.ones_like(logits, dtype=torch.bool)
    empty[:, :N_PLAYER_SLOTS] = kinds.to(logits.device) == 0
    return logits.masked_fill(empty, -1e4)


Params = tuple[float, float, float, float]  # (scale, offset) when the viewer is not to move, then when it is


class Calibration:
    """Per (kind, stage) a logistic map of the network's win chance: p = sigmoid(a * logit(q) + b), with (a, b) for
    the viewer not to move and for the viewer to move. A missing entry means no number is given there."""

    def __init__(self, params: dict[str, list[Params | None]] | None = None, reasons: dict[str, list[str | None]] | None = None):
        self.params = {k: [None if p is None else tuple(p) for p in v] for k, v in (params or {}).items()}
        self.reasons = reasons or {}

    def get(self, kind: str, stage: int, mover: bool) -> tuple[float, float] | None:
        t = self.params.get(kind)
        if t is None or t[stage] is None:
            return None
        p = t[stage]
        return (p[2], p[3]) if mover else (p[0], p[1])

    def reason(self, kind: str, stage: int) -> str:
        r = self.reasons.get(kind)
        if r is not None and r[stage]:
            return r[stage]
        if kind not in self.params:
            return f"no calibrated win chance for this kind of game ({kind})"
        return f"the win chance failed the calibration gate for {kind} games at turns {STAGE_NAMES[stage]}"

    def to_dict(self) -> dict:
        return {"params": {k: [None if p is None else list(p) for p in v] for k, v in self.params.items()},
                "reasons": self.reasons}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        d = d or {}
        return cls(d.get("params"), d.get("reasons"))


def raw_win(logits: torch.Tensor) -> torch.Tensor:
    """The network's own win chance of slot 0."""
    return F.softmax(logits.float(), dim=-1)[:, 0]


def calibrated_win(logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sigmoid(a * logit(q) + b) of slot 0's softmax probability q, per row."""
    q = raw_win(logits).clamp(1e-6, 1 - 1e-6)
    return torch.sigmoid(a * torch.log(q / (1 - q)) + b)


def save_value(path: str | Path, net: ValueNet, cal: Calibration, meta: dict) -> None:
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.startswith(("net.policy_", "net.src_proj",
                                                                                          "net.dst_proj", "net.kind_embed",
                                                                                          "net.arg_embed", "net.off_embed"))}
    torch.save({"format": FORMAT, "model_config": asdict(net.cfg), "model": sd, "calibration": cal.to_dict(),
                "meta": meta}, path)


def load_value(path: str | Path, device: str | torch.device = "cpu") -> tuple[ValueNet, Calibration, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not an analysis value checkpoint")
    net = ValueNet(ModelConfig(**ck["model_config"]))
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if unexpected or any(not k.startswith("net.") or k.startswith("net.value_") or k.startswith("net.pool_") for k in missing):
        raise ValueError(f"{path}: missing {missing}, unexpected {unexpected}")
    return net.to(device).eval(), Calibration.from_dict(ck.get("calibration")), ck.get("meta", {})


def policy_value_net(model: PolyNet) -> ValueNet:
    """The live policy's own value head as a `ValueNet` (the extra inputs stay zero and no slot is masked), so it
    computes exactly the policy's value, for the gate's baseline."""
    v = ValueNet.from_policy(model).to(next(model.parameters()).device).eval()
    v.mask_slots = False
    return v


__all__ = ["ValueNet", "Calibration", "STAGES", "STAGE_NAMES", "stage_of", "kind_of", "save_value", "load_value",
           "calibrated_win", "raw_win", "policy_value_net", "mask_empty_slots", "extra_inputs", "KIND_VOCAB", "EXTRA_DIM", "S"]
