"""KataGo-style residual CNN with a legal-list policy head and a winner value head (design.md section 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import schema as S

MASK_FILL = -1e9


@dataclass
class ModelConfig:
    blocks: int = 6
    channels: int = 96
    gpool_every: int = 3  # every n-th block (3, 6, 9, ...) carries a global-pooling bias
    policy_hidden: int = 128
    value_hidden: int = 128
    g_dim: int = S.G_DIM
    name: str = ""
    # Width of the auxiliary belief heads (one small conv head per truth channel, docs/belief.md); 0 = no heads. The key
    # is left out of `to_dict` when 0, so a config without heads is exactly what it was before the heads existed.
    belief_hidden: int = 0
    # With belief heads, one more small head on the pooled vector for the scalar truth of `schema.TRUTH_G_NAMES` (RL
    # only: the shards carry no scalar labels). Left out of `to_dict` when False.
    belief_scalars: bool = False
    # Engine features (encoding.md section 12) as extra inputs, all zero-initialised so the model starts as the same
    # function without them: tiles through a 3x3 conv into the trunk input, globals through a linear layer into the
    # global vector, actions through a linear layer into the policy MLP's first layer. Left out of `to_dict` when False.
    features: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["belief_hidden"]:
            del d["belief_hidden"]
        if not d["belief_scalars"]:
            del d["belief_scalars"]
        if not d["features"]:
            del d["features"]
        return d


CONFIGS: dict[str, ModelConfig] = {
    "b6c96": ModelConfig(blocks=6, channels=96, name="b6c96"),
    "b10c128": ModelConfig(blocks=10, channels=128, name="b10c128"),
    "b15c192": ModelConfig(blocks=15, channels=192, name="b15c192"),
    "tiny": ModelConfig(blocks=3, channels=24, policy_hidden=32, value_hidden=32, name="tiny"),
}


def eval_pawns(p: torch.Tensor | float) -> torch.Tensor:
    """Chess-style evaluation: 4 * log10(p / (1 - p)), clipped to +/-9.9. 50% is 0.0, 85% about +3.0."""
    p = torch.as_tensor(p, dtype=torch.float32)
    p = p.clamp(1e-6, 1 - 1e-6)
    return (4.0 * torch.log10(p / (1.0 - p))).clamp(-9.9, 9.9)


class ModelOutput(NamedTuple):
    policy_logits: torch.Tensor  # [B, A] float32, padded entries at MASK_FILL
    winner_logits: torch.Tensor  # [B, 9] who wins by elimination, relative slots; softmax gives win probabilities


class BeliefOutput(NamedTuple):
    """A ModelOutput plus the belief logits; returned only by `forward(..., belief=True)` on a model with the heads, so
    the two-field ModelOutput that everything else unpacks stays what it was."""

    policy_logits: torch.Tensor
    winner_logits: torch.Tensor
    belief: tuple[torch.Tensor, ...]  # per truth channel [B, V, H, W] logits
    scalars: torch.Tensor | None = None  # [B, TRUTH_G] predictions of the scalar truth, when the model has that head


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pool [B, C, H, W] over the real board. `x` must be >= 0 (post ReLU) so that zero padding is neutral for
    the max. Returns [B, 2C] = (mean, max)."""
    if mask is None:
        return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
    x = x * mask
    count = mask.sum(dim=(2, 3)).clamp(min=1.0)
    return torch.cat([x.sum(dim=(2, 3)) / count, x.amax(dim=(2, 3))], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c: int, gpool: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gpool = gpool
        if gpool:
            self.gc = max(c // 4, 8)
            self.bn_g = nn.BatchNorm2d(self.gc)
            self.pool_fc = nn.Linear(2 * self.gc, c)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # With a board mask, padded cells are re-zeroed before each conv so a smaller board inside a larger tensor
        # sees the same zero padding at its edge as it would on its own.
        a = F.relu(self.bn1(x))
        y = self.conv1(a if mask is None else a * mask)
        if self.gpool:
            gp = F.relu(self.bn_g(y[:, : self.gc]))
            bias = self.pool_fc(masked_mean_max(gp, mask))
            y = y + bias[:, :, None, None].to(y.dtype)
        a = F.relu(self.bn2(y))
        y = self.conv2(a if mask is None else a * mask)
        x = x + y
        return x if mask is None else x * mask


class PolyNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        # Inputs: one embedding table per categorical field (summed), a 1x1 conv for numeric fields, globals MLP.
        self.cat_embed = nn.ModuleList([nn.Embedding(v, c) for v in S.CAT_VOCABS])
        self.num_proj = nn.Linear(S.N_NUM, c)
        self.global_mlp = nn.Sequential(nn.Linear(cfg.g_dim, c), nn.ReLU(), nn.Linear(c, c))
        self.register_buffer("num_scale", torch.tensor(S.NUM_SCALES, dtype=torch.float32), persistent=False)
        large = torch.zeros(cfg.g_dim, dtype=torch.bool)
        large[[i for i in S.G_LARGE if i < cfg.g_dim]] = True
        self.register_buffer("g_large", large, persistent=False)
        self.blocks = nn.ModuleList(
            [ResBlock(c, gpool=cfg.gpool_every > 0 and (i + 1) % cfg.gpool_every == 0) for i in range(cfg.blocks)]
        )
        self.out_bn = nn.BatchNorm2d(c)
        # Pooled vector g shared by the policy ("no tile") and the value heads.
        self.pool_fc = nn.Linear(2 * c, c)
        # Policy head. The first layer of MLP([h_src, h_dst, E_kind, E_arg, E_off]) is a sum of linear maps, so it
        # is computed per tile once and gathered per action.
        ph = cfg.policy_hidden
        self.src_proj = nn.Linear(c, ph, bias=False)
        self.dst_proj = nn.Linear(c, ph, bias=False)
        self.kind_embed = nn.Embedding(S.N_KINDS, ph)
        self.arg_embed = nn.Embedding(S.N_KINDS * S.ARG_MAX, ph)
        self.n_off = (2 * S.OFFSET_RADIUS + 1) ** 2
        self.off_embed = nn.Embedding(self.n_off + 1, ph)
        self.policy_mlp = nn.Sequential(nn.ReLU(), nn.Linear(ph, ph), nn.ReLU(), nn.Linear(ph, 1))
        # Value head: the winner distribution only (win 1 / loss 0; points are not the objective).
        vh = cfg.value_hidden
        self.value_mlp = nn.Sequential(nn.Linear(c, vh), nn.ReLU(), nn.Linear(vh, S.N_SLOTS))
        # Auxiliary belief heads: what lies under the fog (training only; play never asks for them).
        self.belief_heads = None
        if cfg.belief_hidden > 0:
            bh = cfg.belief_hidden
            self.belief_heads = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(c, bh, 3, padding=1), nn.ReLU(), nn.Conv2d(bh, v, 1)) for v in S.TRUTH_VOCABS]
            )
        self.belief_scalar_head = None
        if cfg.belief_scalars:
            if cfg.belief_hidden <= 0:
                raise ValueError("belief_scalars needs belief_hidden > 0")
            self.belief_scalar_head = nn.Sequential(nn.Linear(c, cfg.belief_hidden), nn.ReLU(),
                                                    nn.Linear(cfg.belief_hidden, S.TRUTH_G))
        # Engine features, zero-initialised additions (see ModelConfig.features).
        self.feat_tile_conv = self.feat_global_proj = self.feat_action_proj = None
        if cfg.features:
            self.feat_tile_conv = nn.Conv2d(S.FEAT_T, c, 3, padding=1, bias=False)
            self.feat_global_proj = nn.Linear(S.FEAT_G, c, bias=False)
            self.feat_action_proj = nn.Linear(S.FEAT_A, ph, bias=False)
            self.register_buffer("feat_scale", torch.tensor(S.FEAT_T_DIVISORS, dtype=torch.float32), persistent=False)
        self._init()

    def _init(self) -> None:
        for emb in list(self.cat_embed) + [self.kind_embed, self.arg_embed, self.off_embed]:
            nn.init.normal_(emb.weight, std=0.1)
            if emb in self.cat_embed:
                with torch.no_grad():
                    emb.weight[0].zero_()  # id 0 is "none"
        # Start with near-uniform policy and neutral values.
        nn.init.zeros_(self.policy_mlp[-1].weight)
        nn.init.zeros_(self.policy_mlp[-1].bias)
        nn.init.zeros_(self.value_mlp[-1].weight)
        nn.init.zeros_(self.value_mlp[-1].bias)
        if self.feat_tile_conv is not None:
            for m in (self.feat_tile_conv, self.feat_global_proj, self.feat_action_proj):
                nn.init.zeros_(m.weight)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(
        self, tiles: torch.Tensor, globals_: torch.Tensor, board_mask: torch.Tensor | None,
        f_tiles: torch.Tensor | None = None, f_globals: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """tiles int [B, H, W, F], globals [B, G] -> (trunk features [B, C, H, W] after BN + ReLU, board mask or None).
        `f_tiles` int [B, H, W, FEAT_T] and `f_globals` [B, FEAT_G] are the optional engine features; a model built
        without `features` ignores them, one built with them treats missing ones as zeros."""
        x = None
        for i, emb in enumerate(self.cat_embed):
            e = emb(tiles[..., i].long().clamp(0, S.CAT_VOCABS[i] - 1))
            x = e if x is None else x + e
        num = tiles[..., S.N_CAT :].to(self.num_scale.dtype) / self.num_scale
        x = x + self.num_proj(num)
        gl = globals_.float()
        gl = torch.where(self.g_large, torch.asinh(gl), gl)  # stars, income and scores run into the tens
        x = x + self.global_mlp(gl.to(x.dtype))[:, None, None, :]
        if self.feat_global_proj is not None and f_globals is not None:
            x = x + self.feat_global_proj(f_globals.to(x.dtype))[:, None, None, :]
        x = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        if self.feat_tile_conv is not None and f_tiles is not None:
            ft = (f_tiles.to(self.feat_scale.dtype) / self.feat_scale).permute(0, 3, 1, 2)
            if board_mask is not None:
                ft = ft * board_mask[:, None].to(ft.dtype)
            x = x + self.feat_tile_conv(ft.contiguous(memory_format=torch.channels_last)).to(x.dtype)
        mask = None
        if board_mask is not None:
            mask = board_mask[:, None].to(x.dtype)
            x = x * mask
        for blk in self.blocks:
            x = blk(x, mask)
        return F.relu(self.out_bn(x)), mask

    def forward(
        self,
        tiles: torch.Tensor,
        globals_: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        board_mask: torch.Tensor | None = None,
        belief: bool = False,
        f_tiles: torch.Tensor | None = None,
        f_globals: torch.Tensor | None = None,
        f_actions: torch.Tensor | None = None,
    ) -> ModelOutput | BeliefOutput:
        """tiles int [B, H, W, F]; globals [B, G]; actions int [B, A, 4] = (kind, arg, src, dst) with -1 for no
        tile; action_mask bool [B, A] (True = real legal action); board_mask optional bool [B, H, W]. With `belief`
        (and a model that has the heads) the output also carries the belief logits. `f_tiles`, `f_globals` and `f_actions`
        (float [B, A, FEAT_A], padded like `actions`) are the optional engine features, used only by a model built
        with `ModelConfig.features`."""
        b, h, w, _ = tiles.shape
        feat, mask = self.encode(tiles, globals_, board_mask, f_tiles, f_globals)
        pooled = masked_mean_max(feat, mask)
        g = self.pool_fc(pooled)  # [B, C]

        tok = feat.permute(0, 2, 3, 1).reshape(b, h * w, -1)  # [B, HW, C]
        tok = torch.cat([tok, g[:, None, :].to(tok.dtype)], dim=1)  # index HW is "no tile": pooled g
        src_f = self.src_proj(tok)
        dst_f = self.dst_proj(tok)

        # Only the real actions go through the policy MLP: legal lists are padded to the longest one in the batch
        # (about 1000 against a mean of 200), so scoring the padding would cost most of the memory.
        bi, ai = action_mask.nonzero(as_tuple=True)
        act = actions[bi, ai].long()
        kind = act[:, 0].clamp(0, S.N_KINDS - 1)
        arg = act[:, 1].clamp(0, S.ARG_MAX - 1)
        src, dst = act[:, 2], act[:, 3]
        none_tile = h * w
        src_i = torch.where(src < 0, torch.full_like(src, none_tile), src)
        dst_i = torch.where(dst < 0, torch.full_like(dst, none_tile), dst)
        r = S.OFFSET_RADIUS
        dy = (dst // w - src // w).clamp(-r, r) + r
        dx = (dst % w - src % w).clamp(-r, r) + r
        off = torch.where((src < 0) | (dst < 0), torch.full_like(src, self.n_off), dy * (2 * r + 1) + dx)

        z = (
            src_f[bi, src_i]
            + dst_f[bi, dst_i]
            + self.kind_embed(kind).to(src_f.dtype)
            + self.arg_embed(kind * S.ARG_MAX + arg).to(src_f.dtype)
            + self.off_embed(off).to(src_f.dtype)
        )
        if self.feat_action_proj is not None and f_actions is not None:
            z = z + self.feat_action_proj(f_actions[bi, ai].to(src_f.dtype)).to(src_f.dtype)
        flat = self.policy_mlp(z).squeeze(-1).float()
        logits = torch.full(action_mask.shape, MASK_FILL, dtype=torch.float32, device=tiles.device)
        logits = logits.index_put((bi, ai), flat)

        value = self.value_mlp(F.relu(g)).float()
        if belief and self.belief_heads is not None:
            scalars = None
            if self.belief_scalar_head is not None:
                scalars = self.belief_scalar_head(F.relu(g)).float()
            return BeliefOutput(logits, value, tuple(head(feat).float() for head in self.belief_heads), scalars)
        return ModelOutput(logits, value)


def build_model(name_or_cfg: str | ModelConfig) -> PolyNet:
    cfg = CONFIGS[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    return PolyNet(cfg)


BELIEF_PREFIXES = ("belief_heads.", "belief_scalar_head.")  # parameter names of the belief heads
FEAT_PREFIX = "feat_"  # parameter names of the engine-feature inputs
FEAT_KEYS = ("f_tiles", "f_globals", "f_actions")


def feature_inputs(batch: dict, model: PolyNet) -> dict:
    """The keyword arguments of `PolyNet.forward` that carry the engine features: those of `batch` (a data batch or
    an observation) when the model uses them, none otherwise."""
    if not model.cfg.features:
        return {}
    missing = [k for k in FEAT_KEYS if k not in batch]
    if missing:
        raise ValueError(f"this model reads the engine features, but the input has no {missing} "
                         "(shards need polyai-gen --features, VecEnv needs features=True)")
    return {k: batch[k] for k in FEAT_KEYS}


def add_features(model: PolyNet) -> PolyNet:
    """A copy of a plain model with the engine-feature inputs added. They start at zero, so the copy computes exactly
    what `model` does until they are trained."""
    if model.cfg.features:
        return model
    new = build_model(replace(model.cfg, features=True))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or any(not k.startswith(FEAT_PREFIX) for k in missing):
        raise ValueError(f"cannot add features: missing {missing}, unexpected {unexpected}")
    return new.to(next(model.parameters()).device).train(model.training)


def add_belief(model: PolyNet, hidden: int = 32, scalars: bool = True) -> PolyNet:
    """A copy of a model with fresh belief heads added (for RL from a plain checkpoint). The policy and value are
    unchanged: the heads only read the trunk."""
    if model.cfg.belief_hidden:
        return model
    new = build_model(replace(model.cfg, belief_hidden=hidden, belief_scalars=scalars))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or any(not k.startswith(BELIEF_PREFIXES) for k in missing):
        raise ValueError(f"cannot add belief heads: missing {missing}, unexpected {unexpected}")
    return new.to(next(model.parameters()).device).train(model.training)


def win_probability(out: ModelOutput) -> torch.Tensor:
    """My win probability (slot 0 of the winner softmax)."""
    return F.softmax(out.winner_logits, dim=-1)[:, 0]


__all__ = [
    "CONFIGS", "ModelConfig", "ModelOutput", "PolyNet", "build_model", "eval_pawns", "masked_mean_max",
    "win_probability", "MASK_FILL", "add_features", "add_belief", "BELIEF_PREFIXES", "feature_inputs", "FEAT_PREFIX", "FEAT_KEYS",
]
