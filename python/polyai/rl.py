"""PPO self-play (design.md sections 6 phase B and 7 item 2).

    nice python -m polyai.rl --name rl1 --init /kvm/polytopia-ai-data/runs/bc3_b6c96_v11 --updates 200

The learner (the network being trained) plays one seat of every game, the seat rotating uniformly; the other seats
are opponents from a pool: frozen earlier checkpoints of the learner (`self:u<update>`), the behaviour-cloning start
checkpoint (`bc`, also the KL reference), and the retail AI at a fixed difficulty (`retail:easy|normal|hard|crazy`).
The learner always has the Normal handicap. Pool entries are sampled in proportion to a base weight times a factor
that grows for opponents the learner beats less than `beat_target` (70%) of the time (`rl_pool.pool_weights`).

Objective (design v1.1): a game is won by elimination only. Reward is 1 for a win and 0 for a loss at the end of the
game and nothing else; a game that hits the 150-turn compute cap is a draw worth 0.5 (`--drop-capped` leaves those
games out of training instead). Credit goes to commands: the trajectory of the learner is the sequence of its
commands, discounted with gamma (default 1) and GAE(lambda). The value is the winner head's probability of the
learner's own win (`model.win_probability`), trained with a binary cross-entropy on the GAE return.

Network mode. BatchNorm runs on its running statistics (eval mode) in rollouts *and* in the PPO update, so the policy
that acted and the policy being updated are the same function before the first step (ratio exactly 1 up to rounding);
the running statistics therefore stay at their behaviour-cloning values and the affine parameters still train.

Run directory (like `train_bc`): <runs-dir>/<name>/{config.json, metrics.jsonl, games.jsonl, ckpt_last.pt (with
optimiser and pool state, for --resume), ckpt_u<update>.pt, ckpt_final.pt, pool/snap_u<update>.pt}.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import schema as S
from .belief import full_belief_loss
from .model import BELIEF_PREFIXES, ModelOutput, PolyNet, add_belief, add_features, feature_inputs
from .policy import collate, collate_features, resolve_checkpoint
from .rl_pool import (
    DEFAULT_POOL_WEIGHTS,
    LEARNER_DIFFICULTY,
    MAX_SEATS,
    RETAIL_NAMES,
    TRUTH_ROW_KEYS,
    EntryInfo,
    NetPolicy,
    OppStats,
    arm_capacity,
    merge_padded,
    obs_len,
    pad_bucket,
    parse_pairs,
    parse_players,
    plan_quotas,
    policy_id_of,
    pool_weights,
    retail_player_weights,
    seat_probs,
    take,
)
from .arena import MODES, feature_kw, mode_kw
from .train_bc import DEFAULT_RUNS_DIR, check_gpu_memory, load_checkpoint, make_optimizer, save_checkpoint

SIZE_IDS = {"tiny": 1, "small": 2, "normal": 3, "large": 4, "huge": 5, "massive": 6}
PRESET_IDS = {"dryland": 1, "lakes": 2, "continents": 3, "archipelago": 4, "waterworld": 5, "pangea": 6}
TRAIN_SEED_MAX = 100_000


@dataclass
class RLConfig:
    name: str = "rl"
    init: str = ""  # run directory or checkpoint the learner starts from
    ref: str = ""  # the behaviour-cloning policy for the KL penalty and the `bc` pool entry (default: init)
    runs_dir: str = DEFAULT_RUNS_DIR
    updates: int = 100  # total number of PPO updates (with --resume: the new total)
    # rollouts
    games: int = 128  # games in flight (all pool entries together)
    rollout_steps: int = 16384  # learner decisions collected per update
    n_players: str = "2:55,3:25,4:20"
    sizes: str = ""  # from tiny,small,normal,large,huge,massive (default: the engine's tiny,small,normal)
    presets: str = ""  # subset of dryland,lakes,continents,archipelago,waterworld,pangea (default: the engine's five, no waterworld)
    turn_limit: int = 150  # compute cap; a game that reaches it is a draw
    mode: str = "domination"  # game mode of every rollout game: domination or might
    temperature: float = 1.0  # learner sampling temperature; the policy is softmax(logits / temperature) throughout
    opp_temperature: float = 1.0  # sampling temperature of model opponents
    drop_capped: bool = False  # leave capped (drawn) games out of training instead of rewarding 0.5
    # opponent pool
    pool_weights: str = ",".join(f"{k}={v}" for k, v in DEFAULT_POOL_WEIGHTS.items())
    beat_target: float = 0.7
    weight_floor: float = 0.1
    weight_cap: float = 2.0
    stats_decay: float = 0.995  # per game against an entry, for its running win rate
    snapshot_every: int = 10  # updates between frozen copies of the learner joining the pool
    max_snapshots: int = 6
    # PPO
    epochs: int = 2
    minibatch: int = 256
    lr: float = 3e-5
    weight_decay: float = 0.0
    grad_clip: float = 0.5
    clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    target_kl: float = 0.05  # stop the epochs early when the mean approximate KL to the behaviour policy passes it
    critic_warmup: int = 2  # first updates train the value head only
    kl_coef: float = 0.1  # penalty on KL(policy || BC policy), decaying linearly to kl_coef_end
    kl_coef_end: float = 0.01
    kl_decay_updates: int = 200
    # run
    ckpt_every: int = 10
    log_games: bool = True
    seed: int = 0
    threads: int = 8  # VecEnv stepping threads plus torch threads
    torch_threads: int = 2
    bf16: bool = True  # autocast in the PPO update on CUDA (inference always runs bf16 on CUDA, as ModelPolicy)
    max_batch: int = 256  # positions per inference forward pass
    pad_sizes: bool = True  # one inference call per network for all map sizes of a bucket (rl_pool.pad_bucket)
    max_lead: int = 4  # steps an arm may run ahead of the slowest arm in a rollout (0: lockstep, see Collector.tick)
    arm_games: int = 0  # games per arm at most (0: games / env threads); smaller arms stall less on one slow game
    min_infer: int = 0  # positions to gather for one inference round before running it (0: run what is ready)
    gather_ms: float = 20.0  # longest wait for min_infer positions
    overlap: bool = False  # play the next rollout (with the weights before this update) while the PPO update runs
    infer_graphs: bool = False  # rollout inference through CUDA graphs on padded fixed shapes (infer_fast.py), CUDA only
    infer_compile: bool = False  # rollout inference through torch.compile on padded fixed shapes (with or without graphs)
    collector: str = "python"  # "rust": arms, batching, learner rows and GAE in the engine module (docs/collector.md)
    need_gb: float = 2.5
    device: str = "cuda"
    resume: bool = False
    features: bool = False  # the learner reads the engine features (encoding.md section 12); zero-initialised on top of init
    # Auxiliary belief loss (docs/belief.md): belief heads on the learner's trunk predict what lies under the fog, trained
    # on the engine's truth labels of the learner's own decisions, masked to hidden tiles. 0 = off (the old behaviour).
    belief_weight: float = 0.0
    belief_hidden: int = 32  # width of each belief head
    belief_lr_mult: float = 1.0  # learning-rate factor of the (fresh) belief heads
    # The first updates train only the fresh heads (they read the trunk through a stop-gradient): random heads would
    # otherwise push large gradients into the trunk and drag the policy away from where it starts.
    belief_warmup: int = 0
    # Self-play recipe options (docs/rl.md "Recipe options"). Every default keeps the behaviour above exactly.
    ema: float = 0.0  # decay of an exponential moving average of the weights, buffers included (0 = off), saved as `ema`
    ent_schedule: bool = False  # entropy coefficient from ent_coef to ent_coef_end, and 0 while the entropy is above ent_max
    ent_max: float = 0.0  # with --ent-schedule: coefficient 0 in an update if the previous update's entropy exceeded it (0 = off)
    ent_coef_end: float = -1.0  # end of the linear decay of the entropy coefficient (negative: stay at ent_coef)
    ent_decay_updates: int = 200
    anchor_every: int = 0  # every N updates the KL reference becomes a frozen copy of the current policy (0 = stay at ref)
    adv_filter: float = 1.0  # fraction of the samples kept in the update, those with the largest |advantage| (1 = all)


# ---------------------------------------------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------------------------------------------


def compute_gae(values: np.ndarray, final_reward: float, terminal: bool, gamma: float, lam: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and return targets of one trajectory of the learner's commands.

    `values[t]` is the value estimate at command t. The only reward is `final_reward` after the last command. A
    terminal trajectory (the game ended) bootstraps from 0; a truncated one (the update's rollout ended mid-game) has
    no later value, so its last command serves as the bootstrap value and is not trained: the result has one entry
    fewer than `values`. The returns are `adv + values`."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if terminal:
        reward = np.zeros(n)
        reward[-1] = final_reward
        v_next = np.append(v[1:], 0.0)
    else:
        n -= 1
        if n <= 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        reward = np.zeros(n)
        v_next = v[1:]
        v = v[:n]
    delta = reward + gamma * v_next - v
    adv = np.zeros(n)
    last = 0.0
    for t in range(n - 1, -1, -1):
        last = delta[t] + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + v).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------------------------------------------


class SizeStore:
    """The learner's decisions on boards of one size: observations, legal lists, and per-decision scalars."""

    def __init__(self) -> None:
        self.tiles: list[np.ndarray] = []
        self.globals: list[np.ndarray] = []
        self.lens: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.feats: dict[str, list[np.ndarray]] = {}  # the engine features, when the observations have them
        self.truth: dict[str, list[np.ndarray]] = {}  # the truth labels (targets of the belief loss), when present
        self.n = 0
        self._cap = 1024
        self._chosen = np.zeros(self._cap, np.int32)
        self._logp = np.zeros(self._cap, np.float32)
        self._value = np.zeros(self._cap, np.float32)
        self._adv = np.zeros(self._cap, np.float32)
        self._ret = np.zeros(self._cap, np.float32)
        self._train = np.zeros(self._cap, bool)

    def _reserve(self, extra: int) -> None:
        if self.n + extra <= self._cap:
            return
        cap = max(self._cap * 2, self.n + extra)
        for name in ("_chosen", "_logp", "_value", "_adv", "_ret", "_train"):
            old = getattr(self, name)
            new = np.zeros(cap, old.dtype)
            new[: self._cap] = old
            setattr(self, name, new)
        self._cap = cap

    def add(self, obs: dict, index: np.ndarray, logp: np.ndarray, value: np.ndarray) -> np.ndarray:
        b = obs_len(obs)
        self._reserve(b)
        s = slice(self.n, self.n + b)
        self._chosen[s], self._logp[s], self._value[s] = index, logp, value
        self.tiles.append(np.asarray(obs["tiles"]))
        self.globals.append(np.asarray(obs["globals"]))
        self.lens.append(np.diff(np.asarray(obs["act_off"], dtype=np.int64)))
        self.actions.append(np.asarray(obs["actions"]))
        for k in ("f_tiles", "f_globals", "f_actions"):
            if k in obs:
                self.feats.setdefault(k, []).append(np.asarray(obs[k]))
        for k in TRUTH_ROW_KEYS:
            if k in obs:
                self.truth.setdefault(k, []).append(np.asarray(obs[k]))
        ids = np.arange(self.n, self.n + b)
        self.n += b
        return ids

    @property
    def value(self) -> np.ndarray:
        return self._value[: self.n]

    def mark(self, ids: np.ndarray, adv: np.ndarray, ret: np.ndarray) -> None:
        self._adv[ids], self._ret[ids], self._train[ids] = adv, ret, True

    def finalize(self) -> dict:
        """One observation-like dict of everything stored, plus the per-decision arrays."""
        lens = np.concatenate(self.lens)
        n = self.n
        return {
            "tiles": np.concatenate(self.tiles), "globals": np.concatenate(self.globals),
            "act_off": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "actions": np.concatenate(self.actions),
            **{k: np.concatenate(v) for k, v in self.feats.items()},
            **{k: np.concatenate(v) for k, v in self.truth.items()},
            "chosen": self._chosen[:n], "logp": self._logp[:n], "value": self._value[:n],
            "adv": self._adv[:n], "ret": self._ret[:n], "train": np.nonzero(self._train[:n])[0],
        }


# ---------------------------------------------------------------------------------------------------------------
# The PPO loss and update
# ---------------------------------------------------------------------------------------------------------------


def ppo_loss(out: ModelOutput, ref_logits: torch.Tensor | None, mb: dict[str, torch.Tensor], cfg: RLConfig,
             kl_coef: float, policy_scale: float = 1.0, ent_coef: float | None = None) -> dict[str, torch.Tensor]:
    """The PPO objective on one minibatch.

    `mb` holds `mask` (legal action mask [B, A]), `chosen`, `old_logp`, `adv` (already normalised) and `ret`
    (return target in [0, 1]). `ref_logits` are the frozen BC policy's logits on the same positions (None: no KL).
    `policy_scale` 0 trains the value head only (critic warm-up). The policy is softmax(logits / cfg.temperature).
    `ent_coef` is the entropy coefficient of this update (None: `cfg.ent_coef`, see `ent_coef_at`).

    With `cfg.belief_weight` > 0 and a belief output (`out` from `model(..., belief=True)`), `mb` also holds `truth`,
    `truth_mask` and `truth_g` and the loss adds `belief_weight` times the belief loss (hidden tiles only). With weight 0
    nothing here changes."""
    mask = mb["mask"]
    logits = out.policy_logits.float() / cfg.temperature
    logp_all = F.log_softmax(logits, dim=1)
    logp = logp_all.gather(1, mb["chosen"][:, None])[:, 0]
    log_ratio = logp - mb["old_logp"]
    ratio = log_ratio.exp()
    adv = mb["adv"]
    pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()

    p = logp_all.exp()
    entropy = -(p * logp_all.masked_fill(~mask, 0.0)).sum(dim=1).mean()
    if ref_logits is not None:
        ref_logp = F.log_softmax(ref_logits.float() / cfg.temperature, dim=1)
        kl = (p * (logp_all - ref_logp).masked_fill(~mask, 0.0)).sum(dim=1).mean()
    else:
        kl = logits.new_zeros(())

    # Value: binary cross-entropy of my win probability (slot 0 of the winner softmax) against the return target.
    wl = out.winner_logits.float()
    lse = torch.logsumexp(wl, dim=1)
    log_win = wl[:, 0] - lse
    log_not = (torch.logsumexp(wl[:, 1:], dim=1) - lse).clamp(min=-40.0)
    ret = mb["ret"]
    value_loss = -(ret * log_win + (1 - ret) * log_not).mean()

    ent_coef = cfg.ent_coef if ent_coef is None else ent_coef
    loss = policy_scale * (pg_loss - ent_coef * entropy + kl_coef * kl) + cfg.vf_coef * value_loss
    belief_metrics: dict[str, torch.Tensor] = {}
    if cfg.belief_weight > 0 and "truth" in mb and getattr(out, "belief", None) is not None:
        bl, per = full_belief_loss(out, mb["truth"], mb["truth_mask"], mb.get("truth_g"))
        loss = loss + cfg.belief_weight * bl
        belief_metrics = {"belief": bl.detach(), **{f"belief_{k}": v.detach() for k, v in per.items()}}
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean()
        value_mse = ((log_win.exp() - ret) ** 2).mean()
    return {"loss": loss, "policy_loss": pg_loss, "value_loss": value_loss, "entropy": entropy, "kl_bc": kl,
            "approx_kl": approx_kl, "clip_frac": clip_frac, "value_mse": value_mse, **belief_metrics}


def make_rl_optimizer(model: PolyNet, cfg: RLConfig) -> torch.optim.Optimizer:
    """The AdamW of `train_bc.make_optimizer`; with belief heads and `belief_lr_mult` != 1 the heads' parameters form
    their own groups at that multiple of the learning rate (the heads start fresh, the trunk is already trained)."""
    if model.belief_heads is None or cfg.belief_lr_mult == 1.0:
        return make_optimizer(model, cfg)
    groups = []
    for is_belief in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith(BELIEF_PREFIXES) == is_belief]
        lr = cfg.lr * (cfg.belief_lr_mult if is_belief else 1.0)
        groups += [{"params": [p for _, p in ps if p.ndim >= 2], "weight_decay": cfg.weight_decay, "lr": lr},
                   {"params": [p for _, p in ps if p.ndim < 2], "weight_decay": 0.0, "lr": lr}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def kl_coef_at(update: int, cfg: RLConfig) -> float:
    """Linear decay of the KL penalty from `kl_coef` to `kl_coef_end` over `kl_decay_updates` updates."""
    frac = min(1.0, update / max(cfg.kl_decay_updates, 1))
    return cfg.kl_coef + (cfg.kl_coef_end - cfg.kl_coef) * frac


def ent_coef_at(update: int, cfg: RLConfig, last_entropy: float | None = None) -> float:
    """The entropy coefficient of an update. Without `--ent-schedule` it is `ent_coef`. With it: a linear move from
    `ent_coef` to `ent_coef_end` over `ent_decay_updates` updates (a negative `ent_coef_end` keeps `ent_coef`), and 0
    for any update that follows one whose mean policy entropy (`last_entropy`) was above `ent_max` (0 = never)."""
    if not cfg.ent_schedule:
        return cfg.ent_coef
    if cfg.ent_max > 0 and last_entropy is not None and last_entropy > cfg.ent_max:
        return 0.0
    end = cfg.ent_coef if cfg.ent_coef_end < 0 else cfg.ent_coef_end
    frac = min(1.0, update / max(cfg.ent_decay_updates, 1))
    return cfg.ent_coef + (end - cfg.ent_coef) * frac


class WeightEMA:
    """Exponential moving average of a network's state (parameters and buffers, so BatchNorm statistics too):
    `state = decay * state + (1 - decay) * live` after every optimiser step. Integer buffers are copied. Decay 0 tracks
    the live weights exactly."""

    def __init__(self, model: PolyNet, decay: float, state: dict[str, torch.Tensor] | None = None):
        self.decay = decay
        src = state if state is not None else model.state_dict()
        self.state = {k: v.detach().clone() for k, v in src.items()}

    @torch.no_grad()
    def update(self, model: PolyNet) -> None:
        for k, v in model.state_dict().items():
            e = self.state[k]
            if e.is_floating_point():
                e.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(v)


def filter_by_advantage(data: dict[tuple, dict], frac: float) -> int:
    """Keep the `frac` of the trained positions with the largest |normalised advantage| (at least 2; ties broken by
    order): replaces each `d["train"]` and returns the number kept."""
    keys = list(data)
    mags = np.concatenate([np.abs(data[k]["adv_n"][data[k]["train"]]) for k in keys])
    keep = min(len(mags), max(2, math.ceil(frac * len(mags))))
    chosen = np.zeros(len(mags), bool)
    chosen[np.argsort(-mags, kind="stable")[:keep]] = True
    pos = 0
    for k in keys:
        n = len(data[k]["train"])
        data[k]["train"] = data[k]["train"][chosen[pos:pos + n]]
        pos += n
    return keep


def ppo_update(model: PolyNet, ref: PolyNet | None, opt: torch.optim.Optimizer, stores: dict[tuple, SizeStore],
               cfg: RLConfig, update: int, device: torch.device, rng: np.random.Generator,
               extra: list[dict] | None = None, ema: WeightEMA | None = None, ent_coef: float | None = None) -> dict:
    """Several epochs of minibatch PPO over everything the rollout stored. Returns the mean metrics.

    `ref` is the network of the KL penalty (the BC policy, or the moving anchor with `--anchor-every`). `ema` is updated
    after every optimiser step. `ent_coef` is the entropy coefficient (None: `cfg.ent_coef`). With `cfg.adv_filter` < 1
    only that fraction of the positions, those with the largest |advantage|, is trained on.

    `extra` is the hook for another source of positions (interfaces.md section 3: live games joining as an off-policy
    stream). Each item has the layout of `SizeStore.finalize()` (one board size: `tiles`, `globals`, `act_off`,
    `actions`, `chosen`, `logp`, `adv`, `ret`, `train`) and is trained on next to the rollout; nothing produces it yet,
    and the truncated importance ratio for such positions is not built."""
    data = {hw: st.finalize() for hw, st in stores.items() if st.n}
    for i, d in enumerate(extra or []):
        data[("extra", i)] = d
    n_train = sum(len(d["train"]) for d in data.values())
    if n_train < 2:
        return {"positions": n_train, "skipped": True}
    adv_all = np.concatenate([d["adv"][d["train"]] for d in data.values()])
    mean, std = float(adv_all.mean()), float(adv_all.std()) + 1e-8
    for d in data.values():
        d["adv_n"] = ((d["adv"] - mean) / std).astype(np.float32)
        d["ret_c"] = np.clip(d["ret"], 0.0, 1.0)
    ret_mean = float(np.concatenate([d["ret_c"][d["train"]] for d in data.values()]).mean())
    n_kept = filter_by_advantage(data, cfg.adv_filter) if cfg.adv_filter < 1.0 else n_train
    ent_coef = cfg.ent_coef if ent_coef is None else ent_coef
    kl_coef = kl_coef_at(update, cfg)
    policy_scale = 0.0 if update < cfg.critic_warmup else 1.0
    sums: dict[str, float] = defaultdict(float)
    count = 0
    epochs_done = 0
    model.eval()  # BatchNorm on running statistics, see the module docstring
    for _ in range(cfg.epochs):
        chunks = []
        for hw, d in data.items():
            perm = rng.permutation(d["train"])
            chunks += [(hw, perm[i:i + cfg.minibatch]) for i in range(0, len(perm), cfg.minibatch)]
        order = rng.permutation(len(chunks))
        ep_kl, ep_n = 0.0, 0
        for ci in order:
            hw, sel = chunks[ci]
            if len(sel) < 2:
                continue
            d = data[hw]
            tiles, globals_, actions, mask = collate(d, sel, device)
            feats = feature_inputs(collate_features(d, sel, device), model)

            def tens(a: np.ndarray, dtype=None) -> torch.Tensor:
                t = torch.from_numpy(np.ascontiguousarray(a[sel])).to(device)
                return t if dtype is None else t.to(dtype)

            mb = {"mask": mask, "chosen": tens(d["chosen"], torch.long), "old_logp": tens(d["logp"]),
                  "adv": tens(d["adv_n"]), "ret": tens(d["ret_c"])}
            use_belief = cfg.belief_weight > 0 and model.belief_heads is not None and "truth" in d
            if use_belief:
                mb.update(truth=tens(d["truth"]), truth_mask=tens(d["truth_mask"]), truth_g=tens(d["truth_g"]))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16 and device.type == "cuda"):
                out = model(tiles, globals_, actions, mask, **feats,
                            **({"belief": True, "belief_detach": update < cfg.belief_warmup} if use_belief else {}))
                ref_logits = None
                if ref is not None:
                    with torch.no_grad():
                        ref_logits = ref(tiles, globals_, actions, mask, **feature_inputs(feats, ref)).policy_logits
            losses = ppo_loss(out, ref_logits, mb, cfg, kl_coef, policy_scale, ent_coef)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            for k, v in losses.items():
                sums[k] += float(v.detach())
            sums["grad_norm"] += float(gn)
            count += 1
            ep_kl += float(losses["approx_kl"])
            ep_n += 1
        epochs_done += 1
        if cfg.target_kl > 0 and ep_n and ep_kl / ep_n > cfg.target_kl:
            break
    res = {k: v / max(count, 1) for k, v in sums.items()}
    res.update(positions=n_train, positions_kept=n_kept, minibatches=count, epochs_done=epochs_done, kl_coef=kl_coef,
               ent_coef=ent_coef, policy_scale=policy_scale, adv_mean=mean, adv_std=std,
               early_stop=bool(epochs_done < cfg.epochs), ret_mean=ret_mean)
    return res


# ---------------------------------------------------------------------------------------------------------------
# The opponent pool
# ---------------------------------------------------------------------------------------------------------------


class Pool:
    """Pool entries, their networks (for model entries) and the running results against them."""

    def __init__(self, cfg: RLConfig, ref_policy: NetPolicy, ref_info: EntryInfo):
        self.cfg = cfg
        self.base = {**DEFAULT_POOL_WEIGHTS, **parse_pairs(cfg.pool_weights, "pool_weights")}
        self.stats = OppStats(cfg.stats_decay)
        self.info: dict[str, EntryInfo] = {"bc": ref_info}
        self.nets: dict[str, NetPolicy] = {"bc": ref_policy}
        self.snapshots: list[str] = []
        for d in RETAIL_NAMES:
            self.info[f"retail:{d}"] = EntryInfo(f"retail:{d}", f"retail:{d}")

    def entries(self) -> list[str]:
        return [e for e in self.info if self.base.get(e, 0.0) > 0 or e.startswith("self:")]

    def model_entries(self) -> list[str]:
        return [e for e in self.entries() if e in self.nets]

    def weights(self) -> dict[str, float]:
        return pool_weights(self.entries(), self.base, self.stats, self.cfg.beat_target, self.cfg.weight_floor,
                            self.cfg.weight_cap)

    def sample_model_entry(self, rng: np.random.Generator) -> str:
        w = self.weights()
        names = [e for e in self.model_entries() if e in w]
        p = np.array([w[e] for e in names])
        return names[int(rng.choice(len(names), p=p / p.sum()))]

    def add_snapshot(self, entry: str, policy: NetPolicy, info: EntryInfo) -> None:
        self.info[entry], self.nets[entry] = info, policy
        self.snapshots.append(entry)

    def drop_oldest(self, keep: int, in_use: set[str]) -> None:
        while len(self.snapshots) > keep:
            old = self.snapshots.pop(0)
            self.info.pop(old, None)
            if old not in in_use:
                self.nets.pop(old, None)

    def collect_garbage(self, in_use: set[str]) -> None:
        for e in [e for e in self.nets if e.startswith("self:") and e not in self.snapshots and e not in in_use]:
            del self.nets[e]


# ---------------------------------------------------------------------------------------------------------------
# Arms: groups of games in one VecEnv
# ---------------------------------------------------------------------------------------------------------------


class Arm:
    """One VecEnv (see `rl_pool` for what an arm is) with `cap` game slots, of which `n_active` are being played.
    An inactive slot holds a fresh game that is never stepped."""

    def __init__(self, key: tuple, env, cap: int):
        self.key, self.env, self.cap = key, env, cap
        self.kind = key[0]
        self.entry = key[1] if self.kind == "retail" else None  # pool entry of every game of a retail arm
        self.seat = key[2] if self.kind == "retail" else None  # the learner's seat in a retail arm
        self.active = np.zeros(cap, bool)
        self.n_active = 0
        self.quota = 0
        self.learner = np.zeros(cap, np.int64)  # model arm: the learner's seat per game
        self.opp = np.full(cap, "", dtype=object)  # model arm: the opponent entry per game
        self.ready: list[dict] = []  # observation groups of the active games awaiting a decision
        self.traj: dict[int, tuple[tuple, list[int]]] = {}  # slot -> (board size, ids in the SizeStore)
        self.choices: list[tuple[np.ndarray, np.ndarray]] = []
        self.future = None  # the engine step in flight on the executor, if any
        self.steps = 0  # engine steps in the current rollout (bounded lead over the slowest arm, see Collector.tick)


class Collector:
    """Plays the games of all arms and records the learner's decisions."""

    def __init__(self, cfg: RLConfig, pe, learner: NetPolicy, pool: Pool, rng: np.random.Generator,
                 executor: ThreadPoolExecutor, games_log=None):
        self.cfg, self.pe, self.learner, self.pool, self.rng, self.executor = cfg, pe, learner, pool, rng, executor
        self.n_players = parse_players(cfg.n_players)
        self.seat_p = seat_probs(self.n_players)
        self.shards = executor._max_workers
        if cfg.arm_games > 0:
            self.shards = max(self.shards, math.ceil(cfg.games / cfg.arm_games))
        self.arms: dict[tuple, Arm] = {}
        self.stores: dict[tuple, SizeStore] = {}
        self.games_log = games_log
        self.update = 0
        self.results: list[dict] = []  # finished games of the current update
        self.n_learner = 0
        self.n_opp = 0
        self.turns_ema = 60.0  # running mean game length in turns, for the length weight of the win statistics
        self.sizes = [SIZE_IDS[s] for s in cfg.sizes.split(",") if s] if cfg.sizes else None
        self.presets = [PRESET_IDS[s] for s in cfg.presets.split(",") if s] if cfg.presets else None
        self.t_infer = 0.0
        self.t_step = 0.0

    # --- arms ---------------------------------------------------------------------------------------------

    def _make_arm(self, key: tuple) -> Arm:
        cap = arm_capacity(key, self.cfg.games, self.seat_p, self.shards)
        seed_start = int(self.rng.integers(1, TRAIN_SEED_MAX + 1))
        kw = dict(split="train", sizes=self.sizes, presets=self.presets, turn_limit=self.cfg.turn_limit,
                  threads=1, auto_reset=True, **mode_kw(MODES[self.cfg.mode]), **feature_kw(self.cfg.features),
                  **({"truth": True} if self.cfg.belief_weight > 0 else {}))
        if key[0] == "retail":
            difficulty = RETAIL_NAMES.index(key[1].split(":")[1])
            seat = key[2]
            kw["n_players"] = retail_player_weights(self.n_players, seat)
            kw["seat_difficulties"] = [LEARNER_DIFFICULTY if s == seat else difficulty for s in range(MAX_SEATS)]
            env = self.pe.VecEnv(cap, seed_start, [seat], **kw)
        else:
            kw["n_players"] = self.n_players
            kw["seat_difficulties"] = [LEARNER_DIFFICULTY] * MAX_SEATS
            env = self.pe.VecEnv(cap, seed_start, "all", **kw)
        return Arm(key, env, cap)

    def set_quotas(self, quotas: dict[tuple, int]) -> None:
        for key, q in quotas.items():
            if q > 0 and key not in self.arms:
                self.arms[key] = self._make_arm(key)
            if key in self.arms:
                self.arms[key].quota = q
        for key, arm in self.arms.items():
            if key not in quotas:
                arm.quota = 0
            if arm.n_active < arm.quota:
                self._settle(arm)  # observe() must not run while the arm's engine is stepping
                self._activate(arm)

    def _assign_roles(self, arm: Arm, g: dict, only: set[int] | None = None) -> None:
        for slot, n in zip(np.asarray(g["game"]).tolist(), np.asarray(g["n_players"]).tolist()):
            if only is None or slot in only:
                arm.learner[slot] = int(self.rng.integers(n))
                arm.opp[slot] = self.pool.sample_model_entry(self.rng)

    def _activate(self, arm: Arm) -> None:
        need = arm.quota - arm.n_active
        taken = 0
        groups = arm.env.observe()
        # pick the slots to activate at random: observe() groups the games by map size, so taking them in order
        # would start an arm on its smallest maps only
        if need <= 0 or not groups:
            return
        idle_slots = np.concatenate([np.asarray(g["game"])[~arm.active[np.asarray(g["game"])]] for g in groups])
        chosen = list(self.rng.permutation(idle_slots)[:need])
        for g in groups:
            slots = np.asarray(g["game"])
            idle = np.nonzero(~arm.active[slots] & np.isin(slots, chosen))[0]
            if not len(idle):
                continue
            sub = take(g, idle)
            arm.active[np.asarray(sub["game"])] = True
            taken += len(idle)
            if arm.kind == "model":
                self._assign_roles(arm, sub)
            arm.ready.append(sub)
        arm.n_active += taken

    # --- one tick: choose for every waiting game, step the engines -----------------------------------------

    def _harvest(self, block: bool) -> None:
        """Take in the engine steps that have finished; with `block`, first wait until at least one has."""
        pending = [arm for arm in self.arms.values() if arm.future is not None]
        if block and pending:
            t0 = time.perf_counter()
            wait([arm.future for arm in pending], return_when=FIRST_COMPLETED)
            self.t_step += time.perf_counter() - t0
        for arm in pending:
            if arm.future.done():
                self._settle(arm)

    def _settle(self, arm: Arm) -> None:
        """Wait for the arm's step in flight, if any, and take in its result."""
        if arm.future is None:
            return
        fut, arm.future = arm.future, None
        groups, finished = fut.result()
        self._after_step(arm, groups, finished)

    def _may_step(self) -> list[Arm]:
        """Arms with games awaiting a decision, no step in flight and no more than `max_lead` steps ahead."""
        live = [a for a in self.arms.values() if a.future is not None or a.ready]
        if not live:
            raise RuntimeError("no game is waiting for a decision")
        floor = min(a.steps for a in live)
        return [a for a in live if a.future is None and a.ready and a.steps <= floor + self.cfg.max_lead]

    def tick(self) -> None:
        """Choose for the waiting games of every arm that may step, and start those steps on the executor.

        Arms step asynchronously: a tick does not wait for every arm's step to come back, only takes in those that
        have, so a slow step (a retail AI playing out a turn on a large map) holds up its own arm only. To keep
        the decisions from tilting towards the arms with fast games, an arm may run at most `cfg.max_lead` steps
        ahead of the slowest arm in the rollout (0 gives lockstep)."""
        cfg = self.cfg
        self._harvest(block=False)
        go = self._may_step()
        if cfg.min_infer > 0 and go:
            # many small arms come back one by one: gather a batch worth a GPU call, but never wait long
            deadline = time.perf_counter() + cfg.gather_ms / 1000
            while sum(obs_len(g) for a in go for g in a.ready) < cfg.min_infer:
                pending = [a.future for a in self.arms.values() if a.future is not None]
                left = deadline - time.perf_counter()
                if not pending or left <= 0:
                    break
                t0 = time.perf_counter()
                wait(pending, timeout=left, return_when=FIRST_COMPLETED)
                self.t_step += time.perf_counter() - t0
                self._harvest(block=False)
                go = self._may_step()
        if not go:
            self._harvest(block=True)
            return
        parts: dict[tuple, list] = defaultdict(list)
        for arm in go:
            arm.choices = []
            for g in arm.ready:
                bucket = pad_bucket(tuple(g["tiles"].shape[1:3])) if cfg.pad_sizes else tuple(g["tiles"].shape[1:3])
                if arm.kind == "retail":
                    parts[("learner", bucket)].append((arm, g))
                    continue
                slots = np.asarray(g["game"])
                is_learner = np.asarray(g["seat"]) == arm.learner[slots]
                keys = np.where(is_learner, "learner", arm.opp[slots])
                for key in np.unique(keys.astype(str)):
                    sel = np.nonzero(keys == key)[0]
                    parts[(str(key), bucket)].append((arm, g if len(sel) == len(keys) else take(g, sel)))
        t0 = time.perf_counter()
        for (key, _), lst in parts.items():
            merged = merge_padded([sub for _, sub in lst])
            if key == "learner":
                ch = self.learner.choose(merged, temperature=cfg.temperature)
            else:
                ch = self.pool.nets[key].choose(merged, temperature=cfg.opp_temperature)
            pos = 0
            for arm, sub in lst:
                b = obs_len(sub)
                s = slice(pos, pos + b)
                pos += b
                arm.choices.append((np.asarray(sub["game"], dtype=np.int32), ch.index[s]))
                if key == "learner":
                    hw = tuple(sub["tiles"].shape[1:3])
                    store = self.stores.setdefault(hw, SizeStore())
                    ids = store.add(sub, ch.index[s], ch.logp[s], ch.value[s])
                    for slot, i in zip(np.asarray(sub["game"]).tolist(), ids.tolist()):
                        arm.traj.setdefault(slot, (hw, []))[1].append(i)
                    self.n_learner += b
                else:
                    self.n_opp += b
        self.t_infer += time.perf_counter() - t0
        for arm in go:
            games = np.concatenate([c[0] for c in arm.choices])
            picks = np.concatenate([c[1] for c in arm.choices]).astype(np.int32)
            arm.future = self.executor.submit(arm.env.step, games, picks)
            arm.steps += 1
            arm.ready = []
            arm.choices = []

    def _after_step(self, arm: Arm, groups: list[dict], finished: list[dict]) -> None:
        reset: set[int] = set()
        for f in finished:
            slot = int(f["game"])
            self._on_finished(arm, slot, f)
            reset.add(slot)
            if arm.n_active > arm.quota:
                arm.active[slot] = False
                arm.n_active -= 1
        for g in groups:
            slots = np.asarray(g["game"])
            keep = arm.active[slots]
            if not keep.any():
                continue
            if not keep.all():
                g = take(g, np.nonzero(keep)[0])
            if arm.kind == "model" and reset:
                self._assign_roles(arm, g, reset)
            arm.ready.append(g)

    # --- trajectories ------------------------------------------------------------------------------------

    def _flush(self, arm: Arm, slot: int, outcome: float, terminal: bool) -> None:
        tr = arm.traj.pop(slot, None)
        if tr is None:
            return
        hw, ids = tr
        ids = np.asarray(ids)
        store = self.stores[hw]
        adv, ret = compute_gae(store.value[ids], outcome, terminal, self.cfg.gamma, self.cfg.lam)
        store.mark(ids[: len(adv)], adv, ret)

    def _on_finished(self, arm: Arm, slot: int, f: dict) -> None:
        n = int(f["n_players"])
        seat = arm.seat if arm.kind == "retail" else int(arm.learner[slot])
        entry = arm.entry if arm.kind == "retail" else str(arm.opp[slot])
        capped = bool(f["capped"])
        winners = [int(w) for w in f["winners"]]
        outcome = 0.5 if (capped or not winners) else (1.0 if seat in winners else 0.0)
        if capped and self.cfg.drop_capped:
            arm.traj.pop(slot, None)
        else:
            self._flush(arm, slot, outcome, terminal=True)
        turns = int(f["turns"])
        self.turns_ema = 0.99 * self.turns_ema + 0.01 * turns
        weight = min(10.0, max(0.1, turns / self.turns_ema))
        self.pool.stats.record(entry, n, outcome, weight)
        rec = {"update": self.update, "seed": int(f["seed"]), "n_players": n, "learner_seat": seat, "opponent": entry,
               "opponent_policy_id": self.pool.info[entry].policy_id if entry in self.pool.info else entry,
               "outcome": outcome, "capped": capped, "turns": turns, "weight": weight, "decisions": int(f["decisions"])}
        self.results.append(rec)
        if self.games_log is not None:
            self.games_log.write(json.dumps(rec) + "\n")

    def cut(self) -> None:
        """End of a rollout: flush every unfinished trajectory as truncated."""
        for arm in self.arms.values():
            for slot in list(arm.traj):
                self._flush(arm, slot, 0.0, terminal=False)

    def in_use(self) -> set[str]:
        used = set()
        for arm in self.arms.values():
            if arm.kind == "model":
                used.update(str(e) for e in arm.opp[arm.active])
        return used

    def collect(self, steps: int) -> None:
        """Play until the learner has made `steps` decisions."""
        self.stores = {}
        self.results = []
        self.n_learner = self.n_opp = 0
        self.t_infer = self.t_step = 0.0
        for arm in self.arms.values():
            arm.steps = 0
        while self.n_learner < steps:
            self.tick()
        for arm in self.arms.values():  # games finishing in the steps still in flight end in this rollout
            self._settle(arm)
        self.cut()


# ---------------------------------------------------------------------------------------------------------------
# Checkpoints and the run
# ---------------------------------------------------------------------------------------------------------------


def save_rl_checkpoint(path: Path, model: PolyNet, update: int, opt: torch.optim.Optimizer, rl_state: dict,
                       ema: WeightEMA | None = None) -> None:
    state = {"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": update,
             "encoding_version": S.ENCODING_VERSION, "optimizer": opt.state_dict(), "rl": rl_state}
    if ema is not None:
        state["ema"] = ema.state
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_plain_checkpoint(path: Path, model: PolyNet, update: int, ema: WeightEMA | None = None) -> None:
    """`train_bc.save_checkpoint`, plus the averaged weights under `ema` when there are any (loaders ignore the key)."""
    if ema is None:
        save_checkpoint(path, model, update)
        return
    state = {"model": model.state_dict(), "model_config": model.cfg.to_dict(), "step": update,
             "encoding_version": S.ENCODING_VERSION, "ema": ema.state}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_ema_checkpoint(path: Path, model: PolyNet, update: int, ema: WeightEMA) -> None:
    """The averaged weights as a plain checkpoint of the same model configuration (arena, h2h and serve load it as is)."""
    state = {"model": ema.state, "model_config": model.cfg.to_dict(), "step": update,
             "encoding_version": S.ENCODING_VERSION}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def frozen_copy(model: PolyNet) -> PolyNet:
    net = copy.deepcopy(model).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def anchor_due(done: int, cfg: RLConfig) -> bool:
    """True after the update that brings the count of finished updates to a multiple of `--anchor-every`."""
    return cfg.anchor_every > 0 and done % cfg.anchor_every == 0


def checkpoint_step(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False).get("step", 0))


def outcome_summary(results: list[dict]) -> dict:
    """Per pool entry: games, wins, draws, losses, the mean outcome of one update (`score`) and the same weighted by the
    games' length weights (`score_w`, corrected for finished games being biased to short ones)."""
    by: dict[str, dict] = {}
    for r in results:
        cell = by.setdefault(r["opponent"], {"games": 0, "wins": 0, "draws": 0, "losses": 0, "score": 0.0,
                                             "score_w": 0.0, "_w": 0.0})
        cell["games"] += 1
        cell["score"] += r["outcome"]
        cell["score_w"] += r["weight"] * r["outcome"]
        cell["_w"] += r["weight"]
        cell["wins" if r["outcome"] == 1.0 else "draws" if r["outcome"] == 0.5 else "losses"] += 1
    for cell in by.values():
        cell["score"] /= cell["games"]
        cell["score_w"] /= cell.pop("_w")
    return by


def train(cfg: RLConfig) -> dict:
    import polyai_engine as pe

    threads = max(2, cfg.threads)
    torch_threads = max(1, min(cfg.torch_threads, threads - 1))
    torch.set_num_threads(torch_threads)
    workers = max(1, threads - torch_threads)
    device = torch.device(cfg.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cap = check_gpu_memory(cfg.need_gb, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if cfg.temperature <= 0 or cfg.opp_temperature <= 0:
        raise SystemExit("sampling temperatures must be positive")

    run_dir = Path(cfg.runs_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pool").mkdir(exist_ok=True)
    init_path = resolve_checkpoint(cfg.init)
    ref_path = resolve_checkpoint(cfg.ref or cfg.init)

    model = load_checkpoint(init_path).to(device).to(memory_format=torch.channels_last).eval()
    ref = load_checkpoint(ref_path).to(device).to(memory_format=torch.channels_last).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if cfg.features:  # the learner gets the feature inputs, zero-initialised; the reference stays as it is
        model = add_features(model).to(memory_format=torch.channels_last).eval()
    if {**model.cfg.to_dict(), "features": False} != {**ref.cfg.to_dict(), "features": False}:  # features may differ
        raise SystemExit("the init and ref checkpoints have different model configurations")
    if cfg.belief_weight > 0:  # fresh belief heads on the learner's trunk (the reference and the pool stay plain)
        model = add_belief(model, cfg.belief_hidden).to(memory_format=torch.channels_last).eval()
    opt = make_rl_optimizer(model, cfg)
    update0 = 0
    rl_state: dict = {}
    if cfg.resume:
        last = run_dir / "ckpt_last.pt"
        if not last.exists():
            raise SystemExit(f"cannot resume: {last} does not exist")
        state = torch.load(last, map_location=device, weights_only=False)
        if state["model_config"] != model.cfg.to_dict():
            raise SystemExit(f"cannot resume: {last} was written by a different model configuration")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        update0 = int(state["step"])
        rl_state = state.get("rl", {})
        print(f"resumed {cfg.name} from update {update0}", flush=True)

    ema = WeightEMA(model, cfg.ema, state.get("ema") if cfg.resume else None) if cfg.ema > 0 else None
    # the KL reference: `ref` (the BC policy) until --anchor-every replaces it by a frozen copy of the learner
    anchor = {"net": ref, "update": 0}
    if cfg.resume and cfg.anchor_every > 0 and "anchor" in rl_state:
        anchor["net"] = frozen_copy(model)
        anchor["net"].load_state_dict(rl_state["anchor"]["model"])
        anchor["update"] = int(rl_state["anchor"]["update"])
    last_entropy: float | None = rl_state.get("last_entropy") if cfg.resume else None

    rng = np.random.default_rng(cfg.seed + 7919 * update0)
    torch.manual_seed(cfg.seed + update0)
    # with overlap the rollouts are played by a copy of the learner (the weights before the update running alongside),
    # so the PPO update can change `model` meanwhile; the stored logp are the copy's, so the ratio stays correct
    actor = copy.deepcopy(model).eval() if cfg.overlap else model
    if actor is not model:
        for p in actor.parameters():
            p.requires_grad_(False)
    ppo_rng = np.random.default_rng(cfg.seed + 7919 * update0 + 1) if cfg.overlap else rng
    fast = dict(infer_graphs=cfg.infer_graphs, infer_compile=cfg.infer_compile)
    learner = NetPolicy(actor, f"{cfg.name}@{update0}", device, max_batch=cfg.max_batch, seed=cfg.seed + update0, **fast)
    ref_id = policy_id_of(ref_path, checkpoint_step(ref_path))
    pool = Pool(cfg, NetPolicy(ref, ref_id, device, max_batch=cfg.max_batch, seed=cfg.seed + 1, **fast),
                EntryInfo("bc", ref_id, str(ref_path)))

    def add_snapshot(update: int, from_file: Path | None = None) -> None:
        entry = f"self:u{update:05d}"
        path = from_file or (run_dir / "pool" / f"snap_u{update:05d}.pt")
        if from_file is None:
            save_checkpoint(path, model, update)
        net = load_checkpoint(path)
        pid = f"{cfg.name}@{update}" if from_file is None or update > 0 or init_path == ref_path else \
            policy_id_of(init_path, checkpoint_step(init_path))
        pool.add_snapshot(entry, NetPolicy(net, pid, device, max_batch=cfg.max_batch,
                                           seed=cfg.seed + update + 2, **fast), EntryInfo(entry, pid, str(path)))

    if cfg.resume:
        if "stats" in rl_state:
            pool.stats.load_state_dict(rl_state["stats"])
        for snap in rl_state.get("snapshots", []):
            p = Path(snap["path"])
            if p.exists():
                add_snapshot(int(snap["update"]), p)
    elif init_path != ref_path:
        # the learner starts away from the BC policy (a later stage): its start is an opponent of its own
        p0 = run_dir / "pool" / "snap_u00000.pt"
        save_checkpoint(p0, model, 0)
        net = load_checkpoint(p0)
        pid = policy_id_of(init_path, checkpoint_step(init_path))
        pool.add_snapshot("self:u00000", NetPolicy(net, pid, device, max_batch=cfg.max_batch,
                                                   seed=cfg.seed + 2, **fast), EntryInfo("self:u00000", pid, str(p0)))

    (run_dir / "config.json").write_text(json.dumps(
        {"train": asdict(cfg), "model": model.cfg.to_dict(), "params": model.n_params(),
         "encoding_version": S.ENCODING_VERSION, "started_at_update": update0,
         "init_policy_id": policy_id_of(init_path, checkpoint_step(init_path)), "ref_policy_id": ref_id}, indent=2))
    metrics = (run_dir / "metrics.jsonl").open("a", buffering=1)
    games_log = (run_dir / "games.jsonl").open("a", buffering=1) if cfg.log_games else None

    executor = ThreadPoolExecutor(max_workers=workers)
    if cfg.collector not in ("python", "rust"):
        raise SystemExit("--collector is python or rust")
    collector_cls = Collector
    if cfg.collector == "rust":
        from .rl_collect import RustCollector as collector_cls
    col = collector_cls(cfg, pe, learner, pool, rng, executor, games_log)
    print(f"{cfg.name}: {model.cfg.name or 'model'} {model.n_params() / 1e6:.2f}M params, {cfg.games} games in flight, "
          f"{cfg.rollout_steps} learner decisions per update, {workers} env threads + {torch_threads} torch threads, "
          f"device {device}, cap {cap / 1024**3:.2f} GiB", flush=True)

    seat_p = col.seat_p

    def snapshot_state(update: int) -> dict:
        st = {"stats": pool.stats.state_dict(),
              "snapshots": [{"update": int(e.split("u")[-1]), "path": pool.info[e].path} for e in pool.snapshots]}
        if anchor["update"] > 0:
            st["anchor"] = {"update": anchor["update"], "model": anchor["net"].state_dict()}
        if cfg.ent_schedule and last_entropy is not None:
            st["last_entropy"] = last_entropy
        return st

    def begin_rollout(update: int, played_by: int) -> dict:
        """Plan the rollout of `update` (the collector must be idle) and return the pool weights it uses."""
        weights = pool.weights()
        col.update = update
        col.set_quotas(plan_quotas(weights, cfg.games, seat_p, col.shards))
        if actor is not model:
            actor.load_state_dict(model.state_dict())
        learner.policy_id = f"{cfg.name}@{played_by}"
        return weights

    def run_rollout() -> float:
        t = time.perf_counter()
        col.collect(cfg.rollout_steps)
        return time.perf_counter() - t

    roll_ex = ThreadPoolExecutor(max_workers=1) if cfg.overlap else None
    last_rec: dict = {}
    t_start = time.perf_counter()
    weights = begin_rollout(update0, update0) if update0 < cfg.updates else {}
    roll_secs = run_rollout() if update0 < cfg.updates else 0.0
    for update in range(update0, cfg.updates):
        # this update's rollout is complete and the collector idle: take its results before the next one starts
        stores, results = col.stores, col.results
        n_learner, n_opp, t_infer, t_step = col.n_learner, col.n_opp, col.t_infer, col.t_step
        behaviour_id = learner.policy_id
        nxt = None
        if roll_ex is not None and update + 1 < cfg.updates:
            next_weights = begin_rollout(update + 1, update)
            nxt = roll_ex.submit(run_rollout)
        t1 = time.perf_counter()
        res = ppo_update(model, anchor["net"], opt, stores, cfg, update, device, ppo_rng, ema=ema,
                         ent_coef=ent_coef_at(update, cfg, last_entropy))
        t2 = time.perf_counter()
        if cfg.anchor_every > 0:
            res["anchor"] = "ref" if anchor["update"] == 0 else f"self:u{anchor['update']:05d}"
        if "entropy" in res:
            last_entropy = res["entropy"]
        next_roll_secs = nxt.result() if nxt is not None else 0.0
        res["overlap_wait_secs"] = time.perf_counter() - t2  # PPO waiting for the next rollout (overlap only)
        done = update + 1
        if anchor_due(done, cfg):  # the next update's KL reference is the learner as it is now
            anchor["net"], anchor["update"] = frozen_copy(model), done
        if done % cfg.snapshot_every == 0:
            add_snapshot(done)
            pool.drop_oldest(cfg.max_snapshots, col.in_use())
        pool.collect_garbage(col.in_use())

        outcomes = outcome_summary(results)
        rec = {
            "kind": "update", "update": done, "learner_policy_id": f"{cfg.name}@{done}",
            "behaviour_policy_id": behaviour_id,
            "learner_decisions": n_learner, "opponent_decisions": n_opp,
            "games_finished": len(results), "capped_games": sum(r["capped"] for r in results),
            "mean_turns": float(np.mean([r["turns"] for r in results])) if results else None,
            "rollout_secs": roll_secs, "update_secs": t2 - t1, "infer_secs": t_infer, "step_secs": t_step,
            "decisions_per_sec": n_learner / max(roll_secs, 1e-9),
            "all_decisions_per_sec": (n_learner + n_opp) / max(roll_secs, 1e-9),
            "win_rate": outcomes, "running_win_rate": {e: pool.stats.rate(e) for e in pool.entries()},
            "pool_weights": weights, "pool_policy_ids": {e: pool.info[e].policy_id for e in pool.entries()},
            **res,
        }
        if device.type == "cuda":
            rec["max_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        metrics.write(json.dumps(rec) + "\n")
        last_rec = rec
        wr = " ".join(f"{e.split(':')[-1]}={c['score']:.2f}/{c['games']}" for e, c in sorted(outcomes.items()))
        print(f"update {done}/{cfg.updates} games {rec['games_finished']} dec/s {rec['decisions_per_sec']:.0f} "
              f"roll {rec['rollout_secs']:.1f}s upd {rec['update_secs']:.1f}s | kl_bc {res.get('kl_bc', float('nan')):.4f} "
              f"ent {res.get('entropy', float('nan')):.3f} clip {res.get('clip_frac', float('nan')):.3f} "
              f"vloss {res.get('value_loss', float('nan')):.3f} | {wr}", flush=True)
        if done % cfg.ckpt_every == 0 and done < cfg.updates:
            save_rl_checkpoint(run_dir / "ckpt_last.pt", model, done, opt, snapshot_state(done), ema)
            save_plain_checkpoint(run_dir / f"ckpt_u{done:05d}.pt", model, done, ema)
            if ema is not None:
                save_ema_checkpoint(run_dir / f"ckpt_ema_u{done:05d}.pt", model, done, ema)
        if nxt is not None:
            weights, roll_secs = next_weights, next_roll_secs
        elif done < cfg.updates:
            weights = begin_rollout(done, done)
            roll_secs = run_rollout()

    if roll_ex is not None:
        roll_ex.shutdown()
    done = max(update0, cfg.updates)
    save_rl_checkpoint(run_dir / "ckpt_last.pt", model, done, opt, snapshot_state(done), ema)
    save_plain_checkpoint(run_dir / "ckpt_final.pt", model, done, ema)
    if ema is not None:
        save_ema_checkpoint(run_dir / "ckpt_ema_final.pt", model, done, ema)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    summary = {"name": cfg.name, "updates": done, "wall_secs": time.perf_counter() - t_start, "peak_mem_mb": peak,
               "last": last_rec}
    metrics.write(json.dumps({"kind": "summary", **{k: v for k, v in summary.items() if k != "last"}}) + "\n")
    metrics.close()
    if games_log is not None:
        games_log.close()
    executor.shutdown()
    return summary


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> RLConfig:
    d = RLConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(RLConfig):
        default = getattr(d, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            if f.name == "resume":
                ap.add_argument(flag, action="store_true", help="continue the run NAME from its ckpt_last.pt")
            elif default:
                ap.add_argument("--no-" + f.name.replace("_", "-"), dest=f.name, action="store_false")
            else:
                ap.add_argument(flag, dest=f.name, action="store_true")
        else:
            ap.add_argument(flag, type=type(default), default=default,
                            required=f.name in ("init",) and "--resume" not in (argv or sys.argv))
    a = ap.parse_args(argv)
    cfg = RLConfig(**vars(a))
    if cfg.mode not in MODES:
        raise SystemExit(f"--mode is one of {', '.join(MODES)}")
    if cfg.resume and not cfg.init:  # a resumed run needs its stored init / ref paths again
        stored = Path(cfg.runs_dir) / cfg.name / "config.json"
        if not stored.exists():
            raise SystemExit(f"cannot resume: {stored} does not exist")
        old = json.loads(stored.read_text())["train"]
        cfg.init, cfg.ref = old["init"], cfg.ref or old.get("ref", "")
    return cfg


def main(argv: list[str] | None = None) -> int:
    summary = train(parse_args(argv))
    print(json.dumps({k: v for k, v in summary.items() if k != "last"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
