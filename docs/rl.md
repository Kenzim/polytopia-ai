# PPO self-play (`polyai.rl`)

Phase B of design v1.1: the imitation model plays games against a pool of opponents and is improved with PPO. The
only reward is the result of the game: win 1, loss 0, and a draw (a game that reaches the 150-turn compute cap) 0.5.
Games are won by elimination only; there is no score reward and no shaping. `--drop-capped` leaves capped games out of
training instead of rewarding them 0.5.

Files: `python/polyai/rl.py` (config, GAE, PPO loss and update, collector, run loop, CLI), `python/polyai/rl_pool.py`
(batched network policy, observation batching, opponent pool statistics, the plan of games per VecEnv),
`python/tests/test_rl.py`.

## Running

```
cd /kvm/polytopia/ai/python
TMPDIR=/kvm/tmp/tmp nice -n 10 /kvm/polytopia/.venv/bin/python -m polyai.rl \
    --name rl1 --init /kvm/polytopia-ai-data/runs/bc3_b6c96_v11 --updates 200
# continue a stopped run (the new --updates is the new total)
... -m polyai.rl --name rl1 --resume --updates 400
```

`--init` is a run directory (its `ckpt_final.pt`, else `ckpt_last.pt`) or a checkpoint file. `--ref` is the behaviour
cloning policy for the KL penalty and the `bc` pool entry; it defaults to `--init`. When `--init` is a later stage
(not the BC policy) pass the BC checkpoint as `--ref`, and the start of the run joins the pool as `self:u00000`.
`--device cpu` for development; on the GPU the run checks free memory first (`--need-gb`, default 2.5) and caps itself
with `check_gpu_memory`, like `train_bc`.

Run directory `/kvm/polytopia-ai-data/runs/<name>/`:

| file | content |
|---|---|
| `config.json` | the `RLConfig`, the model config, parameter count, `init_policy_id`, `ref_policy_id` |
| `metrics.jsonl` | one `kind: update` line per update, a `kind: summary` line at the end |
| `games.jsonl` | one line per finished game: seed, players, learner seat, opponent entry, `opponent_policy_id`, outcome, capped, turns, length weight |
| `ckpt_last.pt` | model, optimiser, pool statistics and snapshot list (what `--resume` loads) |
| `ckpt_u<N>.pt`, `ckpt_final.pt` | plain checkpoints (`train_bc` format; `polyai.policy.ModelPolicy` and the arena load them) |
| `ckpt_ema_u<N>.pt`, `ckpt_ema_final.pt` | with `--ema`: the averaged weights as plain checkpoints (same loaders) |
| `pool/snap_u<N>.pt` | frozen copies of the learner that serve as opponents |

The policy id of the learner after update N is `<name>@N`; every pool entry has one too (`pool_policy_ids` in the
metrics, `opponent_policy_id` in `games.jsonl`).

## How a rollout works

Every game has one learner seat; the learner is the network being trained, with the Normal handicap. Its trajectory
is the sequence of its own commands, and credit is per command. The seat is uniform over the seats of the game
(`P(seat | n)` is 1/n), so the setup mix over players is the requested `n_players` (default `2:55,3:25,4:20`) and
all sizes and presets of the engine are used (`--sizes`, `--presets` restrict them). Seeds are drawn from 1..=100000
by `VecEnv(split="train")`.

VecEnv fixes the model seats of all its games, so the games are grouped in arms (`rl_pool` has the plan):

- **retail arm** `(retail:<difficulty>, learner seat s)`: `model_seats=[s]`, and `seat_difficulties` gives seat s
  Normal and every other seat that difficulty. The engine plays the opponents.
- **model arm** (a few shards, for threads): `model_seats="all"`. Python decides per game which seat is the learner
  and which pool network (`bc` or a `self:` snapshot) plays every other seat, and routes each decision to that network.

Decisions of the same map size are merged across arms and networks and run in batches of `--max-batch`. Every call goes
through `Policy.choose(obs, temperature=...)` of `polyai.policy` (`NetPolicy` is a `ModelPolicy` around a network held
in memory), which returns the index, the log-probability of the sampled move under `softmax(logits / T)`, and the
value. The log-probability is stored and is the old policy of the PPO ratio. The learner samples with `--temperature`,
model opponents with `--opp-temperature` (both default 1). The PPO loss uses the same `softmax(logits / T)`.

Win statistics: finished games are biased to short ones (a game ends at a rate proportional to 1 / length, and short
games are mostly early losses), so the running win rate weights each game by its turns over a running mean of game
length. `metrics.jsonl` has both the plain `score` and the weighted `score_w` per opponent. In the first updates all
games start together, so the numbers are dominated by the shortest games; give them 5 to 10 updates.

## The opponent pool

Entries: `retail:easy|normal|hard|crazy`, `bc`, and `self:u<update>` (a frozen copy of the learner every
`--snapshot-every` updates, newest `--max-snapshots` kept; the `self` weight is shared by them equally). Base weights
`--pool-weights` (default `retail:easy=0.05,retail:normal=0.15,retail:hard=0.15,retail:crazy=0.15,bc=0.15,self=0.35`;
an entry at 0 is off). The sampling weight is the base weight times `clip((1 - win rate) / (1 - beat_target), floor,
cap)` with `beat_target` 0.7, floor 0.1, cap 2.0: opponents the learner beats less than 70% of the time are played
more. The win rate is a running decayed average per opponent and player count (`--stats-decay` per game, a prior of 5
games at parity), mapped so that chance (1/n) counts as 0.5 whatever the number of players, so a four-player win rate
of 25% is parity. The weights are recomputed at the start of every update; the games in flight keep their opponent
until they end.

## PPO

Clipped objective (`--clip` 0.2), GAE (`--gamma` 1.0, `--lam` 0.95) on the learner's commands, advantages normalised
over the update. The value is the winner head's probability that the learner wins (slot 0 of the softmax, as
`model.win_probability`) and is trained with binary cross-entropy against the GAE return (`--vf-coef` 0.5); a draw is
worth 0.5. An unfinished game at the end of a rollout is cut: its last command serves as the bootstrap value and is not
trained, and its later moves start a new trajectory in the next update. Entropy bonus `--ent-coef`. The KL term is
`KL(policy || BC policy)` over the legal actions with a coefficient that decays linearly from `--kl-coef` to
`--kl-coef-end` over `--kl-decay-updates` updates. The first `--critic-warmup` updates train the value head only
(the trunk is shared, so the policy still moves a little). Epochs stop early when the mean approximate KL to the
behaviour policy passes `--target-kl`.

BatchNorm uses its running statistics in rollouts and in the update (eval mode), so the policy that acted and the
policy being updated are the same function before the first step; the statistics stay at their imitation values and
the affine parameters train. A test checks that the stored log-probabilities equal a fresh forward pass.

`ppo_update(..., extra=[...])` takes further batches in the layout of `SizeStore.finalize()`: the hook for the live
games stream of interfaces.md section 3. Nothing fills it yet, and the truncated importance ratio for off-policy
positions is not built.

### Credit per game turn (`--credit-per-turn`, off by default)

A turn is 25 to 30 commands, so with `--lam` 0.95 applied per command the game result reaches a decision 10 turns
earlier at about 1e-6 of its size. With this option gamma and lambda apply once per game turn: the step from one
command to the next is discounted by `gamma ** k` and `lam ** k`, `k` the number of game turns between them (0 within
a turn, so a turn's commands share credit fully). Both collectors store each row's game turn for this.

### Privileged critic (`--priv-critic`, off by default)

A separate value network computes the advantages. It starts as a copy of the learner (so its first estimates are the
learner's own) with extra, zero-initialised inputs for the truth under the fog: the labels of `docs/belief.md` embedded
on the hidden tiles only, and the scalar truth. Before each update it values every stored row; `trajectory_gae`
recomputes each trajectory's `adv` and `ret` from those values with the trajectory's own outcome, terminal flag and
per-turn option (the stores keep `traj`, `traj_out` and `traj_term` for this). It is then trained on every minibatch
against those returns, with its own AdamW at `--critic-lr` (0: `--lr`). The policy never sees the truth and is used in
play as before; its own value head still trains, on the critic's returns. The critic is saved in `ckpt_last.pt` (for
`--resume`) and as `critic_final.pt`. Metrics add `critic_loss`, `critic_value_mean`, `policy_value_mean` and
`critic_policy_value_corr`.

### Value rows from the opponents' side (`--value-opp W`, off by default)

The learner wins 65 to 75% of its training games and only its own positions are trained on, so the value head learns
"the side to move usually wins" (see the calibration table below). With `--value-opp W` (> 0) the positions of the
other seats join the value loss as well, in games against our own networks: pool snapshots `self:u<N>`, `bc` and the
`--extra-opponents`. Those positions are already encoded for the opponents' own inference, from their seat, so they
cost no extra engine work. The retail AI's seats are never encoded and are left out.

- **Target.** The seat's own result, Monte Carlo: 1 win, 0 loss, 0.5 for a capped game (left out with
  `--drop-capped`), as the learner's outcome. A game lasts several updates, so a slot's opponent rows are held across
  rollouts until the game ends and then join the update of the rollout it ended in; a run that stops (or resumes)
  loses the rows of the games in play. TD(lambda) was not used: it needs a value for the opponent's positions, and the
  opponent network's own estimate is the miscalibrated one this option is meant to fix, while the learner's would
  cost an extra forward pass over every held row.
- **Which rows.** `--value-opp-frac F` (default 0.25) of the opponent decisions are kept, chosen by a hash of the
  game seed, the seat and the decision's count in that game (`keep_value_row`), so both collectors keep the same rows
  and no random stream changes. The decisions within a turn are strongly correlated, so a quarter keeps most of the
  information at a quarter of the memory and update time. A row holds the tiles, the globals and the engine feature
  rows (no legal lists: the value reads the trunk only), about 13 to 26 KB on Tiny to Normal maps.
- **Loss.** The rows enter the value loss only, through `value_logits` (the value path of the network without the
  policy head). Each epoch spreads them evenly over its optimiser steps, with a random stream of their own, so the
  learner's minibatches are the same as without the option; a step adds `vf_coef * W * sum(BCE of its rows) / (its
  learner positions)`, so `W` 1 weighs an opponent row like a learner row. They take no part in the policy, entropy and
  KL terms, the advantage normalisation, `--adv-filter`, the critic warm-up's policy scale or `--priv-critic` (which
  trains on the learner's rows only).
- **Metrics.** `value_opp_positions` (rows in this update), `value_opp_trained` (row passes over the epochs),
  `value_opp_loss` (their mean binary cross-entropy), `value_opp_ret_mean` and `value_opp_pending` (rows held for games
  still in play); the stdout line adds `vopp <rows> held <pending>`.

With the option off nothing is held and the update is unchanged (same samples, same random streams); the Rust
collector is then built without the new arguments, so an older engine module still runs it.

Cost, measured 2026-11-20 on the shared RTX 3060 (b6c96 with features): a value row costs about as much in the update as
a learner row (the trunk dominates), so the update grows by the share of value rows. The live loop's rollout (32,768
learner and about 29,700 model-opponent decisions per update, a game lasting about 10 updates) would give about
7,400 value rows per update at the default fraction 0.25, and hold about 150,000 x F rows (about 20 KB each with the
features, so 0.75 GB at 0.25 and 3 GB at 1) for the games in play. GPU memory does not change.

### Calibration on human games (`polyai.calibrate`)

`python -m polyai.calibrate --shards DIR... --model NAME=CKPT...` runs the value head of policy checkpoints, and
privileged critics (`critic_final.pt`, on shards with the truth keys), over finished games and reports how often the
sides given probability p won. A sample is one side of one game in one window of game turns (the mean estimate over
that side's positions); the report has the Brier score, log loss, accuracy, expected calibration error (ECE) and the
win share per probability bucket, per model and window. `--players 2` (the default) needs the `meta.jsonl` of
`polyai-human`.

Measured 2026-11-20 on the calibration set of docs/human-data.md (8,859 two-player human games, 1,786,775 positions,
winner by resignation), for round 77 and the three arms trained from it for 300 updates (`ctrl` as the live loop,
`turn` with `--credit-per-turn`, `crit` with `--credit-per-turn --priv-critic`, and the critic of `crit`):

| Model | Turn 0: Brier | Turn 0: share given >= 0.8, and won | Turns 11-15: ECE | Turns 11-15: given >= 0.8 (mean), won |
|---|---|---|---|---|
| round 77 | 0.327 | 37%, 50.3% | 0.043 | 0.92, 93.6% |
| ctrl | 0.266 | 0%, n/a | 0.104 | 0.91, 98.0% |
| turn | 0.334 | 49%, 49.4% | 0.038 | 0.91, 94.9% |
| crit (policy) | 0.285 | 1%, 54.7% | 0.036 | 0.90, 94.8% |
| crit (critic) | 0.349 | 62%, 51.0% | 0.152 | 0.93, 81.6% |

A coin flip has Brier 0.25. No estimate means anything in the first five turns of a game between equals: the learner
wins 65 to 75% of its training games (over half the pool is weaker opponents) and only its own rows are trained on, so
every model gives the mover 0.55 to 0.85 from either seat, and the sides given 0.8 or more win half the time. From
turn 11 the policy value heads of round 77, `turn` and `crit` are calibrated and right about the winner in 80 to 88%
of the samples; `ctrl` is too low there (sides given 0.4 to 0.6 win 69 to 79%). The privileged critic is the worst of
the five at every stage and overconfident throughout (sides given 0.93 win 82% at turns 11 to 15), so it fails the
check "sides given over 80% at the start win about 90%". A value estimate for play between equals needs symmetric
training games (self-play rows of both seats), not only the learner's rows against the pool.

## Metrics (`kind: update`)

`update`, `learner_policy_id`, `learner_decisions`, `opponent_decisions`, `games_finished`, `capped_games`,
`mean_turns`, `rollout_secs`, `update_secs`, `infer_secs`, `step_secs`, `decisions_per_sec` (learner decisions per
rollout second), `all_decisions_per_sec`, `win_rate` (per opponent entry: `games, wins, draws, losses, score,
score_w`), `running_win_rate` (parity adjusted, per entry), `pool_weights`, `pool_policy_ids`, and from the update:
`loss, policy_loss, value_loss, value_mse, entropy, kl_bc, approx_kl, clip_frac, grad_norm, positions, minibatches,
epochs_done, early_stop, kl_coef, policy_scale, adv_mean, adv_std, ret_mean`; `max_mem_mb` on the GPU.

## Configuration (`RLConfig`, every field is a flag)

Rollouts: `games` 128 in flight, `rollout_steps` 16384 learner decisions per update, `n_players`, `sizes`, `presets`,
`turn_limit` 150, `temperature`, `opp_temperature`, `drop_capped`. Pool: `pool_weights`, `beat_target`,
`weight_floor`, `weight_cap`, `stats_decay`, `snapshot_every` 10, `max_snapshots` 6. PPO: `epochs` 2, `minibatch` 256,
`lr` 3e-5, `grad_clip` 0.5, `clip`, `gamma`, `lam`, `vf_coef`, `ent_coef` 0.005, `target_kl` 0.05, `critic_warmup` 2,
`kl_coef` 0.1 to `kl_coef_end` 0.01 over `kl_decay_updates` 200. Recipe options (see above): `ema`, `ent_schedule`,
`ent_max`, `ent_coef_end`, `ent_decay_updates`, `anchor_every`, `adv_filter`. Value rows of the opponents: `value_opp`
0 (off), `value_opp_frac` 0.25. Run: `ckpt_every` 10, `threads` (default 8; VecEnv workers plus
torch threads), `torch_threads`, `bf16` (autocast in the update on CUDA; inference is bf16 on CUDA as in
`ModelPolicy`), `max_batch`, `need_gb`, `device`, `seed`, `resume`.

## Faster rollouts (opt-in, defaults unchanged)

- `--infer-graphs` / `--infer-compile`: rollout inference on a few fixed padded shapes, captured in CUDA graphs and/or
  `torch.compile`d (`polyai/infer_fast.py`; one shared runner per architecture, so a pool snapshot costs no capture).
  Same logits as the eager path up to bf16 rounding (`tests/test_infer_fast.py`). `tools/bench_infer.py CKPT` times
  `NetPolicy.choose` per call on realistic positions. On the shared RTX 3060 at 100% load (b6c96, median ms per call,
  speed-up of `--infer-graphs --infer-compile` over eager): 4 rows x1.22, 8 x1.00, 16 x1.03, 32 x1.04, 64 x1.14,
  128 x1.63; graphs alone were not faster there (0.7 to 1.1x). Needs about 0.3 GB more GPU memory. Worth trying on a
  free GPU; measure before relying on it.
- `--collector rust`: the arm scheduling, inference batching, learner rows and GAE in the engine module
  (`docs/collector.md`). Same rollout content, same `ppo_update`; 2 to 6% more decisions per second with the retail AI in
  the pool, 4 times in a collector-bound self-play profile, no difference when a loaded GPU dominates.
  `tools/profile_collector.py [--collector rust] [--ckpt CKPT]` measures it.

## Recipe options

Four options for the self-play recipe, each behind a flag whose default reproduces the update as it was (a test and a
one-off comparison with the commit before check that losses and parameters after an update are bit-for-bit equal).
Larger batches need no option: `--rollout-steps` (learner decisions per update, default 16384) and `--minibatch`
(default 256) are already flags; `--epochs` sets the passes over a rollout.

- **`--ema DECAY`** (for example 0.999; 0 = off). An exponential moving average of all weights *and buffers*
  (`avg = DECAY * avg + (1 - DECAY) * live` after every optimiser step; integer buffers are copied). The average is saved
  as the `ema` entry (a state dict) of `ckpt_last.pt`, `ckpt_u<N>.pt` and `ckpt_final.pt`, and as plain loadable checkpoints
  `ckpt_ema_u<N>.pt` and `ckpt_ema_final.pt` next to them. Rollouts, opponents' snapshots and the KL term use the live
  weights, so the option changes nothing about training; it only gives a smoother model to evaluate. Note that with
  BatchNorm in eval mode the running statistics never change in this trainer, so the buffers stay equal to the live ones.
  `--resume` restores the average (a run first started without `--ema` starts it from the live weights).
- **`--ent-schedule`** with `--ent-coef-end`, `--ent-decay-updates` (default 200) and `--ent-max`. The coefficient of the
  entropy bonus moves linearly from `--ent-coef` to `--ent-coef-end` over `--ent-decay-updates` updates (the default
  `--ent-coef-end -1` means "stay at `--ent-coef`", so the decay is off unless an end is given). If `--ent-max` is set
  (> 0) and the mean policy entropy of the *previous* update was above it, the coefficient is 0 for this update; it
  returns when the entropy is back under the threshold. The coefficient in effect is logged as `ent_coef` every update.
  Without `--ent-schedule` the coefficient is `--ent-coef` throughout and `--ent-max`, `--ent-coef-end` are ignored.
- **`--anchor-every N`** (R-NaD style moving anchor; 0 = off). The KL reference starts as `--ref`; after every N-th update
  it is replaced by a frozen copy of the learner, so updates 1 to N use `--ref`, updates N+1 to 2N use the learner as it
  was after update N, and so on. The `bc` pool entry stays the original `--ref`. The
  metric `kl_bc` is then the KL to the anchor in use, and `anchor` (`ref` or `self:u<N>`) names it. The anchor is saved in
  `ckpt_last.pt` (`rl.anchor`) and restored by `--resume`. The KL coefficient schedule is unchanged, so with a moving
  anchor it is worth a larger `--kl-coef-end`.
- **`--adv-filter FRAC`** (default 1.0 = all). After the advantages are normalised over the whole rollout, only the FRAC
  of the trained positions with the largest absolute advantage (at least two) are kept for the epochs of that update.
  The value loss, entropy and KL terms also see only the kept positions, which makes the update cheaper and the policy
  gradient less noisy but biased towards decisive moves. `positions_kept` in the metrics counts them.
- **`--polytopai-games N`** (0 = off). Keeps N games against the PolytopAI website's bot in flight next to the rollout;
  their decisions join every update on top of `--rollout-steps` (`site_decisions` in the metrics), and their results
  are logged under the opponent `site:polytopai<level>`. `--polytopai-level` (1 easier, 3 standard),
  `--polytopai-sizes` and `--polytopai-maps` choose the games. Needs the network and bindings with
  `Game.from_position`. See `polytopai.md`.

## Measurements and sizing

CPU smoke run from `bc3_b6c96_v11` (6 threads: 4 env, 2 torch; 32 games, 1500 learner decisions per update): about
320 learner decisions per second in the rollout (inference 3.5 s, engine stepping 1.1 s of 4.7 s) and 20 s for the
update (2 epochs of minibatches of 128), so inference and the update dominate on CPU and a GPU is needed for a real
run. The GPU was full during development, so no GPU number was measured.

Memory estimate for a GPU run (not measured; from the `train_bc` peaks at batch 256 with bf16, 0.79 GB for b6c96 and
1.53 GB for b10c128, plus the frozen BC reference and pool networks, a few MB each, and inference activations):
b6c96 about 1.3 GB in the allocator, set `--need-gb 2.0`; b10c128 about 2.3 GB, set `--need-gb 3.0`. Keep `--minibatch`
at 256 and lower `--max-batch` if memory is tight.

Suggested real run: `--games 128 --rollout-steps 16384 --epochs 2 --minibatch 256 --lr 3e-5 --threads 8`, which makes
a game last about 8 updates; raise `--threads` if the host allows, since the engine stepping is the part that scales
with them. Watch `kl_bc` (should stay well under 0.1 early on), `clip_frac` (0.05 to 0.2), `entropy` (should not
collapse) and `score_w` against `retail:normal` and `bc`.
