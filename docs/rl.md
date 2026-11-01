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
`ent_max`, `ent_coef_end`, `ent_decay_updates`, `anchor_every`, `adv_filter`. Run: `ckpt_every` 10, `threads` (default 8; VecEnv workers plus
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
