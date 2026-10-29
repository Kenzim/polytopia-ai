# Belief heads: predicting what lies under the fog

Experiment A of gameplan v2 (workstream 1): does teaching the network to predict the hidden map make it play better?
Every generated game knows the whole map, so the labels cost nothing.

## Labels

`polyai-core/src/truth.rs` builds, for a position and its mover, one label row per tile and a mask of hidden tiles (the
tiles the mover has not explored, exactly the tiles whose `unexplored` input column is 1). Labels are zero on explored
tiles. Nothing in `encode` reads them: the observation is unchanged, and a unit test rewrites a hidden tile and checks
that the encoding stays the same while the truth changes.

| channel | classes | meaning |
|---|---|---|
| terrain | 10 | engine terrain idx (water, ocean, land, forest, mountain, ...) |
| resource | 11 | engine resource idx (0 none), whether or not the mover's techs would show it |
| site | 3 | 0 nothing, 1 village (unowned city), 2 ruin |
| enemy_city | 2 | a city of another player stands here (capitals included) |
| enemy_capital | 8 | the mover's slot (1..7) of the player whose capital this tile is, 0 otherwise |
| unit | 3 | 0 none, 1 a unit of another player, 2 a Nature unit |

## Shards

`polyai-gen --truth` appends two optional keys to every shard (`truth_shard.rs`, so `shard.rs`, `npz.rs` and the
client's `shardify` stay as they are, and shards made without the flag are unchanged):

| key | dtype | shape |
|---|---|---|
| truth | int16 | [N, H, W, 6] |
| truth_mask | uint8 | [N, H, W] |

`polyai.data.Shard` loads them when present and applies the board symmetries to them with the tiles.

## Model and training

`ModelConfig.belief_hidden > 0` adds one head per channel on the trunk features: `Conv3x3(c, hidden) - ReLU -
Conv1x1(hidden, classes)`. The key is left out of the saved config when 0, so checkpoints without heads are byte for
byte what they were. `model(..., belief=True)` returns the logits in `ModelOutput.belief`; play never asks for them.

`train_bc --belief-weight W` (default 0 = off) adds `W * mean over channels of CE(hidden tiles only)` to the loss and
sets `--belief-hidden` (default 32) as head width. `--init CKPT` starts from a checkpoint's weights (new heads are fresh;
the trunk must match). `load_checkpoint(path)` drops the heads, so arena, policy, serve and rl load either kind of
checkpoint as the plain network; `load_checkpoint(path, belief=True)` keeps them.

## Measuring

`python -m polyai.belief --model CKPT --shards DIR` reports, per channel on hidden tiles of held-out shards, accuracy
against the always-most-common-value baseline, cross entropy against the cross entropy of the class frequencies, and
non-majority precision and recall. The enemy capital is also scored as a location: how often the true capital is among
the top 1 or top 3 hidden land tiles by predicted probability, against picking uniformly among the hidden land tiles.
The same numbers appear as `belief` in the `val` records of `metrics.jsonl`.

## Results (2026-10-29)

Data: `bc-might-truth`, 20,000 Might games from seeds 50001..70000 (default 1v1-weighted mix), written with `--truth`;
135 shards to train and 10 held out (every 15th). Both arms fine-tune `might1_b6c96/ckpt_final.pt` (b6c96) by behaviour
cloning: 12,000 steps, batch 256, lr 2e-4, warm-up 100, seed 0. Arm A (`truthab_b0`) has belief weight 0; arm B
(`truthab_b1`) has weight 0.5 and 32-wide heads (1.59M params; the heads are dropped at play time).

Belief accuracy of `truthab_b1` on the held-out shards, hidden tiles only (25.0M tiles). Baseline is always the most
common value; skill is 1 minus CE over the CE of the class frequencies.

| channel | acc | baseline acc | CE | baseline CE | skill |
|---|---|---|---|---|---|
| terrain | 0.453 | 0.332 | 1.285 | 1.513 | 0.150 |
| resource | 0.683 | 0.683 | 1.158 | 1.245 | 0.070 |
| site | 0.966 | 0.966 | 0.141 | 0.172 | 0.179 |
| enemy_city | 0.967 | 0.966 | 0.105 | 0.150 | 0.300 |
| enemy_capital | 0.991 | 0.991 | 0.045 | 0.062 | 0.279 |
| unit | 0.925 | 0.925 | 0.229 | 0.266 | 0.138 |

Enemy capital as a location (163,530 positions): true capital is the top-1 hidden land tile in 29.2% of positions
(uniform 3.6%) and in the top 3 in 52.4% (uniform 10.6%). Accuracy beats the majority baseline only for terrain, since
the other channels are mostly one class; the cross entropy shows the heads do learn.

Play, Might 1v1 suite, 200 seeds, both seats (400 games), CPU, greedy:

| match | W-L-D | score (95% CI) |
|---|---|---|
| belief 0.5 vs belief 0 (head to head) | 202-197-1 | 50.6% (45.7 to 55.5) |
| belief 0 vs retail:crazy | 38-359-3 | 9.9% (7.3 to 13.2) |
| belief 0.5 vs retail:crazy | 57-341-2 | 14.5% (11.4 to 18.3) |

Reading: head to head is a tie. Against Crazy belief 0.5 scores 4.6 points higher (z about 2.0, borderline, one seed of
each arm). More important, both arms are far below their starting point: `might1_b6c96` scores 81.4% against Crazy, so
behaviour cloning on this dataset wrecks play whatever the belief weight. The retail-AI moves in the data are a worse
teacher than the self-play policy it starts from. The A/B therefore says little about whether belief heads help a
strong policy; a fair test needs the belief loss added to RL (or BC on stronger-play data) and a policy that starts
and stays strong.
