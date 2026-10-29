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
