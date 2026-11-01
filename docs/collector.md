# Rollout collector in Rust (`--collector rust`)

Design of plan item `rust-collector`. Status and measurements are at the end.

## 1. What the Python collector costs

`polyai.rl.Collector` plays up to 512 games in "arms" (one `VecEnv` each, `docs/rl.md`). The engine already does the
heavy work in Rust: stepping, legal moves, encoding, features, truth labels, the retail AI seats. What the collector adds
in Python, per tick and per game:

- `VecEnv.step` and `observe` turn each map-size group into a dict of numpy arrays (a copy of every array);
- `take` (select the rows of a group that belong to one policy), `merge_padded` (concatenate, pad boards, re-base the
  action tile indices) and `SizeStore.add` (keep the learner's rows) copy the same observations three more times;
- per-game bookkeeping: trajectories, GAE at the end of a game, roles of a fresh game;
- the arm scheduler (`_harvest`, `_may_step`, `max_lead`, `min_infer` gathering) and futures on a thread pool.

Measured with a stand-in policy that answers in microseconds (`tools/profile_collector.py`), so that only the engine and
the collector remain, the main thread spends 10 to 15% of the rollout outside engine waits. With a real network the
inference calls themselves dominate (about 90% of a rollout on a GPU), so the ceiling for this item is the share of the
Python side of the loop, not the engine, which was already native.

## 2. Division of labour

Stays in Python (SEED RL style: one central, batched inference):

- all `torch` work: `policy.choose` on batches, the PPO update, the opponent pool (weights, snapshots, statistics);
- every random draw of the run: which slots to activate, the learner's seat and the opponent entry of each fresh game.
  Keeping them in Python keeps the numpy stream and its order the same as the Python collector's;
- `Arm` creation (`VecEnv(...)` arguments), quotas, `RLConfig`.

Moves into Rust (`crates/polyai-py/src/collector.rs`, class `RolloutCore`):

- the arms' scheduling: which arms may step (`max_lead`), gathering for `min_infer`, running the steps on a worker pool
  and taking in the results as they come back (no GIL involved);
- the ready lists of games awaiting a decision, as slot indices; observations are read straight from the engine's
  pending decisions, never converted to Python objects one group at a time;
- the inference batches: rows of one policy and board bucket, built once (padded boards, re-based action tiles,
  `board_mask`, features) and returned as numpy arrays;
- the rollout store: the learner's rows go straight from the engine into per-board-size columns; trajectories per
  game and GAE (`compute_gae`, in `f64` like the Python one) when a game ends or the rollout is cut.

Python calls per inference round: `next_round()` (events and batches), one `policy.choose` per batch, then
`finish_round(results)` (choices, learner rows, start of the engine steps). Roles of fresh games take one extra round
trip, `set_roles`, only in rounds where a game of a model arm has ended.

```
collect(steps):
    core.begin_rollout()
    while core.n_learner < steps:
        r = core.next_round()            # harvest finished steps, schedule, build batches (GIL released while waiting)
        for f in r.finished: on_finished(f)          # pool statistics, results, games.jsonl (as the Python collector)
        for (arm, slot, n) in r.requests: draw roles # rng, pool.sample_model_entry
        core.set_roles(...)
        results = [policy_of(b).choose(b.obs, temperature) for b in r.batches]
        core.finish_round(results)       # store learner rows, start the engine steps
    drain()                              # settle the steps in flight (their finished games and roles), then cut()
    stores = core.take_stores()          # one finalised dict per board size, same keys as SizeStore.finalize()
```

## 3. What is recorded, and the equivalence contract

The rollout the trainer receives must have the same content as the Python collector's:

- per learner decision: tiles, globals, legal list (`act_off`, `actions`), features and truth rows when the env has them,
  the chosen index, the behaviour `logp`, the value, and (after GAE) `adv`, `ret` and whether it is trained
  (`SizeStore.finalize()` keys, dtypes and shapes, so `ppo_update` is unchanged);
- per finished game: the `results` record (seed, players, learner seat, opponent entry and policy id, outcome, turns,
  decisions) and the pool statistics it feeds.

Not the same, by nature: the order of rows inside a store and which decisions are in flight when the rollout stops.
Both depend on how the worker threads finish, in either collector. With one arm and `max_lead` 0 the process is
lockstep and deterministic in both, and the two collectors then record the same rows in the same order; the tests rely
on that, plus per-game equality (same seed, same trajectory, same outcome) for several arms.

Batches are built in the same row order as the Python collector's (arms in creation order, groups by board size, keys
sorted by name, rows by slot), so a network policy with a seeded generator samples the same moves from either.

## 4. Flags

- `--collector python|rust` (default `python`; the Python path is untouched);
- `max_lead`, `min_infer`, `gather_ms`, `pad_sizes`, `drop_capped`, `gamma`, `lam` keep their meaning;
- the Rust workers count is the same `threads - torch_threads` as the Python executor's.

## 5. Tests

`python/tests/test_collector_rust.py`: both collectors with `polyai.standin.StandInPolicy` (position-deterministic) on
one model arm and lockstep give identical stores (every array) and results; on several arms (retail and model) every game
that finishes in both has the same record; features and truth rows; `drop_capped`; padded and unpadded batches; roles
and `in_use`; one PPO update on the stores of each collector. Rust unit checks of GAE live in the same file through the
module's `gae` function.

## 6. Results

Learner decisions per second, `tools/profile_collector.py`, 512 games, 16384 decisions per rollout (a warm-up rollout,
then two more; the later rollouts play longer games, so compare within a rollout), `--threads 14`, Might, standard mix,
`nice -n 10`, on the 96-core epyc host shared with training runs (load average 12 to 15, noise about 5%).

| Setup (stand-in policy, no network) | python | rust |
|---|---|---|
| default pool (retail and model arms), rollout 1 / 2 / 3 | 2133 / 1496 / 887 | 2229 / 1476 / 944 |
| pure self-play (`--pool-weights bc=1,...=0`), rollouts 1 / 2 / 3 | 9100 / 11500 / 11000 | 48000 / 44300 / 43900 |

The default pool is bound by the engine (the retail AI is 80 to 90% of the step wait, and the collector's Python work
already overlapped with it on other arms), so the Rust collector gains 2 to 6%. With no retail AI the collector is
what is left: 4 times as many decisions per second, and the "infer" bucket of the profile (batch building, stores) drops
from 0.6 s to 0.1 s per rollout.

End to end with a real network (b6c96 `bc3_b6c96_v11`, 256 games, 8192 decisions, 2 snapshots, RTX 3060 at 100%
utilisation from other users): both collectors reach 1100 to 1750 decisions/s and the difference is inside the noise
(python 1747 / 587 / 1525 / 1131, rust 1435 / 1292 / 1526 / 1255 in the repeated reps). Inference is 4 to 5 s of a
5 to 7 s rollout, so the collector cannot show there; the gain appears when inference gets cheaper (fewer or smaller
networks, a free GPU, `--infer-graphs`) or the pool is mostly model arms.

A first version stepped the arms on a rayon pool and was 30 to 50% slower than Python in later rollouts: a rayon worker
waiting in the `install` of the arm's own engine pool runs other queued jobs on top of itself, so a finished step of one
arm waits for a slow retail turn of another. The workers are now plain threads (`Workers` in `collector.rs`).
