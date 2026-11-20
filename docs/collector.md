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
    opp = core.take_opp_stores()         # --value-opp only: the opponents' value rows of games that ended
```

## 3. What is recorded, and the equivalence contract

The rollout the trainer receives must have the same content as the Python collector's:

- per learner decision: tiles, globals, legal list (`act_off`, `actions`), features and truth rows when the env has them,
  the chosen index, the behaviour `logp`, the value, and (after GAE) `adv`, `ret` and whether it is trained
  (`SizeStore.finalize()` keys, dtypes and shapes, so `ppo_update` is unchanged);
- per finished game: the `results` record (seed, players, learner seat, opponent entry and policy id, outcome, turns,
  decisions) and the pool statistics it feeds;
- with `--value-opp` (docs/rl.md): the kept opponents' rows of every game that ended, with their seat's result
  (`ValueStore.finalize()` keys), and the count of rows still held (`opp_pending`). The core holds them per slot across
  rollouts; the choice of rows is a hash, not a random draw, so both collectors keep the same ones.

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

## 7. Per-decision cost and the profile-guided build (2026-11-20)

`cargo run --release -p polyai-core --example bench_decision [GAMES] [TURNS] [PLAYERS]` plays retail AI games on one
thread and, before every command, does the collector's work for that position: slot map, legal list, observation,
action encoding, engine features, truth labels and a copy of the game. It prints the time of each part and a checksum
of everything the parts produce, so a change that only makes them faster leaves the checksum line as it was. With
`PART=<name> REPEAT=<n>` it then runs one part alone over every fourth position, the loop to point `perf` at.

`tools/pgo_build.sh` builds the extension with a profile from that example (fat LTO, one codegen unit). With
`OUT=<dir>` it writes `<dir>/polyai_engine.so` (put the directory first on `PYTHONPATH`); without it, it installs into
the active virtualenv with `maturin develop --release`. It needs `rustup component add llvm-tools-preview` and stops
if the instrumented and the optimised example print different checksums.

Microseconds per decision, 20 standard-mix Might games to turn 30 (19,500 decisions, 106 legal moves on average), one
core of the shared epyc host, lowest of five alternating runs. "Before" is ai `de9ff28` on engine `2e85b41`; "after" is
this branch on engine `0a60918`; "PGO" is "after" built by `tools/pgo_build.sh`.

| Part | before | after | PGO |
|---|---|---|---|
| legal list (not changed; same instruction count before and after) | 23.3 | 27.1 | 19.7 |
| observation (`encode`) | 34.7 | 20.9 | 18.5 |
| action encoding | 5.1 | 4.3 | 3.5 |
| engine features | 122.5 | 76.2 | 53.2 |
| truth labels | 5.9 | 5.1 | 2.9 |
| copy of the game | 9.7 | 5.5 | 4.3 |
| sum of the above, with the slot map | 201 | 140 | 102 |
| retail AI pick, per command | 343 | 252 | 194 |
| apply, per command | 19.7 | 18.6 | 16.1 |

The host was busy (load average about 100 on 96 cores), so the instruction counts of the `PART` loops (`perf stat`,
user space, per call; they do not move with the load) are the firmer comparison: features 629k / 430k / 333k,
observation 295k / 211k / 188k, copy 49.5k / 27.5k / 21.9k, retail AI pick with a copy 1,618k / 1,077k / 858k, and the
legal list 113k / 113k / 93k.

Collector, `tools/profile_collector.py --collector rust --games 256 --steps 8192 --threads 8 --warmup 1 --reps 3`
(stand-in policy, default pool, cores 40 to 47), median of three alternating runs: the three reps take 30.6 s before,
19.9 s after and 16.1 s with PGO (user CPU of the whole run 84.5 / 60 / 50 s).

What changed: the features build the copy of the game for enemy movement only when a hostile attacker is in sight, and
only the parts a move search reads; the move searches share the engine's unit grid and zone-of-control maps
(`Engine::query_scope`); a unit's combat figures are worked out once for all its commands; hostile neighbours are
counted once per tile. `encode` asks for resource visibility once per resource kind and takes the scores from one pass
over the map. `ai_pick` checks the retail AI's pick with `Engine::is_legal` instead of building the whole legal list.
The rest is in the engine (copy of the game, the retail AI's position ratings).

Tried and left out: mimalloc as the extension's allocator. Game stepping gets about 10% faster, but a collector run
then spends 8 to 15 s in the kernel instead of about 1 s and 5 to 8 s more outside game stepping, so the reps took
22 to 27 s instead of 20 to 21 s. The cause was not tracked down; the large per-rollout buffers going back to the
system and being faulted in again is the likely one.

Not done: a legal list kept across decisions. The list is 23 of the 140 microseconds, every command changes the
position it was made for, and an incremental update would need to know what each command can change for every other
unit (fog, zone of control, borders, routes), which is where a behaviour difference would come from.
