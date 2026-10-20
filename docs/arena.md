# Arena: evaluating agents against each other

`crates/polyai-arena` plays two agents against each other on fixed seeds and writes one JSONL line per game.
`tools/elo.py` turns those lines into win rates and Elo. Nothing here uses a model yet; the `Agent` trait is the
place one plugs in.

## Setup of every game

1v1, Small map, Continents, Domination, 50-turn limit (adjudicated on score when the limit is hit, as in
`polyai-gen`). Two different tribes are drawn from the 13 (12 normal plus Cymanti) per seed, in `eval_setup`
(`src/lib.rs`). Seeds start at 1,000,000 and the runner refuses lower ones (`EVAL_SEED_MIN`; seeds 1 to 100,000 are
training data). Every seed is played twice with the agents swapping seats: the map and the tribe in each seat stay
the same, so the swap removes the seat and tribe advantage from the comparison. Games are deterministic given the
agents (random and MCTS are seeded from the game seed and seat).

## Agents

| name | what it is |
|---|---|
| `retail:easy\|normal\|hard\|crazy` | the engine's retail AI at that difficulty (the seat's handicap) |
| `random` | uniform over polyai-core's `legal_actions` (engine legal list plus `Stay`) |
| `mcts:<iters>[:depth=D][:topk=K]` | weekly's generic UCT (`polytopia_weekly::mcts`) over single engine commands, uniformly random rollouts of at most `D` commands (default 60), reward = win/loss at the end or score share when cut off; `topk=K` prunes the root to the retail AI's `K` best-scored candidates |

**The MCTS agent cheats.** It searches clones of the true engine state, so it sees through fog (full information),
and the retail AI also reads the true state. It is still weak: with random rollouts and a modest budget it cannot
plan a 50-turn game. Treat its numbers as a sanity check of the adapter and a rough speed reference, not as a
strength target.

An agent is `trait Agent { fn name; fn pick(&mut Engine, &Ctx) -> Command; fn trusted }`. The runner checks every
command from an untrusted agent against the engine's legal list and replaces an illegal one with `EndTurn`, counting
it in the `illegal` field (all runs below have zero). To add a model agent (Python or ONNX), implement `Agent`, add
a variant to `AgentSpec` (the `model:` prefix is reserved and currently returns an error) and build it in
`AgentSpec::build`.

## Running

```
cd /kvm/polytopia/ai
export TMPDIR=/kvm/tmp/tmp UV_CACHE_DIR=/kvm/tmp/uv-cache
cargo build --release -p polyai-arena
nice -n 10 target/release/polyai-arena \
    --agents random,retail:easy,retail:normal,retail:hard,retail:crazy,mcts:50 \
    --seeds 50 --threads 20 --out /kvm/polytopia-ai-data/arena/rr.jsonl
/kvm/polytopia/.venv/bin/python tools/elo.py /kvm/polytopia-ai-data/arena/rr.jsonl
```

Every pair of the listed agents plays every seed twice. `--threads` is capped at 24 (rayon pool). Output is appended
and games already in the file are skipped, so an interrupted run resumes and a longer `--seeds` extends a run. Data
goes under `/kvm/polytopia-ai-data/arena/`, never into git.

JSONL fields: `seed, swap, agents` (seat 0 first), `tribes, setup, winner` (seat, `-1` none), `scores, turns,
adjudicated, over, commands, illegal, wall_ms, think_ms` (per seat, time inside `pick`), `engine` (engine commit,
`-dirty` if the working tree had changes), `game_version`.

`tools/elo.py FILE...` prints per pair: games, W-L-D, win rate with a Wilson 95% interval (a draw is half a win),
mean score margin, share adjudicated and seconds per game; then a Bradley-Terry fit anchored at `retail:normal` =
1000, with 95% intervals from a bootstrap over seeds (300 resamples, both seatings of a seed kept together). A weak
prior of one drawn game per pair keeps unbeaten agents finite, so a 100-0 pair still gives a finite rating whose
interval is wide. Tests: `/kvm/polytopia/.venv/bin/python -m pytest tools/test_elo.py`.

## First table (2026-10-20)

Engine commit `729db890c58f` (working tree dirty), game version 123. Round robin of random, the four retail
difficulties and `mcts:50` over 50 seeds x 2 seatings = 100 games per pair, plus `mcts:50:topk=5` (20 seeds, against
Easy and Normal) and `mcts:200` (20 seeds, against Easy). 1,620 games, 15% adjudicated at the turn limit overall.
Raw data: `/kvm/polytopia-ai-data/arena/all.jsonl` (merged), `rr.jsonl`, `mcts_topk.jsonl`, `mcts200.jsonl`.

| agent | Elo | 95% CI |
|---|---|---|
| retail:crazy | 1418 | 1359 to 1492 |
| retail:hard | 1237 | 1192 to 1294 |
| retail:normal (anchor) | 1000 | 1000 to 1000 |
| mcts:50:topk=5 | 914 | 845 to 983 |
| retail:easy | 785 | 730 to 827 |
| mcts:200 | 267 | 199 to 340 |
| mcts:50 | 181 | 132 to 216 |
| random | -40 | -115 to 20 |

Difficulties order as expected, Easy < Normal < Hard < Crazy, with every adjacent gap far outside the intervals
(Crazy beats Hard 73%, Hard beats Normal 80%, Normal beats Easy 79%). Pairs:

| pair | games | W-L-D (first) | win rate | 95% CI | mean margin | adjudicated | s/game |
|---|---|---|---|---|---|---|---|
| retail:crazy vs retail:easy | 100 | 100-0-0 | 100.0% | 96.3 to 100.0 | +10549 | 3% | 0.2 |
| retail:crazy vs retail:hard | 100 | 73-27-0 | 73.0% | 63.6 to 80.7 | +5091 | 17% | 0.5 |
| retail:crazy vs retail:normal | 100 | 93-7-0 | 93.0% | 86.3 to 96.6 | +9692 | 13% | 0.3 |
| retail:easy vs retail:hard | 100 | 6-94-0 | 6.0% | 2.8 to 12.5 | -8967 | 11% | 0.3 |
| retail:easy vs retail:normal | 100 | 21-79-0 | 21.0% | 14.2 to 30.0 | -6583 | 46% | 0.4 |
| retail:hard vs retail:normal | 100 | 80-20-0 | 80.0% | 71.1 to 86.7 | +6427 | 26% | 0.4 |
| random vs retail:crazy | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5512 | 0% | 0.1 |
| random vs retail:easy | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5449 | 0% | 0.1 |
| random vs retail:hard | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5241 | 0% | 0.1 |
| random vs retail:normal | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5435 | 0% | 0.1 |
| mcts:50 vs random | 100 | 80-20-0 | 80.0% | 71.1 to 86.7 | +2135 | 99% | 41.9 |
| mcts:50 vs retail:crazy | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5276 | 0% | 9.4 |
| mcts:50 vs retail:easy | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5426 | 0% | 31.0 |
| mcts:50 vs retail:hard | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -4907 | 0% | 10.0 |
| mcts:50 vs retail:normal | 100 | 0-100-0 | 0.0% | 0.0 to 3.7 | -5185 | 0% | 18.1 |
| mcts:200 vs retail:easy | 40 | 0-40-0 | 0.0% | 0.0 to 8.8 | -5031 | 0% | 131.0 |
| mcts:50:topk=5 vs retail:easy | 40 | 28-12-0 | 70.0% | 54.6 to 81.9 | +3960 | 48% | 216.5 |
| mcts:50:topk=5 vs retail:normal | 40 | 15-25-0 | 37.5% | 24.2 to 53.0 | -2527 | 25% | 170.3 |

Scores are in the engine's score units (hundreds per city and so on), so margins of thousands are normal.

### Reading it

- **Adjudication.** 15% of all games hit the 50-turn limit; among retail pairs it is 3% to 46% (highest for Easy v
  Normal, 46%, where neither side finishes the other quickly). The other games end under the Domination rule
  before the limit. Random loses before the limit every time, so it is never adjudicated; random against the passive plain
  MCTS is adjudicated in 99% of games.
- **Retail speed.** A retail game takes 0.1 to 0.5 s of CPU, so a 1,000-game retail table takes about 13 s on 20
  threads. The cost of a table is all in MCTS.
- **MCTS speed.** Plain random-rollout MCTS spends about 90 ms per own command at 50 iterations and about 380 ms at
  200 (rough: think time divided by half the game's commands), so a game costs 20 to 130 s of CPU
  (`mcts:50`: 22 s thinking per game on average, `mcts:200`: 131 s). `topk=5` spends about 280 ms per command at 50
  iterations (193 s per game) because of the retail candidate scoring.
- **MCTS strength.** Plain MCTS loses every game to every retail difficulty (0 of 440 games) and only beats random, in a
  slow adjudicated game. More iterations help little (`mcts:200` 267 vs `mcts:50` 181 Elo, different opponents so
  only roughly comparable). Pruning the root to the retail AI's top 5 candidates is the big gain: `mcts:50:topk=5`
  beats Easy 70% and loses to Normal, about 914 Elo, but this rests on 20 seeds and mostly measures the retail
  candidate list. The cheating (full information) is part of that number.
- **Caveats.** Elo values for agents that win or lose every game (random, plain MCTS against retail) are set by the
  prior and are only ordinal. The Wilson interval of a 0-100 pair (0 to 3.7%) is the honest statement there. Pairs
  involving the topk and 200-iteration variants have 40 games, not 100. The engine ranks Domination winners by
  score, whereas retail may rank by city count (ledger to-do 8e4f7017); this could shift who wins an adjudicated
  game, and so the absolute win rates, but it applies equally to every agent.
