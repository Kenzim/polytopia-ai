# Arena: evaluating agents against each other

`crates/polyai-arena` plays two agents against each other on fixed seeds and writes one JSONL line per game.
`tools/elo.py` turns those lines into win rates and Elo. Nothing here uses a model yet; the `Agent` trait is the
place one plugs in.

## Setup of every game

The main suite is **1v1 Domination across varied maps** (objective v1.1: win by elimination only, points never
decide). Each seed draws, in `eval_setup` (`src/lib.rs`), a map size (Tiny, Small or Normal), a preset (Dryland, Lakes,
Continents, Pangea or Archipelago) and two different tribes from the 13 (12 normal plus Cymanti). The compute cap is
`DEFAULT_TURN_LIMIT` = 150 turns: a game still running then, with both players alive, is a **draw** (`capped`), never
decided on score, and counts as half a win for each side. Seeds start at 1,000,000 and the runner refuses lower ones
(`EVAL_SEED_MIN`; seeds 1 to 100,000 are training data). Every seed is played twice with the agents swapping seats: the
map and the tribe in each seat stay the same, so the swap removes the seat and tribe advantage from the comparison.
Games are deterministic given the agents (random and MCTS are seeded from the game seed and seat). FFA tables (one
model against two or three retail seats) are a second suite and not built yet.

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

JSONL fields: `seed, swap, agents` (seat 0 first), `tribes, setup` (`size, preset, mode, turn_limit`), `winner`
(seat that won by elimination; **`-1` for a draw**), `scores` (logged, never decisive), `turns, capped` (true when the
game reached the turn cap with both players alive, which is exactly a draw), `over, commands, illegal, wall_ms,
think_ms` (per seat, time inside `pick`), `engine` (engine commit, `-dirty` if the working tree had changes),
`game_version`. Lines written before the change (suite v1: Small, Continents, 50 turns) have `adjudicated` instead of
`capped` and the top-score player as `winner`; `elo.py` reads those as draws.

`tools/elo.py FILE...` prints per pair: games, W-L-D, win rate with a Wilson 95% interval (a draw is half a win), the
draw rate, seconds per game and, as an information-only last column, the mean final score difference; then a
Bradley-Terry fit (a draw counts half a win each way) anchored at `retail:normal` = 1000, with 95% intervals from a
bootstrap over seeds (300 resamples, both seatings of a seed kept together). A weak prior of one drawn game per pair
keeps unbeaten agents finite, so a 100-0 pair still gives a finite rating whose interval is wide. The headline says
how many games were draws. Tests: `/kvm/polytopia/.venv/bin/python -m pytest tools/test_elo.py`.

## Retail table on the 1v1 suite (2026-10-21)

Engine commit `729db890c58f` (working tree dirty), game version 123. Round robin of the four retail difficulties over
200 seeds x 2 seatings = 400 games per pair, 2,400 games, 13 draws (0.5%) at the 150-turn cap. Run:
`nice -n 10 target/release/polyai-arena --agents retail:easy,retail:normal,retail:hard,retail:crazy --seeds 200
--threads 16 --out /kvm/polytopia-ai-data/arena-v2/retail.jsonl` (about 55 s). Raw data and the report:
`/kvm/polytopia-ai-data/arena-v2/`.

| agent | Elo | 95% CI |
|---|---|---|
| retail:crazy | 1472 | 1430 to 1523 |
| retail:hard | 1282 | 1243 to 1324 |
| retail:normal (anchor) | 1000 | 1000 to 1000 |
| retail:easy | 697 | 654 to 735 |

| pair | games | W-L-D (first) | win rate | 95% CI | draws | s/game | score diff (info) |
|---|---|---|---|---|---|---|---|
| retail:crazy vs retail:easy | 400 | 398-1-1 | 99.6% | 98.4 to 99.9 | 0% | 0.3 | +9701 |
| retail:crazy vs retail:hard | 400 | 288-108-4 | 72.5% | 67.9 to 76.6 | 1% | 0.5 | +5606 |
| retail:crazy vs retail:normal | 400 | 381-15-4 | 95.8% | 93.3 to 97.3 | 1% | 0.4 | +10267 |
| retail:easy vs retail:hard | 400 | 12-388-0 | 3.0% | 1.7 to 5.2 | 0% | 0.3 | -8729 |
| retail:easy vs retail:normal | 400 | 62-337-1 | 15.6% | 12.4 to 19.5 | 0% | 0.7 | -9532 |
| retail:hard vs retail:normal | 400 | 322-75-3 | 80.9% | 76.7 to 84.4 | 1% | 0.5 | +6662 |

The order is Easy < Normal < Hard < Crazy with every gap far outside the intervals, as in the first table, and the
spread is a little wider than on the old Small/Continents suite (Crazy 1472 against 1418, Easy 697 against 785).
Draws are rare (0.5% of games, at most 1% in any pair): retail games between these agents end by elimination long
before the cap. The 13 draws fall on 11 of the 200 seeds, mostly Normal maps; a first run of 50 seeds had none.

## First imitation model on the 1v1 suite (2026-10-21)

`bc3_b6c96_v11`: b6c96 (1.4M parameters), behaviour cloning on bc-v1 shards 0 to 148 relabelled for v1.1 (score
adjudicated games are draws), 40,000 steps at batch 256; held-out top-1 0.547, top-3 0.806. It plays greedily (argmax
policy, no search) at the Normal handicap, against each retail level on the same 200 seeds x 2 seatings. Run on CPU
because the shared GPU was full: `python -m polyai.arena --model /kvm/polytopia-ai-data/runs/bc3_b6c96_v11 --opponent
retail:easy,retail:normal,retail:hard,retail:crazy --seeds 200 --threads 8 --torch-threads 4 --device cpu --out
/kvm/polytopia-ai-data/arena-v2/model/bc3.jsonl` (about 35 min). Report fitted together with the retail table:
`/kvm/polytopia-ai-data/arena-v2/model/bc3_report.md`.

| opponent | W-L-D | score | 95% CI |
|---|---|---|---|
| retail:easy | 343-56-1 | 85.9% | 82.1 to 88.9 |
| retail:normal | 225-172-3 | 56.6% | 51.7 to 61.4 |
| retail:hard | 101-293-6 | 26.0% | 21.9 to 30.5 |
| retail:crazy | 40-355-5 | 10.6% | 8.0 to 14.0 |

Elo 1057 (1031 to 1084), between Normal (1000) and Hard (1260) in the joint fit. The held-out Brier of bc-v1 runs is
optimistic: with the old 50-turn cap, positions after turn 30 that keep a label come from games that ended soon after,
so late-turn Brier is near zero by selection. Judge value heads on bc-v2 (150-turn cap, 1.6% draws) instead.

### Larger imitation model (2026-10-21)

`bc4_b10c128_v2`: b10c128 (3.4M parameters), behaviour cloning on bc-v1 (relabelled) plus bc-v2, 100,000 steps at batch
256; held-out top-1 0.626, top-3 0.850, Brier 0.071 (bc-v2 validation shards). Same suite and settings as bc3, on the
GPU (peak 50 MiB, about 15 min): `--threads 6 --device cuda --gpu-gb 1.0 --out
/kvm/polytopia-ai-data/arena/bc4/bc4_vs_retail.jsonl`.

| opponent | W-L-D | score | 95% CI |
|---|---|---|---|
| retail:easy | 351-45-4 | 88.2% | 84.7 to 91.0 |
| retail:normal | 230-167-3 | 57.9% | 53.0 to 62.6 |
| retail:hard | 95-295-10 | 25.0% | 21.0 to 29.5 |
| retail:crazy | 46-349-5 | 12.1% | 9.3 to 15.7 |

Joint fit with the retail table and bc3 (5,600 games): Crazy 1437, Hard 1255, bc4 1065 (1042 to 1086), bc3 1053 (1028
to 1079), Normal 1000, Easy 711. Better imitation (top-1 0.547 to 0.626) bought almost no strength: greedy imitation
of a mixed-difficulty retail AI plateaus near Normal, so further gains have to come from self-play and search.

### First self-play model (2026-10-21)

`rl1_b6c96_cpu`: PPO self-play (`polyai.rl`, `docs/rl.md`) from bc3, 200 updates of about 16,000 learner decisions,
pool of retail Easy to Crazy, bc3 and its own snapshots, KL to bc3 decaying 0.1 to 0.01 (updates 1 to 100 on CPU, 101
to 200 on the GPU, about 70 min there, peak 0.97 GB). Same suite and settings as bc3 and bc4: greedy, no search, model
seat at Normal, 200 seeds x 2 seats per opponent, 0 illegal commands. Results:
`/kvm/polytopia-ai-data/arena/rl1/rl1_raw.jsonl`.

| opponent | W-L-D | score | 95% CI |
|---|---|---|---|
| retail:easy | 396-2-2 | 99.2% | 97.8 to 99.7 |
| retail:normal | 386-13-1 | 96.6% | 94.4 to 98.0 |
| retail:hard | 338-60-2 | 84.8% | 80.9 to 87.9 |
| retail:crazy | 273-122-5 | 68.9% | 64.2 to 73.2 |

Joint fit with the retail table and bc4 (5,600 games): rl1 1573 (1529 to 1615), Crazy 1443, Hard 1267, bc4 1068,
Normal 1000, Easy 703. Both seats contribute (vs Crazy: 144-56 in seat 0, 129-71 in seat 1). Self-play took the same
1.4M network from about 1050 to about 1570 in 200 updates. Caveat: the pool contained the retail AI, so part of the
gain may be specific to exploiting it; human and live games are the check on that. The policy entropy rose from 1.5
to 2.6 as the KL weight fell to 0.01, so the follow-up run (rl3) keeps the KL weight at 0.03 to 0.02 and the entropy
bonus at 0.0005.

### Might 1v1 suite (2026-10-29)

Same 1v1 suite with `--mode might` (win by holding every capital), greedy, no search, model seat at Normal, 200 seeds
x 2 seats, 0 illegal commands, CPU. `might1_b6c96` is 600 updates of Might self-play from rl3 u480 (rl3 settings, 3060,
6.4 h). `rl3 u480` has had no Might training. Results: `/kvm/polytopia-ai-data/arena/might/`.

| model | vs retail:hard | vs retail:crazy |
|---|---|---|
| rl3 u480 (Domination only) | 370-30-0, 92.5% (89.5 to 94.7) | 303-97-0, 75.8% (71.3 to 79.7) |
| might1 final | 367-33-0, 91.8% (88.6 to 94.1) | 325-74-1, 81.4% (77.3 to 84.9) |

Head to head, might1 against rl3 u480 in Might: 179-218-3, **45.1%** (40.3 to 50.0).

Reading it: the Domination model carries over to Might almost unchanged, since the two modes differ mainly in the
win condition. Might self-play raised the score against retail Crazy by about 6 points but lost head to head, so the
gain looks specific to the retail AI (which is in the opponent pool) rather than general strength. The Might
baseline stays **rl3 u480** (`policy_id rl3_b6c96@480`), for live and BPolait games too.

### Older results (suite v1: Small, Continents, 50 turns, adjudicated on score)

The first table (2026-10-20) used the old suite and is not comparable with the one above: 15% of its games were
adjudicated, up to 46% for Easy v Normal, and the adjudicated winner was the top-score player, which is no longer a
result. Its numbers for the agents not re-run are kept for reference only: `mcts:50:topk=5` 914, `mcts:200` 267,
`mcts:50` 181, random -40 Elo (anchor `retail:normal` = 1000; retail Crazy 1418, Hard 1237, Easy 785). The MCTS notes
below still hold qualitatively but should be re-measured on the new suite before they are quoted.

### Reading it (from the first table)

- **Retail speed.** A retail game takes 0.1 to 0.7 s of CPU, so a 1,000-game retail table takes under a minute on 16
  threads. The cost of a table is all in MCTS.
- **MCTS speed.** Plain random-rollout MCTS spends about 90 ms per own command at 50 iterations and about 380 ms at
  200 (rough: think time divided by half the game's commands), so a game costs 20 to 130 s of CPU
  (`mcts:50`: 22 s thinking per game on average, `mcts:200`: 131 s). `topk=5` spends about 280 ms per command at 50
  iterations (193 s per game) because of the retail candidate scoring. The 150-turn cap lets an MCTS game run longer
  than the 50-turn limit it was timed under, so its cost per game on the new suite is higher and unmeasured.
- **MCTS strength.** Plain MCTS lost every game to every retail difficulty (0 of 440 games) and only beat random, in a
  slow game decided on score. More iterations help little. Pruning the root to the retail AI's top 5 candidates is the
  big gain: `mcts:50:topk=5` beat Easy 70% and lost to Normal, but this rests on 20 seeds and mostly measures the
  retail candidate list. The cheating (full information) is part of that number. Under elimination-only scoring a
  passive agent no longer wins by holding more points, so the MCTS-versus-random numbers will change.
- **Caveats.** Elo values for agents that win or lose every game are set by the prior and are only ordinal; the Wilson
  interval of a 0-100 pair (0 to 3.7%) is the honest statement there. The engine ranks Domination winners by score,
  whereas retail may rank by city count (ledger to-do 8e4f7017); this no longer matters for results here, since the
  engine's top-score winner is not used.
