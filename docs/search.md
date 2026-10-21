# Per-turn search (`polyai.search`)

`SearchPolicy` wraps a `ModelPolicy` and looks ahead through the rest of our own turn with the value head. It has the
`Policy` interface of `docs/interfaces.md` (`choose(obs, temperature=...)` returning `Choices`) with one addition: it
needs the live games, so `choose` takes `games=[polyai_engine.Game, ...]`, one per position of `obs`, in order.
Without `games=` it raises. Files: `python/polyai/search.py`, `python/tests/test_search.py`.

A Polytopia turn is many commands (moves, attacks, trains, techs) ending in an end turn. The network picks one command
at a time and its value head says P(we win) for the position. The search asks, for the few moves the network likes
best, where the turn would end up if it carried on as the network would.

## Algorithm

At a decision, if the network's top move has probability below `skip` (and there is more than one legal move):

1. Take the network's top `k` legal moves.
2. Clone the game once per candidate and apply the candidate.
3. Play every clone on greedily with the network until the network chooses end turn (that clone is valued at the
   position it chose to end in), or `budget` commands are used up (valued where it stopped). The value comes from our
   seat's view (`winner_logits[0]`, P(our seat wins)). A line that ends the game is valued 1, 0 or 0.5 (win, loss, draw)
   without the network. Ending the turn straight away is valued at the position as it stands.
4. Score a candidate `log prior + w * logit(value)` (prior is the network's probability at T = 1; values are clipped to
   [0.001, 0.999]) and play the best. With a temperature the move is sampled from softmax(score / T) over the
   candidates, and `logp` and `top` are those of that distribution.

All rollouts of all positions given to one `choose` call advance in lockstep, so every network call is one batch over
candidates and games (split at `max_batch` rows).

**Plan cache.** The greedy line of the chosen candidate is kept as a plan. The next commands of the turn are taken
from it without a network call (such a step is answered with the network's probability from the rollout), until a step
where the network was unsure (probability below `skip`) and the turn still has searches left (`turns`), where the
search runs again. A plan is checked against the live legal list at every step (hash of the list and presence of the
action) and dropped if the game is not where the rollout saw it. Plans are only followed for greedy moves.

With `k = 1` nothing is searched and the answers are exactly `ModelPolicy`'s.

## Parameters

| option (`PATH:search=K:...`) | default | meaning |
|---|---|---|
| `search=K` | 4 | candidates per decision; 1 is the plain policy |
| `w` | 1.0 | weight of logit(value) against log(prior); 0 keeps the network's order |
| `skip` | 0.9 | no search when the top move has at least this probability |
| `budget` | 200 | most commands in a rollout after its first move |
| `turns` | 4 | most searches per game turn (0 = unlimited) |
| `plan` | 1 | follow the chosen line instead of asking the network again |

`PATH:sample=T` still sets the sampling temperature and can be combined: `RUN:sample=0.3:search=4:w=1.5`.

## Running

```
cd /kvm/polytopia/ai/python
TMPDIR=/kvm/tmp/tmp nice -n 10 /kvm/polytopia/.venv/bin/python -m polyai.arena --device cpu --threads 8 \
    --model /kvm/polytopia-ai-data/runs/bc3_b6c96_v11:search=4 --opponent retail:normal,retail:hard \
    --seeds 50 --out /kvm/polytopia-ai-data/arena/search/k4.jsonl
python ../tools/elo.py --anchor retail:normal --anchor-elo 1000 raw.jsonl k4.jsonl
```

(`scripts/run_arena.py` is the same entry point.) Search specs need `--device cpu`. The `VecEnv` does not hand out its
games to clone, so a search spec plays single `Game`s in `--threads` worker processes of one torch thread each, four
games per chunk, and writes the same JSONL as the other specs (agent name
`model:RUN:search=4[:option=value]`). Existing specs are untouched. `play_games` and `play_search_pairing` are the
library entry points; `SearchPolicy.stats` has decisions, searched, plan steps followed, network rows and seconds.

## Speed

bc3_b6c96_v11 (1.4M parameters), CPU, one torch thread per process, 1v1 suite, `nice -n 10`, host shared with other
jobs. Time is inside `choose` only (the retail AI's moves are not counted).

| | ms per decision | decisions per second | share searched | share from plan | policy seconds per game |
|---|---|---|---|---|---|
| raw (`search=1`, 8 games probe) | 5.4 | 185 | 0% | 0% | 3.7 |
| `search=4` (200 games) | 63 to 65 | 15 to 16 | 24 to 30% | 70 to 76% | 31 to 45 |
| `search=8` (200 games) | 119 to 122 | 8 | 24 to 29% | 71 to 76% | 61 to 82 |

A decision here is one command of ours (about 650 per game), so a `search=4` game costs about 30 to 45 seconds of
policy time on one core, against about 4 seconds raw. The cost is the rollouts: about 55 network rows per searched
decision at `k=4`, 110 at `k=8`. The root pass is under 2% of the rows. The plan cache answers about three quarters of
all decisions without a network call, but the saving is small because the rollouts dominate: on the 8-game probe
`plan=0` plays the identical games (same 5699 decisions, same 1429 searches, same rollout rows) at 75.6 ms per decision
against 69.2 ms with the plan, 9% slower. The plan is kept on because it is free and the moves are the same.

## Result

bc3_b6c96_v11 raw and with search, against retail Normal and Hard on the 1v1 suite: 50 seeds x 2 seats per opponent
(100 games each), seeds from the evaluation range, draws (turn cap) half a win. `tools/elo.py`, 1000 bootstrap
resamples, Normal anchored at 1000. Data in `/kvm/polytopia-ai-data/arena/search/{raw,k4,k8}.jsonl`.

| agent | vs Normal W-L-D | score (95% CI) | vs Hard W-L-D | score (95% CI) | Elo (95% CI) |
|---|---|---|---|---|---|
| raw | 51-48-1 | 51.5% (41.8 to 61.1) | 22-74-4 | 24.0% (16.7 to 33.2) | 1011 (953 to 1065) |
| `search=4` | 66-32-2 | 67.0% (57.3 to 75.4) | 37-61-2 | 38.0% (29.1 to 47.8) | 1122 (1067 to 1178) |
| `search=8` | 68-31-1 | 68.5% (58.9 to 76.8) | 38-60-2 | 39.0% (30.0 to 48.8) | 1131 (1079 to 1182) |
| retail Hard | | | | | 1207 (1150 to 1270) |

Search adds about 110 Elo (about 15 points of win rate against both opponents). Pooled over the two opponents (200
games each) raw scores 37.8% and `search=4` 52.5%, a difference larger than the sampling error; each opponent alone
has overlapping intervals. `search=8` is no better than `search=4` within the noise (+9 Elo, CIs overlapping) at twice
the cost, so `k=4` is the default. Still below retail Hard.

## Limits and ideas

- The search sees only our own turn: the value head stands in for everything the opponent does afterwards. The network
  was trained to imitate the retail AI, so its value head is only as good as that imitation's results; a value head
  trained on game outcomes (the PPO runs, `docs/rl.md`) should make `w` larger and the search worth more.
- Rollouts follow the network greedily, so a candidate is only as good as the network's own continuation.
- The network dominates the time (rollout rows), not `Game.clone`. Cheaper rollouts would need a smaller or batched
  network call, not an engine change; a batched `clone` plus `step` in `polyai-py` would only trim the Python loop.
- Not tuned: `w`, `skip` and `turns` are the first values tried. A sweep of `w` and `skip` at `k=4` is the cheapest next
  experiment.
