# polytopia-ai game plan v2: beyond "more self-play"

Status 2026-10-30. v1 (design.md) is complete: imitation, PPO self-play, per-turn search, arena, live-play interface. Training is now in Might mode.

## Where v1 stopped

- rl3 u480 (1.4M params, Domination) beats retail Crazy 79% and Hard 94.5% on the 1v1 suite. In Might, untrained for it, it scores 75.8% and 92.5%.
- Two seeds of the same run, and three runs of a 2.4 times bigger network, all stop at the same level. More updates and more parameters on the same recipe are not buying strength.
- The per-turn search added about 110 Elo to the imitation model but nothing to rl3. Its value estimates are the limit, and it only looks through our own turn.
- All three wide-mix runs (big maps, up to 8 players) got worse than rl3.

So the next gains need better information and better targets, not just more of the same.

## Workstreams

### 1. Belief model: what is under the clouds

The engine knows the full map in every self-play and imitation game, so labels for hidden tiles are free.

- Targets, per tile under fog: terrain, resource, village or ruin, enemy city, enemy capital, enemy unit present. Per opponent: capital tile, city count, rough army size.
- Step A: auxiliary heads on the existing trunk, trained alongside policy and value. This is the cheapest test; KataGo found auxiliary targets (ownership) improve the main network even when the heads are not used at play time.
- Step B: if A helps, a separate small belief network whose predicted planes are fed to the main network as extra input channels, so the policy reads "probably a capital here" directly.
- Step C: honest search under fog. Sample complete maps from the belief model and search over those samples. Today's MCTS cheats by seeing through fog; this replaces that.
- Caveat: map generation is random, so exact hidden terrain is partly unpredictable. Capitals, villages and coastlines follow generator rules and should be predictable. Measure accuracy against a "most common value" baseline before relying on it.

### 2. Engine-computed features (heuristics the network does not have to learn)

- Per action: combat preview (damage dealt and taken, kill or not, retaliation), captures a city or village, ends adjacent to how many enemies. The engine already computes these.
- Per tile: threat map (tiles enemies can attack next turn), our reach, city defence and siege state, distance to each known capital.
- Per player: projected income, tech affordability, unit counts by type.
- The retail AI's own candidate score per action is already in the data (cand_score) and can be an input.
- Measure each group by ablation in the arena, not by imitation accuracy.

### 3. Set plays

- Opening book per tribe and map size for the first 3 to 5 turns, built offline with deep search from the best network; checked in the arena before use.
- Scenario curriculum: self-play started from chosen mid-game positions (rush defence, naval landing, capital siege, Might capital races), taken from self-play, live games and later human replays.
- Not planned: hand-written macro actions. They tend to lock in weak play; the scenarios teach the same patterns without that.

### 4. Better training targets

- Search-improved targets (AlphaZero style): train the policy towards the search's choice and the value towards the search result, not only the sampled move. This is the standard way past a PPO plateau, and it needs the belief model for honest search under fog.
- League training: exploiter runs whose only job is to beat the current best, so the main run learns not to have holes; keep retail, imitation and BPolait-like styles in the pool.

### 5. Speed

- Rollouts: the per-step loop is Python lockstep. Moving the collector into Rust (VecEnv drives games and batches inference requests) removes most of it.
- Retail AI opponents take 93% of engine CPU. Use more network opponents (GPU) and fewer retail ones in the pool.
- Inference: CUDA graphs or torch.compile for the fixed batch sizes; fp16 everywhere.
- Search: batch the turn rollouts of many games into one GPU call; today it runs per game on CPU at about 64 ms a decision.
- Engine: a cheaper state clone only if search profiling shows clone time matters (it did not in v1).
- One learner, many actors across 3060, 3090 and 4090, instead of three independent runs.

### 6. Live play and external benchmarks

- Verify the headless client in public random matchmaking with our model (client repo, client manager). Needs Kenzi's go-ahead for public games.
- BPolait (Might, Tiny and Small): benchmark through private games, 3 to 5 at once, opponent tagged in every record.

## Order

1. Belief auxiliary heads (1A) and per-action combat features (2), each measured in the arena against the current best. Both use existing data plus new label columns.
2. Rust-side collector and batched search (5), so 3 and 4 are affordable.
3. Belief inputs and sampled-map search (1B, 1C), then search-improved targets (4).
4. Opening book and scenario curriculum (3), using live and BPolait losses to choose scenarios.

Every step is judged the same way: the Might 1v1 suite against retail Hard and Crazy, plus head to head against the current best, 400 games each.
