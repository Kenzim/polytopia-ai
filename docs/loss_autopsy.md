# Loss autopsy: where our model loses to BPolait

Run on 17 November 2026 over every finished BPolait game in `client/games/` (games played 2 to 17 November): 368
games that ended by the rules, 118 won and 250 lost (32%). Abandoned games and games resigned on a divergence are
left out. 352 of the 368 were played by `wrl1_water_b6c96@300` (112 won, 32%); the rest are 11 by `rl3_b6c96@480`
and 5 by `hrl1_human_b6c96@300`. No game by `auto_r0060@300` had finished yet, so rerun the tool with
`--model auto_r0060@300` once it has a sample.

## The tool

```
cd ai
python3 tools/loss_autopsy.py [--opponent bpolait|polytopai|...] [--model ID] [--map NAME] [--size N] [--no-replay]
```

It replays each game's command stack on the engine with the `replay_stats` example of polyai-core
(`crates/polyai-core/examples/replay_stats.rs`, built on first use), which prints per-turn rows for both seats
(score, stars in hand, income, cities, city levels, population, villages and enemy cities captured, units, army value
as the sum of unit costs, techs, kills, losses, territory, explored tiles, and the turn's moves, attacks, trains,
builds and researches) plus every research, train, build, capture and city reward. Replays are cached in
`/kvm/tmp/autopsy/games/`, a per-game table goes to `/kvm/tmp/autopsy/games.tsv`, and the report goes to stdout.
The whole corpus replays in about 7 seconds on epyc. A game that has ended stays at its last row in the later-turn
columns, so the turn 20 and 25 columns mix finished and running games; read the turn 2 to 12 columns as the clean ones.

## Findings

1. **Every game is decided by a capital capture, and losses are short.** All 250 losses end with BPolait taking our
   capital (median turn 22, range 8 to 71 apart from one 361-turn outlier); all 118 wins end with us taking theirs
   (median turn 25). No game reaches a score decision. Example quick losses: pangea Tiny on 16 November, lost on turn
   8 at 845 to 2,535; dryland Small on 17 November, lost on turn 9 at 1,050 to 2,715.

2. **The curves split between turns 4 and 8, economy first.** Measured as us minus BPolait, losses fall behind wins
   by an effect size of 0.5 or more first in city levels (turn 4), then units and units trained (turn 6), then
   score, income, population, army value and territory (turn 8), then cities and villages (turn 10). At turn 8 in
   losses our income is 10.1 against BPolait's 10.4, and at turn 12 it is 12.6 against 15.7; in wins it is 11.8
   against 9.6 at turn 8. After turn 10 our city count in losses stops growing (3.0 at turn 10, 3.1 at turn 12)
   while BPolait goes from 3.7 to 4.7 by turn 15. Who leads at turn 8 matters a lot: ahead on income we win 43% of
   183 games, behind 21% of 185; ahead on city levels 49% of 132, behind 22% of 236.

3. **We train almost nothing but warriors.** By turn 10 we have trained about 10 to 11 warriors and under one
   catapult, and essentially no archers, riders, swordsmen or defenders, in wins and losses alike. BPolait trains
   about 5 warriors plus archers (3.9 per game in our losses, 2.2 in our wins), riders and swordsmen, so its army is
   worth more with fewer units: in losses at turn 10 its army value is 30.7 against our 21.6. We never research
   archery; our early techs are forestry, organisation, fishing and mathematics, while BPolait's first research is
   often hunting then archery, or mining. Fighting starts around turn 6, and we trade badly: in Small-map losses by
   turn 15 we have made 5.7 kills for 11.6 units lost.

4. **We explore much less than BPolait, in wins and losses.** At turn 6 we have seen 54 tiles against its 68 in
   losses (52 against 66 in wins); by turn 10 the gap is 30 tiles in losses and 15 in wins. Village capture keeps
   pace early (first village on turn 3 in half the games for both sides) and then stalls in losses at about 2 to 2.5
   villages.

5. **The value head is overconfident and late.** The model's own P(win) averages 0.76 at turn 2 in both wins and
   losses, against a real win rate of 32%, and only separates losses (0.72) from wins (0.83) at turn 8, at the same
   time as the economy curves. It does not foresee the losses earlier than a simple income count would.

6. **Matchups and maps.** We win 43 to 56% against Oumaji, Vengir, Quetzali and Zebasi but 16 to 25% against
   Luxidoor, Bardur, Hoodrick, Imperius and Kickoo (32 to 42 games each). Water maps are kinder (archipelago 40%,
   waterworld 40%) than land maps (pangea 26%, continents 26%, lakes 29%, dryland 32%). Tiny maps are worse than
   Small (28% of 197 against 37% of 171); on Tiny the losses already trail on units at turn 2.

What is not the problem: we do not over-research (tech counts are level with BPolait up to turn 15) and we do not
hoard stars (we hold 2 to 3 at the end of a turn, BPolait 4 to 6).

## Most likely training fixes

1. **Put a BPolait-like opponent in the pool.** The pool is half retail AI, which never punished a warrior-only army
   or slow growth. Train a small imitation policy on the BPolait corpus (`tools/bpolait_corpus.py` already makes the
   shards) and give it a share of the pool, or weight retail Crazy openings that go archery. Test: one autoloop
   round with and without it, judged by live win rate against BPolait (at least 100 games each, Wilson interval) and
   by the turn 10 unit mix from this tool (archers and riders above zero).
2. **Fix the value target before trusting it.** The value head says 0.76 in games we win a third of the time. Use
   the all-seeing critic and per-turn credit from ledger note ca179c04, and add these records as a calibration
   check: mean predicted P(win) by turn against the real win rate, and how early losses separate from wins. Test:
   rerun this tool's value section and `polyai.value_rank` on the new head; pass if turn 2 to 4 predictions are
   within 10 points of the real rate and losses separate by turn 6.
3. **Reward the turn 4 to 10 economy and a mixed army.** Losses are already visible in city levels, income and army
   value by turn 8. A small potential-based shaping term on income and city levels relative to the opponent (zero-sum,
   so it cannot change the optimal policy) or a KL term to the human imitation policy on train and research choices
   would push against the warrior-only habit. Test: an A/B round of 300 updates each, compared on the turn 8 income
   gap and turn 10 unit mix from this tool, then on live win rate.
