# polytopia-ai

A machine-learned bot for The Battle of Polytopia: one neural network that ranks the legal moves of a position and
estimates who is winning, trained on games played by our own engine (`polytopia-engine`). This repository holds the
training code, the models, the results of every experiment and the notes written along the way.

State on 20 November 2026: the current best model ("round 77") beats the game's built-in AI at its hardest level in
89% of 1v1 Might games, playing its single best-rated move with no search. In live online games against BPolait, the
rival bot we use as an outside benchmark, our models win about a third. Training has been flat since round 77; the
section "Where it stands and what is open" says what we know about why.

## Contents

| Path | What it is |
|---|---|
| `crates/` | Rust: position and action encoding (`polyai-core`), Python bindings to the engine and the rollout collector (`polyai-py`), imitation data generator (`polyai-gen`), human game converter (`polyai-human`), match runner (`polyai-arena`) |
| `python/polyai/` | the model, imitation training, PPO self-play, evaluation, search, calibration, the policy server used in live play |
| `tools/` | the automatic training loop, A/B evaluation, reports and one-off analysis scripts |
| `deploy/` | systemd units for the training loop and its helpers |
| `models/` | 24 trained models with their training configuration (`models/README.md`) |
| `results/` | raw game records and logs behind every number quoted here (`results/README.md`) |
| `docs/` | design, encoding, method notes and dated experiment reports |

## Approach

**The game as the network sees it.** Every decision is one engine command (move a unit, train, research, end the
turn), so a turn is 10 to 40 decisions. The observation is from the point of view of the player to move and honours
the fog: a tile grid of categorical and numeric fields at the map's own size, plus a vector of global facts (stars,
techs, scores, what is known of each opponent). Polytopia has no fixed action set, so the network scores the legal
list: each legal command is described by its kind, argument, source tile and target tile, and gets one logit.
`docs/design.md` and `docs/encoding.md` have the detail.

**The model** is a residual convolutional network with global pooling blocks (as in KataGo), which lets one set of
weights serve every map size. The policy head reads the trunk features at a command's source and target tiles; the
value head gives the probability that each player wins. The model used throughout is 6 blocks of 96 channels,
1.42 million parameters. Larger ones (3.4M and 10.6M) imitate better but do not play better.

**Training in stages.**

1. *Imitation.* The engine plays its built-in AI against itself at mixed difficulties and the network learns to
   predict its moves. This gives a legal, sensible player at about the built-in Normal level.
2. *Self-play with PPO.* The network plays whole games against a pool of opponents: the built-in AI at four levels,
   the imitation model, and frozen earlier copies of itself. Opponents it beats less than 70% of the time are played
   more. The only reward is the result (win 1, loss 0, a game that reaches the 150-turn cap 0.5); there is no score
   term and no shaping. A KL penalty towards the imitation model keeps the policy from drifting into nonsense.
   Credit is assigned with GAE.
3. *Human games.* Finished tournament games between people are fetched, replayed through our engine to check them
   command by command, and turned into training positions. The self-play model is fine-tuned to predict the human
   moves, then goes back to self-play.
4. *An automatic loop.* A round trains a candidate for 300 updates (9.8 million decisions, about an hour on an RTX
   4090 laptop), then plays it against the champion. The candidate replaces the champion only if it passes the gate.

**How anything is judged.** Every comparison is a 400-game match: 200 map seeds, each played twice with the seats
swapped so that the map and tribes cancel out, both sides playing their top move, seeds never used in training. A
match of 400 games resolves about 5 points either way, and results are quoted with 95% intervals. The loop's gate
is 55% against the champion (two standard errors above even) plus a guard that the score against built-in Crazy has
not dropped by more than 8 points. A change to the training recipe is tested as an A/B: two runs from the same
starting model, same seed and settings, differing in the one option. Besides win rates we use paired sign tests on
identical games, per-tribe and per-map tables, a calibration check of the value head on human games, and a per-turn
autopsy of live losses.

**Live play.** `polyai.serve` serves a model over a local socket; the headless game client (its own repository)
asks it for every move of real online games and records which model made each one.

## What we did and what came of it

Scores are against the built-in AI at Crazy unless stated, in 1v1 on the evaluation suite. Until 27 October the
mode was Domination (win by elimination); since then Might (win by taking every capital), the format of
tournament play.

| Date (2026) | Step | Result |
|---|---|---|
| 20 Oct | Engine bindings, encoding, data generator, match runner | Built-in AI ladder on our suite: Easy 697, Normal 1000, Hard 1282, Crazy 1472 Elo |
| 21 Oct | Imitation of the built-in AI, 1.4M parameters | 10.6%; Elo 1057, just above Normal |
| 21 Oct | The same with 3.4M parameters and more data | 12.1%: better imitation, no stronger |
| 21 Oct | First self-play run, 200 updates | 68.9%; Elo 1573, above Crazy |
| 21 Oct | Per-turn search on top of the network | +110 Elo for the imitation model; later, nothing for rl3 |
| 24 Oct | Longer self-play run ("rl3", update 480) | 79% (94.5% against Hard). Two seeds and a 2.4 times larger network all stop here |
| 27 Oct | Training moves to Might | rl3 carries over untrained: 75.8% |
| 29 Oct | Value-rank test: can the value head rank candidate turns? | No (rank correlation +0.02), so search work stops |
| 29 Oct to 1 Nov | Experiments on the Might recipe (table below) | Engine features and the advantage filter help; the rest does not |
| 1 Nov | Rollout collector moved into Rust | Up to 4 times more decisions per second when the collector is the limit |
| 1 to 9 Nov | Human games: collect ids, fetch, replay, convert | 1,256 clean tournament games, 413,504 positions |
| 9 Nov | Fine-tune on human games, then 300 updates of self-play | Beats rl3 63.2% head to head; 77.8% |
| 9 Nov | Automatic loop starts | 49 rounds without a promotion while each round restarted from the champion |
| 9 Nov | Weight averaging of models | No average passes the gate. Side finding: every model scores 14% on Water World maps |
| 14 Nov | Round with water maps in the training mix | Beats the previous champion 56.0%; 86.2%; Water World 41.8% |
| 16 Nov | Training moves to an RTX 4090 laptop | 7.9 s an update against 23.8 s on the RTX 3060 |
| 16 Nov | Loop carries on from failed candidates for up to eight rounds | Rounds 53 to 77: seven promotions in 25 rounds |
| 17 Nov | Autopsy of 368 live games against BPolait (32% won) | Losses split at turns 4 to 8 on economy; we build almost only warriors; the value head is overconfident |
| 17 Nov | Training aimed at the benchmark: 85% 1v1 games, and an imitation of BPolait's play as a pool opponent | In place from round 66 |
| 18 Nov | Round 77 becomes champion | 55.1% against round 75; 89.0% |
| 19 to 20 Nov | A/B of per-turn credit and of a critic that sees under the fog | Neither beats the control |
| 20 Nov | Calibration check on 8,859 human games | Value head sound from turn 11, meaningless in the first five turns |
| 20 Nov | A/B of engine features, and of the advantage filter switched off | Features beat the control 55.4% and are on in the loop from round 98; the filter stays on |

The champions in order, each measured against the one before it over 400 games:

| Champion | Date | Against the previous champion | Against Crazy |
|---|---|---|---|
| rl3 (update 480) | 24 Oct | | 76.0% |
| human fine-tune plus self-play | 9 Nov | 63.2% | 77.8% |
| water round | 14 Nov | 56.0% | 86.2% |
| round 53 | 16 Nov | 55.4% | 83.5% |
| round 55 | 16 Nov | 55.6% | 81.0% |
| round 60 | 17 Nov | 58.2% | 83.5% |
| round 66 | 17 Nov | 57.0% | 83.0% |
| round 72 | 18 Nov | 57.6% | 86.8% |
| round 75 | 18 Nov | 57.2% | 84.5% |
| round 77 | 18 Nov | 55.1% | 89.0% |

### Experiments on the training recipe

Each row is a 300-update run compared with a control from the same starting model over 400 games.

| Idea | Head to head against its control | Verdict |
|---|---|---|
| Engine-computed features as extra inputs (combat previews, threat maps and the like) | 58.2% and 63.1% on two seeds; 55.4% from round 77 | Helps. In the live loop from round 98 |
| Auxiliary loss: predict what is under the fog | 58.1% on one seed, 46.2% on a second | Not reproducible |
| Features and the fog loss together | 46.4% against features alone | Worse |
| Keep only the half of positions with the largest advantage in each update | 54.2%; switching it off from round 77 scored 47.5% | Adopted: half the update cost for no loss |
| Averaged weights (EMA) as the exported model | 52.1% to 53.5% | Within noise |
| Entropy bonus decayed to zero | 53.0% | Within noise |
| Moving KL anchor (R-NaD style) | 46.0% | Worse |
| Might-only self-play from rl3, 600 updates | 45.1% | Better against Crazy (81.4%), worse head to head: a gain against the built-in AI is not a gain in strength |
| Exploiter trained only against a frozen rl3 | 43.1% | Finds no hole in rl3's top-move play |
| Bigger networks, big maps with up to 8 players | | All at or below rl3 |
| Weight averaging of champions | 46.0% to 53.4% | None passes the gate |
| Credit that fades once per game turn, not per command | 45.9% | Not adopted |
| Per-turn credit plus a critic that sees under the fog | 41.8% | Not adopted; its value estimates are also the worst calibrated |

Reports with the method and the full tables: `docs/arena.md`, `docs/rl.md`, `docs/belief.md`, `docs/features-ab.md`,
`docs/search.md`, `docs/human-data.md`, `docs/loss_autopsy.md`, `docs/autoloop.md`, `docs/collector.md`.

## Where it stands and what is open

**Training is flat.** Rounds 78 to 97 all failed the gate, averaging 46.3% against round 77 (best 53.9%, round 97),
while their scores against Crazy kept rising, to 89.5% on average. A lower gate would not help: the candidates are
losing. What the diagnostics of 20 November show:

- A control run from round 77 scores 48.6%, 40.8%, 51.2% and 46.1% against it at updates 20, 60, 140 and 300. There
  is no steady slide, only a level a few points below round 77 from the start.
- With both sides sampling their moves, as in training, the same control scores 52.5% against round 77. Training
  optimises sampled play, the gate measures top-move play, and the two have come apart.
- Round 77 playing its top move beats itself sampling at temperature 1.0 by 57.6%, and is level with itself
  sampling at 0.5 (49.8%). Training samples at 1.0, so the learner and the model opponents in its pool play
  noticeably worse in training than the model plays live. An A/B of training at temperature 0.5 is queued.
- Round 77's promotion was not luck: it scores 55.1% against round 75 again on fresh maps. Round 95 scores 49.9%
  against round 75, so the 18 rounds since have drifted back to round 75's level.

**The value head cannot yet support search.** On human games it is well calibrated from turn 11 (sides it gives
about 90% win 94%), but in the first five turns it gives 55 to 85% to whoever is to move. The cause is in the
training data: only the learner's own positions are trained on, and the learner wins 65 to 75% of its training games
because half the pool is weaker than it. A value that means something between equals needs positions from both
seats of self-play games. The lookahead planned for live play waits for that.

**Against BPolait** we win about a third of games. The autopsy points at the opening (growing cities and income more
slowly through turns 4 to 8) and at an army of warriors with no archers or riders. Our imitation of BPolait is a weak
stand-in for it: the model beats the imitation 67 to 84% of the time in training.

**Against the PolytopAI website's bot** (played through the site's own browser game, `docs/polytopai.md`) round 77
won 9 of 199 games at the site's easier setting. The bridge can run about 950 such games an hour and feed them into
training (`--polytopai-games`), which has not yet been tested as an A/B.

Next, in order: training against the PolytopAI website as an A/B (running), value training on both seats, an exploiter
trained to beat the champion and added to the pool (league training as in AlphaStar), then throughput (the GPU is
about 40% busy during training).

## Running it

The crates build against the engine at `../engine` and the search baselines at `../weekly`, so this repository does
not build on its own. Python needs PyTorch and the `polyai_engine` module built from `crates/polyai-py` with
maturin.

```
cd python
python -m polyai.arena --model ../models/auto_r0077 --opponent retail:crazy --mode might --seeds 200 \
    --threads 16 --out /tmp/r77_crazy.jsonl                       # 400 games against built-in Crazy
python -m polyai.rl --name myrun --init ../models/auto_r0077 --ref ../models/bc3_b6c96_v11 --mode might \
    --updates 300                                                 # self-play; docs/rl.md lists every option
python -m polyai.calibrate --shards DIR --model r77=../models/auto_r0077/ckpt_final.pt
python -m polyai.polytopai --ckpt ../models/auto_r0077 --games 100 --parallel 8 --level 1 \
    --out /tmp/r77_site.jsonl                                     # games against the PolytopAI website
python -m pytest -q                                               # the Python tests
```

`tools/autoloop.py` is the training loop (`docs/autoloop.md`), `tools/ab_eval.sh` plays every match of an A/B, and
`tools/elo.py` turns game records into win rates with intervals and Elo.

Datasets, run directories and logs stay outside git (`/kvm/polytopia-ai-data/` on the training hosts). The human
games themselves are not in this repository, only the code that fetches and converts them and the aggregate results.
