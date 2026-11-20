# Game analysis: the win chance, the gate, and the deeper check (2026-11-20)

What the website's game analysis gets from the AI service (`evaluate`, `interfaces.md` section 1.3.1): for any point
of a game, each player's chance of winning the game by that game's own rule, read only from that player's own fogged
view, and the mover's preferred moves. The number must be honest: of the players given p, about p must win. Where we
cannot show that, the service gives null and says why.

## 1. What counts as winning

The win chance is the chance of being the engine's winner when the game ends (`Engine::winners`, one player;
`check_game_over` and `pick_winner` in engine/src/engine/turn.rs):

| rules | the game ends | winner |
|---|---|---|
| Perfection (turn limit, 30 on the weekly) | at the turn limit (it plays on after the last opponent falls) | the best living player by score |
| Domination (no turn limit and no win flag, as the website and `Engine::try_new_game` set it up) | when one player is left | that player |
| Might (win by capital and by extermination, no turn limit) | the moment one player holds every capital, or one player is left | that player |
| any, once no person is left alive | when one player is left | that player |
| a resignation (human corpus) | at once | the other player of a two-player game |

A game with one player (solo practice) has no opponent: the service gives null with that reason. A finished game
gives the exact answer (1 for the winner, 0 for the others) and an eliminated player gets 0.

## 2. Data

Value rows (`polyai-gen/src/value_shard.rs`): at sampled positions, one row per living player, encoded from that
player's own view (`encoding.md`), labelled with the winner's slot relative to that player, plus who plays each slot
(a person, or a bot and its level) and whether the row's player is the one to move. Two sources:

- **Human games** (`polyai-human --value-rows`): the 12,488 fetched games. 10,650 replay cleanly; all are two people
  against each other, rules mode Might; 8,906 are decided (most by resignation). One command in 8 is sampled.
- **Website-style games** (`polyai-webgen`): generated as the web's engine service sets up a game
  (`Engine::try_new_game`, one person's seat, sometimes two, among bots at easy to crazy), modes Perfection
  (30 turns), Domination and Might, 2 to 6 players, every map type, tiny to large maps. The person's seat is played by
  the retail AI at a level drawn per game (easy to crazy, or with no handicap), so the person's side has a spread of
  strengths. 78,000 games in three batches (the third weighted to two-player Domination and Might); one command in 32
  is sampled.

The website itself has only three games so far (one finished), so website games are checked on generated games only.
A real person who is much stronger or weaker than the stand-in mix could see a number that is off for them.

Splits are by game, never by position: a hash of the game key puts one game in ten in `test` and one in ten in `fit`.

## 3. The gate (`python -m polyai.value_gate`)

Per game kind and stage, on the `test` games:

- Kinds: `human2` (two people), and for games with bots `<mode>-<players>` with players grouped as 2, 3-4 and 5-8.
- Stages, by game turn: 0-2, 3-5, 6-10, 11-20, 21-30, 31-50, 51+.
- Reported: rows, games, the base rate (the share of rows whose player won), the Brier score and the Brier score of
  always saying the base rate (skill = 1 - Brier / base Brier), the expected calibration error over ten buckets
  (ECE), and the worst gap between the mean prediction and the share that won in any bucket holding at least 2%
  (and 200) of the rows.
- A stage passes with at least 100 games, ECE <= 0.05, Brier no worse than the base rate's, and no bucket off by more
  than 0.10. A stage with fewer games is "too few games" and gets no number.

## 4. The analysis value

A separate network (`polyai.analysis.ValueNet`): the trunk and value head of the champion policy (auto_r0077),
fine-tuned on the value rows of every player (both sides of every game, each from its own view; `polyai.train_value`),
with two more public inputs: who plays each player slot, and whether the viewer is to move. Slots without a player
are masked out of the softmax. No truth input: it sees exactly what that player sees.

Then a calibration layer: per (kind, stage, viewer to move or not) a logistic map sigmoid(a logit(q) + b) of the
network's win chance q, fitted on the `fit` games. A (kind, stage) keeps its map only when it passes the gate on the
`fit` games and on the `test` games; elsewhere the checkpoint records the reason and the service gives null.

## 4.1 Gate results (2026-11-20)

Held-out `test` games (one game in ten): 1,365,378 rows. Full tables (Brier score, expected calibration error and the
base rate's Brier score per kind, stage and checkpoint, raw and calibrated): `results/calibration/
analysis-gate-2026-11-20.md`; the short JSON next to it. Each cell below is live / v0b / v1 / v2, then the checkpoint
the service uses there and the number of held-out games (P passes, F fails, - too few games or no number):

- **live**: the champion auto_r0077's own value head, raw.
- **v0b**: the same head with the calibration layer (`--steps 0`).

- **v1**: fine-tuned 30,000 steps on the human rows and the first two generated batches (`models/analysis_v1`).
- **v2**: v1 fine-tuned 15,000 more steps with the third batch, weighted to two-player Domination and Might
  (`models/analysis_v2`).

live and v0b were gated on the earlier test set (the human rows and the first 12,000 generated games), so for them
"-" mostly means fewer than 100 held-out games; the game counts shown are v2's.

| kind | 0-2 | 3-5 | 6-10 | 11-20 | 21-30 | 31-50 | 51+ |
|---|---|---|---|---|---|---|---|
| domination-2p | -/-/-/P: **v2** (783) | -/-/-/P: **v2** (1000) | -/-/P/P: **v2** (1104) | -/-/P/P: **v2** (1099) | -/-/P/P: **v2** (980) | -/-/P/P: **v2** (696) | -/-/-/P: **v2** (230) |
| domination-3-4p | F/P/P/P: **v2** (1111) | F/P/P/P: **v2** (1118) | F/-/P/P: **v2** (1120) | F/P/P/P: **v2** (1120) | F/P/P/P: **v2** (1092) | F/P/P/P: **v2** (984) | -/-/P/P: **v2** (569) |
| domination-5-8p | -/-/P/P: **v2** (590) | -/-/P/-: **v1** (590) | -/-/P/-: **v1** (590) | -/-/P/P: **v2** (590) | -/-/P/P: **v2** (584) | -/-/P/P: **v2** (530) | -/-/-/-: **null** (302) |
| human2 | F/P/P/P: **v2** (918) | F/P/P/P: **v2** (912) | F/P/P/P: **v2** (898) | F/P/P/P: **v2** (731) | F/-/P/P: **v2** (131) | -/-/-/-: **null** (11) | -/-/-/-: **null** (1) |
| might-2p | -/-/-/P: **v2** (684) | -/-/-/P: **v2** (885) | -/-/-/P: **v2** (997) | -/-/P/P: **v2** (980) | -/-/P/P: **v2** (822) | -/-/-/P: **v2** (508) | -/-/-/-: **null** (154) |
| might-3-4p | F/P/P/P: **v2** (972) | F/-/P/P: **v2** (982) | F/P/P/P: **v2** (984) | F/P/P/P: **v2** (983) | F/-/P/P: **v2** (968) | -/-/P/P: **v2** (839) | -/-/-/P: **v2** (432) |
| might-5-8p | -/-/P/P: **v2** (432) | -/-/P/-: **v1** (432) | -/-/P/P: **v2** (432) | -/-/-/P: **v2** (432) | -/-/P/P: **v2** (427) | -/-/P/P: **v2** (389) | -/-/P/-: **v1** (219) |
| perfection-2p | -/-/-/P: **v2** (354) | F/-/-/P: **v2** (460) | F/P/P/P: **v2** (517) | F/P/P/P: **v2** (515) | P/P/P/P: **v2** (486) | -/-/-/-: **null** (0) | -/-/-/-: **null** (0) |
| perfection-3-4p | F/P/P/P: **v2** (1185) | F/P/P/P: **v2** (1202) | F/P/P/P: **v2** (1203) | F/P/P/P: **v2** (1203) | F/P/P/P: **v2** (1190) | -/-/-/-: **null** (0) | -/-/-/-: **null** (0) |
| perfection-5-8p | F/P/P/P: **v2** (693) | F/P/P/P: **v2** (693) | F/P/P/P: **v2** (693) | F/P/P/P: **v2** (693) | F/P/P/P: **v2** (687) | -/-/-/-: **null** (0) | -/-/-/-: **null** (0) |

The service loads v2 then v1 (`--value models/analysis_v2/value.pt models/analysis_v1/value.pt`). Each (kind,
stage) is scored by the first checkpoint whose calibration passed the gate there; elsewhere the answer is null with
the reason. Every kind the website hosts gets a number at every stage it reaches, except:

- turns 51 and later of five- to eight-player Domination and of two-player Might (neither checkpoint passed: a
  bucket is off by more than 0.10, or the fit split failed);
- human two-player games after turn 30 (too few held-out games);
- solo games, games with more than one person and no bot except `human2`, and any kind not in the table;
- Perfection games have no turns after 30.

What the gate shows: the live head's opening numbers mean nothing (two-player Domination at turns 0 to 2: Brier
0.305 against 0.25 for always saying the base rate), and a calibration layer alone does not fix that (v0b passes only
where the head already ranked positions). Training on both sides of every game fixes the opening: v2's two-player
Domination at turns 0 to 2 has Brier 0.143 and ECE 0.033.

Caveat: website-style games are gated on generated games in which the retail AI, at a level drawn per game, stands
in for the person. The website has only two games so far; both get a number at every point for every living player
(below). A person much stronger or weaker than the stand-ins could see
numbers that are off for them; recheck once there are a few hundred real website games.

Real dev games (`/kvm/tmp/analysis/data/web.sqlite3`, CPU, 8 threads, every turn start and every person move in one
request, with the stored turn positions): the finished five-player game (one person, four Normal bots, 2,767
commands, 210 points) answers in 1.4 s (1.7 s with the top 3 moves), and every living player gets a number at every
point (the person was eliminated at turn 10, after 85 points); the four-player game in progress (156 commands, 57
points) answers in 0.3 s, numbers everywhere.

## 5. The deeper check (`search`, built 2026-11-20)

The website's present deeper check plays every legal command out to the end of the game with the retail AI and
compares final scores. With about six play-outs a move over 170 turns the noise is far larger than the differences
between moves, so the ranking is close to random.

`search` (`interfaces.md` section 1.3.2, `Evaluator.search`) replaces it, for finished games as with `after` (it plays
on the true state):

1. Candidates: the policy's top k moves (default 6), the move played, and any the caller names.
2. For each candidate, n short play-outs (default 16): the candidate, then the rest of the mover's turn by the
   sampled policy, then every other player's turn (bots by the retail AI at their own level, which is their real
   behaviour; people by the policy), up to the start of the mover's next turn. Play-out j of every candidate uses the
   same random draws, so the differences between candidates are paired. The retail AI is deterministic given the
   state, so in a game against bots the spread comes from the mover's own rest of turn.
3. Each leaf is scored by the calibrated analysis value from the mover's own view; where the gate gives no number,
   the search gives none either.
4. Per candidate: the mean leaf win chance and its standard error, the mean paired difference from the move played
   and its standard error, and the shares of play-outs in which it ends better or worse.

Measured on the dev database's finished five-player game (one person, four Normal bots), CPU, 8 threads, k 6 and 16
play-outs: 0.7 s at turn 0, 1.1 s at turn 5, 2.3 s at turn 8. At turn 0 the move played comes out best, the
others 0.5 to 2.8 points behind with paired standard errors of 0.2 to 0.4 points; at turn 5 three candidates reach
the same end-of-turn state and tie exactly, and an early `EndTurn` is 21 points worse. Mid-turn values move by a few
points from command to command (the value is read from positions with part of the turn played), so differences
below about two standard errors should be shown as "no clear difference".
