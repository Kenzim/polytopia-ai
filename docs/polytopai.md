# Games against the PolytopAI website

PolytopAI (polytopai.com) lets anyone play its bot in the browser without an account. `polyai.polytopai` plays those
games with one of our networks instead of a person, so the site's bot is an opponent we can measure ourselves against
and train against, without going through the game's own servers.

## How it works

The site's page talks to four calls under `/api/play/`, identified by a random `x-play-id` header (one game per id):

| Call | What it does |
|---|---|
| `POST new` | starts a game: two tribes, map type, size, the bot's strength (`level` 1 easier, 3 standard). The person is always the first player. |
| `POST move` | plays one of the legal moves the last answer listed. |
| `POST bot` | has the bot play (with `turn: true`, the rest of its turn). |
| `GET state` | the current game again. |

Every answer holds the whole state (tiles, both tribes with their cities, units, techs and stars) and the mover's legal
moves. There is no fog in the data: the state is the true board, with the tiles each player has explored marked.

The bridge has three parts:

* **`polytopai_state.py`** turns the site's state into the engine's position document (`Engine::from_position`). The
  two use the game's own ids for tribes, units, techs, terrain, resources and buildings, so this is a field-by-field
  translation. It also names every site move and every engine action in a common form, which joins the two legal lists.
* **`Game.from_position`** (new in the Python bindings) loads such a document, so the position is encoded for the
  network and its legal actions listed exactly as in training. The engine never steps the game. Every decision starts
  from a fresh translation of what the site sent, so the two rule sets cannot drift apart over a game.
* **`polytopai.py`** plays the games: `SiteClient` makes the calls, `SiteGame` is one game, `SiteArm` runs several at
  once on threads with one batched forward pass for the decisions that are waiting.

The policy only ever chooses among actions that both the engine and the site call legal. An engine action the site does
not offer is dropped from the list before the network sees it; a site move the engine has no counterpart of is never
played. The one engine action without a site move is "stay" (a unit passes): it costs no call, and the bridge marks the
unit as done for the rest of the turn in the positions that follow.

## What differs from our own games

* **The win rule is the same.** The site plays Might: the game ends once one player holds every capital, as in our
  training games.
* **Seat.** We are always the first player.
* **Diplomacy.** The site has none: the engine's peace treaty is never offered.
* **A few rules.** The site lets a road be built on a village tile and the engine does not. The two disagree about
  where a Cymanti centipede and its segments may move, and the site lets a segment explode. Now and then they offer
  different rewards for a city that grew two levels at once; the bridge then takes the site's first. Only what both
  list is ever played.
* **Maps.** Sizes 11, 14, 16 and 18 and the six map types; two players only. The 13 tribes of our encoding only.
* **Stay** is ours alone, as above.

## Playing a match

    python -m polyai.polytopai --ckpt models/auto_r0077 --games 100 --parallel 8 --level 3 --out results.jsonl

needs bindings that have `Game.from_position` (built from `crates/polyai-py` at or after this change, see below).
`--temperature 0` (the default) plays the top move, like the gate. The run prints one line per game, writes one JSON
line per game to `--out` and a summary beside it (`results.summary.json`): the score, games an hour, where the time
went, and how often the two legal lists agreed. `--trace DIR` keeps every call of every game.

A game that fails (the site unreachable, an answer the bridge cannot use, an error in the bridge) is dropped and not
counted; the summary says how many were and why. After a failure a worker waits before its next game, longer each time the site keeps failing.

## Training against it

    python -m polyai.rl ... --polytopai-games 8 [--polytopai-level 3]

keeps that many games against the site in flight next to the rollout. The learner plays them with its current
weights while the rollout runs; their decisions are stored like a rollout's (same advantage estimate, same
`--credit-per-turn`) and join the update on top of `--rollout-steps`. A game runs across many updates, so it is cut at
each one like any unfinished game. The site's results are counted in the log and in `metrics.jsonl` under the opponent
`site:polytopai3`, and written to `games.jsonl` with `--log-games`. The site is not a member of the opponent pool: its
share of the training data is whatever its games produce, set by `--polytopai-games`.

Off by default. Like every change it needs an A/B from the same starting model before it goes into the live loop.

## Measured (20 November 2026)

* **Translation.** 6,373 positions of our own games gave the same encoding and the same legal list after
  `position()` and `from_position()`. All 3,685 moves of eight recorded site games (both sides) were in the engine's
  legal list for the translated position before them. Over 12,986 of our positions on the site the two legal lists
  were identical in 92%; the rest are the rule differences above.
* **Strength.** Round 77, top move, against the site's easier setting (`level` 1): 9 wins in 199 games (4.5%), mean
  length 17 turns; 10.9% on 11 by 11 maps, under 2% on 14 and 16. By turn 12 the site's bot has 15.6 units to our 9.0.
* **Speed.** At level 1 with 8 games in flight: 950 games an hour from one machine, 24 seconds a game, of which 19 are
  the site's answers (about 215 calls a game). At level 3 the bot searches: its turn takes 0.1 s at the start, 4 s by
  turn 9 with one game in flight, and several games at once slow each other down on the site's side.
* **Levels.** The site's page offers Easier (1: its network's first choice, no search), PolytopAI Jr (5: a weaker
  network looking 2 turns ahead) and Standard (3: 4 turns ahead), plus Cheat (4: Standard seeing through the fog).
  Nothing is weaker than level 1.

## The training A/B (20 November 2026)

Two runs of 300 updates from round 77 with the live loop's settings and seed 900, on poly4090 with the bindings of
ai 3735b8f: `abe_ctrl` (as the loop) and `abe_site` (`--polytopai-games 8 --polytopai-level 1`). The site run played
3,374 games against the site while training, of which it won about 86 (2.6%, sampling its moves); their 703,780
decisions were 6.7% of the learner's. 400 games per match, 200 against the site:

| Match | Score |
|---|---|
| `abe_site` against `abe_ctrl` | 38.9% (150-239-11) |
| `abe_site` against round 77 | 34.8% (132-254-14) |
| `abe_ctrl` against round 77 | 43.9% (172-221-7) |
| Against retail Crazy: `abe_site`, `abe_ctrl`, round 77 | 86.0%, 88.8%, 88.5% |
| Against the site at level 1: `abe_site`, `abe_ctrl`, round 77 | 5, 4 and 8 wins in 200 |

Not adopted. Games that are lost 97% of the time carry almost only one message ("this went badly"), and with
`--adv-filter 0.5` their large negative advantages are likely to take more than their 6.7% share of the positions
kept. Results: `results/polytopai/ab-2026-11-20/`.

## Building the bindings

The bindings must come from this repository at or after the commit that added `Game.from_position`:

    cd crates/polyai-py && maturin build --release -o /some/folder/wheels
    pip install --no-deps --target /some/folder/engine /some/folder/wheels/*.whl
    PYTHONPATH=/some/folder/engine:python python -m polyai.polytopai ...

Installing into a folder of its own leaves the shared environment's bindings alone.

## Tests

`python/tests/test_polytopai.py` replays a recorded game (`tests/fixtures/polytopai_game.json.gz`) through the bridge with
a stub policy and a client that answers from the recording, so it needs no network. It checks that every recorded
position loads, that the move played is in the engine's legal list, that the game is stored like a rollout and that an
update trains on it.

## Being a good guest

The site is one person's project and every game costs it a search for each of the bot's turns. The bridge sends a
plain `polyai-bridge` user agent, backs off when the site answers busy, and never retries a move. Keep `--parallel`
and `--polytopai-games` modest.
