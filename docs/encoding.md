# polyai encoding spec (version 2)

Authority for the tensor, action and shard layout. The Rust constants live in `crates/polyai-core/src/encoding.rs`
and `action.rs`; this file is what `python/polyai/schema.py` must match. Where this file and `design.md` disagree,
this file wins (the differences from `schema.py` are listed at the end). Ids come from the embedded rules data of game
version 123 (logic data v28); `ENCODING_VERSION = 2` is stored in every shard (and is the `version` of the Python module).
The changes from version 1 are collected in section 11; read that first when moving code from v1 to v2.

## 1. Player slots

Everything is seen from the player to move ("me"). Slots are relative to the mover:

- slot 0 is me;
- slots 1..7 are the other human-or-bot players in turn order starting after me (slot 1 moves next after me);
- slot 8 is Nature (units and nothing else; engine player id 255).

Dead players keep their slot. A game has 2 to 4 players in v1 but the encoding allows up to 8. Slot-valued tile
fields store `0` for "none / unknown" and `1 + slot` otherwise, so their vocabulary is 10.

## 2. Tile tensor: `tiles: int16[H, W, 41]`

`H, W` are the map's own size (Tiny 11, Small 14, Normal 16; the engine supports up to 30). Index `[y, x]`, tile
number `y * W + x` (the engine's tile index). The last axis is 9 categorical columns (embedded by the model, id 0 =
none / unknown) followed by 32 numeric columns (stored as integers, the model multiplies by `1 / divisor`).

**Fog.** A tile is visible when `Engine::is_explored(me, idx)` (Polytopia never re-fogs). An unexplored tile is all
zeros except `unexplored = 1`. Units are read the way `visible_unit_for` reads them: explored tile, and not an
invisible enemy that is not at peace with me.

### Categorical columns (index, name, vocabulary size)

| col | name | vocab | value |
|---|---|---|---|
| 0 | terrain | 10 | engine terrain idx (1 water, 2 ocean, 3 field, 4 mountain, 5 forest, 6 ice, 7 wetland, 8 mangrove) |
| 1 | climate | 19 | engine `Tile.climate` (tribe idx of the land, 1 to 17) |
| 2 | resource | 11 | engine resource idx (1 game, 2 crop, 3 fish, 4 whale, 5 metal, 6 fruit, 7 spores, 8 starfish, 9 aquacrop), only when the resource is visible to me (`resource_visible_to`) |
| 3 | owner | 10 | `1 + slot` of the tile owner, 0 unowned |
| 4 | improvement | 56 | engine improvement idx (1 city, 2 ruin, 3 road, ..., 55 algaespawn) |
| 5 | capital_of | 10 | `1 + slot` of the player this tile is a capital for, else 0 |
| 6 | unit_type | 63 | engine unit idx (1 scout ... 62 larva) of the visible unit on the tile |
| 7 | unit_owner | 10 | `1 + slot` of that unit's owner |
| 8 | passenger_type | 63 | unit idx of the unit carried by a boat, or of the second own unit on the tile (Giant with a Cloak), else 0 |

### Numeric columns (index, name, divisor, value)

| col | name | divisor | value |
|---|---|---|---|
| 9 | unexplored | 1 | 1 when I have not explored the tile |
| 10 | flooded | 1 | tile effect bit 1 |
| 11 | swamped | 1 | effect bit 2 |
| 12 | tentacle | 1 | effect bit 3 |
| 13 | algae | 1 | effect bit 4 |
| 14 | foam | 1 | effect bit 5 |
| 15 | has_road | 1 | `Tile.has_road` |
| 16 | has_route | 1 | `Tile.has_route` |
| 17 | has_ruin | 1 | `Tile.has_ruin` |
| 18 | ruling_city_mine | 1 | the tile's ruling city is owned by me |
| 19 | in_my_border | 1 | `Tile.owner == me` |
| 20 | improvement_level | 8 | `improvement_level(at)` for a non-city improvement, 0 otherwise |
| 21 | is_city | 1 | improvement is a city |
| 22 | is_capital | 1 | `Tile.is_capital` |
| 23 | city_level | 8 | city level |
| 24 | population | 8 | city xp towards the next level (my cities only, else 0) |
| 25 | population_needed | 8 | `level + 1`, the xp at which the city levels up (my cities only, else 0) |
| 26 | walls | 1 | city has the wall reward |
| 27 | park | 1 | city has the park reward |
| 28 | workshop | 1 | city has the workshop reward |
| 29 | border_size | 3 | city border radius |
| 30 | connected_to_capital | 1 | my city is connected to my capital (my cities only) |
| 31 | under_siege | 1 | a visible enemy unit stands on the city tile |
| 32 | health | 100 | `100 * health / max_health` of the visible unit (can exceed 100 after a promotion) |
| 33 | veteran | 1 | unit has been promoted |
| 34 | moved | 1 | `UnitState.moved` |
| 35 | attacked | 1 | `UnitState.attacked` |
| 36 | frozen | 1 | |
| 37 | poisoned | 1 | |
| 38 | boosted | 1 | |
| 39 | invisible | 1 | unit is cloaked (only ever set on my own or allied units, enemy ones are hidden) |
| 40 | can_act | 1 | my unit appears in the legal list with at least one command (Stay included) |

So `F_TILE = 41`, columns 0..9 categorical, 9..41 numeric. This is the same column order as `schema.py`.

## 3. Globals: `globals: float32[247]`

| offset | size | content |
|---|---|---|
| 0 | 13 | my tribe, one-hot over `TRIBES` (below) |
| 13 | 1 | my stars / 10 |
| 14 | 1 | my income per turn / 10 |
| 15 | 1 | my score / 1000 |
| 16 | 1 | turn number / 50 |
| 17 | 1 | turns left / 50: always 0 in version 2 (the turn limit of a Domination game is a compute cap, see section 11) |
| 18 | 1 | has a turn limit: always 0 in version 2 |
| 19 | 1 | game mode / 5 (Domination 2) |
| 20 | 1 | map width / 30 |
| 21 | 1 | living players / 8 |
| 22 | 1 | players at the start / 8 |
| 23 | 1 | my cities / 8 |
| 24 | 1 | my units / 32 |
| 25 | 51 | my techs, a bit per engine tech idx 0..50 (idx 11 and 34 do not exist and stay 0) |
| 76 | 7 x 24 | slots 1..7, below |
| 244 | 3 | pending trigger of the mover: [none, city reward, peace request] |

Tribe order (engine tribe idx): aimo 2, bardur 4, hoodrick 6, imperius 7, kickoo 8, luxidoor 9, oumaji 10, quetzali
11, vengir 12, xinxi 13, yadakk 14, zebasi 15, cymanti 17. The out-of-scope tribes (aquarion 3, elyrion 5, polaris
16) are all zeros.

Per other slot (24 values), zeros for a slot that has no player:

| offset | size | content |
|---|---|---|
| 0 | 1 | present (the slot has a player) |
| 1 | 1 | alive (known only once met, else 0) |
| 2 | 1 | met (I know the player) |
| 3 | 13 | tribe one-hot if met |
| 16 | 1 | score / 1000 if met |
| 17 | 1 | cities / 8 if met |
| 18 | 4 | my relation to them: neutral, peace, war, cooldown (one-hot, if met) |
| 22 | 1 | I have an embassy with them |
| 23 | 1 | they have an embassy with me |

Opponent stars and techs are not visible and are not encoded.

## 4. Actions

An action is `int16[4] = (kind, arg, src_tile, dst_tile)`; tiles are `y * W + x`, `-1` means none. The legal list
(`legal_actions`) is the engine's `legal_commands()` plus one `Stay` per own unit that `can_move` (the retail AI
chooses Stay about as often as a move and the engine does not list it; it is inserted just before `EndTurn`). While a
trigger is pending the list is just the choices, no Stays.

| kind id | name | arg | src | dst |
|---|---|---|---|---|
| 1 | Move | stack rank | unit tile | target tile |
| 2 | Attack | stack rank | unit tile | target tile |
| 3 | Capture | stack rank | unit tile | city tile (same as src) |
| 4 | Promote | stack rank | unit tile | unit tile |
| 5 | Recover | stack rank | unit tile | unit tile |
| 6 | HealOthers | stack rank | unit tile | unit tile |
| 7 | Explode | stack rank | unit tile | unit tile |
| 8 | Disband | stack rank | unit tile | unit tile |
| 9 | ExamineRuins | stack rank | unit tile | unit tile |
| 10 | Stay | stack rank | unit tile | unit tile |
| 11 | Swarm | stack rank | unit tile | unit tile |
| 12 | Boost | stack rank | unit tile | unit tile |
| 13 | Disembark | stack rank | unit tile | unit tile |
| 14 | Upgrade | target unit idx | unit tile | unit tile |
| 15 | Train | unit idx | city tile | city tile |
| 16 | Build | improvement idx | tile | tile |
| 17 | Destroy | 0 | tile | tile |
| 18 | Decompose | 0 | tile | tile |
| 19 | ClearTileEffect | effect bit (1 flooded, 4 algae) | tile | tile |
| 20 | CityReward | reward id (below) | city tile | city tile |
| 21 | Research | tech idx | -1 | -1 |
| 22 | BreakPeace | slot | -1 | -1 |
| 23 | PeaceTreaty | slot | -1 | -1 |
| 24 | EstablishEmbassy | slot | -1 | -1 |
| 25 | PeaceResponse | `slot * 2 + accept` | -1 | -1 |
| 26 | EndTurn | 0 | -1 | -1 |

Kind 0 is padding in a batch. `N_KINDS = 32` embedding rows. City reward ids: 1 CityWall, 2 Park, 3 Workshop, 4
Explorer, 5 BorderGrowth, 6 SuperUnit, 7 Resources, 8 PopulationGrowth, 9 Rebellion, 10 Poison, 11
TutorialExplorer. All arguments are below 64.

Units are named by their tile. A boat's passenger is inside the boat. The only case of two own units on one tile is a
Giant with a Cloak: for every unit command except Upgrade, `arg` is then the unit's rank among the mover's units on that
tile (engine unit-list order; 0 in all other cases), and the tile shows the second unit as `passenger_type`. An Upgrade
names the target type in `arg`, so it is resolved to the unit on the tile that can upgrade to it.
`decode(encode(cmd)) == cmd` holds on every legal list (`cargo test -p polyai-core`, thousands of positions, all 13
tribes), and two different commands never share a tuple.

## 5. Outcome labels

For every recorded position (a decision of a player who is not Nature):

- `winner`: slot (relative to the mover) of the player who won by elimination, `-1` when nobody won. Points never
  decide a game: when the game reaches the turn cap with more than one player alive it is a draw, `winner = -1` for
  every position of the game, and the engine's own top-score winner is discarded;
- `final_scores`: final scores by slot, float32[8], zeros for absent slots. Logged for analysis only; it is not a
  target;
- `capped`: 1 when the game ended without an elimination winner: it reached the turn cap with more than one player
  alive (or was cut off by the generator's command safety cap). `capped == 1` exactly when `winner == -1`. Positions of
  a capped game still train the policy; they carry no value target.

## 6. Shards (`.npz`, deflate)

One shard holds one map size; `N` positions, `T` = total legal-list length, `K = 16` candidate slots. Written by
`polyai-gen` to `/kvm/polytopia-ai-data/<dir>/shard-<n>-<H>x<W>.npz`.

| key | dtype | shape | meaning |
|---|---|---|---|
| tiles | int16 | [N, H, W, 41] | section 2 |
| globals | float32 | [N, 247] | section 3 |
| act_off | int32 | [N + 1] | legal list of position i is `actions[act_off[i]:act_off[i+1]]` |
| actions | int16 | [T, 4] | section 4 |
| chosen | int16 | [N] | index of the played command inside its legal list |
| cand_idx | int16 | [N, 16] | legal-list indices of the AI's best candidates, best first, -1 padded |
| cand_score | float32 | [N, 16] | their scores (noise free) |
| winner | int8 | [N] | section 5 (`-1` for a draw) |
| final_scores | float32 | [N, 8] | section 5 (logged only) |
| capped | int8 | [N] | section 5 |
| turn | int16 | [N] | game turn of the position |
| seed | int32 | [N] | game seed (training seeds 1..100000) |
| cmd_index | int32 | [N] | index of the command in the game record |
| difficulty | int8 | [N] | mover's difficulty: 0 easy, 1 normal, 2 hard, 3 crazy |
| version | int32 | [1] | `ENCODING_VERSION` (2) |

Only a fraction of the positions of each game is written (default one in four, chosen by a seeded hash), because
neighbouring commands of a turn are strongly correlated. `ai_candidates()` is only evaluated for written positions.

## 7. Game records

`GameRecord { format, engine_version, engine_commit, setup, commands, outcome }`, one binary file per generator
thread batch (`records-<n>.polyrec`, a length-prefixed list). Record `format` is 2 (v2 writes it, v1 and v2 files
load). The layout is unchanged; what changed is the meaning of the outcome: flag bit 1 is `capped` (it was
`adjudicated`) and a capped game has no winners. A format 1 record with that bit set stored the engine's top-score
winner; the loader sets `capped`, empties `winners` and keeps the scores, so an old adjudicated game is a draw
(`bc-v1` relabelling needs nothing else). The setup holds the seed, map size, preset, game mode,
turn limit and per seat tribe and difficulty; the command list holds every command of every player (Nature's
`EndTurn` included) in order. `replay(&record)` rebuilds the game and must reproduce the final scores and winners.
The engine commit is the `git rev-parse` of `../engine` at build time, with `-dirty` appended when it has local
changes.

## 8. `polyai-gen`

`target/release/polyai-gen --out DIR --games N [--seed-start S] [--threads T] [--keep K] [--shard-size P]
[--difficulty D] [--turn-limit L] [--players 2:55,3:25,4:20]` plays games `S..S+N-1` (setup drawn from the seed by
`GameSetup::sample`: 2 to 4 players weighted 55/25/20, 13 tribes, Tiny/Small/Normal, five presets, per-seat difficulty
weights 10/20/35/35 for easy/normal/hard/crazy, Domination, a 150-turn compute cap) with every seat on the retail AI.
`--players` takes `count:weight` pairs (`2:1,4:3`) or plain counts (`2,3`, uniform); the weights live in
`SetupSpec.n_player_weights` (empty means uniform, the old behaviour for an explicit list). A game still running at
the cap is a draw (`capped = 1`, `winner = -1`); the summary prints the number of capped games and the player-count mix. `T` is capped at 24 and is the number of worker threads (shard writing happens on the
workers, so there are no others). Seeds must lie in 1..=100000; evaluation seeds (>= 1,000,000) are refused. A shard
is written whenever one map size has `P` positions (default 50000), so one run writes several shards per size and a
final partial one. `--keep K` writes one position in `K` (default 4).

## 9. `polyai_engine` (Python, built with `maturin develop --release` in `crates/polyai-py`)

- Constants: `ENCODING_VERSION`, `F_TILE`, `G_DIM`, `N_KINDS`, `TRIBES`, `KIND_NAMES`, `CAT_VOCABS`, `ENGINE_COMMIT`.
- `sample_setup(seed, n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=150)` returns a setup dict
  `{seed, size, preset, mode, turn_limit, seats: [(tribe, difficulty)]}`; size ids 1 tiny, 2 small, 3 normal.
  `n_players` is a list (uniform over it, as before) or a dict `{count: weight}`; `None` is the default 55/25/20 mix
  of 2, 3 and 4 players. `turn_limit` is the compute cap (default 150): a game still running then is a draw.
- `Game.sample(seed, ...)` / `Game.from_setup(dict)`: `legal_actions()` int16[N, 4], `encode()` (tiles, globals),
  `step(index)`, `step_action((kind, arg, src, dst))`, `ai_move()`, `play_ai_move()`, `ai_candidates(k=16)`,
  `current_seat()` (`None` on Nature's turn, call `play_ai_move()`), `scores()`, `winners()` (empty for a draw), `capped()` (`adjudicated()` is an alias for one version), `is_over()`,
  `clone()`,
  `record()` (bytes of the section 7 format). Seats are indices into the setup's seat list.
- `VecEnv(n_games, seed_start=1, model_seats="rotate" | "all" | [seats], split="train" | "eval", n_players, sizes,
  presets, difficulty, turn_limit=150, threads=24, auto_reset=True)`. Seats the model does not control (and Nature) are
  played by the retail AI at their handicap on a rayon pool (at most 24 threads, GIL released in `step`). `"rotate"`
  gives each game one model seat, `seed % n_players`. `split="train"` draws seeds from 1..=100000 (wrapping),
  `"eval"` needs `seed_start >= 1000000`.
  - `observe()` returns a list with one dict per map size of every game awaiting a decision; `step(games, choices)`
    (int32 arrays, `choices[i]` an index into game `games[i]`'s legal list) returns `(observations of the stepped games,
    finished games)`. A dict has `game`, `seat`, `n_players`, `turn`, `seed` (all `[B]`), `tiles` `[B, H, W, 41]`,
    `globals` `[B, 247]`, `act_off` `[B + 1]`, `actions` `[T, 4]`.
  - A finished game is a dict with `game, seed, n_players, tribes, difficulties, model_seats, winners` (seats),
    `scores` (by seat), `turns, capped, decisions, commands`, and `adjudicated` (a copy of `capped`, kept for one
    version). `winners` is empty and `capped` true for a draw at the cap. With `auto_reset` the slot restarts at once on the
    next seed and shows up in the returned observations.
  - Observations are only for games awaiting the model; it is up to the caller to give every awaiting game a choice
    (a game that gets none simply waits).

## 10. Differences from `python/polyai/schema.py` (at commit 30ad097)

Everything not listed here matches: the 9 categorical then 32 numeric column order and names, divisors, slot
convention (`0` none, `1 + slot`), action kind names and order, `KIND_PAD = 0`, `N_KINDS = 32`, `ARG_MAX = 64`, and the
shard keys and dtypes of the first eleven rows of the section 6 table.

1. **Vocabulary sizes.** `improvement` is 56 (schema: 50) because engine improvement idx goes up to 55 (algaespawn);
   `unit_type` and `passenger_type` are 63 (schema: 60) because unit idx goes up to 62 (larva). Counts of 49 and 59
   in design.md are numbers of entries, not the largest idx. Terrain 10, climate 19, resource 11, owner, capital_of and
   unit_owner 10 are unchanged.
2. **Techs.** 51 bits indexed by engine tech idx (0..50, two idx values unused), schema: `N_TECHS = 49`.
3. **Global scalars.** 12 scalars (stars, income, score, turn, turns left, has turn limit, mode, map width, living
   players, players at the start, my cities, my units); schema has 8. `N_SCALARS = 12`.
4. **Per other slot.** 24 values: the schema's single `embassy` is split into "I have an embassy with them" and
   "they have one with me" (`PER_SLOT = 24`, schema 23).
5. **Trigger kinds.** 3 (none, city reward, peace request), schema `N_TRIGGER_KINDS = 8`.
6. **`G_DIM = 247`** (schema 239): 13 + 12 + 51 + 7 * 24 + 3.
7. **Stay.** The engine does not list `Stay`, but the retail AI plays it constantly, so the encoder's legal list adds
   one `Stay` per own unit that can move. Imitation targets are then always inside the legal list.
8. **ClearTileEffect arg** is the engine effect bit (1 flooded, 4 algae); **CityReward arg** is the compact reward id of
   section 4.
9. **Action `arg` of unit commands** is the stack rank (0 except for a Giant with a Cloak on one tile; Upgrade keeps the
   target type), so every arg still fits `ARG_MAX = 64`.
10. **Extra shard keys:** `final_scores float32[N, 8]`, `capped int8[N]` (v1: `adjudicated`), `seed int32[N]`, `cmd_index int32[N]`,
   `difficulty int8[N]`, `version int32[1]`. `K = 16` candidate slots.
11. **`winner`** is a relative slot 0..7 or -1; the Nature slot 8 never wins, so the winner head can be 8 wide, but 9 is
    harmless.

## 11. Version 1 to version 2 (objective v1.1: win by elimination only)

Points never decide a game. The bot wins by eliminating the other players, a game that is not decided by the turn cap
is a draw, and the cap exists for compute only.

| what | v1 | v2 |
|---|---|---|
| `ENCODING_VERSION`, shard `version` | 1 | **2** |
| tile tensor, action tuples, `G_DIM = 247`, global layout | | unchanged (same shapes and offsets) |
| globals 17 (turns left) and 18 (has limit) | filled when `turn_limit > 0` | **always 0** in Domination; the slots stay so offsets do not move |
| turn cap | 50, then the engine's top-score player won (adjudication) | **150**, then a draw: `winner = -1`, no winner of any kind |
| shard `adjudicated` | int8, 1 at the limit with several alive | replaced by **`capped`** (int8): 1 for a draw, exactly when `winner == -1` |
| shard `margin`, `margin5` | float32 labels | **removed** (points are not a target) |
| shard `final_scores` | float32[N, 8] | unchanged, logged only |
| shard `winner` | slot or -1; a game adjudicated at the limit carried the top-score player's slot | slot of the elimination winner, **-1 for every position of a capped game** |
| Python `Game.winners()`, finished-game `winners` | engine winners (top score at the limit) | empty for a draw |
| finished-game dict | `adjudicated` | **`capped`** (and `adjudicated`, an alias for one version) |
| `turn_limit` default of `sample_setup`, `Game.sample`, `VecEnv` | 50 | **150** |
| setup sampler player counts | uniform over 2, 3, 4 | **55% / 25% / 20%** (an explicit list is still uniform; a dict `{count: weight}` is weighted) |
| record format | 1 | **2**, same layout; flag bit 1 is `capped`, a capped game has no winners; v1 files load with the adjudicated winner dropped |
| arena JSONL | `adjudicated`, winner = top-score player | `capped`, `winner` -1 for a draw; 1v1 on Tiny/Small/Normal, five presets (docs/arena.md) |

What the Python side should do:

- Read `version` from the shard and accept 1 and 2. For a v1 shard use `adjudicated` where v2 has `capped`, and treat
  `adjudicated == 1` as `winner = -1` (the stored winner of an adjudicated game is the top-score player, not a
  result). `margin` and `margin5` exist only in v1 shards; do not train on them.
- Value target: win/loss from `winner` only; skip the value loss where `winner == -1` (equivalently `capped == 1`).
  The policy loss still uses those positions.
- Do not rely on the turn limit as a feature: globals 17 and 18 are 0 in every game.
- `bc-v1` (50-turn limit, adjudication) keeps its shape; relabelling is `winner = -1 where adjudicated == 1`.

Measured basis for the cap (retail vs retail, 300-turn limit, 600 games over the sampler mix and 600 1v1 games, train
seeds, Tiny/Small/Normal; elimination turn = the turn the game ended): median 38 turns (mix) and 33 (1v1), p90 68 and
57; still alive after turn 100: 3.2% and 2.2%, after 150: 1.8% and 1.3%, after 200: 1.3% and 1.2%, after 300: 1.2% and
1.0%. The tail beyond 150 is almost all games that never end (7 of 8 games alive at 200 were still alive at 300), and
a stuck game costs 6 to 11 s of CPU at 300 turns against 0.5 to 0.9 s for an average game. The cap is therefore 150: a
draw rate of about 1.3% (1v1) to 1.8% (mix) with 4-player games the worst (3.9%).
