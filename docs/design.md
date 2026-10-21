# polytopia-ai design (v1.1, 2026-10-21)

Goal: an ML Polytopia bot with a learned value function (a chess-style +/- evaluation) and a learned policy (move
ranking and best-move suggestions), trained on games played by `polytopia-engine`: first by imitating the retail AI,
then by self-play. It is measured against the retail AI at each difficulty and against MCTS from `../weekly`.

**Objective (Kenzi, 2026-10-20, ledger decision 0e246e45).** The bot should be strong against real players, not
just against the retail AI. It plays to win by elimination only (Domination); points are ignored. The value target
and the RL reward are win 1 / loss 0 and nothing else. Scores stay in the observation because players can see them.
It must be general across setups (tribes, map sizes, presets, player counts), with 1v1 weighted at about half or
more of all games, because top-level human play is mostly 1v1. Might mode is out of scope unless Kenzi adds it.

v1.1 changes from v1 (2026-10-20): no score-margin value heads or shaped score reward; no top-score adjudication
at a turn limit (a high cap for compute only, capped games are draws); 1v1-weighted setup sampler; 1v1 main eval
suite; a human-replay track (section 6, phase C).

Scope: the 12 normal tribes plus Cymanti (ledger to-do c2666790). Elyrion, Polaris and Aquarion are out of scope.

## 1. What the engine gives us

Facts checked against `../engine` (crate `polytopia-sim`) on 2026-10-20:

- New game: `Engine::new_game(&GameOptions { version, seed, size, preset, players: Vec<PlayerSetup { tribe, human }>,
  difficulty, mode, turn_limit, score_limit })` (`src/engine/setup.rs:14-93`). Player ids are 1..n, Nature is 255.
  Non-human players get the handicap for `difficulty`; it can be changed per player afterwards through
  `e.s.players[i].handicap`.
- Moves: `legal_commands()` for the current player (`src/engine/apply.rs:13`), `apply` / `apply_unchecked`,
  `current_player()`, `Command::EndTurn` ends the turn. `Command` is a small `Copy` enum (`src/engine/mod.rs:49-78`).
  Pending triggers (city level-up rewards, peace answers) restrict the legal list to the choices.
- Retail AI: `ai_get_move()`, `step_ai_cmd()`, and `ai_candidates() -> Vec<(Command, f32)>`, scored candidates best
  first (`src/engine/ai_move.rs:30-140`). Deterministic given the seed.
- End: `is_over()`, `winners`, `score(id)`, `reward_for(id)`. A turn limit ends the game with the top-scoring living
  player as winner. We do not use that rule as a result (section 6).
- Fog: per-tile explorer lists, `is_explored(player, idx)` (`src/engine/vision.rs`). Polytopia has no re-fogging: an
  explored tile stays visible, so "explored" is the whole fog model. `visible_unit_for(at, viewer)` also hides
  invisible (cloaked) enemy units.
- `Engine: Clone`, tens of KB on a Normal map. No undo. `RefCell` fields make it `Send` but not `Sync`.
- Speed: `examples/play_ai` plays a 3-player Small game of 30 turns (about 900 commands, AI decisions included) in
  0.21 to 0.31 s on one thread, so about 3,500 AI commands per second per thread. At 24 threads that is about
  80,000 positions per second for imitation data, far more than training can consume.
- Weekly: `polytopia_weekly::mcts` is a generic UCT (`trait Game`, uniform random rollouts, iterations not time,
  full information, returns one action). No Polytopia adapter exists yet. The Polytopia searchers in weekly
  (`solver`, `line_search`, `macro_search`, `heur`) maximise the player's own score, not winning.

## 2. How Python reaches the engine

Decision: **one Rust workspace in `ai/` with the encoder in Rust, exposed both as a PyO3 module and as a native
data generator.** The encoder is written once, so the training data and the live bot can never disagree.

```
ai/
  Cargo.toml                 workspace (members below), path deps ../engine and ../weekly
  crates/polyai-core/        encoding, action vocabulary, game setup, game records, runner (pure Rust, tested)
  crates/polyai-py/          PyO3 module `polyai_engine` (built with maturin into /kvm/polytopia/.venv)
  crates/polyai-gen/         binary: imitation and evaluation games to dataset shards, multithreaded
  crates/polyai-arena/       Agent trait, match runner, MCTS adapter for weekly's `mcts::Game`
  python/polyai/             model, dataset, training, evaluation (Elo), CLI scripts
  docs/                      this file, encoding.md (exact spec, owned by the core crate)
```

- Imitation data does not need Python in the loop, so `polyai-gen` runs retail-vs-retail games on N threads and
  writes shards directly.
- Self-play and evaluation need the network in the loop. `polyai_engine.VecEnv` holds many games (for example 512),
  steps them on a rayon pool (capped at 24 threads) with the GIL released, auto-plays every seat that is not a
  model seat (retail AI at its handicap, MCTS, scripted), and returns batched NumPy observations only for the games
  waiting on a model decision. Python batches inference on the GPU and calls `step(actions)`.
- A pure Rust inference path (ONNX or `tch`) is not needed for v1. Revisit if Python stepping becomes the
  bottleneck.

## 3. Observation encoding (v1)

Everything is from the point of view of the player to move ("me") and honours fog: an unexplored tile is all
zeros except the `unexplored` flag, and enemy units are read through `visible_unit_for`. Opponent stars and techs
are not visible in the game and are not encoded; scores and city counts are visible and are.

**Player slots.** Me is slot 0. Other players get slots 1..7 in turn order starting after me (so slot 1 moves next).
Nature is slot 8. Games with more than 8 tribes are out of scope for v1. Slot order makes the encoding invariant to
which player id we are.

**Spatial tensor** `tiles: int16[H, W, F_tile]`, the map's own size (11 to 30 per side). Categorical fields are
stored as small integer ids (0 = none/unknown) and embedded by the model; numeric fields are stored scaled to
integers. Fields:

| Group | Fields |
|---|---|
| Fog | unexplored |
| Terrain | terrain (9), climate as tribe-of-land (18), resource (10), effects bits (flooded, swamped, tentacle, algae, foam), has_road, has_route, has_ruin |
| Ownership | owner slot (0..8, plus "none"), ruling city is mine, in my border |
| Improvement | improvement id (49), improvement level, is_city, is_capital, capital_of slot |
| City | city level, population, population needed, walls, park, workshop, border size, connected to capital, under siege (enemy unit on the city) |
| Unit | type id (59), owner slot, health / max health (x100), veteran, moved, attacked, frozen, poisoned, boosted, invisible, passenger type id (for boats), can still act this turn (from my legal list) |

**Global vector** `globals: float32[G]`: my tribe (one-hot over the 13 tribes), my stars, my income, my score,
turn number, turns left (if a limit), game mode, map size, number of living players; my tech set (49 bits); for each
slot 1..7: present, alive, met, tribe if met, score, city count, relation (neutral, peace, war, cooldown), embassy;
pending trigger kind.

**Variable map sizes.** The model is fully convolutional with masked global pooling, so one network serves every
size. Batches are bucketed by map size (11, 14, 16, 18, 20, 30) to avoid padding. Training starts on Tiny, Small and
Normal; larger sizes come in once the pipeline works.

**Augmentation.** The 8 square symmetries (rotations and mirrors) of the grid are valid if the rules are symmetric
apart from coordinates. To be confirmed by a test: play a game and its mirrored twin and compare outcomes; if the
map generator or tie-breaks depend on coordinates, the symmetry still holds for the rules, which is what the
network needs.

## 4. Action encoding (v1)

Polytopia has no fixed action space, so the policy scores the legal list rather than a fixed vector. Every legal
`Command` is encoded as a tuple:

`(kind, arg, src_tile, dst_tile)`, all integers, tiles as `y * W + x`.

| Commands | kind | arg | src | dst |
|---|---|---|---|---|
| Move, Attack, Capture | own kind each | 0 | unit tile | target tile |
| Promote, Recover, HealOthers, Explode, Disband, ExamineRuins, Stay, Swarm, Boost, Disembark | own kind each | 0 | unit tile | unit tile |
| Upgrade | Upgrade | target unit type | unit tile | unit tile |
| Train | Train | unit type | city tile | city tile |
| Build | Build | improvement id | tile | tile |
| Destroy, Decompose | own kind each | 0 | tile | tile |
| ClearTileEffect | ClearTileEffect | effect | tile | tile |
| CityReward | CityReward | reward id | city tile | city tile |
| Research | Research | tech id | none (-1) | none |
| BreakPeace, PeaceTreaty, EstablishEmbassy | own kind each | player slot | none | none |
| PeaceResponse | PeaceResponse | slot * 2 + accept | none | none |
| EndTurn | EndTurn | 0 | none | none |

Units are named by their tile (one unit per tile; a boat's passenger is the boat), so unit ids never reach the
model. The encoding must be a bijection on each legal list: `decode(encode(cmd)) == cmd` is a core-crate test over
many random positions.

Decision granularity is one command, as in the engine. A turn is usually 10 to 40 commands. The policy sees the
same position after each command, so it plans the turn one step at a time; the value head evaluates the position
at every step.

## 5. Model

A KataGo-style residual CNN, which is the standard answer for variable board sizes:

- Input: tile categoricals through embedding tables (summed), numeric tile fields through a 1x1 conv, plus the
  global vector through an MLP broadcast over the board.
- Trunk: residual blocks of 3x3 convs with global-pooling bias blocks every third block (masked mean and max over
  the real board), so information crosses a 30x30 map without needing 15 layers.
- Policy head: per-tile features `h` (C channels) and a pooled vector `g`. For each legal action,
  `logit = MLP([h[src], h[dst], E_kind[kind], E_arg[kind, arg], E_off[clip(dst - src)]])`, with `h[none] = g`.
  Softmax over the legal list (padded with a mask).
- Value head from pooled features: the winner distribution over the 9 slots (softmax), giving my win probability
  in any number of players. Positions from games that ended without an elimination winner (draws at the compute
  cap) carry no winner target. There are no score-margin heads (removed in v1.1: points are not the objective).
- Evaluation shown to people: `eval = 4 * log10(p / (1 - p))` clipped to +/-9.9, so 50% shows 0.0 and 85% shows
  about +3.0, like chess pawns. Best moves: the policy's top 3 with the value after each.

Sizes, all within the GPU budget below:

| Name | Blocks x channels | Params | Use |
|---|---|---|---|
| b6c96 | 6 x 96 | about 1.0M | pipeline tests, imitation baseline |
| b10c128 | 10 x 128 | about 3M | main v1 model |
| b15c192 | 15 x 192 | about 10M | if b10 saturates and memory allows |

**GPU budget.** The RTX 3060 has 12 GB; on 2026-10-20 at 03:08 other processes (desktop session, a llama-server,
browsers) used 8.9 GB, leaving about 3.3 GB. Training uses bf16 autocast, batch 256 on 16x16, and must stay under
2.5 GB peak (`torch.cuda.max_memory_allocated`, checked at start-up with a clear error if free memory is short).
b10c128 at batch 256 needs well under 1.5 GB of activations. We never touch other processes on the GPU.

## 6. Data generation

**Phase A, imitation of the retail AI.** `polyai-gen` plays retail-vs-retail games and records, for every command
of every AI seat: the observation, the legal action list, the action the AI chose, and the top of
`ai_candidates()` with scores (a soft target). After the game it fills in the outcome: the elimination winner, or
none if the game hit the compute cap.

- Setups drawn per game from a seeded generator: 1v1 in about half or more of games, otherwise 3 or 4 players;
  tribes uniformly from the 13; map size Tiny, Small or Normal; presets Dryland, Lakes, Continents, Pangea,
  Archipelago; difficulty per seat from Easy, Normal, Hard, Crazy (the teacher signal is mostly Hard and Crazy, but
  weaker seats give varied positions); Domination.
- **No adjudication.** The turn limit is a high safety cap for compute only (on the order of 150 to 200 turns, set
  from the measured game-length distribution). A game that reaches it is a draw: `winner = -1`, and its positions
  are dropped from value training (policy imitation still uses them). The engine's top-score rule is never used as
  a result.
- **bc-v1 relabel.** The existing 20,000-game set was generated with a 50-turn limit and top-score adjudication
  (6,389 games adjudicated) and 2 to 4 players without the 1v1 weighting. Its adjudicated games are relabelled
  `winner = -1` (the `adjudicated` flag marks them), so they still train the policy but not the value. The next
  dataset (bc-v2) uses the v1.1 sampler and cap.
- Records: one compact game record per game (setup + command list + outcome), from which any encoding version can
  be regenerated by replay; plus encoded shards (`.npz`, zstd or `savez_compressed`) of about 50,000 positions each
  for training. Both live under `/kvm/polytopia-ai-data/` (outside git).
- First target: 20,000 games, about 15 to 20 million positions. Subsample to about 1 position in 4 per game to
  reduce correlation.
- Seeds 1 to 100,000 are for training data; seeds from 1,000,000 up are reserved for evaluation and never used in
  training.

**Phase B, self-play.** The `VecEnv` runs games where some seats are the current model (sampling from the policy
with temperature) and the rest come from an opponent pool: earlier checkpoints, the imitation policy, and retail AI
at random difficulty, and later a human-style policy (phase C). Trajectories go to a replay buffer on disk.

**Phase C, human track.** Capture top-level 1v1 Domination replays, check that each replays exactly in the engine,
and encode them like any other record. Uses: a benchmark (move-match rate against the human's move, outcome
prediction), fine-tuning data, and a human-style opponent in the self-play pool. Later: live tests on the website
and through the headless client. Sourcing the replays depends on the `re`, `vision` and `client` work and is not
started in v1.

## 7. Training loop

1. **Behaviour cloning.** Policy loss: cross-entropy on the AI's chosen command plus 0.5 x KL to
   `softmax(candidate scores / T)` where candidates exist. Value loss: winner cross-entropy only, on games with an
   elimination winner.
   AdamW, cosine schedule, bf16. Held-out metrics: top-1 and top-3 agreement with the AI, value calibration
   (Brier score) by turn.
2. **Reinforcement from self-play.** Start with PPO (clipped, GAE over the per-command trajectory, reward = win
   1 / loss 0 at the end and nothing else; a game at the compute cap is a draw worth 0.5 to both or is dropped)
   with a KL penalty towards the behaviour-cloning policy that decays over training, as AlphaStar did. Opponent pool sampling weighted
   towards opponents we beat less than 70% of the time.
   Live games played by our two accounts through the client are a second, smaller data stream next to self-play
   (value targets, off-policy PPO on our moves, imitation of strong human opponents); the policy interface the client
   calls and the record format those games come back in are defined in `docs/interfaces.md`.
3. **Search on top (later).** A small per-turn search using the policy as prior and the value head at leaves
   (Gumbel MuZero-style root sampling with tens of simulations over the current turn's commands). If it beats the raw
   policy clearly, use it as an expert-iteration teacher.

Each run writes `runs/<name>/` under `/kvm/polytopia-ai-data/`: config, metrics (JSONL, plus TensorBoard if
installed), checkpoints. Jobs run under `nice`, with at most 24 CPU threads (rayon pool, PyTorch threads and data
loader workers counted together).

## 8. Evaluation harness

- **Agents:** `retail:easy|normal|hard|crazy`, `mcts:<iterations>[:rollout=random|ai]`, `random`, and
  `model:<checkpoint>[:greedy|sample|search]`.
- **Matches:** fixed seed lists from 1,000,000 up. Each seed is played with seats swapped (and tribes swapped with
  them), so map and tribe advantages cancel. The main suite is 1v1 Domination across varied map sizes, presets
  and tribes, with the high compute cap; capped games are draws (half a win each). FFA tables (1 model vs 2 or 3
  retail) are a second suite.
- **Results:** one JSONL line per game (agents, seed, setup, winner, scores, turns, capped, wall time).
- **Metrics:** win rate (draws as half) with a Wilson 95% interval per pairing, draw rate, and Elo from a
  Bradley-Terry fit over all games, anchored at retail Normal = 1000. Scores are logged but are not a metric.
- **Human benchmark (phase C):** move-match rate and outcome prediction on held-out top-level 1v1 replays.
- **MCTS caveat:** weekly's MCTS sees through fog and uses uniform random rollouts over single commands, so it is
  likely weak and slow at Polytopia's horizon. The adapter (`polyai-arena`) offers two variants: plain uniform
  rollouts, and the legal list pruned to the retail AI's top-k candidates. It is a baseline, and a cheating one;
  results say so.
- **Engine bots baseline:** first run a retail-vs-retail table (Easy, Normal, Hard, Crazy) to calibrate the Elo
  scale and to check that the engine's difficulties order as expected.

## 9. Milestones

1. Core crate: encoding, action vocabulary, round-trip tests, game records. Arena with retail and MCTS agents and
   the retail table. Python model and training on synthetic data.
2. Imitation dataset (20,000 games) and a behaviour-cloned b6c96: top-1 agreement with the AI, then win rate vs
   retail Easy and Normal.
3. b10c128 imitation model; eval and best-move API in `polyai_engine` for web and client to use later.
4. PPO self-play; beat retail Crazy more than half the time on the 1v1 suite.
5. Search on top; MCTS and Elo table.
6. Human track: replay capture, human benchmark, fine-tuning, human-style pool opponent.

## 10. Open questions and risks

- The retail AI's own move choice may use hidden information. Imitation from the fogged view just learns a noisier
  policy; the value head is trained on outcomes, which are fair.
- Long games: Domination between equal AIs can run long, and without adjudication some never end. The compute cap
  keeps data finite; measure the length distribution and the draw rate at the cap, and set the cap from it.
- Engine changes from the other manager (data format, versions, Might mode) may shift behaviour. Records store the
  engine commit and game version so data can be regenerated.
- Engine requests (relayed through the project chat, not done by us): a public, cheap per-player "currently
  observable" accessor, if the AI encoding ever needs more than explored tiles; a cheaper state clone for search
  (copy-on-write map) only if search profiling shows clones dominate.
