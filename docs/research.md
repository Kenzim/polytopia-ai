# Research: methods, prior art and Polytopia agents (2026-10-29)

Research only; nothing in the code changed. Written for gameplan-v2 (`docs/gameplan-v2.md`). British spelling, own
words, no long quotes. Where I could not open a source or could not verify a claim, the text says so.

How the evidence is graded:

- **Read** means I opened the source in this session (paper abstract, README, write-up or research log).
- **Recalled** means a well-known paper I did not re-open for the detail quoted; the arXiv id was checked to exist and
  to carry the title given.
- **Blocked** means the site refused me (OpenReview, Fandom, Reddit, AAAI's PDF host); I did not work round it.
- Numbers of ours come from `docs/arena.md`, `docs/search.md` and `docs/rl.md`. "Gain" estimates are my judgement and
  are marked as such; none has been measured on our system.

## 0. Short version

1. The closest comparable project (section 2.1, `bengreff/polytopia`) is public and very informative. Same recipe
   (clone the built-in AI, then PPO self-play), same plateau (self-play adds only about +90 Elo after the first
   generations), same failure of value-head turn planning (30% against the raw network), same use of the game's
   per-seed deterministic engine. Its one big jump came from **fine-tuning on human tournament replays** (7,732 games):
   cloning the built-in AI got about 10% against Crazy, and the human fine-tune took the same network to 94% in its
   format (1v1 Might, Lakes, Small). That jump was larger than everything our PPO runs have bought.
2. Its human-midgame start curriculum was pre-registered and **failed** (45.6% against the control over 800 games).
   Our "scenario curriculum" item should be demoted until a diagnosis says which scenarios we lose.
3. The remaining standard tools split into cheap training-recipe changes (weight EMA, entropy bonus to zero, a moving
   KL anchor, larger batches), medium ones (belief heads, per-action features, exploiters) and expensive ones
   (search-improved targets under fog). The expensive ones depend on a value head that can rank candidate turns, and
   neither our search (no gain on rl3) nor the comparable project's (30%) shows one yet. A cheap test of that is the
   first thing to build before any search work (recommendation 1).
4. BPolait: see 2.9. Status of the search is stated there.

## 1. Methods from similar games and general game AI

Format per method: source, what it is, what it needs in our codebase, expected gain and why, cost, plan item
(gameplan-v2 section numbers: 1 belief, 2 features, 3 set plays, 4 targets and league, 5 speed, 6 live and external).

### 1.1 Search-improved targets

**AlphaZero and Expert Iteration.** [AlphaZero, arXiv 1712.01815](https://arxiv.org/abs/1712.01815); [Expert Iteration,
arXiv 1705.08439](https://arxiv.org/abs/1705.08439) (both recalled).
- What: the policy is trained towards the visit distribution of a tree search that uses the policy as prior and the
  value head at leaves; the value is trained towards game outcomes. The search is a policy-improvement operator, so the
  loop does not need PPO's noisy outcome gradient.
- Needs: a search over our *turn* (10 to 40 commands) that is fast enough to generate millions of targets, a target
  field in the replay data, a distillation loss next to the PPO loss.
- Gain: potentially the largest of any item here, and it is the textbook route past a PPO plateau (gameplan 4). But it
  only works if the search beats the raw policy, and our per-turn search gave +110 Elo on the imitation model and
  nothing on rl3 (`docs/search.md`). My estimate for now: 0 to +100 Elo, wide uncertainty, because the limiting factor
  (value quality, and a search that sees only our own turn) is not touched by the method itself.
- Cost: high (new search, new data path, GPU batching).
- Plan item: 4 (search-improved targets), gated by recommendation 1 below.

**MuZero.** [arXiv 1911.08265](https://arxiv.org/abs/1911.08265) (recalled). A learned dynamics model so search needs
no simulator. We have an exact, fast simulator, so this adds nothing for known states. The one place a learned model
could matter is the fogged part of the state; the belief route (1.5) is simpler. Drop.

**Gumbel MuZero / Gumbel AlphaZero** (Danihelka et al., ICLR 2022, "Policy improvement by planning with Gumbel").
Source page: [OpenReview bERaNdoegnO](https://openreview.net/forum?id=bERaNdoegnO) (**blocked**: browser check; content
below is from memory, not re-read).
- What: sample the root candidates without replacement with the Gumbel-top-k trick, spend a small simulation budget
  with sequential halving, and build an improved policy as softmax(logits + a monotone function of the "completed"
  Q-values). It is designed so the improved policy is a real improvement even with 2 to 16 simulations, which is the
  regime we are in (we can afford tens of rollouts per decision).
- Needs: our `SearchPolicy` (`python/polyai/search.py`) is already a degenerate form (top-k, one rollout each, scored
  by log prior plus w times logit value). The additions are: Gumbel sampling at the root, sequential halving over the
  candidates with more than one rollout each, completed Q for unvisited candidates (the value of the position or the
  mixed value), and writing the improved policy out as a *training target*.
- Gain: it fixes how the budget is spent and gives a principled target, but it does not fix a value head that
  mis-ranks. Polyfish (2.3) uses Gumbel search and, in its research log (EXP 3, read), found that going from 64 to 256
  simulations fixed only a "never captures" tail and cost 2.3 times the wall-clock. My estimate: small until the
  value head test in recommendation 1 passes.
- Cost: medium on top of what exists.
- Plan item: 4.

**Sampled MuZero** ([arXiv 2104.06303](https://arxiv.org/abs/2104.06303), title checked, content recalled). Plans over a
*sample* of actions drawn from the policy, with a correction so that the improvement step stays valid, for huge or
continuous action sets. We already restrict to the policy's top k, which is a biased version of the same idea. Use the
paper only if search ever needs more than top-k breadth; no separate item.

**KataGo** ([arXiv 1902.10565](https://arxiv.org/abs/1902.10565), recalled).
- What: AlphaZero-style training made roughly an order of magnitude cheaper by several stacked changes: auxiliary
  targets (final board ownership, score, the opponent's next move), global pooling blocks, variable board sizes in one
  net, playout cap randomisation (a few positions get a deep search and generate policy targets, the rest a shallow
  one and generate only value data), forced playouts with policy target pruning, and surprise-weighted sampling of
  training positions.
- Needs and mapping: the auxiliary-target idea is exactly gameplan 1A (belief heads). Add the other free engine
  labels in the same pass: final city ownership per tile, income and city count some turns ahead, the opponent's tech
  set. Playout cap randomisation and forced playouts only matter once we have search targets.
- Gain: auxiliary targets are reported as a clear help in KataGo's setting (dense supervision per position,
  tens of millions of positions). In ours the supervised part is BC data plus PPO batches, and the closest independent
  test I found (the Polyfish aux heads, 2.3) logged the aux losses but never turned them into a strength verdict. My
  estimate for 1A: 0 to +40 Elo on its own; its larger value is as an enabler for honest search and belief inputs.
- Cost: low to medium (labels already exist, `docs/belief.md`).
- Plan item: 1A, plus extra label channels.

### 1.2 League training, exploiters, surgery, large-scale PPO

**AlphaStar** ([Nature 2019, doi 10.1038/s41586-019-1724-z](https://doi.org/10.1038/s41586-019-1724-z); offline
follow-up [arXiv 2308.03526](https://arxiv.org/abs/2308.03526); both recalled).
- What: supervised start from human replays, then a league: main agents (self-play plus prioritised play against
  opponents they lose to), main exploiters (trained only against the current mains, to expose holes) and league
  exploiters (against the whole league). A KL term to the human policy keeps strategies sane. Actions are
  autoregressive (action type, then arguments, then which units). Fog is handled with recurrent memory; there is no
  explicit belief. A statistic of the human build order (the "z") conditions the policy and gives opening diversity.
- Needs: a pool that already exists (retail, bc, self snapshots); exploiter runs that train a fresh copy against the
  current best with no KL anchor and are added to the pool when they find a winning line; a way to record exploiter
  win rates over time.
- Gain: robustness, not retail strength. Our might1 run is a warning sign: +6 points against retail Crazy, but 45.1%
  against its own starting point, i.e. gains specific to one pool member. Exploiters are the standard cure and also
  the best test of whether rl3 has holes a human would find. My estimate: little change in the retail suite, a real
  change against humans and BPolait-like opponents; hard to put an Elo figure on it.
- Cost: medium (training time; the code is mostly pool configuration).
- Plan item: 4 (league). Start with one exploiter, a short run (about 100 updates) from rl3 with a higher entropy
  coefficient and no BC anchor; add it to the pool at a fixed weight if it beats rl3 more than 55%.

**Adversarial Policies Beat Superhuman Go AIs** ([arXiv 2211.00241](https://arxiv.org/abs/2211.00241), title checked,
content recalled). A small adversary trained against a *frozen* KataGo found a winning line that humans can copy,
although the network is superhuman in normal play. It is the strongest argument that a frozen-opponent exploiter
(cheap) tells you something self-play does not. Same plan item as above.

**PSRO / prioritised sampling.** [arXiv 1711.00832](https://arxiv.org/abs/1711.00832) (title checked). General
framework behind league pools. Our pool already weights opponents we beat less than 70% of the time (`docs/rl.md`),
which is a light version of prioritised sampling. No new item.

**OpenAI Five** ([arXiv 1912.06680](https://arxiv.org/abs/1912.06680), recalled).
- What: plain PPO with GAE at very large batch sizes and long horizons, a shaped reward early on that was annealed,
  an opponent pool of mostly the current agent plus some past versions, and "surgery": tooling to carry a trained
  network across changes to the observation space, architecture or environment instead of restarting.
- Needs and mapping: surgery is directly useful for gameplan 1B and 2. Adding belief planes or per-action features
  changes the input width; a zero-initialised new input slice (and zero-initialised new head weights) leaves rl3's
  behaviour unchanged at step 0, so each new feature group can be trained from rl3 instead of from BC. Our
  checkpoints already tolerate new heads (`load_checkpoint`), the input side is the missing piece.
- Gain: it removes a restart cost and makes A/B tests of features cheaper; no direct strength effect.
- Cost: low.
- Plan item: supports 1B and 2. Also note their batch size is far larger than ours (16k learner decisions, about 25
  games, per update); see recommendation 3.

**Phasic Policy Gradient** ([arXiv 2009.04416](https://arxiv.org/abs/2009.04416), title checked) and **UNREAL**
([arXiv 1611.05397](https://arxiv.org/abs/1611.05397), title checked). Both are about auxiliary losses on a shared
trunk with PPO-type learners; PPG separates the auxiliary phase from the policy phase so that aux gradients do not
fight the policy. Relevant if the belief heads (1A) are trained inside PPO: do aux updates in their own phase with a
small policy-KL penalty rather than adding the loss to every minibatch. Plan item 1A.

**Superhuman Generals.io by plain self-play** ([arXiv 2606.23348](https://arxiv.org/abs/2606.23348), abstract read).
Fog of war, many units, huge action space, 1v1; sparse win/loss reward only; a transformer policy trained by
plain policy-gradient self-play for seven days on four GPUs; reached first place on the public ladder and beat top
humans. The ablations found that an **exponential moving average of the weights is the stronger deployment policy**
and that **training only on the highest-advantage samples halves the cost of an iteration**.
- Needs: EMA of the parameters (and, for our BatchNorm model, of the buffers or a recalibration pass) kept alongside the
  learner and exported as the arena checkpoint; an advantage filter in `ppo_update`.
- Gain: EMA is nearly free and often worth a few points of win rate in noisy policy-gradient training; I would try it
  on the rl3 line first. Estimate +10 to +40 Elo, low confidence.
- Cost: very low.
- Plan item: new (training recipe).

**Artificial Generals Intelligence** ([arXiv 2507.06825](https://arxiv.org/abs/2507.06825), abstract read). Supervised
pre-training then self-play on a fast Gymnasium environment, potential-based reward shaping and memory features,
top 0.003% of the human ladder after 36 hours on one H100. Same family as the above and as our pipeline; supports
the case for dense signal (shaping or aux) and shows what raw speed buys.

**Lux AI Season 1, Toad Brigade write-up** ([README of the repository](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021),
read). Fully convolutional ResNet with squeeze-excitation (about 20M parameters), IMPALA plus UPGO and TD(lambda),
shaped reward for the first 20M steps and then sparse win/loss, **a frozen teacher with a KL term, with each larger
network trained against the smaller one as teacher**, per-unit policy heads on a dense grid, log-probability of a turn
as the sum over units, and a rotate-by-180-degrees average at test time. Takeaways for us: (a) the teacher-chain (a smaller or earlier net as KL anchor for the next) is
the same device as our KL to BC, and they credit it with stopping strategic cycles; (b) a cheap test-time
augmentation: average the policy over the 8 board symmetries (`docs/design.md` lists them as valid augmentations; I did not check whether training uses them). Estimate for (b): +0 to
+20 Elo for 8 times the inference cost, so only for live play; not for training.

### 1.3 Imperfect information

Facts about our fog (from `docs/design.md`): explored tiles stay visible for ever, so "explored" is the whole fog
model; what is hidden is unexplored terrain, villages, ruins and enemy cities and units there, plus opponents' stars
and techs. Map generation is the main random element. This is much milder than hidden cards: it is a one-off,
mostly static unknown that shrinks as exploration proceeds.

**ReBeL** ([arXiv 2007.13544](https://arxiv.org/abs/2007.13544), title checked) and **Student of Games**
([arXiv 2112.03178](https://arxiv.org/abs/2112.03178), title checked). Both search over public belief states with
counterfactual-regret machinery and a learned value on the *belief*, giving provable guarantees in poker-like games.
They need a tractable public state and a fixed, enumerable information structure. Polytopia's hidden state (map
content of a 400 to 900 tile grid, plus opponent techs) is vast and the game is long; I see no practical path here.
Borrow only the idea that the value should be a function of what the player *knows* (a belief summary), which
gameplan 1B already does by feeding predicted planes as input. Drop as a method.

**DeepNash / Regularised Nash Dynamics (R-NaD)** ([arXiv 2206.15378](https://arxiv.org/abs/2206.15378), title checked,
content recalled). Model-free self-play for Stratego (huge hidden state, no search). A reward transformation
regularises the policy towards a reference policy; the reference is periodically *replaced by the current policy*, which
produces convergence to a Nash fixed point instead of cycling. Mapping: our KL term pulls towards the fixed BC policy
and then decays; R-NaD suggests a *moving* anchor (periodically set the reference to the current best) as a cure for
the entropy drift and cycling that both we and the comparable project saw. It is a small change to `rl.py`. Estimate:
helps stability more than strength; +0 to +30 Elo, low confidence. Cost: low. Plan item: new (training recipe).

**Information-set MCTS and determinisation.** Cowling, Powley and Whitehouse, "Information Set Monte Carlo Tree
Search", IEEE TCIAIG 2012 (recalled; an application with results is [arXiv 2005.07156](https://arxiv.org/abs/2005.07156),
abstract read: SO-ISMCTS matched simple rule-based agents in a hidden-role game). Perfect-information Monte Carlo
(determinise the hidden state, search each sample as a perfect-information game, average) is the pragmatic baseline and
works well when the hidden information is mostly static, which is our case; its known failure modes (strategy fusion,
no value for information-gathering moves) matter less when scouting is cheap and the unknown shrinks every turn.
- Needs: a sampler of full maps consistent with the player's observation; a belief-aware search root that runs the
  existing rollout machinery on each sample and aggregates candidate scores across samples; a hard rule that the
  sampled state may not copy anything hidden from the true state.
- **Existing example of this exact design:** PolyEnv (2.4) ships `make_belief_env`, which builds a detached world from a
  complete hypothesis of the map, rejects any hypothesis that differs from the player's observed tiles, uses a
  synthetic seed so hidden RNG outcomes are not correlated with the true world, and offers a batched belief-MCTS
  self-play pool. That is gameplan 1C written down as an API; its validation rules are worth copying.
- Gain: removes the cheat in our search (it sees through fog) and in any search used for targets; without it, search
  targets teach the policy to rely on information it will not have. Strength gain by itself is unknown; correctness
  gain is certain.
- Cost: high (map sampler, engine hook to build a world from a hypothesis, 1B first).
- Plan item: 1C.

### 1.4 Multi-unit and combinatorial action spaces

Our policy is already the *autoregressive* form that AlphaStar uses: one command at a time, the state re-encoded after
each, scores over the engine's legal list (`docs/design.md` section 4). That avoids the joint action blow-up, and it is
the right choice while per-command inference is affordable. The alternatives and what each would give us:

- **Per-cell dense heads (Lux AI, Gym-microRTS).** One forward pass proposes an action for every unit at once.
  [Gym-microRTS, arXiv 2105.13807](https://arxiv.org/abs/2105.13807) (title checked) and the Toad Brigade write-up
  (read) both use it, with invalid-action masking. It makes inference one pass per turn instead of about 650 per game
  of ours. The cost is that units act on a stale picture within a turn (a move changes what the next unit should do),
  which matters in Polytopia's tight tactical turns. A hybrid worth knowing about: one trunk pass per *state change that
  matters*, with cheap head evaluations in between. Gain: speed only (inference is already a minor part of rollout time
  versus the retail opponent, see 1.7); skip.
- **Per-action input features, DouZero style.** [DouZero, arXiv 2106.06135](https://arxiv.org/abs/2106.06135) (title
  checked) scores every legal action with a network that takes the action's own feature vector. Gameplan 2 (combat
  preview, captures, exposure, retail's `cand_score`) is this. The policy head already receives
  (kind, arg, src, dst) embeddings; adding a small real-valued vector per action to the MLP input is cheap.
  Expected gain: tactical errors (attacking into a losing exchange, leaving a unit adjacent to a lethal enemy) are
  where a 1.4M-parameter conv net with one-command decisions is most likely to be sloppy, and these features hand it the
  answer. Estimate +20 to +80 Elo, moderate confidence, because the same information is partly recoverable from the
  tiles. Cost: low to medium. Plan item: 2.
- **Portfolio search with progressive unpruning (the Polytopia-specific one).** Perez-Liebana et al., [arXiv
  2104.08641](https://arxiv.org/abs/2104.08641) (abstract read): MCTS over a portfolio of scripts for the game Tribes,
  tuned by MAP-Elites for different play styles. Not a strength route for us, but the *action portfolio* idea matters
  for search breadth: when we search, candidates could be "the policy's top move", "retail's top candidate", "best
  attack by preview", "cheapest capture". A mixed candidate set is more robust than policy top-k. Cost: low once the
  features exist. Plan item: part of search (4).
- **Freeciv.** [CivRealm, arXiv 2401.10568](https://arxiv.org/abs/2401.10568) (title checked): a benchmark and
  environment for Civilization-like play with RL and language-model agents. I read only the title; I found no strong RL
  result there that transfers to us.
- **Wesnoth.** I found no useful paper or strong open implementation (one tiny GitHub repository, `tgambim/wesnoth-rl-agent`,
  2 stars; not read). Stated as not found.
- **Generals.io** is the best analogue: see 1.2.

### 1.5 Belief models in practice

- Predicting hidden map content from the visible part is a supervised task with free labels (our `truth.rs`).
- Polyfish (2.3) trains four auxiliary heads (final ownership, hidden enemy units, income five turns ahead, opponent
  techs) and logs flat or slightly rising losses without a strength verdict (their EXP 1, read).
- bengreff's network has a territory-ownership head; whether it helped was not isolated in their log (I found no
  ablation).
- PolyEnv exposes the *consumer* of a belief model (2.4) but ships no predictor.

Honest reading: nobody I found has shown that belief heads make a Polytopia bot stronger. Our own plan has the right
gate (measure against a most-common-value baseline, then an arena A/B), so I would keep 1A but not expect a jump, and
add the labels that are nearly free (opponent tech set, income and city count a few turns ahead, final ownership).

### 1.6 Opponent modelling and human-like play

- **Learning opponent models in deep RL.** [Opponent Modeling in Deep Reinforcement Learning, arXiv
  1609.05559](https://arxiv.org/abs/1609.05559) (title checked): an opponent encoder feeds a mixture of expert heads.
  For us the cheap form is global features about the *observed* opponent: how fast it expanded, whether its units were
  seen near our border, how many attacks it made, and the tribe. Polytopia has no hidden cards, but style (rush,
  turtle, naval) is a real hidden variable that humans read. Estimate: small; skip until human or BPolait games show we
  lose to a specific style. Plan item: 3 or 6, later.
- **KL-regularised search towards a human-like policy.** [Jacob et al., arXiv 2112.07544](https://arxiv.org/abs/2112.07544)
  (title checked, content recalled): search that trades value against closeness to a human-imitation policy gives
  strong yet human-like play in Diplomacy and Go. If we later train a human-imitation policy (1.8), this is the way
  to use it at play time: search with a KL penalty to that policy. Plan item: 4/6, later.

### 1.7 Speed and systems

- Our own measurement: retail opponents take 93% of engine CPU (gameplan 5). Anything that swaps retail seats for
  network seats in the pool lowers cost per game; this is a config change.
- Architectures for the actor-learner split: [SEED RL, arXiv 1910.06591](https://arxiv.org/abs/1910.06591) and [Podracer,
  arXiv 2104.06272](https://arxiv.org/abs/2104.06272) (titles checked). Central batched inference with actors
  stepping environments is exactly the Rust collector plan (gameplan 5). The main lesson is to keep inference in one
  place with large batches and to make the environment step asynchronous.
- Reference point: the Generals.io projects (1.2) get their speed from a purpose-built fast environment and a single
  GPU; Artificial Generals Intelligence reports 36 hours on one H100. Ours is a CPU engine at about 3,500 retail
  commands per second per thread (`docs/design.md`), which is why the retail share of CPU is the first target.

### 1.8 Curriculum, scenario starts and human data

**Go-Exploit** ([arXiv 2302.12359](https://arxiv.org/abs/2302.12359), published as "Targeted Search Control in
AlphaZero for Effective Policy Improvement"; title checked, content recalled). Keep an archive of states visited in
earlier games and start some training games from states sampled from it, so the agent practises positions it would
otherwise rarely reach. Reported to improve sample efficiency in Go and in small games; the archive is the agent's own,
which keeps the distribution close to what it will face.

**Backplay, reverse curriculum, single-demonstration starts.** [Backplay arXiv 1807.06919](https://arxiv.org/abs/1807.06919),
[Reverse Curriculum arXiv 1707.05300](https://arxiv.org/abs/1707.05300), [Salimans and Chen arXiv 1812.03381](https://arxiv.org/abs/1812.03381),
[Go-Explore arXiv 1901.10995](https://arxiv.org/abs/1901.10995) (titles checked). Start from states near the goal
(or from a demonstration) and move the start back. Natural for sparse-reward problems with a long path to the reward.
Polytopia's reward is sparse, but the starts that matter are mid-game contested positions, not "near the goal".

**Direct evidence from the comparable project (read).** bengreff pre-registered "30% of training games start from 1,715
close, complex human midgames" (S4b) against an identical control (S4a). Over 800 games at matching generations the
treatment won **45.6%**; the falsifier fired and they dropped it. A larger set (53,272 varied positions, track E) was
queued; I found no result for it in the log. Their combined S3b/S3c change, which included human starts, coincided with
instability (entropy rising, games turning into grinds), but four changes went in together so no cause was
attributed. Take-away: **human-position starts did not help on this evidence** (one project, one format, evaluation
from turn 0 only). Our scenario curriculum item should be scheduled behind diagnostics (recommendation 5), and if
tried, as a pre-registered A/B with the same falsifier shape.

**Imitation of human play.** [VPT, arXiv 2206.11795](https://arxiv.org/abs/2206.11795) (title checked): behaviour
cloning on a large human corpus, then RL fine-tuning, with the human policy kept as a prior; AlphaStar did the same.
The Polytopia-specific datapoint is again bengreff (read): cloning the built-in AI plateaus (their 20k-game BC about
9% against Crazy, the retail AI itself in the same seat about 10%; ours: bc4 12.1% against Crazy, `docs/arena.md`),
whereas fine-tuning on **7,732 human championship games** (3.17M decisions) gave 94% (188/200) against Crazy, and 92.4%
over a 5,760-game gate across all 12 x 12 tribe pairs and both seats. Their human top-1 agreement was about 43%.
Conditions that make it not directly comparable with our numbers: one format (1v1 Might, Lakes, Small 14x14, 12
regular tribes, fixed seat-assignment rules), and the bot has the Crazy handicap behaviour of their engine build
(game version 123). Even so, the gap between "clone the built-in AI" and "clone good humans" is larger than anything
our PPO has delivered.

## 2. Other Polytopia AI agents

Search method: GitHub repository search through the REST API for "polytopia" with ai, bot, simulator, engine and
reinforcement as terms (417 results for the bare word; I looked at everything with a relevant name or description),
arXiv and Crossref searches, and the repositories' own READMEs and logs. Not searched in the browser: Discord, YouTube,
Steam forums, Reddit (HTTP 403), Fandom wiki (Cloudflare block). Star counts and dates are from 2026-10-29.

### 2.1 bengreff/polytopia: the closest comparable (read in depth)

- Links: [repository](https://github.com/bengreff/polytopia), [results site](https://bengreff.github.io/polytopia-ai/),
  [research log](https://github.com/bengreff/polytopia/blob/main/docs/design/research-log.md) (about 1,150 lines,
  pre-registered experiments). Developed with an AI coding agent; active in the last week.
- Approach: the rules engine is **the game's own compiled code** (il2cpp library loaded headlessly, no Unity),
  checked byte for byte against the real app with tracing: same maps, states and built-in bot moves. A 5.9M-parameter
  transformer over the tiles and a global token, pointer policy over the legal list, value head and ownership head,
  fog-correct observations (verified by a leak test). Training: behaviour cloning of the built-in AI, fine-tune on
  human championship replays, then PPO self-play in generations (1,500 games each, about 1 to 2 hours) with a KL limit
  per generation and opponents from earlier generations.
- Data: 7,732 human games in the World Championship format, fetched by game id from the game server (27,548 ids tried,
  7,822 downloaded, 19,726 refused, mostly older games that had expired); 6,072 replay exactly to the server's final
  state. 92% of these games are mirror matches. They note about 14,000 further public tournament games in another map
  type that they did not download.
- Strength (their numbers, 200 games per line, greedy, no search, fog-correct): about 94% against Crazy after the human
  fine-tune; generation 7 of self-play 98% (196/200) against Crazy and 58.5% against generation 0 head to head;
  generation 24 reached 70% against generation 0, their best single result. Gate over all tribe pairs: 92.4%.
- What did not work, all with falsifiers fired (a useful list for us): winners-only imitation; three PPO variants
  against the bots; human-midgame starts (45.6%); value-head turn planning with 8 sampled plans (30%; the value head
  preferred sampled plans on 73% of turns and they were worse, an off-distribution over-estimation effect); a
  self-play run that diverged (entropy climbing from about 1.7 to above 3 after progress stalled; they suspect an
  entropy bonus that no longer cancels once the advantage signal is noise).
- What worked: per-move credit with GAE (58.1% against the control over 800 games; we already use GAE with lambda 0.95);
  small KL-bounded generations; paired sign tests on identical games to read small differences; a per-cell gate that
  found three genuinely weak matchups (11 to 12 wins out of 20) that a pooled number hid.
- Availability: public code and results; the game files and replays are not included. Their weights are not published
  that I could see. No way to play against it that I found.
- What we can learn: (1) human fine-tuning is the large lever; (2) our plateau is the standard plateau, and so is our
  search result; (3) evaluate per cell and use paired tests; (4) watch entropy and set the entropy bonus to zero if it
  climbs; (5) their game-state fidelity (the real compiled rules) removes the doubt we have about retail-AI parity.
  What we have that they lack: all sizes, presets and player counts, and a collector that is faster per core than
  running the real game's library.
- Benchmark use: their published Crazy numbers are for Lakes Small Might, so compare only after running our model on
  that cell (cheap: one arena filter).

### 2.2 GAIGResearch/Tribes (Java, academic framework)

- [Repository](https://github.com/GAIGResearch/Tribes) (108 stars, last push 2022-08, no licence file listed in the API
  result); paper: Perez-Liebana et al., "Tribes: A New Turn-Based Strategy Game for AI Research", AIIDE 2020,
  [doi 10.1609/aiide.v16i1.7438](https://doi.org/10.1609/aiide.v16i1.7438) (AAAI's page returned unreadable binary to
  my fetch, so I did not read the paper itself).
- Approach: an open re-implementation of Polytopia for AI research with a forward model, a Capitals and a Score mode,
  optional fog, and statistical forward-planning agents (MCTS, rolling-horizon evolution, one-step lookahead). A later
  paper builds on it: [arXiv 2104.08641](https://arxiv.org/abs/2104.08641), Portfolio MCTS with progressive unpruning
  and MAP-Elites play styles (abstract read).
- Strength: no comparison with the retail AI that I could find; the rules are an old version of the game, so it is not
  a benchmark for current play.
- What to learn: the forward-model interface and the idea of portfolio candidates (1.4). The 2021 paper's tuning of
  search parameters by quality-diversity is a template for generating *diverse opponents* for the pool.

### 2.3 HenBOMB/Polyfish (Rust, self-play, public)

- [Repository](https://github.com/HenBOMB/Polyfish) (15 stars, last push 2026-10-24). README (read): a TypeScript then Rust
  re-implementation, a web UI, a Gumbel AlphaZero-style MCTS with a candle ResNet, self-play data in safetensors, a
  PyTorch trainer, a frozen-anchor Elo ladder, and a C# PolyMod plugin that extracts live game states from the
  Steam client.
- Strength: its own log (read, 2026-07-11, flagged as provisional by the authors because the gauge had a bug): the
  network lost to its greedy teacher about 2 to 1, though improving by about 500 Elo over the era. I did not
  find a later reading that beats the greedy teacher. Treat it as pre-strong.
- What to learn: the pre-registration style (every change has a hypothesis and a verdict; several "obvious" ideas are
  recorded as rejected), the paired net-vs-net ladder with a non-net floor anchor, and the warning in their own log
  that an unchecked evaluation gauge silently returned nothing for weeks. We already run seeds and logs; the point to
  copy is "check that the instrument actually produced a reading".

### 2.4 FryOne2137/PolyEnv (C++ core, Python API)

- [Repository](https://github.com/FryOne2137/PolyEnv) (6 stars, last push 2026-10-16), [docs](https://polyenv.readthedocs.io/).
  Unofficial engine for the 12 regular tribes, batched vector environments, a native batched PUCT pool and a
  belief-MCTS self-play pool. No trained bot included; a bot module is announced.
- What to learn: the belief-world API (1.3). Their documented rules for it are the best written guide to making
  fog-honest search safe. Not a benchmark.

### 2.5 PolyVision (Java Tribes through Py4J, PPO)

- [NeoLorenzo/PolyVision](https://github.com/NeoLorenzo/PolyVision) (1 star, 2026-09-30), derived from
  ClaireBookworm/polytopia_rl (not opened). Legality-aware PPO with a global action catalogue and a 47-dimensional
  per-action feature vector. Explicitly does not solve the full game: its curriculum is an economy-only Bardur task
  ending at turn 10. Interesting as another example of per-action features (1.4) and of using real extracted maps
  (5,517 genuine 11x11 Bardur maps). Not a benchmark.

### 2.6 LLM agents

- [augustallen/polytopia-claude](https://github.com/augustallen/polytopia-claude) (read): frontier language models play
  through a PolyMod socket plugin, a text observation of about 900 tokens and a numbered legal list; one model call
  plans a whole turn. Results are single games on Tiny maps: one win and one loss against the Normal built-in bot for
  one model, and model-vs-model games each costing about 25 to 55 minutes. Evidence that LLMs play at roughly the Normal
  bot's level, slowly and at cost, with few games. Useful as a *style* opponent for tests, not as a benchmark.
- [andytam101/polytopia-ai](https://github.com/andytam101/polytopia-ai) (0 stars): the same idea, PolyMod plugin plus an
  HTTP backend for an LLM planner. Not read beyond the README head.

### 2.7 Other repositories (seen, not benchmarks)

- [jaydenshi99/polyshark](https://github.com/jaydenshi99/polyshark) (C++ engine, empty README),
  [AshtonCoates/PolyBot](https://github.com/AshtonCoates/PolyBot) (Rust RL agent, README empty, pushed 2026-10-21),
  [DylanHannaScience/polytopia_python](https://github.com/DylanHannaScience/polytopia_python) (Python simulation stub),
  [isaacdchu/polybot](https://github.com/isaacdchu/polybot) (screenshot-driven bot, on hold),
  [Lenguist/polytopia-ai](https://github.com/Lenguist/polytopia-ai) (3D web client with rules),
  [redish2098/Polytopia-AI-Loader](https://github.com/redish2098/Polytopia-AI-Loader) (PolyMod framework for loading a
  custom AI into the real game; not read),
  [PolyModdingTeam/PolyMod](https://github.com/PolyModdingTeam/PolyMod) (the mod loader that the LLM and extraction
  projects above rely on, 96 stars, active).
- Community tools: [Nelluk/Polytopia-ELO-Bot](https://github.com/Nelluk/Polytopia-ELO-Bot) (Discord ELO ladders, match
  history; 68 stars, GPL-3.0, active) is the human community's rating infrastructure, a possible source of who the
  strong human players are; [frothfrenzy/polytopiacalculator](https://github.com/frothfrenzy/polytopiacalculator) and
  [Bibliofile/polytopia](https://github.com/Bibliofile/polytopia) are battle calculators/simulators (useful for
  cross-checking our combat formulas).

### 2.8 Write-ups on how the retail AI works

I found **none** that describe Midjiwan's AI. The Fandom wiki pages (Bots, Difficulty) are behind a Cloudflare check, so
I could not read them. The best available descriptions are indirect: our own `ai_move.rs` and `ai_candidates()`
(`docs/design.md` section 1), and bengreff's note that the built-in AI keeps its full view of the map (reads hidden
information) and that "handicap" and a "stupidity" parameter set difficulty, observed by calling the game's own
`AI.GetMove` (research log common setup, read). Nothing else turned up in the GitHub search.

### 2.9 BPolait

**I could not find who runs BPolait or how it works.** What I tried: GitHub repository and user searches for the name,
for "polait" and for the friend code LLWKL-KOAPO (0 results each); a search of our own tree and ledger (only our docs
mention it); one general web search (nothing relevant; results were the repositories in 2.1 to 2.7 and the Discord
ELO bot); DuckDuckGo and Bing page fetches (empty or unrelated). Reddit (403) and the Fandom wiki (Cloudflare) were not
readable, and I did not try Discord or YouTube.

What can be said from the outside, as inference and not as fact:

- **It is probably not bengreff's bot.** Their site and README show replays only; neither mentions a friend code,
  online play or a hosted opponent, and the repository was created on 2026-10-23.
- **It is probably built the same way as the PolyMod bots** (2.6): the only ways to play through the real multiplayer
  stack are a client automated through PolyMod or a re-implemented headless client with the game's protocol. Play in
  Might on Tiny and Small maps, and the name, suggest a small hobby project. Its strength is unknown; treat "BPolait
  plays like a good human" as untested.
- **What to do about it:** the benchmark plan in gameplan 6 stands (private games, opponent tagged). Because nothing is
  known about it, the first 20 games should be used to *characterise* it (win rate, game length, opening, units built)
  rather than to rate us. Ask Kenzi where he found the friend code; the source may name the author.

## 3. Recommendations

Ranked by (expected gain x confidence) over cost. "Elo" figures are my estimates, not measurements.

1. **Test whether the value head can rank turns, before building any search or search-target work (new, cost: 1 to 2
   days).** Take positions at the start of our turns from rl3 self-play, generate several candidate turns per
   position (policy samples and the top-k plans), play each out to the end of the game a few times, and measure the
   rank correlation between the end-of-turn value and the realised outcome. If correlation is near zero within a
   position, search and search-improved targets (gameplan 4) will not help whatever the budget; if it is clearly
   positive, search is worth fixing. Reason: our search gave nothing on rl3 and the comparable project's value-head
   planning lost (30%, 2.1); this makes the dependency explicit and cheap to test. Gain: avoids a possibly wasted
   multi-week item; uncertainty is about the test itself.
2. **Add a human-data track as the lead strategic item, and find a corpus (new, cost: medium to high, mostly
   acquisition).** The only strong result I found in the field (94% against Crazy, 2.1) came from fine-tuning on
   human championship games after cloning the built-in AI, the same starting point as ours. This is the largest
   documented jump and it is not on our plan (gameplan 3 mentions "later human replays" only as scenario seeds).
   First step: find out how the game exposes finished games (the live-play interface likely already speaks the
   protocol) and whether replays can be fetched by id; do not copy their scraping scale without checking the terms.
   Expected gain: potentially the largest of any item (tens of points of Crazy win rate for them); for us unknown
   because rl3 starts from a higher point (79% against Crazy) and our formats differ.
3. **Do the cheap recipe changes now, each as a short A/B on the rl3 line (cost: hours of GPU each).** (a) EMA of the
   weights as the exported checkpoint (Generals.io result, 1.2); (b) entropy coefficient to zero or a schedule if
   entropy rises (bengreff's divergence, 2.1); (c) a moving KL anchor in the R-NaD style (1.3); (d) larger PPO batches
   (OpenAI Five, 1.2). Gain: +10 to +60 Elo combined, plausible but low confidence; cost: very low; they also
   make later feature work easier to read.
4. **Keep belief heads (1A) and per-action features (2) at the top of the plan, in that order of cost, and add the free
   labels (opponent techs, income and city count ahead, final ownership) to 1A (gameplan 1, 2).** Both are cheap, and
   both can be added to rl3 with zero-initialised new inputs (surgery, 1.2), so each A/B is a short run, not a restart.
   Expect +20 to +80 Elo from features and 0 to +40 from heads; read them with paired tests on identical seeds
   (bengreff's method) at 400 games.
5. **Add a first exploiter run to league training and make it a diagnostic (gameplan 4, cost: one short run).** Train a
   fresh copy from rl3 for about 100 updates with no BC anchor, only against rl3. A win rate above 55% against rl3
   says what rl3 is weak to and gives the scenario curriculum its data; a win rate near 50% says rl3 is more robust
   than the retail suite suggests. Gain: information first, robustness second.
6. **Demote the scenario curriculum and opening book until diagnostics exist (gameplan 3).** The only direct
   test found (human midgame starts, 2.1) lost to its control (45.6%, 800 games), and our own might1 result (45.1%
   against its parent) shows how easily training gains do not transfer. Keep the book (cheap, checkable in the arena);
   keep the curriculum only for positions the exploiter or live games show we lose.
7. **Keep Rust collector and batching (gameplan 5) early.** Every item above that needs many games (exploiters,
   A/Bs at 400 games, any search targets) is gated by games per hour, and retail seats are 93% of CPU. This is the one
   item that makes the others cheaper, and nothing in the literature argues against it.
8. **Move honest-under-fog search (1C) and search-improved targets (gameplan 4) behind recommendation 1.** PolyEnv's
   belief-world API is the best reference for the design (hypothesis checked against what the player has seen,
   synthetic seed for hidden randomness). Do not start before the value test passes.
9. **Drop or defer:** MuZero-style learned dynamics (we have an exact simulator), ReBeL and Student of Games (no
   tractable public state), dense per-cell policy heads (speed is not the limit), hand-written macro actions (already
   not planned), opponent-style encoders (until we know we lose to a style), 8-way test-time augmentation for
   training (use only in live play, 1.2).
10. **Measure and report per cell, as bengreff does.** A pooled 79% hides weak matchups; the Might suite should publish
    a tribe-pair by seat table, so the exploiter and the human-data track have targets. Cost: an arena report change.

### Things I would not rely on

- The size of any Elo estimate above. None has been measured here; they are ranges from reading and judgement.
- The claim that human fine-tuning will transfer to us: one project, one format, one engine build.
- Anything about BPolait beyond "not found".

### Source count

About 45 distinct sources touched: 8 Polytopia repositories read in some depth (bengreff, Polyfish, PolyEnv, Tribes,
PolyVision, polytopia-claude, andytam101, Lux AI write-up) plus 7 more repositories seen only by name or README head;
about 28 papers (arXiv ids resolved for their titles and abstracts where marked "title checked"; most method
descriptions are from memory and are marked "recalled"); 4 searches that returned nothing useful for BPolait.
