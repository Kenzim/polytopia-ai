# Results

The raw records behind the numbers in the top-level README and in `docs/`. A `.jsonl` file is one match or table:
one line per game with the seed, the agents by seat, tribes, map setup, winner (`-1` is a draw at the 150-turn cap),
turns and the engine commit (the fields are listed in `docs/arena.md`). The `.log` next to it ends with the totals.
Paths inside the files are those of the host that produced them.

```
python tools/elo.py results/arena/might/might1_crazy.jsonl     # win rates with 95% intervals and Elo
python tools/cells.py ...                                      # per-tribe and per-map tables, paired sign tests
```

| Folder | Content | Written up in |
|---|---|---|
| `autoloop/` | `history.jsonl`: every decided round of the training loop with its gate, its Crazy guard and the decision. `promotions.jsonl`: the promotions. `config.json`: the loop's settings from round 98 (20 November 2026, engine features on). `champions/`: each champion's 200 games against Crazy on the guard's fixed seeds | `docs/autoloop.md` |
| `arena/suite-v2/` | The built-in AI ladder and the first imitation model on the 1v1 suite | `docs/arena.md` |
| `arena/rr*`, `arena/all*`, `arena/mcts*` | The first tables on the old suite (Small maps, 50 turns) | `docs/arena.md`, "Older results" |
| `arena/bc4/`, `arena/rl1/`, `arena/rl3/`, `arena/rl4*/`, `arena/b10_vs_rl3/`, `arena/wide_big/` | Imitation and Domination self-play models, larger networks, big-map runs | `docs/arena.md` |
| `arena/search/`, `arena/search-rl1/`, `arena/tta/` | Per-turn search and test-time augmentation | `docs/search.md` |
| `arena/might/`, `arena/lakes-small/` | The move to Might: rl3 and might1, and rl3 on Lakes Small | `docs/arena.md` |
| `arena/featab/`, `arena/beliefrl/`, `arena/stack/`, `arena/r4/` | Engine features and the fog loss, first and second seeds, and both together | `docs/features-ab.md`, `docs/belief.md` |
| `arena/recipe/` | EMA, entropy schedule, moving anchor, advantage filter | `docs/rl.md`, "Recipe options" |
| `arena/exploiter/` | The exploiter diagnostic | `docs/arena.md` |
| `arena/human/`, `arena/water/` | The human fine-tune round and the water round, with their gates | `docs/human-data.md` |
| `arena/soup/`, `arena/blend/` | Weight averaging | `docs/arena.md` |
| `arena/bpolait/` | Log of the first BPolait imitation | `docs/human-data.md` |
| `value-rank/` | The value-rank test: realised results of candidate turns | `docs/search.md` |
| `ab-2026-11-19/` | The three-arm A/B from round 77: `run.sh` is the training command, `matches/` the ten 400-game matches, `plateau-diagnostics/` the six matches of 20 November on why rounds stopped passing the gate | `docs/rl.md`, top-level README |
| `ab-2026-11-20/` | The features and advantage-filter arms from round 77: `run.sh` is the training command (the arm's option added at the end), `matches.sh` the match script, `matches/` the ten 400-game matches, `temperature/` round 77's top move against round 77 sampling at temperature 1.0 and 0.5 | Top-level README, `models/README.md` |
| `calibration/` | Output of `polyai.calibrate` for round 77 and the three A/B arms on 8,859 human games: Brier score, calibration error and win share per probability bucket and game stage. Aggregates only, no games | `docs/rl.md` |
| `polytopai/` | Round 77, top move, against the PolytopAI website's bot at its easier setting: one line per game and the summary (score, speed, how often the two legal lists agreed). `ab-2026-11-20/`: the training A/B with site games, `run.sh` the training command, `matches.sh` the match script, `matches/` the 400-game matches and the 200 games of each model against the site, `logs/` the training logs, the site run's metrics and its games | `docs/polytopai.md`, `models/README.md` |

Not here: the human games and anything else under `human/` on the training host, the live games against BPolait
(they are in the client's repository), run directories with their per-update metrics, and the imitation datasets.
