# Models

Each folder holds one trained model: the checkpoint (`ckpt_final.pt`, or the named update) and the `config.json` of
the run that made it, which records every training option, the starting model and the reference model. All are the
6-block, 96-channel network (1.42 million parameters, encoding version 2) unless stated. A checkpoint holds the
weights, the model configuration and the step count; it has no optimiser state, so it is for playing, evaluating
and starting new runs, not for resuming the run that made it.

Load one with `polyai.train_bc.load_checkpoint(path)`, or pass the folder to any tool that takes a model:
`python -m polyai.arena --model ../models/auto_r0077 ...`, `python -m polyai.rl --init ../models/auto_r0077 ...`.
For `rl3_b6c96` pass the checkpoint file itself, since the model that matters there is update 480 and not the end of
the run. The paths inside `config.json` are those of the training host.

Scores are over 400 games (200 seeds, both seats) unless stated; "Crazy" is the built-in AI at its hardest level.

## The champion line

| Folder | What it is | Measured |
|---|---|---|
| `bc3_b6c96_v11` | Imitation of the built-in AI, 40,000 steps. The KL reference and the `bc` pool opponent of every self-play run | Predicts the built-in AI's move 54.7% of the time; 10.6% against Crazy (Domination) |
| `rl1_b6c96_cpu` | First self-play run from `bc3`, 200 updates | 68.9% against Crazy (Domination) |
| `rl3_b6c96` (`ckpt_u00480.pt`) | Self-play continued from `rl1` to update 480. Champion until 9 November | 79% against Crazy in Domination, 75.8% in Might |
| `hbc1_rl3_human` | `rl3` fine-tuned to predict human tournament moves, 6,000 steps | Predicts the human move 39.5% of the time; on its own 39.5% against `rl3` over 200 games |
| `hrl1_human_b6c96` | 300 updates of self-play from `hbc1`. Champion 9 November | 63.2% against `rl3`; 77.8% against Crazy |
| `wrl1_water_b6c96` | 300 updates from `hrl1` with water maps weighted up. Champion 14 November; played most of the live games against BPolait | 56.0% against `hrl1`; 86.2% against Crazy; 41.8% on Water World (14% before) |
| `auto_r0053` | Loop round 53. Champion 16 November | 55.4% against `wrl1`; 83.5% against Crazy |
| `auto_r0055` | Loop round 55. Champion 16 November | 55.6% against round 53; 81.0% |
| `auto_r0060` | Loop round 60. Champion 17 November | 58.2% against round 55; 83.5% |
| `auto_r0066` | Loop round 66, the first with 85% 1v1 games and the BPolait imitation in the pool. Champion 17 November | 57.0% against round 60; 83.0% |
| `auto_r0072` | Loop round 72. Champion 18 November | 57.6% against round 66; 86.8% |
| `auto_r0075` | Loop round 75. Champion 18 November | 57.2% against round 72; 84.5% |
| `auto_r0077` | Loop round 77. **The current champion** since 18 November | 55.1% against round 75 (and 55.1% again on fresh maps); 89.0% against Crazy |

## Other models

| Folder | What it is | Measured |
|---|---|---|
| `bc4_b10c128_v2` | Imitation with the 10-block, 128-channel network (3.4 million parameters), 100,000 steps | Predicts the built-in AI's move 62.6% of the time; 12.1% against Crazy: better imitation, no stronger |
| `bpbc_g0412` | Imitation of BPolait's moves, fine-tuned from `hbc1` on 412 of our live games against it (151,000 positions). A pool opponent since round 66 | Predicts BPolait's move 51.7% of the time on held-out games |
| `auto_r0064` | Loop round 64, not promoted. Arm B of the live tests against BPolait | 53.6% against round 60; 87.5% against Crazy |
| `might1_b6c96` | 600 updates of Might self-play from `rl3` | 81.4% against Crazy but 45.1% against `rl3` |
| `featrl_b6c96` | 300 updates from `rl3` with engine-computed features as extra inputs (run with `--features`; 1.43 million parameters) | 58.2% against the plain control |
| `beliefrl_b6c96` | 300 updates from `rl3` with the auxiliary loss on what is under the fog | 58.1% against the plain control; a second seed scored 46.2% |
| `rec_advf` | 300 updates from `rl3` keeping the half of positions with the largest advantage. The recipe the loop uses | 54.2% against the plain control |
| `expl1_b6c96` | Exploiter: 100 updates against a frozen `rl3` only | 43.1% against `rl3` |

## The A/B of 19 November

Three 300-update runs from `auto_r0077` with the loop's settings and seed 900.

| Folder | Option | Against `abc_ctrl` | Against round 77 | Against Crazy |
|---|---|---|---|---|
| `abc_ctrl` | none (the control) | | 46.1% | 88.8% |
| `abc_turn` | `--credit-per-turn` | 45.9% | 43.5% | 88.8% |
| `abc_crit` | `--credit-per-turn --priv-critic`; `critic_final.pt` is the critic that sees under the fog | 41.8% | 43.1% | 85.0% |

## The A/B of 20 November

Two more 300-update runs from `auto_r0077` with the same settings and seed, trained on the RTX 4090 laptop and
matched against the control `abc_ctrl` of 19 November. In this second set of matches the control scored 45.6%
against round 77 and 86.8% against Crazy, and round 77 scored 87.8% against Crazy.

| Folder | Option | Against `abc_ctrl` | Against round 77 | Against Crazy |
|---|---|---|---|---|
| `abd_feat` | `--features` (1.43 million parameters) | 55.4% | 48.4% | 88.2% |
| `abd_noadv` | `--adv-filter 1.0` (the filter off) | 47.5% (45.8% against `abd_feat`) | 42.1% | 89.8% |

## The website A/B of 20 November

Two 300-update runs from `auto_r0077`, seed 900, the live loop's settings, on the bindings with `Game.from_position`
(`docs/polytopai.md`). Round 77 scored 88.5% against Crazy in this set.

| Folder | Option | Against `abe_ctrl` | Against round 77 | Against Crazy | Against the site (level 1) |
|---|---|---|---|---|---|
| `abe_ctrl` | none (the control) | | 43.9% | 88.8% | 4 of 200 |
| `abe_site` | `--polytopai-games 8 --polytopai-level 1` | 38.9% | 34.8% | 86.0% | 5 of 200 |

## The temperature and overlap A/Bs of 20 November

Three 300-update runs from `auto_r0077` with `--features`, seed 900, the live loop's settings, on the same code and
bindings as `abd_feat`, which is their control (48.4% against round 77, 88.2% against Crazy in this set).

| Folder | Option | Against `abd_feat` | Against round 77 | Against Crazy |
|---|---|---|---|---|
| `abf_t05` | `--temperature 0.5` (the learner samples at 0.5) | 39.6% | 39.0% | 88.5% |
| `abf_t05o` | `--temperature 0.5 --opp-temperature 0.5` | 47.2% (52.2% against `abf_t05`) | 43.5% | 87.5% |
| `abf_ovl` | `--overlap` | 44.6% | 47.4% | 86.8% |

## The value-opp A/B of 20 November

Two 300-update runs from `auto_r0077` with `--features`, seed 900, the live loop's settings, on ai 47167b5 with the
profile-guided bindings of engine 80a68d0. Round 77 scored 88.8% against Crazy in this set.

| Folder | Option | Against `abg_ctrl` | Against round 77 | Against Crazy |
|---|---|---|---|---|
| `abg_ctrl` | none (the control) | | 45.5% | 90.0% |
| `abg_vopp` | `--value-opp 2.0 --value-opp-frac 0.25` | 37.0% | 35.9% | 85.5% |

## Analysis value networks

Not policies: win chance networks for the website's game analysis (`docs/analysis.md`), read from one player's own
view. Each folder holds `value.pt` (`polyai.analysis.load_value`: the weights, the calibration layer per game kind and
stage, and the training record) and `gate.json` (its gate on held-out games). Serve them with
`python -m polyai.serve ... --value models/analysis_v2/value.pt models/analysis_v1/value.pt`.

| Folder | What it is | Measured |
|---|---|---|
| `analysis_v1` | `auto_r0077`'s trunk and value head fine-tuned 30,000 steps on value rows of both sides of 8,906 human games and 48,000 generated website-style games | Passes the gate at most stages of every kind; the service's fallback |
| `analysis_v2` | `analysis_v1` fine-tuned 15,000 more steps with 30,000 more games weighted to two-player Domination and Might | Passes every stage of two-player Domination; the service's first choice |
