# Engine features A/B (2026-11-01)

Question: does giving the network the features the engine already computes (encoding.md section 12) make it play
better? Verdict from RL, not from behaviour cloning (BC fine-tuning on retail-AI Might data weakens the model whatever
the arm, see belief.md).

## Setup

- `featrl_b6c96`: `polyai.rl --features` from `rl3_b6c96/ckpt_u00480.pt`, Might, exactly the config of `might1_b6c96`
  (same seed 0, KL reference, pool, learning rate, 384 games, 32768 decisions per update), 300 updates. The feature
  layers start at zero, so update 0 is the plain rl3 u480 policy. Peak GPU memory 1.07 GB (might1: 1.04 GB).
- Control: the existing `might1_b6c96` at update 300 (`ckpt_u00300.pt`). One run per arm, so run-to-run noise between
  two seeds is not measured; the intervals below are game sampling error only.
- Evaluation: Might 1v1 suite, greedy, no search, 200 seeds x 2 seats = 400 games each, CPU. Files in
  `/kvm/polytopia-ai-data/arena/featab/`.

## Results

| | vs retail:hard | vs retail:crazy |
|---|---|---|
| might1 u300 (control) | 355-45-0, 88.8% (85.3 to 91.5) | 299-100-1, 74.9% (70.4 to 78.9) |
| featrl u300 (features) | 373-27-0, 93.2% (90.4 to 95.3) | 306-94-0, 76.5% (72.1 to 80.4) |
| paired sign test (same 400 games) | features better in 34 games, control in 16, p = 0.015 | 48 against 41, p = 0.53 |

Head to head, featrl u300 against might1 u300, Might: 221-155-24 (24 capped draws), **58.2%** (53.4 to 63.0).

Reading: the features arm is ahead of the plain arm at the same number of updates, clearly head to head and against
Hard, not distinguishable against Crazy. The head-to-head interval excludes 50% but this is one training run per arm.
A second seed of each arm would be needed to separate the effect from training noise.

## BC sanity (not the verdict)

Fine-tune from rl3 u480 on the 10,000-game Might shard set made with `polyai-gen --features` (seeds 70001 to 80000),
4000 steps, same settings, held-out positions:

| | top-1 | top-3 | policy CE | winner CE | Brier |
|---|---|---|---|---|---|
| plain (`featab_bc_plain`) | 0.556 | 0.814 | 1.454 | 0.309 | 0.0637 |
| features (`featab_bc_feat`) | 0.560 | 0.812 | 1.447 | 0.301 | 0.0621 |

Small gains, within the noise of a single run.
