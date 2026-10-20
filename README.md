# polytopia-ai

Machine learning bot research for The Battle of Polytopia: learned policies and value functions trained on games
played by polytopia-engine (self-play, imitation of the retail AI and of recorded games). Private.

Status: design done (`docs/design.md`), first crates in progress.

## Layout
- `crates/polyai-core`: observation and action encoding, game setup, game records.
- `crates/polyai-py`: PyO3 module `polyai_engine` (maturin).
- `crates/polyai-gen`: imitation data generator.
- `crates/polyai-arena`: agents, match runner, MCTS adapter.
- `python/polyai`: model, data loading, training, evaluation.

## Sibling repos (expected)
- polytopia-engine at `../engine`: the simulator that generates games and evaluates bots.
- polytopia-weekly at `../weekly`: search baselines to compare against.

## Data
Datasets, checkpoints and run logs live in `/kvm/polytopia-ai-data/` and stay out of git. Use
`TMPDIR=/kvm/tmp/tmp` and `UV_CACHE_DIR=/kvm/tmp/uv-cache` (the root disk is small). Python is
`/kvm/polytopia/.venv/bin/python`. Long jobs run under `nice` with at most 24 threads.
