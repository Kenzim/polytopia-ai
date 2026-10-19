# polytopia-ai

Machine learning bot research for The Battle of Polytopia: learned policies and value functions trained on games
played by polytopia-engine (self-play, imitation of the retail AI and of recorded games). Private.

Status: empty skeleton. PyTorch with CUDA is available on the host.

## Sibling repos (expected)
- polytopia-engine at `../engine`: the simulator that generates games and evaluates bots.
- polytopia-weekly at `../weekly`: search baselines to compare against.

Keep datasets, checkpoints and run logs out of git (see `.gitignore`), and keep them on `/kvm` (the root disk is
small).
