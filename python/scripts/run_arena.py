"""Play a trained model against the retail AI and write polyai-arena JSONL.

    nice -n 10 python scripts/run_arena.py --model RUN_DIR_OR_CKPT --opponent retail:easy,retail:normal \\
        --seeds 50 --out /kvm/polytopia-ai-data/arena/NAME.jsonl

Same as `python -m polyai.arena` (run from ai/python); see `polyai/arena.py` for the options and the output format.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyai.arena import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
