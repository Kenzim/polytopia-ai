"""Write synthetic shards (same shapes as real ones) for pipeline tests and benchmarks.

    python scripts/make_synthetic.py OUT_DIR --size 16 --shards 2 --n 2048 --seed 0 --max-actions 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyai.data import write_synthetic_shards  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=40)
    a = ap.parse_args()
    for p in write_synthetic_shards(a.out_dir, a.shards, a.n, a.size, a.seed, max_actions=a.max_actions):
        print(p)


if __name__ == "__main__":
    main()
