# Engine profile (2026-11-01)

Plan item `engine-profile`. Retail Hard vs retail Crazy, Might 1v1 arena suite, 60 seeds x 2 = 120 games on 4 threads
(14.8 s, about 0.5 s of CPU per game), recorded with `samply` (installed under `/kvm/tmp/tools/bin`) from the release
build (`debug = 1`). Summarised with `tools/profile_top.py`. Profile: `/kvm/polytopia-ai-data/diag/profile/`.

Inclusive time (share of all samples):

| Where | Share |
|---|---|
| retail AI picking a move (`ai_get_move` and its set-up) | 86% |
| of which `analyze_situation` | 52% |
| of which `rate_position_unit` | 36% |
| candidate generation (`tile_command_candidates`) | 14% |
| `legal_commands` (all callers) | 13% |
| `ensure_opinions` (8.7% self) | 12% |
| `apply_unchecked` | 9% |
| route updates (`update_routes`/`route_between`) | 7% |
| path finding (`get_path`, 6.1% self) | 8% |
| `Vec` growth (`grow_one`) | 7%; malloc/free/realloc together about 8% self |

Top self time: `ensure_opinions` 8.7%, `rate_position_unit` 6.8%, `get_path` 6.1%, `GameState::unit_index_at` 4.6%
(linear scan), `can_build` 4.5%.

Does it matter for training? Not now. The self-play loop is bound by GPU inference and Python (docs/rl.md), and the
engine proper (legal moves, apply, encode) is about 0.05 ms per decision. The retail AI matters only as far as retail
seats stay in the pool (`opponent-mix`).

Possible engine speed-ups, recorded for the project chat, none requested yet:

1. `unit_index_at`: a tile-to-unit index instead of a linear scan (4.6% self, called everywhere).
2. Allocation: reuse buffers in `legal_commands`/`unit_commands`/path finding (`Vec` growth 7%, allocator about 8%).
3. `analyze_situation` and `ensure_opinions` are recomputed per AI move within a turn; caching per turn where the
   inputs do not change would cut most of the retail AI's cost, but must not change its decisions (regress tests).

Expected gain from 1 and 2: 10 to 15% of retail AI time; from 3 possibly 2x on the retail AI. Worth asking only if
retail seats stay a large part of self-play after `opponent-mix`.
