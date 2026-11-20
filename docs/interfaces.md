# polyai interfaces: the policy and live-game records (v1, 2026-10-21)

The AI side (`ai/`) owns both definitions. Anything that wants moves from a trained model (the Python arena, self-play,
the headless client's runner in `client/`) goes through the one policy interface below, and every game the model
plays live comes back to training through the one record path below. Changes to either are made here first.

## 1. The policy (one interface, three callers)

### 1.1 In Python: `polyai.policy`

```python
class Policy(Protocol):
    policy_id: str                       # "<run>@<step>", e.g. "bc4_b10c128_v2@100000"; stored with every move
    def choose(self, obs: dict, *, temperature: float | None = None) -> Choices: ...

@dataclass
class Choices:                           # one entry per position in obs
    index: np.ndarray                    # int32[N], index into that position's legal list
    logp: np.ndarray                     # float32[N], log-probability of `index` under softmax(logits / T); T = 1 when greedy
    value: np.ndarray                    # float32[N], P(the mover wins) from the value head
    top: list[list[tuple[int, float]]]   # per position, the 3 best (index, probability), best first
```

`obs` is the `polyai_engine` observation group (one map size: `tiles`, `globals`, `act_off`, `actions`, ...), the
same dict `VecEnv.observe()` and `VecEnv.step()` return. `temperature=None` is the greedy move; a positive value
samples from softmax(logits / T). A greedy move reports the model's own probability of it (T = 1), not 1; `top`
probabilities are from the same distribution. `value` is slot 0 of the winner softmax. The evaluation shown to people is `polyai.model.eval_pawns(value)` (0.0 at 50%).

Callers:
- `polyai.arena` (model against retail on the fixed 1v1 suite) plays through `Policy`.
- `polyai.rl` (self-play) plays through `Policy` and keeps `logp` for PPO.
- `polyai.serve` (section 1.2) wraps one `Policy` for the client.

So an arena result for `policy_id` X is a measurement of exactly what the client plays when it is serving X.

### 1.2 For the client: `polyai.serve` on a Unix socket

The client is Rust and pins its own engine revision, so it does not link the model or the AI crates. It sends the
server's state blobs; the service rebuilds the game with the AI side's engine, encodes it with `polyai-core` (the same
code as training), runs the policy and answers with the move in the game's own wire format.

**Decision: a separate service, not in the runner's process.** Reasons:
- The model is PyTorch on the shared GPU. One service process holds one copy of the weights and batches requests from
  every game both accounts are in; in-process inference would mean libtorch in the runner and a copy per runner.
- New checkpoints are picked up by restarting or reloading the service (the `policy_id` in every answer says which
  one played); the runner is never rebuilt for a model change.
- The client's pinned engine and the AI side's engine can differ without a link-time clash; a state the AI side
  cannot rebuild comes back as `ok: false` and the runner falls back to the retail AI.
- The cost is one local socket round trip per command, small next to model inference and server latency
  (`think_ms` is measured by the service and reported).

- Socket: `/kvm/tmp/polyai/policy.sock` (the `--socket` option changes it), mode 0600.
- Framing: each message is a 4-byte little-endian length, then that many bytes of UTF-8 JSON. Byte fields are base64.
- One request, one response, on a connection that may be kept open. The service batches requests that arrive together.

Request (`op: "choose"`):

| key | type | meaning |
|---|---|---|
| `v` | int | protocol version, 1 |
| `op` | str | `"choose"` (also `"info"`: returns `policy_id`, encoding and engine versions, no move) |
| `game_id` | str | the server's game id (for logs and the service's per-game cache only) |
| `seat` | int | engine player id of our account in this game |
| `initial`, `current` | base64 | the server's initial and current state blobs, as `SaveFile::parse_server` reads them |
| `per_turn` | int | commands we have already sent this turn (the policy ends the turn at 400, as the retail AI does) |
| `temperature` | float or null | null for greedy (the default for live games) |

Response:

| key | type | meaning |
|---|---|---|
| `v` | int | 1 |
| `ok` | bool | false: no move; the client falls back (below) and records why |
| `error` | str | when `ok` is false |
| `raw` | base64 | the chosen command as `RawCmd::to_bytes` (no version prefix; the client adds it as for the retail AI) |
| `desc` | str | `format!("{cmd:?}")`, for logs |
| `index`, `n_legal` | int | the pick and the length of the legal list it came from |
| `logp` | float | log-probability of the pick |
| `value` | float | P(we win), and `eval` = `eval_pawns(value)` |
| `top` | list | the 3 best moves: `[{"index", "desc", "p"}]` |
| `policy_id` | str | which checkpoint answered |
| `encoding`, `engine` | int, str | `ENCODING_VERSION` and the AI side's engine commit |
| `think_ms` | float | time inside the service |
| `forced` | str | only when the service replaced the model's pick with `EndTurn` because `per_turn` reached the cap |
| `rebuild_ms`, `infer_ms` | float | parts of `think_ms`: replaying the state, and the model's batch |

Client rules:
- Map `raw` back with its own engine's `command_from_raw` and check it is legal there (a `Stay` counts as legal, as
  in `choose_move_with`). If the service is down, answers `ok: false`, times out (default 10 s), or the command does
  not map or is not legal, play the retail AI's move for this one decision and record `fallback` (section 2.1).
- A state the service cannot rebuild (engine version mismatch, unknown tribe) is answered with `ok: false` and a
  clear `error`; the client keeps playing on the retail AI and the game still counts as live data.
- Our account is identified by `seat`; the policy sees only what that seat has explored (the encoding applies fog
  from the state; the full server state never reaches the model's inputs).

### 1.3 For the website: engine lines and whole turns

The website keeps its games as engine lines (`{"position": <polytopia-position>, "steps": [{"text", "ai"}]}`, the
web's `engine-service`). Any request that takes a position may carry `line` instead of `initial` and `current`;
`seat` is then the engine player id to move. The service rebuilds the game with `polyai_engine.Game.from_line`,
playing each step by its command text (`format!("{cmd:?}")`).

- `choose` with `line` answers as above plus `text`, the chosen command's text (`raw` is null when the command has
  no retail encoding).
- `turn` (`line` or blobs, `seat`, `temperature`, `max_commands` from 1 to 400, default 400) plays the model for
  `seat` on its own copy until it plays `EndTurn`, the game is over or another player is to move. The last command is
  forced to `EndTurn` at the retail cap of 400 commands a turn (counting the line's steps) or at `max_commands`.
  A `Stay` the model chooses is played on its copy but left out of `cmds` (`stays` counts them): the engine's legal
  list, which the website plays texts against, has no `Stay`, and leaving one out changes nothing (it only stops that
  unit acting for the rest of the turn, and the model does not use the unit again).
  Answer: `cmds` (`[{"text", "p", "value"}]` in order), `policy_id`, `encoding`, `engine`, `think_ms`, `infer_ms`.
  A turn that outlasts the service's request limit (`--request-seconds`, 30 s) is answered with `ok: false`.
- The website plays the texts on its own line and finishes the turn with the retail AI when one is refused (the
  service and the website were built from different engine commits) or the service fails.

The service reads checkpoints with `torch.load(weights_only=True)`. `python -m polyai.serve --check FILE` loads a
file that way and prints one JSON line (`ok`, `policy_id`, `step`, `encoding_version`, `features`, `params`, or `ok:
false` and `error`); the website runs it on every uploaded checkpoint. `--name NAME` sets the run part of
`policy_id`.

## 2. Live-game records

### 2.1 What the client stores (raw, `client/` owns the files)

Per finished game, as today in the client's `games/` directory: `<game>_init.bin`, `<game>_cur.bin` (the server's
blobs, the archive copy) and `<game>.json` (`SavedGame`). For training, `SavedGame` goes to **format 2**, which adds:

- per command we sent: `policy` = `{policy_id, index, n_legal, logp, value, temperature, fallback}`, where `fallback`
  is null for a model move, or `"retail"` with a `fallback_reason` when the retail AI chose instead;
- per game: `our_seats` (engine player ids of all our accounts in the game, as both accounts can play the same game),
  `mode` (Domination, Perfection, ...), `server_version`, `opponents` (per seat: `"human"`, `"bot:<difficulty>"` or
  `"unknown"`), and the server's result if it gives one (`server_result`: winner ids, and resign or timeout per seat).

Commands from a format-1 file (before the policy was plugged in) are retail moves; they load with `fallback: "retail"`.

### 2.2 What training reads: shards, made by `client/tools/shardify`

Live games become ordinary shards (`encoding.md` section 6, encoding version 2) with these optional keys, which
generated shards may also carry and the loader fills with defaults when they are missing:

| key | dtype | shape | meaning (default) |
|---|---|---|---|
| `source` | int8 | [N] | 0 retail generator, 1 self-play, 2 live, 3 human replay (0) |
| `ours` | int8 | [N] | 1 when the mover is one of our accounts (0) |
| `behaviour_logp` | float32 | [N] | `logp` of the policy that chose the move; NaN for retail, humans and fallbacks (NaN) |
| `policy` | int16 | [N] | index into `policy_ids`, -1 for none (-1) |
| `policy_ids` | str | [P] | the `policy_id` strings of this shard ([]) |
| `game_key` | str | [N] | `live:<game_id>`, so a game is never split across train and validation (`gen:<seed>`) |

Labels follow design v1.1: `winner` is the elimination winner's seat. A game that ended any other way (a score win
in Perfection or at a server turn limit, a resignation or timeout with more than one player left, a game abandoned
unfinished) has `winner = -1`: its positions train the policy but not the value head. A 1v1 resignation or timeout
by the opponent counts as our win only when `server_result` says so. Positions whose legal list does not contain the
played command are skipped and counted, as `shardify` does now.

**Status of `client/tools/shardify` against this section (checked 2026-10-21):**
- Matches: encoding v2 through `polyai_core::encode`, the legal list and candidate scores from `polyai-core`, the
  shard writer taken from `polyai-gen` by path, and the labels. `Outcome::of` drops a top-score winner whenever more
  than one player is alive at the turn limit (so Perfection and server turn-limit games get `winner = -1`), and a game
  replayed without reaching its end is only written with `--unfinished`, also as `winner = -1`, `capped = 1`.
- To add on the client side: the six optional keys above (`source = 2`, `ours`, `behaviour_logp`, `policy`,
  `policy_ids`, `game_key`) from the format-2 `SavedGame`, and the 1v1 `server_result` rule (opponent resigned or
  timed out: our seat is the winner). The writer gains these keys in `polyai-gen/src/shard.rs` without new
  dependencies, so the `#[path]` include keeps working; shards without them still load with the defaults.

## 3. How live games are used in training

Self-play is the main volume; live games are a smaller, more valuable stream (real opponents, including humans).

- Value head: every live position with a winner label, at a fixed share of each batch (start at 10%, reported
  separately in the metrics so a drift between self-play and live play shows).
- Policy, our moves (`ours = 1`, `behaviour_logp` known): an off-policy PPO term with the importance ratio
  pi/behaviour truncated at 1 (V-trace style), as these moves came from an older checkpoint.
- Policy, opponents' moves from humans: imitation (cross-entropy) on the human track (design.md section 6, phase C),
  weighted by the player's strength where the server tells us.
- Live games are never evaluation data: strength is measured only in the arena on the fixed seeds.
