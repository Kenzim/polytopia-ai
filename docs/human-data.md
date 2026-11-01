# Human game data: sources, fetch path, replay pipeline (2026-11-01)

Plan item `human-corpus`. Research rec 2: bengreff's bot went from about 10% to 94% against Crazy after fine-tuning on
7,732 human championship games. Kenzi approved fetching other players' finished games for training on 2026-11-01
(ledger decision b6d05b58), at a modest request rate.

**Status: the replay and shard half is built and tested; the fetch half is blocked on a client change and on a source of
game ids (section 3). No human games are in the corpus yet.**

## 1. What the server gives

- `api/game/get_game_view_model {GameId}` and `api/game/spectate_game {GameId}` return a `GameViewModel`: the initial and
  current state blobs (the current blob holds the whole command stack) plus `GameSettingsJson` and `GameContext`
  (`client/docs/server-api.md:341,449,454`, `client/docs/models.md:455`).
- Non-participants can fetch finished games: bengreff's public `py/fetch_replays.py` does exactly this with a logged-in
  client's token, trying `get_game_view_model` first and `spectate_game` if that fails (error 1005 "can't load game"
  ends the attempt), and reports 27,548 ids tried, 7,822 downloaded, 19,726 refused, mostly older games that had
  expired. Expect most games older than a few months to be gone; fetch newest first. (Read from their public repo; not
  yet confirmed by a call of ours.)
- The blobs replay in our engine: `SaveFile::parse_server` reads them, and `re/` replayed 5 share-link games with 0
  illegal commands (`re/docs/HANDOFF.md:81`).

## 2. Where game ids come from

| source | ids | status |
|---|---|---|
| Weekly leaderboard `ReplayGameId` (`client/docs/models.md:1547`, via `api/game/get_weekly_challenge_data {Date, HighscoreEntryCount}`) | top weekly scores, one game per player per week | needs a client call (section 3). These are single-player weekly challenges against the AI, a fixed map and a turn limit, not player-versus-player games. |
| Challengermode tournaments (bengreff's `py/cm_replays.py`) | every match played through the game's Challengermode integration reports statistics including `ReplayLink` (the share link), mode, map size and tribes; a bracket walk gives all of a tournament's games | needs a Challengermode API application's refresh key, which we do not have. This is where their 7,732 games came from. The game server's own `TournamentViewModel` and member models carry no game ids. |
| Share links posted by the community (`share.polytopia.io/g/<id>`, also in YouTube descriptions) | a handful | manual. One is on disk: `ac49fec4` in `re/share/`. |
| Our own live games | `client/games/` | already shardified with `source=2`, not repeated here |

There is no "games by player" call. `/kvm/polytopia-ai-data/human/ids.jsonl` holds one JSON line per known id
(`id`, `source`, `where`, `fetched`); today it has the one share-link game.

## 3. The fetch path and why it stops here

Two routes exist.

**A. The retail client under gdb (`re/tools/share_grab.sh`, `re/tools/grab_replay.py`).** It starts the real game under
gdb and opens `steam://run/874390//opengame?id=<id>`; the client downloads the game and gdb copies the two blobs out.
Slow (one game at a time through the game window, with waits of tens of seconds each in the script), needs a display,
and I did not run it: the one retail client on this host is already running under the client team's `capture_gdb.py`
on Kenzi's session, and Steam does not start a second instance of the same app on one account. It also gives no ids, since the leaderboard rows are only logged when
the leaderboard screen is opened. Not used.

**B. The headless client over REST (cheapest, recommended).** The ids are plain REST posts with the token the client
already gets from `login_steam`; no hub, no subscription. `client/` has `RestClient::get_game_view_model`, but it
takes an `AllowedGame`, which only our own or matchmaking games can become (`client/src/whitelist.rs:123,153`,
`client/src/rest.rs:131`), and the hub list deliberately bans `Spectat`, `Highscore`, `Challenge` and `Tournament`
(`client/src/hub.rs:962-1015`). I do not edit `client/` and a token must not leave it, so the fetch has to live there.

**Change requested of the client manager** (REST only; the hub list and its test stay as they are):

1. `models.rs`: add `SpectateGameBindingModel { GameId }` (`PolytopiaBackendBase.Game.SpectateGameBindingModel`) and the
   `WeeklyChallengeBindingModel { Date, HighscoreEntryCount }` request plus the response models down to
   `WeeklyChallengeHighscoreViewModel.ReplayGameId` (`client/docs/models.md:1516-1600`).
2. `whitelist.rs`: a `ReplayTarget(Uuid)` type that only the read calls below accept and that cannot be turned into an
   `AllowedGame` (so a replay id can never be subscribed to or sent a command).
3. `rest.rs`: `fetch_finished_game(&ReplayTarget) -> GameViewModel` posting `api/game/get_game_view_model {GameId,
   LobbyId: default}` and, if that fails with anything but code 1005, `api/game/spectate_game {GameId}`; and
   `weekly_replay_ids(date, count)` posting `api/game/get_weekly_challenge_data`.
4. A subcommand, e.g. `polyclient fetch-replays --ids FILE --out DIR [--delay 4]` (and `polyclient weekly-ids --date D
   --count N >> ids`): one request at a time with a 3 to 4 second delay (jittered); ids file is one UUID per line or
   JSON lines with an `id` field; for each id not yet in DIR write `<id>_init.bin` (`InitialGameStateData`),
   `<id>_cur.bin` (`CurrentGameStateData`) and `<id>.json` (`GameSettingsJson`, `State`, `GameContext`, no names);
   failures go to `DIR/errors.tsv` and are not retried; **stop at once on HTTP 401, 403 or 429** and print which; ids
   newest first when the file has a date. Never print the token.
5. Output goes to `/kvm/polytopia-ai-data/human/raw/` and the layout is exactly what `re/tools/share_grab.sh` writes
   (`<id>_init.bin`, `<id>_cur.bin`), so section 4 runs unchanged.

**Ids still need a source** (Kenzi's call): a Challengermode API application's refresh key
(`py/cm_replays.py` is a public reference for the GraphQL walk; I have not run it and would need the key), or lists of
share links from the community. The weekly leaderboard alone gives only solo-versus-AI games.

## 4. Replay and shards (built, tested)

`crates/polyai-human` (binary `polyai-human`, its own crate; `shard.rs` and `npz.rs` of `polyai-gen` are included by
path, as `client/tools/shardify` does, so the shard format cannot drift). For each `<id>_init.bin` + `<id>_cur.bin` in
the raw directory it:

- reads the blobs with `SaveFile::parse_server`, builds the engine, and records mode, map size, seat count, human seat
  count, tribes, command counts, outcome in one JSON line of `meta.jsonl`;
- skips, with a status: `unsupported_tribe` (any tribe outside `polyai_core::TRIBES`, the 12 normal tribes plus
  Cymanti), `too_many_players` (the encoding holds 8 players), `skipped` (blobs unreadable, no human seat, engine
  panic), `dirty` (any command the engine cannot map or finds outside its legal list: the engine state may have
  drifted, so the whole game stays out of the shards but is counted);
- for clean games writes one position before every command of every human seat (a seat not marked `auto_play`), all
  seats, encoded exactly like `polyai-gen` positions, with the legal list and the engine AI's candidate scores;
- labels: winner only by elimination (or capitals for Might), else `winner = -1` and `capped = 1`
  (`Outcome::of`; points never decide); `final_scores` logged;
- writes `source=3`, `ours=0`, `behaviour_logp=NaN`, no policy, `difficulty=-1`, `game_key=human:<id>`, `seed` a hash
  of the id. Kind 35 commands (no engine command, no effect on any verified replay) are counted as `no_command`, not as
  unmapped.

`tools/human_report.py` loads every shard with `polyai.data.Shard` and checks `chosen` inside its legal list, source 3,
the `human:` key prefix, winner/capped agreement and that the shard games are the clean games of `meta.jsonl`, then
prints the counts by status, mode, size, players, humans and tribe.

### Commands

```
cd /kvm/polytopia/ai
export TMPDIR=/kvm/tmp/tmp CUDA_VISIBLE_DEVICES=
H=/kvm/polytopia-ai-data/human            # raw/ (blobs), ids.jsonl, meta.jsonl, shards/
nice -n 10 cargo build --release -p polyai-human
nice -n 10 ./target/release/polyai-human --raw $H/raw --out $H/shards --meta $H/meta.jsonl   # [--keep K] [--shard-size P] [--limit N]
nice -n 10 /kvm/polytopia/.venv/bin/python tools/human_report.py --dir $H
nice -n 10 cargo test --release -p polyai-human
```

It rewrites `human-*.npz` in the output directory each run (stale shards are removed first), so rerunning after more
games arrive is safe; all rows are kept by default (`--keep 4` keeps one in four like the generator).

### Check on local blobs (2026-11-01)

No human games could be fetched, so the pipeline was run on every blob pair in `re/share` and `re/data/regress` (our own
and Kenzi's games, not human-corpus data; the output stayed in `/kvm/tmp/human-test`, not in the corpus): 19 games, 15
clean (0 illegal, 0 unmapped), 3 skipped for more than 8 players (including the share-link game `ac49fec4`: 10 players,
18x18 Might), 1 for an unsupported tribe (tribe 5). 4,054 positions in two shards (14x14, 16x16);
`tools/human_report.py` passed all its checks. The Might games there are mid-game saves, so none was finished and
every winner is -1; finished games are untested until real ones arrive.

## 5. Open

- The client change in section 3 and a source of ids (Challengermode key or share links): both need the client manager
  or Kenzi.
- Games with more than 8 players (the share-link game has 10) cannot be encoded; the Championship format (1v1) is
  unaffected.
- Whether `get_game_view_model` works for non-participants of recent games is taken from bengreff's code, not yet seen
  from ours. The first fetch run should be a handful of ids and read the error codes before going wider.
- Human-versus-human games in the Championship format are the prize (1v1 Might on Lakes Small, 14x14); the replayed
  setup mix should be read from `meta.jsonl` once a first batch exists, before choosing the fine-tune mix
  (`human-finetune`).
