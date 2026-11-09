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

**Ids source: Challengermode** (done, 2026-11-01; see "Collecting ids" below). `tools/cm_ids.py` walks the public
tournament API and has produced 1,893 game ids; fetching them is the client change above. The weekly leaderboard alone
gives only solo-versus-AI games.

### Collecting ids (`tools/cm_ids.py`)

```
cd /kvm/polytopia/ai
nice /kvm/polytopia/.venv/bin/python tools/cm_ids.py            # [--limit N] [--list-only] [--min-interval S]
/kvm/polytopia/.venv/bin/python tools/cm_ids.py --summary       # counts from the files, no network
/kvm/polytopia/.venv/bin/python -m pytest -q tools/test_cm_ids.py   # no network
```

Needs the refresh key in `~/.config/polytopia/challengermode_refresh_key` (read from the file only, never printed or
logged). One request at a time, at least 3.5 s apart (the app's limit is 20 a minute); 429 and 5xx back off 60 s then
double, four tries, then that tournament goes to `cm_errors.tsv` and is retried next run; 401 and 403 stop the run.
A full first walk of 176 tournaments took 12 minutes (182 requests); a rerun with nothing new takes 6 requests.

Files in `/kvm/polytopia-ai-data/human/`:

- `cm_ids.tsv`: one row per game, columns `game_id tournament_id match_id mode size teams turns tribes started
  tournament_name`. `mode` is the name (`Might`), `size` the side in tiles (11 Tiny, 14 Small, 16 Normal), `teams` the
  number of teams, `tribes` the competitors' tribe types, comma separated (empty for 58 games without competitor
  statistics), `started` the tournament's start date.
- `cm_walked.tsv`: one row per walked tournament with its settings (GameMode, MapType, MapSize, the raw settings JSON);
  a tournament listed here is not walked again, so reruns are incremental. `cm_spaces.tsv` holds the organiser spaces
  seen; `cm_errors.tsv` failures; `cm_tournaments.txt` is an optional input, one tournament id per line.

How far back the API goes: the lists are not paged and have no date window, each returns only the 50 most recent
completed tournaments. The tool lists the game's 50 most recent and then the 50 most recent of every organiser space
it meets (4 spaces were found: Grid Haven, PolyChampions, Duelists, Polytopia Real Tournaments), which reached 176
tournaments back to July 2025. Anything older, or from an organiser not yet seen, has to come in through
`cm_tournaments.txt`. Rerun at least every ten days (a weekly timer does this, see below): October 2026 alone holds 635 games, so the 50-most-recent window
moves quickly. Match series inside a tournament are paged fully (no cap).

Counts (2026-11-01): 176 tournaments walked (174 with games), 1,893 games, none failed.

- Mode: Might 1,893 (every game; the tournaments are all Might).
- Teams: 2 for every game, so all are 1v1.
- Map size: 14 (Small) 1,619; 11 (Tiny) 209; 16 (Normal) 65.
- Year: 2025 157, 2026 1,736 (rising: 2026-08 238, 2026-09 329, 2026-10 635 games).
- 1v1 Might on Tiny or Small: 1,828 games in 169 tournaments (2025 105, 2026 1,723). Tournament map settings are mostly
  Lakes Small (68 tournaments), Drylands Small (52), Drylands Tiny (14).
- Tribes recorded for 1,835 games.

**Scheduled reruns** (since 2026-11-02). A systemd timer on the dev host runs the walk weekly, so the
50-most-recent window never moves past games we have not listed:

- Units: `polyai-cm-ids.timer` and `polyai-cm-ids.service`, kept in `deploy/systemd/` and installed in
  `/etc/systemd/system/`. The timer fires on Mondays at 04:30 (up to 30 minutes random delay); `Persistent=true`
  runs a missed week at the next boot.
- The service runs `tools/cm_ids_cron.sh` as root under `nice`, which calls `cm_ids.py --dir
  /kvm/polytopia-ai-data/human`. The key is read only from `/root/.config/polytopia/challengermode_refresh_key`;
  it is not in the units, the script or any argument.
- Full output per run: `/kvm/polytopia-ai-data/human/cm_logs/run-<UTC stamp>.log`. One summary line per run is
  appended to `/kvm/polytopia-ai-data/human/cm_runs.log`: exit code, tournaments walked, new ids, total ids, open
  errors, whether the API refused access (`stopped`), and `key_in_log`. As a safety net the script greps each log
  for the key (without printing it) and empties the log and exits 3 if it is found.
- Check it: `systemctl list-timers polyai-cm-ids.timer`, `tail /kvm/polytopia-ai-data/human/cm_runs.log`,
  `journalctl -u polyai-cm-ids.service`. Run once by hand: `systemctl start polyai-cm-ids.service`.
- After changing the units in the repo: copy them to `/etc/systemd/system/` and `systemctl daemon-reload`.
- First scheduled-style run (2026-11-02 13:40): 1 tournament walked, 4 new ids, 1,897 in total, 0 errors.

**Getting a Challengermode refresh key** (checked 2026-11-01 on challengermode.com/developers):
1. Create a free Challengermode account (`/account/register`).
2. In the developer dashboard (`/developers/applications`) choose Create application.
3. In the application's credentials tab, generate a refresh key. It is shown once.
4. Store it in `~/.config/polytopia/challengermode_refresh_key` on the dev host (mode 0600, never in git).

The key is exchanged at `POST https://publicapi.challengermode.com/mk1/v1/auth/access_keys` for a bot token that
lasts 20 minutes; bot tokens can read public tournament data. The developer pricing page says a free account and
application work but are rate-limited, and applications must be approved by Challengermode before launch; it does not
say whether read-only public-data use needs approval. The paid plans (from EUR 149 a month) are for game integrations
and are not needed here.

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

### First real corpus (2026-11-09)

1,479 Challengermode tournament games fetched by the client into `human/raw`, converted with engine `a28cd84`
(`polyai-human --raw raw --out shards --meta meta_cm.jsonl`, single thread, 3 min 48 s, under `nice`):

| Outcome | Games | Why |
|---|---|---|
| clean | 1,256 | 0 illegal, 0 unmapped; **413,504 positions** in 10 shards (1,154 Small, 90 Tiny, 12 Normal) |
| unsupported tribe | 138 | tribe 5 (73), Polaris 16 (38), tribe 3 (27): outside our 13-tribe encoding |
| dirty (engine mismatch) | 72 | the engine rejects at least one recorded command: 16 games with 1, 21 with 2 to 5, 35 with more |
| skipped | 12 | save versions 112 and 119; the parser reads 121 to 124 |
| too many players | 1 | 10 players |

Only 209 of the clean games end in an elimination our engine sees (`over`); the rest end by resignation, timeout or the
server, so they train the policy only (`winner = -1`). All clean games are 1v1 Might. The dirty games are worth sending to
the engine manager as parity cases; Polaris games need an encoding change (14th tribe) to be used.

Training (`tools/human_round.sh`): `hbc1_rl3_human` is rl3 u480 behaviour-cloned on the human shards (shard 8 held
out, lr 5e-5, 6,000 steps, KL term off since human rows have no candidate scores): held-out human top-1 rose from
0.261 to 0.395 (top-3 0.431 to 0.605). On its own it is weaker than rl3 (39.5% head to head over 200 games, 55.0% vs
retail Crazy against rl3's 76.0%), so it is the start of a 300-update self-play round with the autoloop recipe
(KL to itself), gated like the loop.

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
