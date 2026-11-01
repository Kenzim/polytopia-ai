# Human game data: sources (2026-11-01)

Plan item `human-corpus`. Research rec 2: bengreff's bot went from about 10% to 94% against Crazy after fine-tuning on
7,732 human championship games.

What exists (evidence in the client and re repos):

- Replays by game id: the server returns a full `GameViewModel` (initial and current state blobs, the current blob holds
  the whole command stack, so our engine can replay it) through `api/game/spectate_game {GameId}` and
  `api/game/get_game_view_model` (`client/docs/server-api.md:341,449,454`). Error `ReplayGameStillOngoing`
  (`client/src/error.rs:36`) suggests replays open once a game ends. Not documented whether non-participants may fetch.
- Proven path in `re/`: `re/tools/share_grab.sh` and `re/tools/grab_replay.py` open a share link or game id in the real
  game and dump the blobs; 5 share-link games replayed with 0 illegal commands (`re/docs/HANDOFF.md:81`).
- Sources of game ids: weekly leaderboard `ReplayGameId` (`client/docs/models.md:1547`, top weekly replays already
  grabbed this way), tournaments (`GetTournament`, unconfirmed whether they carry game ids), share links posted by the
  community, and our own live games (`client/games/`, already shardified with `source=2`).
- No "games by player" call; our own lists only.
- The headless client deliberately refuses spectate, highscore, tournament and friend methods
  (`client/src/hub.rs:995-1014`), and the notes call terms of service a policy question for Kenzi
  (`client/docs/notes/matchmaking.md:116-128`).

Blocked on Kenzi: whether fetching other players' finished games (leaderboard and tournament replays, community share
links) is permitted. If yes, the work is: a whitelisted spectate call in the client (client manager), a list of game
ids (leaderboard, tournaments, links), replay through `Game.from_server`, `shardify` with a `source=3` human tag, then
the `human-finetune` A/B.
