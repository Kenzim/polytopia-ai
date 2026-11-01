#!/usr/bin/env python3
"""Collect the game ids (share links) of Polytopia tournament games from Challengermode's public API.

    /kvm/polytopia/.venv/bin/python ai/tools/cm_ids.py [--dir DIR] [--limit N] [--list-only] [--summary] [--min-interval 3.5]

Writes, under DIR (default /kvm/polytopia-ai-data/human):

    cm_ids.tsv       one row per game: game_id tournament_id match_id mode size teams turns tribes started tournament_name
    cm_walked.tsv    one row per walked tournament, with its settings (GameMode, MapType, MapSize, ...) so that games
                     can be filtered later; a tournament in this file is never walked again
    cm_spaces.tsv    the organiser spaces seen so far (the API lists only the 50 most recent tournaments per query,
                     so every known space is listed again on each run)
    cm_errors.tsv    tournaments that failed after retries (not in cm_walked.tsv, so the next run tries them again)
    cm_tournaments.txt (optional input) extra tournament ids, one per line, text after the id ignored

Tournament sources: the game's 50 most recent completed tournaments, the 50 most recent per known organiser space
(spaces are learnt from every tournament seen), and cm_tournaments.txt. The API has no paging or date window for these
lists, so older tournaments only arrive through cm_tournaments.txt; rerun at least every ten days or so to keep up.

Every format is walked (elimination, Swiss, group stages); the mode, map size and team count of each game come from the
match statistics (GameMode is a name such as "Might", MapSize the side in tiles such as 14).

Auth: the refresh key in ~/.config/polytopia/challengermode_refresh_key (never printed or logged) is exchanged for an
access key that lasts about 20 minutes. The app's limit is 20 requests a minute: one request at a time, at least 3.5 s
apart; HTTP 429 and 5xx back off for 60 s, doubling, and give up on that tournament after four tries. HTTP 401 and 403
stop the run.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://publicapi.challengermode.com"
KEY_FILE = os.path.expanduser("~/.config/polytopia/challengermode_refresh_key")
DEFAULT_DIR = "/kvm/polytopia-ai-data/human"
USER_AGENT = "Mozilla/5.0 (polytopia-research)"  # Cloudflare refuses urllib's default
GAME_SLUG = "polytopia"
MIN_INTERVAL = 3.5
PAGE = 200
TOKEN_LIFE = 15 * 60  # access keys last about 20 minutes
RETRIES = 4
BACKOFF = 60.0

IDS_HEADER = ["game_id", "tournament_id", "match_id", "mode", "size", "teams", "turns", "tribes", "started", "tournament_name"]
WALKED_HEADER = ["tournament_id", "state", "game_mode", "map_type", "map_size", "series", "games", "started", "partial_errors", "settings", "name"]
SPACES_HEADER = ["space_id", "name"]
ERRORS_HEADER = ["tournament_id", "when", "error"]

SERIES = """matchSeriesPage(first: $first, after: $after) { pageInfo { hasNextPage endCursor } nodes { id state
  matches(includeFailed: true) { id state statistics { gameSessionStatistics {
    globalStatistics(first: 50) { nodes { name serializedValue } }
    competitorStatistics(first: 50) { nodes { name serializedValue } } } } } } }"""
TOURNAMENT = """query($id: UUID!, $first: Int, $after: String, $from: Int, $to: Int) {
  tournament(tournamentId: $id) { id name state hosts { spaces { id name } }
    schedule { startedAt endedAt } settings { gameSessionSettings }
    stages(fromIndex: $from, toIndex: $to) { index format
      ... on TournamentEliminationStage { brackets { rounds { roundNumber %s } } }
      ... on TournamentSwissStage { rounds { roundNumber %s } }
      ... on TournamentGroupStage { groups { number %s } } } } }""" % (SERIES, SERIES, SERIES)
LIST_FIELDS = "id name state schedule { startedAt } hosts { spaces { id name } }"
LIST_GAME = "query($i: TournamentsForGameInput!) { tournamentsForGame(input: $i) { %s } }" % LIST_FIELDS
LIST_SPACE = "query($i: TournamentsForSpaceInput!) { tournamentsForSpace(input: $i) { %s } }" % LIST_FIELDS
LIST_FILTER = {"completedTournamentSelector": {"tournamentsAfter": "2018-01-01T00:00:00Z"}}
FINISHED = ("COMPLETED", "CANCELLED")


class AccessDenied(Exception):
    """HTTP 401 or 403: stop the run."""


class QueryFailed(Exception):
    """A request that failed after its retries, or a GraphQL error without data."""


# ---------------------------------------------------------------- parsing (no network)


def containers(stage):
    """Yield ((kind, i, j), container) for every match-series container of a stage.

    Elimination stages hold brackets of rounds (key ("bracket", bracket, round)), Swiss stages rounds, group stages
    groups. The key addresses the container again in a later query of the same stage."""
    for i, bracket in enumerate(stage.get("brackets") or []):
        for j, rnd in enumerate(bracket.get("rounds") or []):
            yield ("bracket", i, j), rnd
    for j, rnd in enumerate(stage.get("rounds") or []):
        yield ("round", 0, j), rnd
    for j, grp in enumerate(stage.get("groups") or []):
        yield ("group", 0, j), grp


def statistics(match):
    """(global stats dict with any "General." prefix removed, list of competitor TribeType values) of a match."""
    st = ((match.get("statistics") or {}).get("gameSessionStatistics")) or {}
    glob = {}
    for n in ((st.get("globalStatistics") or {}).get("nodes")) or []:
        glob[str(n["name"]).removeprefix("General.")] = n.get("serializedValue")
    tribes = []
    for n in ((st.get("competitorStatistics") or {}).get("nodes")) or []:
        if str(n["name"]).endswith("TribeType"):
            tribes.append(str(n.get("serializedValue")))
    return glob, tribes


def game_id(link):
    """The id at the end of a share link such as https://share.polytopia.io/g/<id>."""
    return str(link).strip().rstrip("/").split("/")[-1]


def all_series(tour):
    """Every match series of a tournament, over all stages and containers."""
    for stage in tour.get("stages") or []:
        for _, cont in containers(stage):
            yield from ((cont.get("matchSeriesPage") or {}).get("nodes")) or []


def game_rows(tour):
    """One dict per match of a tournament that reported a replay link (a game id seen twice is kept once)."""
    started = ((tour.get("schedule") or {}).get("startedAt") or "")[:10]
    seen = set()
    rows = []
    for series in all_series(tour):
        for m in series.get("matches") or []:
            glob, tribes = statistics(m)
            link = glob.get("ReplayLink")
            gid = game_id(link) if link else ""
            if not gid or gid in seen:
                continue
            seen.add(gid)
            rows.append({
                "game_id": gid,
                "tournament_id": tour["id"],
                "match_id": m["id"],
                "mode": glob.get("GameMode"),
                "size": glob.get("MapSize"),
                "teams": glob.get("NumberOfTeams"),
                "turns": glob.get("CurrentTurnNumber"),
                "tribes": ",".join(tribes),
                "started": started,
                "tournament_name": tour.get("name") or "",
            })
    return rows


def tsv_cell(v):
    return "" if v is None else str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def tsv_line(values):
    return "\t".join(tsv_cell(v) for v in values) + "\n"


# ---------------------------------------------------------------- walking a tournament


def walk_tournament(query, tid, page=PAGE):
    """Fetch a whole tournament, following every match-series container that has a next page.

    query(text, variables) returns the GraphQL data dict. Returns the tournament dict with all pages merged into each
    container's matchSeriesPage nodes."""
    tour = query(TOURNAMENT, {"id": tid, "first": page})["tournament"]
    if tour is None:
        raise QueryFailed("tournament not found")
    for stage in tour.get("stages") or []:
        for key, cont in list(containers(stage)):
            if cont.get("matchSeriesPage"):
                fill_container(query, tid, stage["index"], key, cont["matchSeriesPage"], page)
    return tour


def next_page(query, tid, index, key, cursor, page):
    """The matchSeriesPage after `cursor` of one container, found again by its stage index and key."""
    v = {"id": tid, "first": page, "after": cursor, "from": index, "to": index}
    for stage in query(TOURNAMENT, v)["tournament"].get("stages") or []:
        if stage["index"] == index:
            found = dict(containers(stage)).get(key)
            if found and found.get("matchSeriesPage"):
                return found["matchSeriesPage"]
    raise QueryFailed(f"could not page stage {index} {key}")


def fill_container(query, tid, index, key, pg, page):
    """Append the following pages to `pg` (a matchSeriesPage dict) until hasNextPage is false."""
    while pg["pageInfo"]["hasNextPage"]:
        cursor = pg["pageInfo"]["endCursor"]
        if not cursor:
            raise QueryFailed("page without a cursor")
        new = next_page(query, tid, index, key, cursor, page)
        pg["nodes"] = list(pg["nodes"]) + list(new["nodes"])
        pg["pageInfo"] = new["pageInfo"]


# ---------------------------------------------------------------- network


class Client:
    """One request at a time, at least `interval` seconds apart, with backoff on 429 and 5xx."""

    def __init__(self, interval=MIN_INTERVAL, key_file=KEY_FILE, log=print):
        self.key = open(key_file).read().strip()
        self.interval = max(interval, MIN_INTERVAL)
        self.log = log
        self.token = None
        self.expires = 0.0
        self.last = 0.0
        self.requests = 0
        self.partial = 0  # GraphQL errors tolerated in the last query

    def _post(self, url, body, auth):
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
        if auth:
            headers["Authorization"] = "Bearer " + self.token
        delay = BACKOFF
        for attempt in range(RETRIES):
            wait = self.last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise AccessDenied(f"HTTP {e.code} from {url.split('/', 3)[-1]}")
                if e.code == 429 or e.code >= 500:
                    problem = f"HTTP {e.code}"
                else:
                    raise QueryFailed(f"HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as e:
                problem = f"{type(e).__name__}"
            if attempt == RETRIES - 1:
                raise QueryFailed(f"{problem} after {RETRIES} tries")
            self.log(f"  {problem}; waiting {delay:.0f} s")
            time.sleep(delay)
            delay *= 2
        raise QueryFailed("unreachable")

    def _login(self):
        r = self._post(API + "/mk1/v1/auth/access_keys", {"refreshKey": self.key}, auth=False)
        self.token = r["value"]
        self.expires = time.monotonic() + TOKEN_LIFE

    def query(self, text, variables=None):
        """GraphQL data. Errors alongside data (Challengermode answers 404 for a match without competitor
        statistics) are tolerated and counted in self.partial; errors without data raise QueryFailed."""
        if self.token is None or time.monotonic() > self.expires:
            self._login()
        r = self._post(API + "/graphql", {"query": text, "variables": variables or {}}, auth=True)
        errors = r.get("errors") or []
        self.partial = len(errors)
        for e in errors:
            code = (e.get("extensions") or {}).get("errorCode")
            if code in (401, 403):
                raise AccessDenied(f"GraphQL error {code}")
        data = r.get("data")
        if not data or all(v is None for v in data.values()):
            msg = errors[0].get("message", "no data") if errors else "no data"
            raise QueryFailed(f"GraphQL: {msg}")
        return data


def list_game(cm):
    i = {"gameSlug": GAME_SLUG, "includeUnlisted": True, "includeInviteOnly": True, "tournamentFilter": LIST_FILTER}
    return cm.query(LIST_GAME, {"i": i})["tournamentsForGame"] or []


def list_space(cm, space_id):
    i = {"spaceId": space_id, "includeUnlisted": True, "includeInviteOnly": True, "tournamentFilter": LIST_FILTER}
    return cm.query(LIST_SPACE, {"i": i})["tournamentsForSpace"] or []


# ---------------------------------------------------------------- files


def read_table(path):
    """Rows of a TSV without its header line, as lists of cells; [] when the file does not exist."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.rstrip("\n").split("\t") for line in list(f)[1:] if line.strip()]


def open_table(path, header):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a")
    if new:
        f.write("\t".join(header) + "\n")
        f.flush()
    return f


def read_id_list(path):
    ids = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line.split()[0])
    return ids


def walked_line(tour, rows, partial):
    s = (tour.get("settings") or {}).get("gameSessionSettings") or {}
    return tsv_line([
        tour["id"], tour.get("state"), s.get("GameMode"), s.get("MapType"), s.get("MapSize"), sum(1 for _ in all_series(tour)),
        len(rows), ((tour.get("schedule") or {}).get("startedAt") or "")[:10], partial,
        json.dumps(s, sort_keys=True, separators=(",", ":")), tour.get("name") or "",
    ])


# ---------------------------------------------------------------- summary

TINY_SMALL = ("11", "14")  # side in tiles of the Tiny and Small maps


def summary(d):
    """Text report of cm_ids.tsv and cm_walked.tsv: counts by mode, size, teams, year and tribe-less games."""
    from collections import Counter

    rows = [dict(zip(IDS_HEADER, r)) for r in read_table(os.path.join(d, "cm_ids.tsv"))]
    walked = [dict(zip(WALKED_HEADER, r)) for r in read_table(os.path.join(d, "cm_walked.tsv"))]
    lines = [f"{len(walked)} tournaments walked, {len(rows)} games"]
    for name, key in (("mode", lambda r: r["mode"]), ("map size", lambda r: r["size"]), ("teams", lambda r: r["teams"]),
                      ("year", lambda r: r["started"][:4])):
        c = Counter(key(r) or "?" for r in rows)
        lines.append(f"by {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))))
    lines.append("by game mode x teams x size (top 12): " + ", ".join(
        f"{m}/{t}/{z} {n}" for (m, t, z), n in Counter((r["mode"], r["teams"], r["size"]) for r in rows).most_common(12)))
    one = [r for r in rows if r["mode"] == "Might" and r["teams"] == "2" and r["size"] in TINY_SMALL]
    lines.append(f"Might, 2 teams, Tiny or Small: {len(one)} games in {len({r['tournament_id'] for r in one})} tournaments")
    lines.append("  by year: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["started"][:4] for r in one).items())))
    lines.append(f"games with tribes recorded: {sum(1 for r in rows if r['tribes'])}")
    c = Counter((w["game_mode"], w["map_type"], w["map_size"]) for w in walked)
    lines.append("tournament settings (GameMode, MapType, MapSize): " + ", ".join(
        f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:10]))
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def run(args, cm, log=print):
    d = args.dir
    os.makedirs(d, exist_ok=True)
    ids_path, walked_path = os.path.join(d, "cm_ids.tsv"), os.path.join(d, "cm_walked.tsv")
    spaces_path, errors_path = os.path.join(d, "cm_spaces.tsv"), os.path.join(d, "cm_errors.tsv")
    walked = {r[0] for r in read_table(walked_path)}
    have_games = {r[0] for r in read_table(ids_path)}
    spaces = {r[0]: (r[1] if len(r) > 1 else "") for r in read_table(spaces_path)}
    spaces_f = open_table(spaces_path, SPACES_HEADER)

    cand = {}  # tournament id -> name
    listed_spaces = set()

    def note_spaces(items):
        for t in items:
            for s in ((t.get("hosts") or {}).get("spaces")) or []:
                if s["id"] not in spaces:
                    spaces[s["id"]] = s.get("name") or ""
                    spaces_f.write(tsv_line([s["id"], spaces[s["id"]]]))
                    spaces_f.flush()

    def add(items, label):
        new = 0
        for t in items:
            if t["id"] not in cand:
                cand[t["id"]] = t.get("name") or ""
                new += t["id"] not in walked
        note_spaces(items)
        log(f"{label}: {len(items)} listed, {new} new to walk")

    for tid in read_id_list(os.path.join(d, "cm_tournaments.txt")):
        cand.setdefault(tid, "")
    add(list_game(cm), "recent tournaments of the game")

    out, wlog = open_table(ids_path, IDS_HEADER), open_table(walked_path, WALKED_HEADER)
    errs = open_table(errors_path, ERRORS_HEADER)
    stats = {"walked": 0, "games": 0, "failed": 0, "unfinished": 0}
    skipped = set()  # failed or unfinished this run: tried again on the next run, not in this one

    def pending():
        return [t for t in cand if t not in walked and t not in skipped]

    def list_new_spaces():
        for sid in [s for s in spaces if s not in listed_spaces]:
            listed_spaces.add(sid)
            try:
                add(list_space(cm, sid), f"space {spaces[sid] or sid}")
            except QueryFailed as e:
                log(f"space {sid}: {e}")

    list_new_spaces()
    if args.list_only:
        log(f"{len(cand)} tournaments known, {len(pending())} not walked yet")
        return stats
    while True:
        todo = pending()
        if not todo:
            break
        log(f"{len(cand)} tournaments known, {len(todo)} not walked yet")
        for tid in todo:
            if args.limit is not None and stats["walked"] >= args.limit:
                return stats
            try:
                tour = walk_tournament(cm.query, tid)
            except QueryFailed as e:
                stats["failed"] += 1
                skipped.add(tid)
                log(f"{tid}: {e}")
                errs.write(tsv_line([tid, time.strftime("%Y-%m-%dT%H:%M:%S"), e]))
                errs.flush()
                continue
            note_spaces([tour])
            if tour.get("state") not in FINISHED:
                stats["unfinished"] += 1
                skipped.add(tid)
                log(f"{tid}: state {tour.get('state')}, left for a later run")
                continue
            rows = [g for g in game_rows(tour) if g["game_id"] not in have_games]
            for g in rows:
                out.write(tsv_line(g[c] for c in IDS_HEADER))
                have_games.add(g["game_id"])
            out.flush()
            wlog.write(walked_line(tour, rows, cm.partial))
            wlog.flush()
            walked.add(tid)
            stats["walked"] += 1
            stats["games"] += len(rows)
            log(f"{stats['walked']:4} {len(rows):4} games  {tour.get('name')}")
        list_new_spaces()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--limit", type=int, help="walk at most this many tournaments (for a trial run)")
    ap.add_argument("--list-only", action="store_true", help="only list tournaments, walk none")
    ap.add_argument("--summary", action="store_true", help="print counts from the existing files and exit (no network)")
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL, help="seconds between requests (not below 3.5)")
    args = ap.parse_args()
    if args.summary:
        print(summary(args.dir))
        return 0

    def log(msg):
        print(msg, flush=True)

    cm = Client(args.min_interval, log=log)
    try:
        stats = run(args, cm, log)
    except AccessDenied as e:
        log(f"STOPPED: {e}. Not retrying; check the refresh key and the application's access.")
        return 2
    except KeyboardInterrupt:
        log("interrupted; files are consistent, rerun to continue")
        return 130
    log(f"done: {stats['walked']} tournaments walked, {stats['games']} new games, {stats['failed']} failed, "
        f"{stats['unfinished']} not finished, {cm.requests} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
