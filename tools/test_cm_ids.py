"""Tests for cm_ids.py (no network). Run: /kvm/polytopia/.venv/bin/python -m pytest ai/tools/test_cm_ids.py"""

import argparse
import io
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cm_ids  # noqa: E402


def stat(link=None, mode="Might", size="14", teams="2", turns="11", tribes=None, general=False):
    pre = "General." if general else ""
    nodes = [{"name": pre + "GameMode", "serializedValue": mode}, {"name": pre + "MapSize", "serializedValue": size},
             {"name": pre + "NumberOfTeams", "serializedValue": teams}, {"name": pre + "CurrentTurnNumber", "serializedValue": turns}]
    if link:
        nodes.append({"name": pre + "ReplayLink", "serializedValue": link})
    comp = None
    if tribes is not None:
        comp = {"nodes": [x for t in tribes for x in ({"name": "Player.TribeType", "serializedValue": t}, {"name": "Player.Score", "serializedValue": "9"})]}
    return {"gameSessionStatistics": {"globalStatistics": {"nodes": nodes}, "competitorStatistics": comp}}


def match(mid, **kw):
    return {"id": mid, "state": "COMPLETED", "statistics": stat(**kw)}


def series(sid, *matches):
    return {"id": sid, "state": "COMPLETED", "matches": list(matches)}


def page(nodes, more=False, cursor=None):
    return {"pageInfo": {"hasNextPage": more, "endCursor": cursor}, "nodes": nodes}


LINK = "https://share.polytopia.io/g/%s"


def fixture():
    """A tournament with an elimination stage (two brackets), a Swiss stage and a group stage."""
    return {
        "id": "T1", "name": "Test\tCup", "state": "COMPLETED", "hosts": {"spaces": [{"id": "S1", "name": "Org"}]},
        "schedule": {"startedAt": "2026-10-31T19:01:04.027Z", "endedAt": None},
        "settings": {"gameSessionSettings": {"GameMode": 1, "MapType": 1, "MapSize": 1}},
        "stages": [
            {"index": 0, "format": "SINGLE_ELIMINATION", "brackets": [
                {"rounds": [{"roundNumber": 1, "matchSeriesPage": page([
                    series("a", match("m1", link=LINK % "g1", tribes=["Bardur", "Kickoo"])),
                    series("b", match("m2", link=None)),  # no replay link
                    series("c"),  # no matches
                ])}]},
                {"rounds": [{"roundNumber": 1, "matchSeriesPage": page([
                    series("d", match("m3", link=LINK % "g2/", general=True, mode="Domination", size="16", teams="3"),
                           match("m4", link=LINK % "g3", tribes=[])),
                ])}]},
            ]},
            {"index": 1, "format": "SWISS", "rounds": [
                {"roundNumber": 1, "matchSeriesPage": page([series("e", match("m5", link=LINK % "g1"))])},  # duplicate id
            ]},
            {"index": 2, "format": "ROUND_ROBIN", "groups": [
                {"number": 1, "matchSeriesPage": page([series("f", match("m6", link=LINK % "g4"))])},
            ]},
        ],
    }


def test_statistics_strips_general_prefix_and_reads_tribes():
    glob, tribes = cm_ids.statistics(match("m", link=LINK % "x", general=True, tribes=["Bardur", "Kickoo"]))
    assert glob["ReplayLink"] == LINK % "x" and glob["GameMode"] == "Might" and glob["NumberOfTeams"] == "2"
    assert tribes == ["Bardur", "Kickoo"]


def test_statistics_tolerates_missing_parts():
    assert cm_ids.statistics({"id": "m", "statistics": None}) == ({}, [])
    assert cm_ids.statistics({"id": "m", "statistics": {"gameSessionStatistics": None}}) == ({}, [])
    glob, tribes = cm_ids.statistics(match("m", link=None))  # competitorStatistics null, as for most 404 matches
    assert "ReplayLink" not in glob and tribes == []


@pytest.mark.parametrize("link,want", [
    ("https://share.polytopia.io/g/abc-123", "abc-123"),
    ("https://share.polytopia.io/g/abc-123/", "abc-123"),
    (" https://share.polytopia.io/g/abc-123\n", "abc-123"),
    ("abc-123", "abc-123"),
])
def test_game_id(link, want):
    assert cm_ids.game_id(link) == want


def test_game_rows_walks_every_format_and_dedupes():
    rows = cm_ids.game_rows(fixture())
    assert [r["game_id"] for r in rows] == ["g1", "g2", "g3", "g4"]
    r1, r2 = rows[0], rows[1]
    assert r1 == {"game_id": "g1", "tournament_id": "T1", "match_id": "m1", "mode": "Might", "size": "14", "teams": "2",
                  "turns": "11", "tribes": "Bardur,Kickoo", "started": "2026-10-31", "tournament_name": "Test\tCup"}
    assert (r2["mode"], r2["size"], r2["teams"], r2["tribes"]) == ("Domination", "16", "3", "")
    assert sum(1 for _ in cm_ids.all_series(fixture())) == 6


def test_containers_keys():
    keys = [k for st in fixture()["stages"] for k, _ in cm_ids.containers(st)]
    assert keys == [("bracket", 0, 0), ("bracket", 1, 0), ("round", 0, 0), ("group", 0, 0)]


def test_tsv_line_has_no_stray_tabs_or_newlines():
    line = cm_ids.tsv_line(["a", None, "b\tc\nd", 3])
    assert line == "a\t\tb c d\t3\n"


def test_walked_line_keeps_settings():
    tour = fixture()
    rows = cm_ids.game_rows(tour)
    cells = cm_ids.walked_line(tour, rows, 2).rstrip("\n").split("\t")
    assert len(cells) == len(cm_ids.WALKED_HEADER)
    row = dict(zip(cm_ids.WALKED_HEADER, cells))
    assert (row["game_mode"], row["map_type"], row["map_size"], row["series"], row["games"], row["partial_errors"]) == ("1", "1", "1", "6", "4", "2")
    assert json.loads(row["settings"]) == {"GameMode": 1, "MapType": 1, "MapSize": 1}


class PagedServer:
    """Serves one Swiss stage whose only round has 5 series, `first` at a time, with opaque cursors."""

    def __init__(self, total=5, bad_stage=False):
        self.total, self.calls, self.bad_stage = total, [], bad_stage

    def query(self, text, v):
        self.calls.append(dict(v))
        start = int(v["after"].split(":")[1]) if v.get("after") else 0
        n = v["first"]
        nodes = [series("s%d" % i, match("m%d" % i, link=LINK % ("g%d" % i))) for i in range(start, min(start + n, self.total))]
        end = start + len(nodes)
        pg = page(nodes, end < self.total, "c:%d" % end)
        idx = 7 if self.bad_stage and v.get("after") else 0
        return {"tournament": {"id": v["id"], "name": "P", "state": "COMPLETED", "schedule": {"startedAt": "2026-01-02T00:00:00Z"},
                               "settings": None, "stages": [{"index": idx, "format": "SWISS", "rounds": [{"roundNumber": 1, "matchSeriesPage": pg}]}]}}


def test_walk_tournament_follows_pages():
    srv = PagedServer()
    tour = cm_ids.walk_tournament(srv.query, "T", page=2)
    assert [r["game_id"] for r in cm_ids.game_rows(tour)] == ["g0", "g1", "g2", "g3", "g4"]
    assert len(srv.calls) == 3
    assert [c.get("after") for c in srv.calls] == [None, "c:2", "c:4"]
    assert srv.calls[1]["from"] == srv.calls[1]["to"] == 0


def test_walk_tournament_single_page_is_one_request():
    srv = PagedServer(total=3)
    cm_ids.walk_tournament(srv.query, "T", page=200)
    assert len(srv.calls) == 1


def test_walk_tournament_fails_when_stage_cannot_be_found_again():
    with pytest.raises(cm_ids.QueryFailed):
        cm_ids.walk_tournament(PagedServer(bad_stage=True).query, "T", page=2)


# ---------------------------------------------------------------- client


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def client(tmp_path, monkeypatch):
    kf = tmp_path / "key"
    kf.write_text("not-a-real-key\n")
    sleeps = []
    monkeypatch.setattr(cm_ids.time, "sleep", lambda s: sleeps.append(s))
    cm = cm_ids.Client(key_file=str(kf), log=lambda m: None)
    cm.sleeps = sleeps
    return cm


def script(monkeypatch, steps):
    """Make urlopen answer from `steps`: an int is an HTTP error, anything else a JSON body."""
    seen = []

    def fake(req, timeout=None):
        seen.append(req)
        step = steps.pop(0)
        if isinstance(step, int):
            raise urllib.error.HTTPError(req.full_url, step, "x", {}, io.BytesIO(b"{}"))
        return FakeResponse(json.dumps(step).encode())

    monkeypatch.setattr(cm_ids.urllib.request, "urlopen", fake)
    return seen


def test_client_backs_off_on_429_then_succeeds(client, monkeypatch):
    seen = script(monkeypatch, [{"value": "tok"}, 429, 503, {"data": {"x": 1}}])
    assert client.query("{ x }") == {"x": 1}
    assert [s for s in client.sleeps if s >= 60] == [60.0, 120.0]
    assert seen[1].headers["Authorization"] == "Bearer tok" and "Mozilla" in seen[1].headers["User-agent"]


def test_client_gives_up_after_four_tries(client, monkeypatch):
    script(monkeypatch, [{"value": "tok"}, 500, 500, 500, 500])
    with pytest.raises(cm_ids.QueryFailed):
        client.query("{ x }")
    assert [s for s in client.sleeps if s >= 60] == [60.0, 120.0, 240.0]


@pytest.mark.parametrize("code", [401, 403])
def test_client_stops_on_401_and_403(client, monkeypatch, code):
    script(monkeypatch, [{"value": "tok"}, code])
    with pytest.raises(cm_ids.AccessDenied):
        client.query("{ x }")


def test_client_enforces_minimum_interval(client, tmp_path):
    assert client.interval == cm_ids.MIN_INTERVAL
    assert cm_ids.Client(interval=0.5, key_file=str(tmp_path / "key")).interval == cm_ids.MIN_INTERVAL
    assert cm_ids.Client(interval=6, key_file=str(tmp_path / "key")).interval == 6


def test_client_tolerates_partial_errors_but_not_missing_data(client, monkeypatch):
    err = {"message": "Failed to load data.", "extensions": {"errorCode": 404}}
    script(monkeypatch, [{"value": "tok"}, {"data": {"x": {"a": 1}}, "errors": [err, err]}, {"data": None, "errors": [err]}])
    assert client.query("{ x }") == {"x": {"a": 1}} and client.partial == 2
    with pytest.raises(cm_ids.QueryFailed):
        client.query("{ x }")


def test_client_sleeps_between_requests(client, monkeypatch):
    script(monkeypatch, [{"value": "tok"}, {"data": {"x": 1}}, {"data": {"x": 2}}])
    client.query("{ x }")
    client.query("{ x }")
    assert client.requests == 3


# ---------------------------------------------------------------- whole run


class FakeCM:
    """Answers the three kinds of query from a dict of tournaments; counts walks."""

    partial = 0
    requests = 0

    def __init__(self, tours, game_list, space_lists):
        self.tours, self.game_list, self.space_lists = tours, game_list, space_lists
        self.walks = []

    def query(self, text, v=None):
        if "tournamentsForGame" in text:
            return {"tournamentsForGame": [self.brief(t) for t in self.game_list]}
        if "tournamentsForSpace" in text:
            return {"tournamentsForSpace": [self.brief(t) for t in self.space_lists.get(v["i"]["spaceId"], [])]}
        self.walks.append(v["id"])
        return {"tournament": self.tours.get(v["id"])}

    def brief(self, tid):
        t = self.tours[tid]
        return {"id": tid, "name": t["name"], "state": t["state"], "hosts": t["hosts"]}


def make_tour(tid, space, state="COMPLETED", gid=None):
    t = fixture()
    t["id"], t["state"], t["name"] = tid, state, "T " + tid
    t["hosts"] = {"spaces": [{"id": space, "name": "Org " + space}]}
    if gid:
        for s in cm_ids.all_series(t):
            for m in s["matches"]:
                for n in m["statistics"]["gameSessionStatistics"]["globalStatistics"]["nodes"]:
                    if n["name"].endswith("ReplayLink"):
                        n["serializedValue"] = LINK % (gid + "-" + m["id"])
    return t


def read(path):
    return [l.rstrip("\n").split("\t") for l in open(path)]


def test_run_discovers_spaces_is_incremental(tmp_path):
    tours = {"A": make_tour("A", "S1", gid="a"), "F": make_tour("F", "S3", gid="f")}
    (tmp_path / "cm_tournaments.txt").write_text("# extra\nF some name\n")
    lists = {"S1": ["A"], "S3": ["F"]}
    args = argparse.Namespace(dir=str(tmp_path), limit=None, list_only=False)
    stats = cm_ids.run(args, FakeCM(tours, ["A"], lists), log=lambda m: None)
    assert {r[0] for r in read(tmp_path / "cm_walked.tsv")[1:]} == {"A", "F"}
    ids = read(tmp_path / "cm_ids.tsv")
    assert ids[0] == cm_ids.IDS_HEADER
    # five matches per tournament carry a link; ids are unique and tied to their tournament
    assert len({r[0] for r in ids[1:]}) == len(ids) - 1 == 10 == stats["games"]
    assert {r[1] for r in ids[1:]} == {"A", "F"}
    assert {r[0] for r in read(tmp_path / "cm_spaces.tsv")[1:]} == {"S1", "S3"}

    before = (tmp_path / "cm_ids.tsv").read_text()
    cm2 = FakeCM(tours, ["A"], lists)
    stats2 = cm_ids.run(args, cm2, log=lambda m: None)
    assert (tmp_path / "cm_ids.tsv").read_text() == before
    assert cm2.walks == [] and stats2["games"] == 0


def test_run_walks_space_found_through_a_tournament(tmp_path):
    tours = {"A": make_tour("A", "S1", gid="a"), "B": make_tour("B", "S1", gid="b")}
    cm = FakeCM(tours, game_list=["A"], space_lists={"S1": ["A", "B"]})
    cm_ids.run(argparse.Namespace(dir=str(tmp_path), limit=None, list_only=False), cm, log=lambda m: None)
    assert {r[0] for r in read(tmp_path / "cm_walked.tsv")[1:]} == {"A", "B"}
    assert [r[0] for r in read(tmp_path / "cm_spaces.tsv")[1:]] == ["S1"]


def test_run_leaves_running_tournament_unwalked_and_failures_in_errors(tmp_path):
    tours = {"C": make_tour("C", "S1", state="RUNNING"), "A": make_tour("A", "S1", gid="a")}

    class Flaky(FakeCM):
        def query(self, text, v=None):
            if v and v.get("id") == "A" and "tournament(" in text:
                raise cm_ids.QueryFailed("HTTP 500 after 4 tries")
            return super().query(text, v)

    cm = Flaky(tours, game_list=["C", "A"], space_lists={})
    stats = cm_ids.run(argparse.Namespace(dir=str(tmp_path), limit=None, list_only=False), cm, log=lambda m: None)
    assert stats["unfinished"] == 1 and stats["failed"] == 1 and stats["walked"] == 0
    assert not (tmp_path / "cm_walked.tsv").exists() or len(read(tmp_path / "cm_walked.tsv")) == 1
    assert read(tmp_path / "cm_errors.tsv")[1][0] == "A"


def test_summary_counts(tmp_path):
    def row(gid, mode, size, teams, started, tribes=""):
        return cm_ids.tsv_line([gid, "T", "m", mode, size, teams, "9", tribes, started, "n"])

    (tmp_path / "cm_ids.tsv").write_text("\t".join(cm_ids.IDS_HEADER) + "\n" + row("1", "Might", "14", "2", "2026-01-01", "Bardur,Kickoo")
                                         + row("2", "Might", "11", "2", "2025-05-05") + row("3", "Might", "16", "2", "2025-05-05")
                                         + row("4", "Domination", "14", "2", "2025-05-05") + row("5", "Might", "14", "3", "2025-05-05"))
    (tmp_path / "cm_walked.tsv").write_text("\t".join(cm_ids.WALKED_HEADER) + "\n" + cm_ids.walked_line(fixture(), [], 0))
    text = cm_ids.summary(str(tmp_path))
    assert "1 tournaments walked, 5 games" in text
    assert "Might, 2 teams, Tiny or Small: 2 games" in text
    assert "by year: 2025 1, 2026 1" in text
    assert "games with tribes recorded: 1" in text
