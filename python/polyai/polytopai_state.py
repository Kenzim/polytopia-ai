"""The PolytopAI website's game state as an engine position, and engine actions as the site's moves.

The site (polytopai.com) runs its games on its own simulator and sends the whole state as JSON with every answer,
together with the mover's legal moves. Its ids for tribes, units, techs, terrain, resources and buildings are the
game's own, and so are the engine's, so a state translates field by field into a position document
(`Engine::from_position`, `polyai_engine.Game.from_position`). The engine never steps such a game: every decision
starts from a fresh translation, so the two rule sets cannot drift apart. What can differ is the legal list; the
bridge only plays moves both sides call legal (`match_actions`).

`position(state, legal_moves, stayed)` builds the document. `move_key` and `action_keys` name a site move and an
engine action in a common form, which is how the two legal lists are joined.
"""

from __future__ import annotations

NATURE = 255
CITY = 1

# engine action kinds (crates/polyai-core/src/action.rs)
K_MOVE, K_ATTACK, K_CAPTURE, K_PROMOTE, K_RECOVER, K_HEAL_OTHERS, K_EXPLODE, K_DISBAND, K_EXAMINE_RUINS = range(1, 10)
K_STAY, K_SWARM, K_BOOST, K_DISEMBARK, K_UPGRADE, K_TRAIN, K_BUILD, K_DESTROY, K_DECOMPOSE = range(10, 19)
K_CLEAR_TILE_EFFECT, K_CITY_REWARD, K_RESEARCH = 19, 20, 21
K_END_TURN, K_FREEZE_AREA, K_BREAK_ICE = 26, 27, 28

# site move types
M_STEP, M_ATTACK, M_ABILITY, M_TRAIN, M_HARVEST, M_BUILD, M_RESEARCH, M_CAPTURE, M_REWARD, M_END = range(1, 11)

# site ability ids of a unit acting where it stands, by the engine kind that does the same
UNIT_ABILITY = {7: K_RECOVER, 8: K_DISBAND, 9: K_HEAL_OTHERS, 13: K_FREEZE_AREA, 14: K_BOOST, 15: K_EXPLODE, 16: K_PROMOTE,
                17: K_BREAK_ICE, 104: K_DISEMBARK}
# site abilities on a tile that the engine has as commands of their own
TILE_ABILITY = {4: K_DESTROY, 5: K_DECOMPOSE, 102: K_CLEAR_TILE_EFFECT}
# site abilities on a tile that the engine has as builds: ability id -> improvement id
ABILITY_BUILD = {1: 11, 2: 10, 3: 14}  # burn forest, clear forest, grow forest
STARFISHING = 46  # the site has a unit "capture" the starfish it stands on
# The buildings the site names by type. Every other engine build (picking fruit, fishing, hunting, ...) is a "harvest"
# on the site, which names only the tile.
SITE_STRUCTURES = frozenset({1, 2, 3, 5, 6, 8, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 37,
                             38, 39, 41, 47, 48, 49, 50, 51, 52, 53, 54, 70, 123}) | frozenset(ABILITY_BUILD.values())
# The site's ids of finished tasks -> the engine's task kinds (each unlocks one monument). Seen in play: 0 the Altar
# of Peace, 3 the Eye of God, 4 the Gate of Power, 5 the Grand Bazaar, 6 the Park of Fortune; 1 and 2 are taken to be
# the two left over.
TASK_KIND = {0: (1,), 1: (2,), 2: (4,), 3: (7, 8), 4: (5,), 5: (3,), 6: (6,)}

# unit effects in the game's id order, which the site sends as numbers
UNIT_FLAGS = ("frozen", "poisoned", "boosted", "invisible", "bubble", "petrified", "swift", "double_ready", "charmed")


def _pair(c: dict | None) -> list[int] | None:
    if not c or c.get("x", -1) < 0 or c.get("y", -1) < 0:
        return None
    return [int(c["x"]), int(c["y"])]


def _unit(u: dict, stayed: frozenset, notes: list[str]) -> dict:
    at = u["coords"]
    out = {
        "type": int(u["type"]),
        "owner": int(u["owner"]),
        "x": int(at["x"]),
        "y": int(at["y"]),
        "health": int(round(float(u["health"]) * 10)),
        "xp": int(u.get("kills", 0)),
        "promotion": int(bool(u.get("veteran"))),
        "moved": bool(u.get("moved")) or at["idx"] in stayed,
        "attacked": bool(u.get("attacked")) or at["idx"] in stayed,
        "created": int(u.get("createdTurn", 0)),
        "direction": int(u.get("direction", 0)),
    }
    home, prev = _pair(u.get("homeCoords")), _pair(u.get("prevCoords"))
    out["home"] = home or [-1, -1]
    if prev:
        out["previous"] = prev
    flags = []
    for e in u.get("effects") or []:
        if isinstance(e, int) and 0 <= e < len(UNIT_FLAGS):
            flags.append(UNIT_FLAGS[e])
        else:
            notes.append(f"unit effect {e!r}")
    if flags:
        out["flags"] = flags
    if u.get("passengerType") is not None:
        out["passenger"] = {"type": int(u["passengerType"]), "owner": int(u["owner"])}
    return out


def position(state: dict, legal_moves: list[dict] | tuple = (), stayed: frozenset = frozenset()) -> tuple[dict, list[str]]:
    """The site's `state` as a position document, and notes on anything that had no translation.

    `legal_moves` is the mover's legal list: a pending city reward only shows there. `stayed` holds the tile indices
    of the mover's units that were told to stay this turn (the site has no such move); they count as done.
    """
    notes: list[str] = []
    st = state["settings"]
    size = int(st["size"])
    mover = int(st["currentPlayerTurnId"])
    tribes = sorted(state["tribes"].values(), key=lambda t: int(t["id"]))

    game = {
        "game_version": int(st.get("version", 123)),
        "seed": int(st.get("seed", 0)) & 0x7FFFFFFF,
        "turn": int(st["turn"]),
        "current_player": mover,
        "width": size,
        "height": size,
        "settings": {
            "base_game_mode": int(st.get("mode", 4)),
            "rules_game_mode": int(st.get("mode", 4)),
            "win_by_capital": bool(st.get("winByCapital", True)),
            "win_by_extermination": bool(st.get("winByExtermination", True)),
            "opponent_count": len(tribes) - 1,
        },
    }

    players = []
    for t in tribes:
        pid = int(t["id"])
        rel = []
        for other, r in (t.get("relations") or {}).items():
            rel.append({"other": int(other), "state": int(r.get("state", 0)), "last_attack": int(r.get("lastAttackTurn", -100)),
                        "embassy": int(r.get("embassyLevel", 0)), "peace_broken": int(r.get("lastPeaceBrokenTurn", -100)),
                        "first_meet": int(r.get("firstMeet", 0)), "embassy_built": int(r.get("embassyBuildTurn", -100)),
                        "previous_attack": int(r.get("previousAttackTurn", -100))})
        players.append({
            "id": pid,
            "tribe": int(t["type"]),
            "human": True,
            "stars": int(t["stars"]),
            "start": _pair(t.get("startingTileCoords")) or [-1, -1],
            "tech": [int(x["type"]) for x in t.get("tech_vanilla", []) if x.get("discovered", True)],
            "known": [int(x) for x in t.get("knownPlayers", [])],
            "diplomacy": rel,
            "tasks": [{"kind": kind, "started": True, "completed": True}
                      for k in t.get("tasksDone", []) for kind in TASK_KIND.get(int(k), ())],
            "unique": [int(x) for x in t.get("builtUniqueImprovements", [])],
            "climate": int(t.get("climate", t["type"])),
            "kills": int(t.get("kills", 0)),
            "casualties": int(t.get("casualties", 0)),
            "killer": max(int(t.get("killerId", -1)), 0),
            "killed_turn": max(int(t.get("killedTurn", -1)), 0),
            "resigned_turn": int(t.get("resignedTurn", -1)),
        })
    players.append({"id": NATURE, "tribe": 1, "human": False, "handicap": 0, "stars": 0, "tech": [0, 20], "known": [],
                    "diplomacy": [], "climate": 1})

    structures, resources = state.get("structures") or {}, state.get("resources") or {}
    tiles = []
    for t in state["tiles"]:
        idx = int(t["coords"]["idx"])
        out = {"x": int(t["coords"]["x"]), "y": int(t["coords"]["y"]), "terrain": int(t["type"]),
               "climate": int(t.get("climate", 1)), "explored": [int(p) for p in t.get("explorers", [])]}
        if t.get("skinType"):
            out["skin"] = int(t["skinType"])
        if str(idx) in resources:
            out["resource"] = int(resources[str(idx)]["type"])
        if str(idx) in structures:
            out["improvement"] = int(structures[str(idx)]["type"])
        if t.get("owner"):
            out["owner"] = int(t["owner"])
        if t.get("capitalOf"):
            out["capital"] = True
            out["capital_of"] = int(t["capitalOf"])
        ruling = _pair(t.get("rulingCityCoords"))
        if ruling:
            out["ruling"] = ruling
        if t.get("hasRoad"):
            out["road"] = True
        # No "route": the site leaves `hasRoute` false everywhere, and without the field the engine works the trade
        # routes out from the roads, ports and cities, as in our own games.
        effects = [e for e in t.get("effects") or [] if isinstance(e, int) and 0 <= e < 16]
        if effects:
            out["effects"] = effects
        for e in t.get("effects") or []:
            if e not in effects:
                notes.append(f"tile effect {e!r}")
        tiles.append(out)

    # Only owned cities carry runtime state worth giving; the engine derives the rest (villages, ruins, building levels).
    improvements = []
    for t in tribes:
        for c in t.get("cities", []):
            idx = int(c["idx"])
            improvements.append({
                "x": idx % size, "y": idx // size, "owner": int(c["owner"]), "founder": int(c["owner"]),
                "level": int(c["level"]), "xp": int(c.get("progress", 0)), "population": int(c.get("population", 0)),
                "production": int(c.get("production", 1)), "border_size": int(c.get("borderSize", 1)),
                "connected": int(c["owner"]) if c.get("connectedToCapital") else 0,
                "rewards": [int(r) for r in c.get("rewards", [])],
                "founded": int((structures.get(str(idx)) or {}).get("founded", 0)),
            })

    units = [_unit(u, stayed if int(t["id"]) == mover else frozenset(), notes) for t in tribes for u in t.get("units", [])]

    triggers = []
    for m in legal_moves:
        if m.get("moveType") == M_REWARD:
            at = int(m["target"])
            trig = {"player": mover, "x": at % size, "y": at // size, "peace_from": -1}
            if trig not in triggers:
                triggers.append(trig)

    doc = {"format": "polytopia-position", "version": 1, "game": game, "players": players, "tiles": tiles,
           "improvements": improvements, "units": units, "triggers": triggers}
    return doc, notes


def move_key(m: dict) -> tuple | None:
    """A site move in the common form, `None` for a move the engine's action list has no counterpart of (or one
    without the fields its type should have)."""
    try:
        return _move_key(m)
    except KeyError:
        return None


def _move_key(m: dict) -> tuple | None:
    t = m.get("moveType")
    if t == M_STEP:
        return (K_MOVE, m["src"], m["target"])
    if t == M_ATTACK:
        return (K_ATTACK, m["src"], m["target"])
    if t == M_ABILITY:
        a = m.get("type")
        if a in UNIT_ABILITY:
            return (UNIT_ABILITY[a], m["src"])
        if a == 6:  # a mind bender converting is the engine's attack
            return (K_ATTACK, m["src"], m["target"])
        at = m.get("target", m.get("src"))  # an ability on the tile a unit stands on names it as `src`
        if a in TILE_ABILITY and at is not None:
            return (TILE_ABILITY[a], at)
        if a in ABILITY_BUILD and at is not None:
            return (K_BUILD, at, ABILITY_BUILD[a])
        return None
    if t == M_TRAIN:
        # a move in the legal list carries `upgrade`; a played one only says so in its label
        upgrade = m.get("upgrade") or str(m.get("label", "")).startswith("Upgrade")
        return (K_UPGRADE if upgrade else K_TRAIN, m["src"], m["type"])
    if t == M_HARVEST:
        return ("harvest", m["target"])
    if t == M_BUILD:
        return (K_BUILD, m["target"], m["type"])
    if t == M_RESEARCH:
        return (K_RESEARCH, m["type"])
    if t == M_CAPTURE:
        return (K_CAPTURE, m["src"])
    if t == M_REWARD:
        return (K_CITY_REWARD, m["target"], m["type"])
    if t == M_END:
        return (K_END_TURN,)
    return None


def action_keys(a) -> list[tuple]:
    """The common forms an engine action `(kind, arg, src, dst)` may have on the site, most specific first."""
    kind, arg, src, dst = (int(x) for x in a)
    if kind in (K_MOVE, K_ATTACK):
        return [(kind, src, dst)]
    if kind in (K_CAPTURE, K_EXAMINE_RUINS):
        return [(K_CAPTURE, src)]
    if kind in (K_UPGRADE, K_TRAIN):
        return [(kind, src, arg)]
    if kind == K_BUILD:
        if arg == STARFISHING:
            return [(K_CAPTURE, src), ("harvest", src)]
        return [(K_BUILD, src, arg)] if arg in SITE_STRUCTURES else [("harvest", src)]
    if kind == K_CITY_REWARD:
        return [(K_CITY_REWARD, src, arg)]
    if kind == K_RESEARCH:
        return [(K_RESEARCH, arg)]
    if kind == K_END_TURN:
        return [(K_END_TURN,)]
    if kind == K_SWARM:  # the site's "boost allies" of a shaman
        return [(K_BOOST, src)]
    return [(kind, src)]


def match_actions(actions, legal_moves: list[dict]) -> list[dict | None | bool]:
    """For every engine action the site move that does the same, `True` for a stay (which needs no move) and `None`
    for an action the site does not offer."""
    by_key = {}
    for m in legal_moves:
        k = move_key(m)
        if k is not None:
            by_key.setdefault(k, m)
    out: list[dict | None | bool] = []
    for a in actions:
        if int(a[0]) == K_STAY:
            out.append(True)
            continue
        out.append(next((by_key[k] for k in action_keys(a) if k in by_key), None))
    return out
