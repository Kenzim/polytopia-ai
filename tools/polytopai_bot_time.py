"""How long the PolytopAI site's bot takes for a turn: one game in which our side only ends its turns.

    PYTHONPATH=python python3 tools/polytopai_bot_time.py [LEVEL] [TURNS]
"""

import sys
import time

from polyai.polytopai import SiteClient

level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
turns = int(sys.argv[2]) if len(sys.argv) > 2 else 14
c = SiteClient()
v = c.call("new", {"mode": "play", "tribes": [7, 10], "map_type": 2, "size": 14, "skill": None, "steered": False,
                   "level": level, "levels": None, "win": None}, repeatable=True)
rows = []
t_all = time.time()
while not v["play"]["over"] and v["state"]["settings"]["turn"] < turns:
    end = next(m for m in v["legalMoves"] if m["moveType"] == 10)
    t = time.time()
    v = c.call("move", end)
    t_move = time.time() - t
    t, n = time.time(), 0
    while not v["play"]["over"] and v["state"]["settings"]["currentPlayerTurnId"] != v["play"]["human"]:
        v = c.call("bot", {"turn": True})
        n += 1
    rows.append((v["state"]["settings"]["turn"], round(t_move, 2), n, round(time.time() - t, 2)))
print(f"level {level}: (turn, seconds for our move, bot calls, bot seconds)")
for r in rows:
    print("  ", r)
print("total", round(time.time() - t_all, 1), "s")
