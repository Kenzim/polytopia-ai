# Combat cross-check: engine vs community calculators (2026-11-01)

Plan item `combat-crosscheck`. The engine (`engine/src/rules/combat.rs:259-327`, spec `engine/docs/specs/combat.md`)
uses the community formula in fixed-point integers (HP and attack x10, bonus x10, `round_to_10` = truncate to tenths,
then round half up). Source for the community side: frothfrenzy's calculator
(https://frothfrenzy.github.io/polytopiacalculator/); the Fandom Combat page returned 403.

| # | Fight | Engine (dmg / retaliation, HP) | Calculator |
|---|---|---|---|
| 1 | Warrior vs warrior, plain | 5 / 5 | 4.5→5 / 4.5→5 |
| 2 | Warrior vs warrior on forest (Archery) | 4 / 5 | 3.6→4 / 5.4→5 |
| 3 | Swordsman vs defender in walled city | 3 / 11 | 2.7 / 10.8 |
| 4 | Knight vs defender in city, no wall | 7 / 8 | 6.89 / 7.59 |
| 5 | Archer vs warrior at 5/10 HP | 6, kill, no retaliation | 6, kill |
| 6 | Catapult vs warrior on mountain (Climbing) | 10, kill | 10.29→10 |
| 7 | Veteran warrior at 10/15 vs warrior | 4 / 5 | 3.6 / 5.4 |
| 8 | Boosted warrior vs warrior | 6 / 4 | 6.25 / 4.0 |
| 9 | Warrior vs poisoned warrior on forest | 5 / 4 | 5.14→5 (bonus 0.75) |

All nine agree. Points where calculators may differ from the engine, none confirmed as an engine bug:

- Poison: the engine halves the bonus with a shift (version ≥115: 1.5 → 0.7, not 0.75; ≤114 flat 0.7) and keeps the
  unhalved defence for retaliation. A calculator using 0.75, or halved defence in retaliation, can differ by 1 HP in
  edge cases. Whether retail uses 0.7 or 0.75 is worth confirming against the decompiled code (`re/`).
- Rounding: half up, as JS `Math.round`; a round-half-to-even calculator would give 4 for 4.5.
- City bonus replaces the terrain bonus and needs the Fortify ability; no fortify bonus outside cities.
- Splash re-runs the calculation per neighbour, halves and truncates to whole HP; splash plus poison only poisons.

Conclusion: no engine discrepancy found. The poison 0.7 vs 0.75 question goes to the project chat for the engine
manager to confirm against the decompiled game.
