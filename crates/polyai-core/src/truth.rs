//! Truth labels for the tiles a player has not explored (docs/belief.md).
//!
//! The engine knows the whole map, so what lies under the fog of a recorded position is free to label. The labels
//! are training targets for belief heads; they are never part of the observation (`encode` does not read this
//! module, and a test checks that changing a hidden tile changes the truth but not the encoding).
//!
//! Labels are seen from the mover like the rest of the encoding: slot-valued labels use the mover's slots. They are
//! written for hidden tiles only; an explored tile has all labels 0 and mask 0.

use crate::slots::SlotMap;
use polytopia_sim::engine::{Engine, NATURE};

/// Label channels per tile.
pub const TRUTH_K: usize = 6;
pub const TRUTH_NAMES: [&str; TRUTH_K] = ["terrain", "resource", "site", "enemy_city", "enemy_capital", "unit"];
/// Number of values of each channel (the labels are class ids below these).
pub const TRUTH_VOCABS: [i16; TRUTH_K] = [10, 11, 3, 2, 8, 3];

/// Channel indices.
pub const T_TERRAIN: usize = 0;
/// Raw engine resource idx (1 game ... 9 aquacrop), 0 none, whether or not an improvement stands on it and whether
/// or not the mover could see it with its techs.
pub const T_RESOURCE: usize = 1;
/// 0 nothing, 1 village (an unowned city), 2 ruin.
pub const T_SITE: usize = 2;
/// 1 when a city owned by another player stands on the tile (the capital included).
pub const T_ENEMY_CITY: usize = 3;
/// The mover's slot (1..7) of the player this tile was placed as capital for, 0 when it is no one's capital or the
/// mover's own (an explored tile).
pub const T_ENEMY_CAPITAL: usize = 4;
/// 0 no unit, 1 a unit of another player, 2 a Nature unit (an animal).
pub const T_UNIT: usize = 5;

/// Labels of one position, `[h, w, TRUTH_K]` row major, and the mask of hidden tiles `[h, w]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truth {
    pub h: usize,
    pub w: usize,
    pub labels: Vec<i16>,
    /// 1 where the mover has not explored the tile (exactly the tiles with `unexplored = 1` in the encoding).
    pub mask: Vec<u8>,
}

fn clamp(v: i32, vocab: i16) -> i16 {
    if v >= 0 && v < i32::from(vocab) {
        v as i16
    } else {
        0
    }
}

/// Labels for every tile of the map as if all of it were explored. `truth` keeps the hidden ones.
pub fn truth_all(e: &Engine, me: u8, slots: &SlotMap) -> Vec<i16> {
    let (w, h) = (e.width() as usize, e.height() as usize);
    let n = w * h;
    let mut labels = vec![0i16; n * TRUTH_K];
    let mut unit_class = vec![0i16; n];
    for u in &e.s.units {
        let Some(k) = e.tile_index(u.coordinates) else { continue };
        if u.owner == me {
            continue;
        }
        let class = if u.owner == NATURE { 2 } else { 1 };
        unit_class[k] = unit_class[k].max(class);
    }
    for k in 0..n {
        let t = &e.s.map.tiles[k];
        let row = &mut labels[k * TRUTH_K..(k + 1) * TRUTH_K];
        row[T_TERRAIN] = clamp(t.terrain, TRUTH_VOCABS[T_TERRAIN]);
        row[T_RESOURCE] = clamp(t.resource, TRUTH_VOCABS[T_RESOURCE]);
        let is_city = t.improvement == 1;
        row[T_SITE] = if is_city && t.owner == 0 {
            1
        } else if t.improvement == 2 || t.has_ruin {
            2
        } else {
            0
        };
        row[T_ENEMY_CITY] = i16::from(is_city && t.owner != 0 && t.owner != me && t.owner != NATURE);
        if t.capital_of != me {
            row[T_ENEMY_CAPITAL] = clamp(i32::from(slots.tile_value(t.capital_of)) - 1, TRUTH_VOCABS[T_ENEMY_CAPITAL]);
        }
        row[T_UNIT] = unit_class[k];
    }
    labels
}

/// The truth labels of the position for `me`: labels on the tiles `me` has not explored, zeros elsewhere.
pub fn truth(e: &Engine, me: u8, slots: &SlotMap) -> Truth {
    let (w, h) = (e.width() as usize, e.height() as usize);
    let mut labels = truth_all(e, me, slots);
    let mut mask = vec![0u8; w * h];
    for k in 0..w * h {
        if e.is_explored(me, k) {
            labels[k * TRUTH_K..(k + 1) * TRUTH_K].fill(0);
        } else {
            mask[k] = 1;
        }
    }
    Truth { h, w, labels, mask }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::*;
    use crate::game::play_ai_game;
    use crate::setup::{GameSetup, SetupSpec, MIGHT};
    use polytopia_sim::engine::Command;

    /// Positions along a few retail games (Domination and Might, 2 and 4 players), as (engine, mover) clones.
    fn positions() -> Vec<(Engine, u8)> {
        let mut out = Vec::new();
        for (seed, mode, players) in [(11, MIGHT, vec![2usize]), (12, crate::setup::DOMINATION, vec![2]), (13, MIGHT, vec![4])] {
            let spec = SetupSpec { mode, ..SetupSpec::default() }.with_players(players, vec![]).unwrap();
            let mut setup = GameSetup::sample(seed, &spec);
            setup.turn_limit = 14;
            let mut i = 0;
            play_ai_game(&setup, |e: &mut Engine, _c: &Command| {
                i += 1;
                if i % 40 == 1 && e.current_player() != NATURE {
                    out.push((e.clone(), e.current_player()));
                }
            });
        }
        out
    }

    #[test]
    fn mask_is_exactly_the_unexplored_tiles_of_the_encoding() {
        let mut hidden_seen = 0;
        for (e, me) in positions() {
            let slots = SlotMap::new(&e, me);
            let obs = encode(&e, me, &slots, &legal_actions(&e));
            let t = truth(&e, me, &slots);
            assert_eq!((t.h, t.w), (obs.h, obs.w));
            for k in 0..t.h * t.w {
                let unexplored = obs.tiles[k * F_TILE + C_UNEXPLORED];
                assert_eq!(i16::from(t.mask[k]), unexplored, "tile {k}");
                let row = &t.labels[k * TRUTH_K..(k + 1) * TRUTH_K];
                if t.mask[k] == 0 {
                    assert!(row.iter().all(|&v| v == 0), "explored tile {k} carries labels {row:?}");
                }
                hidden_seen += usize::from(t.mask[k]);
            }
        }
        assert!(hidden_seen > 0, "the test positions had no fog");
    }

    #[test]
    fn labels_match_the_full_state() {
        let (mut cities, mut capitals, mut units, mut villages, mut resources) = (0, 0, 0, 0, 0);
        for (e, me) in positions() {
            let slots = SlotMap::new(&e, me);
            let t = truth(&e, me, &slots);
            for k in 0..t.h * t.w {
                if t.mask[k] == 0 {
                    continue;
                }
                let tile = &e.s.map.tiles[k];
                let row = &t.labels[k * TRUTH_K..(k + 1) * TRUTH_K];
                assert_eq!(i32::from(row[T_TERRAIN]), tile.terrain);
                assert_eq!(i32::from(row[T_RESOURCE]), tile.resource);
                let here: Vec<_> = e.s.units.iter().filter(|u| e.tile_index(u.coordinates) == Some(k)).collect();
                let want_unit = if here.iter().any(|u| u.owner != NATURE && u.owner != me) {
                    1
                } else if here.iter().any(|u| u.owner == NATURE) {
                    2
                } else {
                    0
                };
                assert_eq!(row[T_UNIT], want_unit, "unit on tile {k}");
                if tile.improvement == 1 && tile.owner == 0 {
                    assert_eq!(row[T_SITE], 1);
                    assert_eq!(row[T_ENEMY_CITY], 0);
                    villages += 1;
                }
                if tile.improvement == 1 && tile.owner != 0 {
                    assert_eq!(row[T_ENEMY_CITY], 1, "city of player {} on tile {k}", tile.owner);
                    assert_ne!(tile.owner, me, "my own city is never hidden");
                    cities += 1;
                }
                if tile.capital_of != 0 {
                    let slot = slots.slot_of(tile.capital_of).unwrap();
                    assert!((1..=7).contains(&slot));
                    assert_eq!(row[T_ENEMY_CAPITAL], slot as i16, "capital on tile {k}");
                    capitals += 1;
                } else {
                    assert_eq!(row[T_ENEMY_CAPITAL], 0);
                }
                units += usize::from(want_unit > 0);
                resources += usize::from(tile.resource != 0);
                for (v, vocab) in row.iter().zip(TRUTH_VOCABS) {
                    assert!((0..vocab).contains(v));
                }
            }
        }
        assert!(cities > 0 && capitals > 0 && units > 0 && villages > 0 && resources > 0, "{cities} {capitals} {units} {villages} {resources}");
    }

    /// On the tiles the mover has explored, the full-map labels agree with what the encoder shows.
    #[test]
    fn labels_agree_with_the_encoder_on_explored_tiles() {
        let mut checked = 0;
        for (e, me) in positions() {
            let slots = SlotMap::new(&e, me);
            let obs = encode(&e, me, &slots, &legal_actions(&e));
            let all = truth_all(&e, me, &slots);
            for k in 0..obs.h * obs.w {
                let c = &obs.tiles[k * F_TILE..(k + 1) * F_TILE];
                if c[C_UNEXPLORED] == 1 {
                    continue;
                }
                let l = &all[k * TRUTH_K..(k + 1) * TRUTH_K];
                assert_eq!(l[T_TERRAIN], c[C_TERRAIN]);
                if c[C_RESOURCE] != 0 {
                    assert_eq!(l[T_RESOURCE], c[C_RESOURCE]);
                }
                let cap = c[C_CAPITAL_OF]; // 0 none, 1 + slot
                assert_eq!(l[T_ENEMY_CAPITAL], if cap >= 2 { cap - 1 } else { 0 }, "capital of tile {k}");
                let owner = c[C_OWNER];
                let enemy_city = c[C_IS_CITY] == 1 && owner >= 2 && owner <= 8;
                assert_eq!(l[T_ENEMY_CITY] == 1, enemy_city, "city on tile {k}");
                if c[C_UNIT_OWNER] >= 2 {
                    assert!(l[T_UNIT] >= 1, "a visible foreign unit on tile {k}");
                }
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    /// Fog changes nothing in the observation: rewriting a hidden tile changes the truth and leaves the encoding as
    /// it was, and computing the truth does not touch the engine.
    #[test]
    fn hidden_tiles_do_not_leak_into_the_encoding() {
        let mut flipped = 0;
        for (e, me) in positions() {
            let slots = SlotMap::new(&e, me);
            let legal = legal_actions(&e);
            let before = encode(&e, me, &slots, &legal);
            let t = truth(&e, me, &slots);
            let after = encode(&e, me, &slots, &legal);
            assert_eq!((&before.tiles, &before.globals), (&after.tiles, &after.globals));
            let Some(k) = (0..t.h * t.w).find(|&k| t.mask[k] == 1 && e.s.map.tiles[k].terrain != 3) else { continue };
            let mut other = e.clone();
            other.s.map.tiles[k].terrain = 3;
            other.s.map.tiles[k].resource = 2;
            let obs = encode(&other, me, &slots, &legal);
            assert_eq!((&before.tiles, &before.globals), (&obs.tiles, &obs.globals), "tile {k} leaked");
            let t2 = truth(&other, me, &slots);
            assert_ne!(t.labels, t2.labels);
            assert_eq!(t2.labels[k * TRUTH_K + T_TERRAIN], 3);
            assert_eq!(t2.labels[k * TRUTH_K + T_RESOURCE], 2);
            flipped += 1;
        }
        assert!(flipped > 0);
    }
}
