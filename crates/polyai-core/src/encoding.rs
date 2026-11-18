//! Observation encoding (encoding.md sections 2 and 3).

use crate::slots::{SlotMap, N_OTHER_SLOTS};
use polytopia_sim::engine::{Command, Engine, NATURE};
use polytopia_sim::{CityReward, DiplomacyRelationState};

pub const ENCODING_VERSION: i32 = 2;

pub const N_CAT: usize = 9;
pub const N_NUM: usize = 32;
pub const F_TILE: usize = N_CAT + N_NUM;

pub const CAT_NAMES: [&str; N_CAT] =
    ["terrain", "climate", "resource", "owner", "improvement", "capital_of", "unit_type", "unit_owner", "passenger_type"];
pub const CAT_VOCABS: [i16; N_CAT] = [10, 19, 11, 10, 56, 10, 63, 10, 63];
pub const NUM_NAMES: [&str; N_NUM] = [
    "unexplored", "flooded", "swamped", "tentacle", "algae", "foam", "has_road", "has_route", "has_ruin", "ruling_city_mine",
    "in_my_border", "improvement_level", "is_city", "is_capital", "city_level", "population", "population_needed", "walls",
    "park", "workshop", "border_size", "connected_to_capital", "under_siege", "health", "veteran", "moved", "attacked",
    "frozen", "poisoned", "boosted", "invisible", "can_act",
];
pub const NUM_DIVISORS: [f32; N_NUM] = [
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 100.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];

// Column indices.
pub const C_TERRAIN: usize = 0;
pub const C_CLIMATE: usize = 1;
pub const C_RESOURCE: usize = 2;
pub const C_OWNER: usize = 3;
pub const C_IMPROVEMENT: usize = 4;
pub const C_CAPITAL_OF: usize = 5;
pub const C_UNIT_TYPE: usize = 6;
pub const C_UNIT_OWNER: usize = 7;
pub const C_PASSENGER: usize = 8;
const N0: usize = N_CAT; // first numeric column
pub const C_UNEXPLORED: usize = N0;
pub const C_FLOODED: usize = N0 + 1;
pub const C_HAS_ROAD: usize = N0 + 6;
pub const C_HAS_ROUTE: usize = N0 + 7;
pub const C_HAS_RUIN: usize = N0 + 8;
pub const C_RULING_MINE: usize = N0 + 9;
pub const C_IN_MY_BORDER: usize = N0 + 10;
pub const C_IMP_LEVEL: usize = N0 + 11;
pub const C_IS_CITY: usize = N0 + 12;
pub const C_IS_CAPITAL: usize = N0 + 13;
pub const C_CITY_LEVEL: usize = N0 + 14;
pub const C_POPULATION: usize = N0 + 15;
pub const C_POP_NEEDED: usize = N0 + 16;
pub const C_WALLS: usize = N0 + 17;
pub const C_PARK: usize = N0 + 18;
pub const C_WORKSHOP: usize = N0 + 19;
pub const C_BORDER: usize = N0 + 20;
pub const C_CONNECTED: usize = N0 + 21;
pub const C_SIEGE: usize = N0 + 22;
pub const C_HEALTH: usize = N0 + 23;
pub const C_VETERAN: usize = N0 + 24;
pub const C_MOVED: usize = N0 + 25;
pub const C_ATTACKED: usize = N0 + 26;
pub const C_FROZEN: usize = N0 + 27;
pub const C_POISONED: usize = N0 + 28;
pub const C_BOOSTED: usize = N0 + 29;
pub const C_INVISIBLE: usize = N0 + 30;
pub const C_CAN_ACT: usize = N0 + 31;

/// Engine tribe idx of the 13 supported tribes: 12 normal ones and Cymanti.
pub const TRIBES: [i32; 13] = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17];
pub const N_TRIBES: usize = TRIBES.len();
pub const N_TECHS: usize = 51;
pub const N_SCALARS: usize = 12;
pub const PER_SLOT: usize = 24;
pub const N_TRIGGER_KINDS: usize = 3;

pub const G_TRIBE: usize = 0;
pub const G_SCALARS: usize = G_TRIBE + N_TRIBES;
pub const G_TECHS: usize = G_SCALARS + N_SCALARS;
pub const G_SLOTS: usize = G_TECHS + N_TECHS;
pub const G_TRIGGER: usize = G_SLOTS + N_OTHER_SLOTS * PER_SLOT;
pub const G_DIM: usize = G_TRIGGER + N_TRIGGER_KINDS;

pub fn tribe_index(tribe: i32) -> Option<usize> {
    TRIBES.iter().position(|&t| t == tribe)
}

/// One encoded position.
#[derive(Debug, Clone)]
pub struct Obs {
    pub h: usize,
    pub w: usize,
    /// `[h, w, F_TILE]`, row major.
    pub tiles: Vec<i16>,
    pub globals: Vec<f32>,
}

/// The unit a command acts with, if it is a unit command.
pub fn command_unit(c: &Command) -> Option<u32> {
    use Command::*;
    match *c {
        Move { unit, .. } | Attack { unit, .. } | Capture { unit, .. } | Promote { unit } | Recover { unit } | HealOthers { unit }
        | Explode { unit } | Disband { unit } | Upgrade { unit, .. } | ExamineRuins { unit } | Stay { unit } | Swarm { unit }
        | Boost { unit } | Disembark { unit } => Some(unit),
        _ => None,
    }
}

/// The commands the model chooses from: [`engine_actions`] less builds on tiles the mover has not explored. The
/// server accepts those, but offering them would reveal hidden terrain (road against water) to the model.
pub fn legal_actions(e: &Engine) -> Vec<Command> {
    let me = e.current_player();
    let mut v = engine_actions(e);
    v.retain(|c| !matches!(*c, Command::Build { at, .. } if !e.is_explored_at(me, at)));
    v
}

/// The engine's legal list plus a `Stay` for every own unit that can move, inserted before the final `EndTurn`
/// (encoding.md section 4). With a trigger pending it is just the choices. Use it to check recorded commands,
/// which may include builds the model is never offered.
pub fn engine_actions(e: &Engine) -> Vec<Command> {
    let mut v = e.legal_commands();
    let me = e.current_player();
    if e.is_over() || e.pending_trigger(me).is_some() {
        return v;
    }
    let at = v.iter().rposition(|c| matches!(c, Command::EndTurn)).unwrap_or(v.len());
    let stays: Vec<Command> = e.s.units.iter().filter(|u| u.owner == me && e.can_move(u)).map(|u| Command::Stay { unit: u.id }).collect();
    v.splice(at..at, stays);
    v
}

/// Encode the position for the player to move (`me` must be the current player for `legal` to make sense).
pub fn encode(e: &Engine, me: u8, slots: &SlotMap, legal: &[Command]) -> Obs {
    let (w, h) = (e.width() as usize, e.height() as usize);
    let n = w * h;
    let mut tiles = vec![0i16; n * F_TILE];

    // The first unit on a tile is the one shown; a second unit of the same owner (a Giant with a Cloak) shows as the
    // passenger.
    let mut unit_at: Vec<Option<usize>> = vec![None; n];
    let mut second_at: Vec<Option<usize>> = vec![None; n];
    for (i, u) in e.s.units.iter().enumerate() {
        if let Some(k) = e.tile_index(u.coordinates) {
            if unit_at[k].is_none() {
                unit_at[k] = Some(i);
            } else if second_at[k].is_none() {
                second_at[k] = Some(i);
            }
        }
    }
    let mut rt_at: Vec<Option<usize>> = vec![None; n];
    for (i, r) in e.s.improvements.iter().enumerate() {
        if let Some(slot) = rt_at.get_mut(r.tile as usize) {
            if slot.is_none() {
                *slot = Some(i);
            }
        }
    }
    let can_act: std::collections::HashSet<u32> = legal.iter().filter_map(command_unit).collect();
    let clamp = |v: i32, vocab: i16| -> i16 { if v >= 0 && v < i32::from(vocab) { v as i16 } else { 0 } };

    for k in 0..n {
        let row = &mut tiles[k * F_TILE..(k + 1) * F_TILE];
        if !e.is_explored(me, k) {
            row[C_UNEXPLORED] = 1;
            continue;
        }
        let t = &e.s.map.tiles[k];
        row[C_TERRAIN] = clamp(t.terrain, CAT_VOCABS[C_TERRAIN]);
        row[C_CLIMATE] = clamp(t.climate, CAT_VOCABS[C_CLIMATE]);
        row[C_RESOURCE] = e.get_resource(t.coordinates, me).map_or(0, |r| clamp(r, CAT_VOCABS[C_RESOURCE]));
        row[C_OWNER] = slots.tile_value(t.owner);
        row[C_IMPROVEMENT] = clamp(t.improvement, CAT_VOCABS[C_IMPROVEMENT]);
        row[C_CAPITAL_OF] = slots.tile_value(t.capital_of);
        for bit in 0..5 {
            row[C_FLOODED + bit] = i16::from(t.effects & (1 << (bit + 1)) != 0);
        }
        row[C_HAS_ROAD] = i16::from(t.has_road);
        row[C_HAS_ROUTE] = i16::from(t.has_route);
        row[C_HAS_RUIN] = i16::from(t.has_ruin);
        row[C_RULING_MINE] = i16::from(e.tile(t.ruling_city).is_some_and(|c| c.owner == me));
        row[C_IN_MY_BORDER] = i16::from(t.owner == me);
        let is_city = t.improvement == 1;
        row[C_IS_CITY] = i16::from(is_city);
        row[C_IS_CAPITAL] = i16::from(t.is_capital);
        if t.improvement != 0 && !is_city {
            row[C_IMP_LEVEL] = e.improvement_level(t.coordinates).clamp(0, 255) as i16;
        }
        let rt = rt_at[k].map(|i| &e.s.improvements[i]);
        if is_city {
            if let Some(r) = rt {
                row[C_CITY_LEVEL] = r.level.min(255) as i16;
                row[C_WALLS] = i16::from(r.rewards.contains(&CityReward::CityWall));
                row[C_PARK] = i16::from(r.rewards.contains(&CityReward::Park));
                row[C_WORKSHOP] = i16::from(r.rewards.contains(&CityReward::Workshop));
                row[C_BORDER] = r.border_size.min(255) as i16;
                if t.owner == me {
                    row[C_POPULATION] = r.xp.clamp(0, 255);
                    row[C_POP_NEEDED] = (r.level as i16 + 1).min(255);
                    row[C_CONNECTED] = i16::from(r.connected_to_capital_of_player == me);
                }
            }
        }
        if let Some(u) = unit_at[k].map(|i| &e.s.units[i]) {
            if !u.invisible || u.owner == me || e.at_peace(me, u.owner) {
                row[C_UNIT_TYPE] = clamp(u.type_idx, CAT_VOCABS[C_UNIT_TYPE]);
                row[C_UNIT_OWNER] = slots.tile_value(u.owner);
                if let Some(p) = u.passenger.as_ref() {
                    row[C_PASSENGER] = clamp(p.type_idx, CAT_VOCABS[C_PASSENGER]);
                } else if let Some(v) = second_at[k].map(|i| &e.s.units[i]).filter(|v| v.owner == u.owner && (!v.invisible || v.owner == me)) {
                    row[C_PASSENGER] = clamp(v.type_idx, CAT_VOCABS[C_PASSENGER]);
                }
                let max = e.max_health(u).max(1);
                row[C_HEALTH] = (100 * i32::from(u.health) / max).clamp(0, 1000) as i16;
                row[C_VETERAN] = i16::from(u.promotion > 0);
                row[C_MOVED] = i16::from(u.moved);
                row[C_ATTACKED] = i16::from(u.attacked);
                row[C_FROZEN] = i16::from(u.frozen);
                row[C_POISONED] = i16::from(u.poisoned);
                row[C_BOOSTED] = i16::from(u.boosted);
                row[C_INVISIBLE] = i16::from(u.invisible);
                row[C_CAN_ACT] = i16::from(u.owner == me && (can_act.contains(&u.id) || second_at[k].is_some_and(|i| can_act.contains(&e.s.units[i].id))));
                if is_city && t.owner != 0 && u.owner != t.owner {
                    row[C_SIEGE] = 1;
                }
            }
        }
    }
    Obs { h, w, tiles, globals: encode_globals(e, me, slots) }
}

fn encode_globals(e: &Engine, me: u8, slots: &SlotMap) -> Vec<f32> {
    let mut g = vec![0f32; G_DIM];
    let player = e.player(me);
    if let Some(i) = player.and_then(|p| tribe_index(p.tribe)) {
        g[G_TRIBE + i] = 1.0;
    }
    let s = &e.s.settings;
    let turn = e.s.current_turn as f32;
    let sc = G_SCALARS;
    g[sc] = player.map_or(0, |p| p.currency) as f32 / 10.0;
    g[sc + 1] = e.income(me) as f32 / 10.0;
    g[sc + 2] = e.score(me) as f32 / 1000.0;
    g[sc + 3] = turn / 50.0;
    // The turn limit of a Domination game is a compute cap that players do not have, so the model never sees it:
    // both values stay 0 (encoding.md section 3, version 2).
    if s.turn_limit > 0 && s.rules_game_mode != crate::setup::DOMINATION {
        g[sc + 4] = (s.turn_limit as f32 - turn).max(0.0) / 50.0;
        g[sc + 5] = 1.0;
    }
    g[sc + 6] = s.rules_game_mode as f32 / 5.0;
    g[sc + 7] = e.width() as f32 / 30.0;
    g[sc + 8] = e.living_players().len() as f32 / 8.0;
    g[sc + 9] = (e.s.players.len().saturating_sub(1)) as f32 / 8.0;
    g[sc + 10] = e.city_count(me) as f32 / 8.0;
    g[sc + 11] = e.s.units.iter().filter(|u| u.owner == me).count() as f32 / 32.0;
    for &t in e.techs(me) {
        if let Some(x) = g.get_mut(G_TECHS + t as usize).filter(|_| (0..N_TECHS as i32).contains(&t)) {
            *x = 1.0;
        }
    }
    for slot in 1..=N_OTHER_SLOTS {
        let Some(id) = slots.id_of(slot) else { continue };
        let b = G_SLOTS + (slot - 1) * PER_SLOT;
        g[b] = 1.0;
        if !e.knows_player(me, id) {
            continue;
        }
        g[b + 1] = f32::from(u8::from(e.is_alive(id)));
        g[b + 2] = 1.0;
        if let Some(i) = e.player(id).and_then(|p| tribe_index(p.tribe)) {
            g[b + 3 + i] = 1.0;
        }
        g[b + 16] = e.score(id) as f32 / 1000.0;
        g[b + 17] = e.city_count(id) as f32 / 8.0;
        let rel = polytopia_sim::rules::diplomacy::get_relation(&e.s, me, id).state;
        let r = match rel {
            DiplomacyRelationState::Neutral => 0,
            DiplomacyRelationState::Peace => 1,
            DiplomacyRelationState::War => 2,
            DiplomacyRelationState::CoolDown => 3,
        };
        g[b + 18 + r] = 1.0;
        g[b + 22] = f32::from(u8::from(polytopia_sim::rules::diplomacy::has_embassy_with(&e.s, me, id)));
        g[b + 23] = f32::from(u8::from(polytopia_sim::rules::diplomacy::has_embassy_with(&e.s, id, me)));
    }
    let _ = NATURE;
    match e.pending_trigger(me) {
        None => g[G_TRIGGER] = 1.0,
        Some(t) if t.peace_from.is_some() => g[G_TRIGGER + 2] = 1.0,
        Some(_) => g[G_TRIGGER + 1] = 1.0,
    }
    g
}
