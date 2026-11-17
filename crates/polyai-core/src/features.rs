//! Engine-computed features: quantities the engine already knows how to work out, given to the network so that it
//! does not have to learn them (docs/encoding.md section 12).
//!
//! Everything is seen from the mover and uses only what the mover can see:
//!
//! * a unit counts when the encoder shows it (explored tile, not a cloaked enemy);
//! * features of tiles the mover has not explored are 0 (the distance columns are the exception: they are computed
//!   from known capitals and cities only);
//! * the defence bonus that an enemy gets from its techs is not visible, so every combat figure gives the defender
//!   the mover's own tech bonus for the terrain it stands on (own units: their real bonus);
//! * an enemy's reach is the engine's movement options for that unit, asked of [`honest_view`]: a copy of the game in
//!   which only what the mover has explored is known to anybody, units the mover cannot see are gone, and every
//!   other player has the mover's techs (their real ones are hidden).
//!
//! The engine functions behind the figures are `get_battle_results` (what `Attack` applies), `unit_info`,
//! `movement_options_ignoring_state` (reach next turn) and `income`. A test compares every figure with direct engine
//! calls on positions from AI games.

use crate::encoding::command_unit;
use crate::slots::SlotMap;
use polytopia_sim::engine::{Command, Engine, Wc, NATURE};
use polytopia_sim::rules::combat::{
    defence_bonus, get_battle_results, BattleQuery, CombatUnit, DefenceBonusInput, CITY_IMPROVEMENT, FORTIFY_ABILITY,
};
use polytopia_sim::rules::path::{attack_options, tile_being_captured};
use polytopia_sim::{CityReward, UnitState};

// ---------- column layouts ----------

/// Per-action columns, `float32`, one row per entry of the legal list (zeros for commands that are not unit commands
/// of a unit that can fight or move).
pub const A_F: usize = 13;
pub const A_NAMES: [&str; A_F] = [
    "dmg_dealt",
    "dmg_taken",
    "kills",
    "attacker_dies",
    "target_hp_after",
    "trade",
    "captures_city",
    "captures_village",
    "end_threat_n",
    "end_threat_max",
    "end_threat_sum",
    "end_adjacent_enemies",
    "end_could_die",
];
pub const A_DMG_DEALT: usize = 0;
pub const A_DMG_TAKEN: usize = 1;
pub const A_KILLS: usize = 2;
pub const A_ATTACKER_DIES: usize = 3;
pub const A_TARGET_HP_AFTER: usize = 4;
pub const A_TRADE: usize = 5;
pub const A_CAPTURES_CITY: usize = 6;
pub const A_CAPTURES_VILLAGE: usize = 7;
pub const A_END_THREAT_N: usize = 8;
pub const A_END_THREAT_MAX: usize = 9;
pub const A_END_THREAT_SUM: usize = 10;
pub const A_END_ADJACENT: usize = 11;
pub const A_END_COULD_DIE: usize = 12;

/// Per-tile columns, `int16`, `[h, w, T_F]`; the model multiplies by `1 / divisor`.
pub const T_F: usize = 7;
pub const TF_NAMES: [&str; T_F] = ["threat_n", "threat_dmg", "reach_n", "attack_reach_n", "siege", "dist_enemy_capital", "dist_enemy_city"];
pub const TF_DIVISORS: [f32; T_F] = [4.0, 100.0, 4.0, 4.0, 3.0, 30.0, 30.0];
pub const TF_THREAT_N: usize = 0;
pub const TF_THREAT_DMG: usize = 1;
pub const TF_REACH_N: usize = 2;
pub const TF_ATTACK_REACH_N: usize = 3;
pub const TF_SIEGE: usize = 4;
pub const TF_DIST_CAPITAL: usize = 5;
pub const TF_DIST_CITY: usize = 6;

/// Extra globals, `float32[G_F]`.
pub const G_F: usize = 14;
pub const GF_NAMES: [&str; G_F] = [
    "stars_next",
    "techs_researchable",
    "techs_affordable",
    "cheapest_tech",
    "techs_affordable_next",
    "hostile_units_slot1",
    "hostile_units_slot2",
    "hostile_units_slot3",
    "hostile_units_slot4",
    "hostile_units_slot5",
    "hostile_units_slot6",
    "hostile_units_slot7",
    "my_cities_threatened",
    "my_units_threatened",
];
pub const GF_HOSTILE0: usize = 5;

const COUNT_CAP: i16 = 15;
/// Health and damage figures are in tenths of a hit point scale: a full-health Warrior has 100.
const HP_SCALE: f32 = 100.0;

/// The extra inputs of one position.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub h: usize,
    pub w: usize,
    /// `[legal.len(), A_F]`, row major, aligned with the legal list the features were computed for.
    pub actions: Vec<f32>,
    /// `[h, w, T_F]`, row major.
    pub tiles: Vec<i16>,
    pub globals: Vec<f32>,
}

// ---------- helpers ----------

/// An enemy unit the mover can see and is not at peace with (Nature is not an enemy here: animals do not attack).
fn hostile_visible(e: &Engine, me: u8, u: &UnitState) -> bool {
    if u.owner == me || u.owner == NATURE || u.owner == 0 || e.at_peace(me, u.owner) {
        return false;
    }
    let Some(k) = e.tile_index(u.coordinates) else { return false };
    e.is_explored(me, k)
}

/// A copy of the game that holds only what `me` can know, for asking the engine about the movement of others: the
/// engine reads the exploration, the techs and the (hidden) units of the unit's own player when it works out where
/// that unit can go.
pub fn honest_view(e: &Engine, me: u8) -> Engine {
    let mut v = e.clone();
    let everyone: Vec<u8> = v.s.players.iter().map(|p| p.id).chain([NATURE]).collect();
    for k in 0..v.s.fog.len() {
        v.s.fog[k].explorers = if e.is_explored(me, k) { everyone.as_slice().into() } else { Default::default() };
    }
    v.s.units.retain(|u| {
        u.owner == me || e.tile_index(u.coordinates).is_some_and(|k| e.is_explored(me, k)) && (!u.invisible || e.at_peace(me, u.owner))
    });
    let mine = e.techs(me).to_vec();
    for r in v.s.tech.iter_mut().filter(|r| r.player != me) {
        r.tech = mine.clone();
    }
    v
}

/// Combat figures of a unit as the mover can know them, with the unit standing on `at`: the defence bonus is the one
/// the mover's own techs unlock for that terrain (the real bonus for the mover's own units, the best visible guess
/// for the others), with the city and wall bonus of the tile.
pub fn visible_combat_unit(e: &Engine, me: u8, unit: &UnitState, at: Wc) -> CombatUnit {
    let mut cu = e.combat_unit(unit);
    let tile = e.tile(at);
    let mut input = DefenceBonusInput {
        version: e.s.version,
        poisoned: unit.poisoned,
        tile_present: tile.is_some(),
        owner: unit.owner,
        player_resolved: e.player(unit.owner).is_some(),
        unlocked_defence_bonus: 1,
        improvement_known: false,
        is_city: false,
        same_owner: false,
        has_fortify: e.has_ability(unit, FORTIFY_ABILITY),
        has_city_wall: false,
    };
    if let Some(tile) = tile {
        input.unlocked_defence_bonus = e.t.unlocked_defence_bonus(e.techs(me), tile.terrain);
        input.improvement_known = tile.improvement != 0;
        input.is_city = tile.improvement == CITY_IMPROVEMENT;
        input.same_owner = tile.owner == unit.owner;
        input.has_city_wall = e.improvement_rt(tile.coordinates).is_some_and(|r| r.rewards.contains(&CityReward::CityWall));
    }
    let bonus = defence_bonus(&input);
    cu.defence = cu.raw_defence * bonus;
    cu.defence_bonus = bonus;
    cu
}

/// Damage a unit deals to a defender, ignoring retaliation.
fn damage(e: &Engine, attacker: &CombatUnit, defender: &CombatUnit) -> i32 {
    let q = BattleQuery {
        game_version: e.s.version,
        account_for_peace: false,
        has_active_peace_treaty: false,
        simple_data: true,
        attacker_in_attack_options: false,
        can_path_onto_defender: false,
    };
    get_battle_results(attacker, defender, &q).attack_damage.max(0)
}

/// A full-health Warrior on open ground: the yardstick for the per-tile threat figure.
fn reference_defender() -> CombatUnit {
    CombatUnit {
        health: 100,
        max_health: 100,
        attack: 20,
        defence: 20,
        defence_bonus: 10,
        raw_attack: 2,
        raw_defence: 2,
        range: 1,
        weapon: 0,
        has_stiff: false,
        has_convert: false,
        has_surprise: false,
        has_freeze: false,
        has_tentacles: false,
        has_leader: false,
        has_infiltrate: false,
        frozen: false,
        invisible: false,
    }
}

/// The result of attacking the unit on `target` with `attacker` from where it stands: the engine's own calculation
/// (`get_battle_results` with the query `Engine::battle_damage_query` builds), with the defender's tech bonus as the
/// mover can know it. Returns `(attack_damage, retaliation_damage)`; retaliation is 0 when the defender dies.
pub fn attack_preview(e: &Engine, me: u8, attacker: &UnitState, defender: &UnitState) -> (i32, i32) {
    let a = e.combat_unit(attacker);
    let d = visible_combat_unit(e, me, defender, defender.coordinates);
    let in_options = e.unit_info(defender).is_some_and(|info| {
        let mut probe = defender.clone();
        probe.leader = 0;
        attack_options(&e.s, &probe, e.unit_range(defender), true, info.data.attack, &info.abilities, true, |t| tile_being_captured(&e.s, t), |_| false)
            .contains(&attacker.coordinates)
    });
    let q = BattleQuery {
        game_version: e.s.version,
        account_for_peace: true,
        has_active_peace_treaty: defender.owner == attacker.owner || e.at_peace(defender.owner, attacker.owner),
        simple_data: true,
        attacker_in_attack_options: in_options,
        can_path_onto_defender: false,
    };
    let r = get_battle_results(&a, &d, &q);
    (r.attack_damage.max(0), r.retaliation_damage.max(0))
}

struct Ctx<'a> {
    e: &'a Engine,
    me: u8,
    explored: Vec<bool>,
    /// Indices into `e.s.units` of the hostile visible units.
    hostile: Vec<usize>,
    /// Per hostile unit (parallel to `hostile`): combat figures and the damage it does to the reference defender.
    hostile_cu: Vec<CombatUnit>,
    hostile_ref_dmg: Vec<i32>,
    /// Per tile: positions in `hostile` of the units that can attack the tile next turn.
    threats: Vec<Vec<u16>>,
}

impl<'a> Ctx<'a> {
    fn idx(&self, c: Wc) -> Option<usize> {
        self.e.tile_index(c)
    }

    /// Damage the threats on `tile` could do to `unit` standing there with `health` (strongest, total, count),
    /// leaving out the unit with id `except`.
    fn end_threat(&self, unit: &UnitState, health: i32, tile: Wc, except: Option<u32>) -> (usize, i32, i32) {
        let Some(k) = self.idx(tile) else { return (0, 0, 0) };
        let mut cu = visible_combat_unit(self.e, self.me, unit, tile);
        cu.health = health;
        let (mut n, mut max, mut sum) = (0, 0, 0);
        for &h in &self.threats[k] {
            let u = &self.e.s.units[self.hostile[h as usize]];
            if Some(u.id) == except {
                continue;
            }
            let d = damage(self.e, &self.hostile_cu[h as usize], &cu);
            n += 1;
            max = max.max(d);
            sum += d;
        }
        (n, max, sum)
    }

    fn adjacent_hostiles(&self, tile: Wc, except: Option<u32>) -> usize {
        self.hostile
            .iter()
            .filter(|&&i| {
                let u = &self.e.s.units[i];
                Some(u.id) != except && Engine::chebyshev(u.coordinates, tile) <= 1
            })
            .count()
    }
}

fn cap(v: usize) -> i16 {
    (v as i16).min(COUNT_CAP)
}

/// Compute the features of the position for `me`; `legal` is the legal list the actions are aligned with
/// ([`crate::legal_actions`]).
pub fn features(e: &Engine, me: u8, slots: &SlotMap, legal: &[Command]) -> Features {
    let (w, h) = (e.width() as usize, e.height() as usize);
    let n = w * h;
    let explored: Vec<bool> = (0..n).map(|k| e.is_explored(me, k)).collect();

    // Hostile visible units and the tiles they can attack next turn.
    let mut hostile = Vec::new();
    for (i, u) in e.s.units.iter().enumerate() {
        if hostile_visible(e, me, u) && (!u.invisible || e.at_peace(me, u.owner)) {
            hostile.push(i);
        }
    }
    let view = honest_view(e, me);
    let reference = reference_defender();
    let hostile_cu: Vec<CombatUnit> = hostile.iter().map(|&i| e.combat_unit(&e.s.units[i])).collect();
    let hostile_ref_dmg: Vec<i32> = hostile_cu.iter().map(|c| damage(e, c, &reference)).collect();
    let mut threats: Vec<Vec<u16>> = vec![Vec::new(); n];
    let mut seen = vec![u16::MAX; n];
    for (hi, &ui) in hostile.iter().enumerate() {
        let u = &e.s.units[ui];
        if e.unit_attack(u) <= 0 || u.leader != 0 {
            continue;
        }
        let range = e.unit_range(u).max(1);
        let mut from = vec![u.coordinates];
        if let Some(vu) = view.unit(u.id) {
            for d in view.movement_options_ignoring_state(vu) {
                if e.tile_index(d).is_some_and(|k| explored[k]) {
                    from.push(d);
                }
            }
        }
        for d in from {
            for c in e.area_iter(d, range) {
                let Some(k) = e.tile_index(c) else { continue };
                if explored[k] && seen[k] != hi as u16 {
                    seen[k] = hi as u16;
                    threats[k].push(hi as u16);
                }
            }
        }
    }
    let ctx = Ctx { e, me, explored, hostile, hostile_cu, hostile_ref_dmg, threats };

    // Our reach: the tiles our units can move onto, and the tiles they can attack after moving.
    let mut reach = vec![0i16; n];
    let mut atk_reach = vec![0i16; n];
    {
        let mut moves: std::collections::HashMap<u32, Vec<Wc>> = std::collections::HashMap::new();
        for c in legal {
            if let Command::Move { unit, to } = *c {
                moves.entry(unit).or_default().push(to);
                if let Some(k) = e.tile_index(to) {
                    reach[k] += 1;
                }
            }
        }
        let mut seen = vec![u32::MAX; n];
        for u in e.s.units.iter().filter(|u| u.owner == me) {
            if e.unit_attack(u) <= 0 || u.leader != 0 || !e.can_attack(u) {
                continue;
            }
            let range = e.unit_range(u).max(1);
            let mut from = vec![u.coordinates];
            if let Some(m) = moves.get(&u.id) {
                from.extend(m.iter().copied());
            }
            for d in from {
                for c in e.area_iter(d, range) {
                    let Some(k) = e.tile_index(c) else { continue };
                    if ctx.explored[k] && seen[k] != u.id {
                        seen[k] = u.id;
                        atk_reach[k] += 1;
                    }
                }
            }
        }
    }

    // Known enemy capitals and cities.
    let mut capitals = Vec::new();
    let mut cities = Vec::new();
    for k in 0..n {
        if !ctx.explored[k] {
            continue;
        }
        let t = &e.s.map.tiles[k];
        if t.improvement != CITY_IMPROVEMENT || t.owner == 0 || t.owner == me || t.owner == NATURE || e.at_peace(me, t.owner) {
            continue;
        }
        cities.push(t.coordinates);
        if t.capital_of == t.owner {
            capitals.push(t.coordinates);
        }
    }
    let nearest = |from: Wc, set: &[Wc]| -> i16 { set.iter().map(|&c| Engine::chebyshev(from, c)).min().map_or(0, |d| (d + 1).min(255) as i16) };

    // Tile features.
    let mut tiles = vec![0i16; n * T_F];
    let (mut cities_threatened, mut units_threatened) = (0usize, 0usize);
    for k in 0..n {
        let c = e.s.map.tiles[k].coordinates;
        let row = &mut tiles[k * T_F..(k + 1) * T_F];
        row[TF_DIST_CAPITAL] = nearest(c, &capitals);
        row[TF_DIST_CITY] = nearest(c, &cities);
        if !ctx.explored[k] {
            continue;
        }
        let th = &ctx.threats[k];
        row[TF_THREAT_N] = cap(th.len());
        row[TF_THREAT_DMG] = th.iter().map(|&x| ctx.hostile_ref_dmg[x as usize]).max().unwrap_or(0).clamp(0, 1000) as i16;
        row[TF_REACH_N] = reach[k].min(COUNT_CAP);
        row[TF_ATTACK_REACH_N] = atk_reach[k].min(COUNT_CAP);
        let t = &e.s.map.tiles[k];
        let unit_here = e.unit_at(c);
        if t.improvement == CITY_IMPROVEMENT && t.owner == me {
            let occupied_by_enemy = unit_here.is_some_and(|u| hostile_visible(e, me, u));
            row[TF_SIEGE] = if occupied_by_enemy {
                3
            } else if !th.is_empty() && unit_here.is_none_or(|u| u.owner != me) {
                2
            } else {
                i16::from(!th.is_empty())
            };
            cities_threatened += usize::from(row[TF_SIEGE] > 0);
        }
        if unit_here.is_some_and(|u| u.owner == me) && !th.is_empty() {
            units_threatened += 1;
        }
    }

    // Action features.
    let mut actions = vec![0f32; legal.len() * A_F];
    for (i, cmd) in legal.iter().enumerate() {
        let row = &mut actions[i * A_F..(i + 1) * A_F];
        let Some(uid) = command_unit(cmd) else { continue };
        if matches!(cmd, Command::Disband { .. } | Command::Explode { .. }) {
            continue;
        }
        let Some(u) = e.unit(uid) else { continue };
        let mut end = u.coordinates;
        let mut health = i32::from(u.health);
        let mut except = None;
        match *cmd {
            Command::Move { to, .. } => end = to,
            Command::Capture { at, .. } => {
                if let Some(t) = e.tile(at) {
                    if t.owner == 0 {
                        row[A_CAPTURES_VILLAGE] = 1.0;
                    } else {
                        row[A_CAPTURES_CITY] = 1.0;
                    }
                }
            }
            Command::Attack { target, .. } => {
                if let Some(d) = e.unit_at(target) {
                    let cu_a = e.combat_unit(u);
                    if !cu_a.has_infiltrate && !cu_a.has_convert {
                        let (dealt, taken) = attack_preview(e, me, u, d);
                        let dh = i32::from(d.health);
                        let kills = dealt >= dh;
                        let dies = taken >= i32::from(u.health);
                        row[A_DMG_DEALT] = dealt as f32 / HP_SCALE;
                        row[A_DMG_TAKEN] = taken as f32 / HP_SCALE;
                        row[A_KILLS] = f32::from(u8::from(kills));
                        row[A_ATTACKER_DIES] = f32::from(u8::from(dies));
                        row[A_TARGET_HP_AFTER] = (dh - dealt).max(0) as f32 / e.max_health(d).max(1) as f32;
                        let (cost_d, cost_a) = (e.full_cost(d) as f32, e.full_cost(u) as f32);
                        let gain = if kills { cost_d } else { cost_d * dealt as f32 / e.max_health(d).max(1) as f32 };
                        let loss = if dies { cost_a } else { cost_a * taken as f32 / e.max_health(u).max(1) as f32 };
                        row[A_TRADE] = (gain - loss) / 10.0;
                        health = i32::from(u.health) - taken;
                        if kills {
                            except = Some(d.id);
                            if e.unit_range(u) == 1 && e.can_path_onto(u, target) {
                                end = target;
                            }
                        }
                        if dies {
                            continue;
                        }
                    }
                }
            }
            _ => {}
        }
        let (tn, tmax, tsum) = ctx.end_threat(u, health, end, except);
        row[A_END_THREAT_N] = tn as f32 / 4.0;
        row[A_END_THREAT_MAX] = tmax as f32 / HP_SCALE;
        row[A_END_THREAT_SUM] = tsum as f32 / HP_SCALE;
        row[A_END_ADJACENT] = ctx.adjacent_hostiles(end, except) as f32 / 4.0;
        row[A_END_COULD_DIE] = f32::from(u8::from(tn > 0 && tsum >= health));
    }

    // Globals.
    let mut globals = vec![0f32; G_F];
    let stars = e.player(me).map_or(0, |p| p.currency);
    let income = e.income(me);
    let researchable = e.researchable(me);
    let prices: Vec<i32> = researchable.iter().map(|&t| e.tech_price(me, t)).collect();
    globals[0] = (stars + income) as f32 / 10.0;
    globals[1] = researchable.len() as f32 / 10.0;
    globals[2] = prices.iter().filter(|&&p| p <= stars).count() as f32 / 10.0;
    globals[3] = prices.iter().copied().min().unwrap_or(0) as f32 / 10.0;
    globals[4] = prices.iter().filter(|&&p| p <= stars + income).count() as f32 / 10.0;
    for &i in &ctx.hostile {
        if let Some(s) = slots.slot_of(e.s.units[i].owner).filter(|s| (1..=7).contains(s)) {
            globals[GF_HOSTILE0 + s - 1] += 1.0 / 16.0;
        }
    }
    globals[12] = cities_threatened as f32 / 8.0;
    globals[13] = units_threatened as f32 / 16.0;
    Features { h, w, actions, tiles, globals }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::legal_actions;
    use crate::game::play_ai_game;
    use crate::setup::{GameSetup, SetupSpec, DOMINATION, MIGHT};
    use polytopia_sim::engine::NATURE;

    /// Positions along a few retail games (Domination and Might, 2 and 4 players), as (engine, mover) clones, every
    /// 25th decision so that fights are among them.
    fn positions() -> Vec<(Engine, u8)> {
        let mut out = Vec::new();
        for (seed, mode, players) in [(11, MIGHT, vec![2usize]), (12, DOMINATION, vec![2]), (13, MIGHT, vec![4]), (14, MIGHT, vec![2])] {
            let spec = SetupSpec { mode, ..SetupSpec::default() }.with_players(players, vec![]).unwrap();
            let mut setup = GameSetup::sample(seed, &spec);
            setup.turn_limit = 30;
            let mut i = 0;
            play_ai_game(&setup, |e: &mut Engine, _c: &Command| {
                i += 1;
                if i % 25 == 1 && e.current_player() != NATURE {
                    out.push((e.clone(), e.current_player()));
                }
            });
        }
        out
    }

    fn feats(e: &Engine, me: u8) -> (Vec<Command>, Features) {
        let slots = SlotMap::new(e, me);
        let legal = legal_actions(e);
        let f = features(e, me, &slots, &legal);
        (legal, f)
    }

    #[test]
    fn shapes_and_ranges() {
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            assert_eq!(f.actions.len(), legal.len() * A_F);
            assert_eq!(f.tiles.len(), f.h * f.w * T_F);
            assert_eq!(f.globals.len(), G_F);
            assert!(f.actions.iter().all(|v| v.is_finite() && v.abs() < 50.0));
            for k in 0..f.h * f.w {
                let row = &f.tiles[k * T_F..(k + 1) * T_F];
                assert!(row.iter().all(|&v| (0..=1000).contains(&v)), "{row:?}");
            }
            assert!(f.globals.iter().all(|v| v.is_finite() && *v >= 0.0));
        }
    }

    /// Attack figures equal the engine's own `rate_battle_input` numbers, which come from `battle_damage_query`,
    /// whenever the defender's real tech bonus is the one we assume (always, for the mover's own units; elsewhere the
    /// figures are allowed to differ only through that bonus).
    #[test]
    fn attack_figures_match_the_engine() {
        let (mut attacks, mut exact, mut kills, mut retaliations) = (0, 0, 0, 0);
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            for (i, c) in legal.iter().enumerate() {
                let Command::Attack { unit, target } = *c else { continue };
                let (a, d) = (e.unit(unit).unwrap(), e.unit_at(target).unwrap());
                let input = e.rate_battle_input(a, target);
                let row = &f.actions[i * A_F..(i + 1) * A_F];
                attacks += 1;
                let cu = e.combat_unit(d);
                let mine = visible_combat_unit(&e, me, d, d.coordinates);
                if cu.defence_bonus != mine.defence_bonus {
                    continue; // a hidden enemy tech bonus: not comparable
                }
                if e.combat_unit(a).has_convert || e.combat_unit(a).has_infiltrate {
                    continue;
                }
                exact += 1;
                assert_eq!((row[A_DMG_DEALT] * HP_SCALE).round() as i32, input.attack_damage, "{c:?}");
                assert_eq!((row[A_DMG_TAKEN] * HP_SCALE).round() as i32, input.retaliation_damage.max(0), "{c:?}");
                assert_eq!(row[A_KILLS] == 1.0, input.attack_damage >= i32::from(d.health));
                kills += usize::from(row[A_KILLS] == 1.0);
                retaliations += usize::from(row[A_DMG_TAKEN] > 0.0);
            }
        }
        assert!(attacks > 20 && exact > 10 && kills > 0 && retaliations > 0, "{attacks} {exact} {kills} {retaliations}");
    }

    /// A kill or a hit is what applying the command does: play the attack on a clone and compare.
    #[test]
    fn attack_preview_agrees_with_applying_it() {
        let mut checked = 0;
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            for (i, c) in legal.iter().enumerate() {
                let Command::Attack { unit, target } = *c else { continue };
                let (a, d) = (e.unit(unit).unwrap(), e.unit_at(target).unwrap());
                if e.combat_unit(d).defence_bonus != visible_combat_unit(&e, me, d, d.coordinates).defence_bonus {
                    continue;
                }
                if e.combat_unit(a).has_convert || e.combat_unit(a).has_infiltrate {
                    continue;
                }
                let row = &f.actions[i * A_F..(i + 1) * A_F];
                let mut after = e.clone();
                after.apply(*c).unwrap();
                let (dh, ah) = (i32::from(d.health), i32::from(a.health));
                let d_after = after.unit(d.id).map_or(0, |u| i32::from(u.health));
                let a_after = after.unit(a.id).map_or(0, |u| i32::from(u.health));
                assert_eq!(row[A_KILLS] == 1.0, after.unit(d.id).is_none(), "kill flag for {c:?}");
                if row[A_KILLS] == 0.0 && d_after > 0 {
                    assert_eq!(dh - d_after, (row[A_DMG_DEALT] * HP_SCALE).round() as i32, "damage for {c:?}");
                }
                if row[A_ATTACKER_DIES] == 0.0 && a_after > 0 && row[A_KILLS] == 0.0 {
                    assert_eq!(ah - a_after, (row[A_DMG_TAKEN] * HP_SCALE).round() as i32, "retaliation for {c:?}");
                }
                checked += 1;
            }
        }
        assert!(checked > 10, "{checked}");
    }

    #[test]
    fn captures_follow_the_tile_owner() {
        let (mut villages, mut cities) = (0, 0);
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            for (i, c) in legal.iter().enumerate() {
                let row = &f.actions[i * A_F..(i + 1) * A_F];
                match *c {
                    Command::Capture { at, .. } => {
                        let owner = e.tile(at).unwrap().owner;
                        assert_eq!(row[A_CAPTURES_VILLAGE] == 1.0, owner == 0);
                        assert_eq!(row[A_CAPTURES_CITY] == 1.0, owner != 0);
                        villages += usize::from(owner == 0);
                        cities += usize::from(owner != 0);
                    }
                    _ => assert!(row[A_CAPTURES_CITY] == 0.0 && row[A_CAPTURES_VILLAGE] == 0.0),
                }
            }
        }
        assert!(villages > 0, "no village capture in the test positions");
        let _ = cities;
    }

    /// Threat and reach agree with a direct, slow recomputation from engine calls.
    #[test]
    fn tile_threat_and_reach_match_direct_engine_calls() {
        let mut threatened = 0;
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            let n = f.h * f.w;
            let view = honest_view(&e, me);
            let mut want_threat = vec![0i16; n];
            for u in e.s.units.iter().filter(|u| hostile_visible(&e, me, u) && !u.invisible && e.unit_attack(u) > 0 && u.leader == 0) {
                let range = e.unit_range(u).max(1);
                let mut pos = vec![u.coordinates];
                pos.extend(view.movement_options_ignoring_state(view.unit(u.id).unwrap()).into_iter().filter(|d| e.is_explored(me, e.tile_index(*d).unwrap())));
                let mut hit = vec![false; n];
                for d in pos {
                    for k in 0..n {
                        let c = e.s.map.tiles[k].coordinates;
                        let dist = Engine::chebyshev(d, c);
                        if dist >= 1 && dist <= range && e.is_explored(me, k) {
                            hit[k] = true;
                        }
                    }
                }
                for k in 0..n {
                    want_threat[k] += i16::from(hit[k]);
                }
            }
            let mut want_reach = vec![0i16; n];
            for c in &legal {
                if let Command::Move { to, .. } = *c {
                    want_reach[e.tile_index(to).unwrap()] += 1;
                }
            }
            for k in 0..n {
                let row = &f.tiles[k * T_F..(k + 1) * T_F];
                assert_eq!(row[TF_THREAT_N], want_threat[k].min(COUNT_CAP), "threat at tile {k}");
                assert_eq!(row[TF_REACH_N], want_reach[k].min(COUNT_CAP), "reach at tile {k}");
                if !e.is_explored(me, k) {
                    assert!(row[..TF_DIST_CAPITAL].iter().all(|&v| v == 0), "unexplored tile {k}: {row:?}");
                }
                threatened += usize::from(row[TF_THREAT_N] > 0);
            }
        }
        assert!(threatened > 0, "no threatened tile in the test positions");
    }

    #[test]
    fn end_tile_figures_are_consistent() {
        let (mut stays_in_danger, mut checked) = (0, 0);
        for (e, me) in positions() {
            let (legal, f) = feats(&e, me);
            for (i, c) in legal.iter().enumerate() {
                let Command::Stay { unit } = *c else { continue };
                let u = e.unit(unit).unwrap();
                let k = e.tile_index(u.coordinates).unwrap();
                let row = &f.actions[i * A_F..(i + 1) * A_F];
                // The number of threats at the end tile is the tile's own threat count.
                assert_eq!((row[A_END_THREAT_N] * 4.0).round() as i16, f.tiles[k * T_F + TF_THREAT_N], "{c:?}");
                assert!(row[A_END_THREAT_MAX] <= row[A_END_THREAT_SUM] + 1e-6);
                stays_in_danger += usize::from(row[A_END_THREAT_N] > 0.0);
                checked += 1;
            }
        }
        assert!(checked > 0 && stays_in_danger > 0, "{checked} {stays_in_danger}");
    }

    #[test]
    fn global_figures() {
        for (e, me) in positions() {
            let (_, f) = feats(&e, me);
            let stars = e.player(me).unwrap().currency;
            assert_eq!(f.globals[0], (stars + e.income(me)) as f32 / 10.0);
            let prices: Vec<i32> = e.researchable(me).iter().map(|&t| e.tech_price(me, t)).collect();
            assert_eq!(f.globals[1], prices.len() as f32 / 10.0);
            assert_eq!(f.globals[3], prices.iter().copied().min().unwrap_or(0) as f32 / 10.0);
            assert!(f.globals[2] <= f.globals[4] + 1e-6);
        }
    }

    /// The features are computed from the position alone: building them twice gives the same numbers, and they do not
    /// touch the engine.
    #[test]
    fn deterministic() {
        for (e, me) in positions().into_iter().take(8) {
            let a = feats(&e, me).1;
            let b = feats(&e, me).1;
            assert_eq!(a, b);
        }
    }

    /// Fog: moving or removing an enemy unit on a tile the mover has not explored, or changing an enemy's techs,
    /// leaves every feature as it was.
    #[test]
    fn hidden_information_does_not_leak() {
        let (mut units_moved, mut techs_changed) = (0, 0);
        for (e, me) in positions() {
            let (legal, before) = feats(&e, me);
            // A hidden enemy unit: remove it, and put it somewhere else hidden.
            if let Some(i) = e.s.units.iter().position(|u| u.owner != me && u.owner != NATURE && !e.is_explored(me, e.tile_index(u.coordinates).unwrap())) {
                let mut other = e.clone();
                other.s.units.remove(i);
                let slots = SlotMap::new(&other, me);
                assert_eq!(before, features(&other, me, &slots, &legal), "removing a hidden unit changed the features");
                let mut moved = e.clone();
                moved.s.units[i].health = 10;
                moved.s.units[i].type_idx = 1;
                let slots = SlotMap::new(&moved, me);
                assert_eq!(before, features(&moved, me, &slots, &legal), "changing a hidden unit changed the features");
                units_moved += 1;
            }
            // An enemy's techs do not show through.
            let mut other = e.clone();
            for r in other.s.tech.iter_mut().filter(|r| r.player != me) {
                r.tech = (1..=50).collect();
            }
            let slots = SlotMap::new(&other, me);
            assert_eq!(before, features(&other, me, &slots, &legal), "an enemy's techs changed the features");
            techs_changed += 1;
        }
        assert!(units_moved > 0 && techs_changed > 0, "{units_moved} {techs_changed}");
    }
}
