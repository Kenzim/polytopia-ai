//! Action encoding `(kind, arg, src_tile, dst_tile)` (encoding.md section 4).

use crate::slots::SlotMap;
use polytopia_sim::engine::{Command, Engine, Wc};
use polytopia_sim::CityReward;

pub type Action = [i16; 4];

pub const N_KINDS: usize = 32;
pub const KIND_PAD: i16 = 0;
pub const NONE_TILE: i16 = -1;

pub const KIND_NAMES: [&str; 26] = [
    "Move", "Attack", "Capture", "Promote", "Recover", "HealOthers", "Explode", "Disband", "ExamineRuins", "Stay", "Swarm", "Boost",
    "Disembark", "Upgrade", "Train", "Build", "Destroy", "Decompose", "ClearTileEffect", "CityReward", "Research", "BreakPeace",
    "PeaceTreaty", "EstablishEmbassy", "PeaceResponse", "EndTurn",
];

pub const K_MOVE: i16 = 1;
pub const K_ATTACK: i16 = 2;
pub const K_CAPTURE: i16 = 3;
pub const K_PROMOTE: i16 = 4;
pub const K_RECOVER: i16 = 5;
pub const K_HEAL_OTHERS: i16 = 6;
pub const K_EXPLODE: i16 = 7;
pub const K_DISBAND: i16 = 8;
pub const K_EXAMINE_RUINS: i16 = 9;
pub const K_STAY: i16 = 10;
pub const K_SWARM: i16 = 11;
pub const K_BOOST: i16 = 12;
pub const K_DISEMBARK: i16 = 13;
pub const K_UPGRADE: i16 = 14;
pub const K_TRAIN: i16 = 15;
pub const K_BUILD: i16 = 16;
pub const K_DESTROY: i16 = 17;
pub const K_DECOMPOSE: i16 = 18;
pub const K_CLEAR_TILE_EFFECT: i16 = 19;
pub const K_CITY_REWARD: i16 = 20;
pub const K_RESEARCH: i16 = 21;
pub const K_BREAK_PEACE: i16 = 22;
pub const K_PEACE_TREATY: i16 = 23;
pub const K_ESTABLISH_EMBASSY: i16 = 24;
pub const K_PEACE_RESPONSE: i16 = 25;
pub const K_END_TURN: i16 = 26;

fn reward_id(r: CityReward) -> i16 {
    match r {
        CityReward::None => 0,
        CityReward::CityWall => 1,
        CityReward::Park => 2,
        CityReward::Workshop => 3,
        CityReward::Explorer => 4,
        CityReward::BorderGrowth => 5,
        CityReward::SuperUnit => 6,
        CityReward::Resources => 7,
        CityReward::PopulationGrowth => 8,
        CityReward::Rebellion => 9,
        CityReward::Poison => 10,
        CityReward::TutorialExplorer => 11,
    }
}

fn reward_from_id(id: i16) -> Option<CityReward> {
    Some(match id {
        1 => CityReward::CityWall,
        2 => CityReward::Park,
        3 => CityReward::Workshop,
        4 => CityReward::Explorer,
        5 => CityReward::BorderGrowth,
        6 => CityReward::SuperUnit,
        7 => CityReward::Resources,
        8 => CityReward::PopulationGrowth,
        9 => CityReward::Rebellion,
        10 => CityReward::Poison,
        11 => CityReward::TutorialExplorer,
        _ => return None,
    })
}

fn tile_of(e: &Engine, c: Wc) -> Option<i16> {
    e.tile_index(c).map(|i| i as i16)
}

fn wc_of(e: &Engine, t: i16) -> Option<Wc> {
    if t < 0 {
        return None;
    }
    let w = e.width();
    let t = i32::from(t);
    let c = Wc { x: t % w, y: t / w };
    e.in_bounds(c).then_some(c)
}

/// Index of the unit among the mover's units on its tile, in engine unit-list order. Nearly always 0; a Giant and a
/// Cloak can share a tile.
fn unit_rank(e: &Engine, id: u32) -> i16 {
    let Some(u) = e.unit(id) else { return 0 };
    let mut rank = 0;
    for o in &e.s.units {
        if o.id == id {
            break;
        }
        if o.owner == u.owner && o.coordinates == u.coordinates {
            rank += 1;
        }
    }
    rank
}

/// Encode a command. `None` only if it names a unit, tile or player that does not exist.
pub fn encode_command(e: &Engine, slots: &SlotMap, cmd: &Command) -> Option<Action> {
    use Command::*;
    let unit_tile = |id: u32| -> Option<i16> { e.unit(id).and_then(|u| tile_of(e, u.coordinates)) };
    let same = |kind: i16, arg: i16, t: i16| -> Option<Action> { Some([kind, arg, t, t]) };
    let unit_kind = |kind: i16, id: u32| -> Option<Action> { same(kind, unit_rank(e, id), unit_tile(id)?) };
    let slot = |id: u8| -> Option<i16> { slots.slot_of(id).map(|s| s as i16) };
    match *cmd {
        Move { unit, to } => Some([K_MOVE, unit_rank(e, unit), unit_tile(unit)?, tile_of(e, to)?]),
        Attack { unit, target } => Some([K_ATTACK, unit_rank(e, unit), unit_tile(unit)?, tile_of(e, target)?]),
        Capture { unit, at } => Some([K_CAPTURE, unit_rank(e, unit), unit_tile(unit)?, tile_of(e, at)?]),
        Promote { unit } => unit_kind(K_PROMOTE, unit),
        Recover { unit } => unit_kind(K_RECOVER, unit),
        HealOthers { unit } => unit_kind(K_HEAL_OTHERS, unit),
        Explode { unit } => unit_kind(K_EXPLODE, unit),
        Disband { unit } => unit_kind(K_DISBAND, unit),
        ExamineRuins { unit } => unit_kind(K_EXAMINE_RUINS, unit),
        Stay { unit } => unit_kind(K_STAY, unit),
        Swarm { unit } => unit_kind(K_SWARM, unit),
        Boost { unit } => unit_kind(K_BOOST, unit),
        Disembark { unit } => unit_kind(K_DISEMBARK, unit),
        Upgrade { unit, to } => same(K_UPGRADE, to as i16, unit_tile(unit)?),
        Train { city, unit_type } => same(K_TRAIN, unit_type as i16, tile_of(e, city)?),
        Build { at, improvement } => same(K_BUILD, improvement as i16, tile_of(e, at)?),
        Destroy { at } => same(K_DESTROY, 0, tile_of(e, at)?),
        Decompose { at } => same(K_DECOMPOSE, 0, tile_of(e, at)?),
        ClearTileEffect { at, effect } => same(K_CLEAR_TILE_EFFECT, effect as i16, tile_of(e, at)?),
        CityReward { city, reward } => same(K_CITY_REWARD, reward_id(reward), tile_of(e, city)?),
        Research { tech } => Some([K_RESEARCH, tech as i16, NONE_TILE, NONE_TILE]),
        BreakPeace { with } => Some([K_BREAK_PEACE, slot(with)?, NONE_TILE, NONE_TILE]),
        PeaceTreaty { with } => Some([K_PEACE_TREATY, slot(with)?, NONE_TILE, NONE_TILE]),
        EstablishEmbassy { with } => Some([K_ESTABLISH_EMBASSY, slot(with)?, NONE_TILE, NONE_TILE]),
        PeaceResponse { with, accept } => Some([K_PEACE_RESPONSE, slot(with)? * 2 + i16::from(accept), NONE_TILE, NONE_TILE]),
        EndTurn => Some([K_END_TURN, 0, NONE_TILE, NONE_TILE]),
    }
}

/// Inverse of [`encode_command`] in the position the action was encoded in. `None` if no such command exists.
pub fn decode_command(e: &Engine, slots: &SlotMap, a: Action) -> Option<Command> {
    use Command::*;
    let [kind, arg, src, dst] = a;
    let me = slots.me;
    let unit = || -> Option<u32> {
        let c = wc_of(e, src)?;
        e.s.units.iter().filter(|u| u.owner == me && u.coordinates == c).nth(usize::try_from(arg).ok()?).map(|u| u.id)
    };
    // An upgrade names the target type in `arg`, so the unit is the one on the tile that can upgrade to it.
    let upgrader = || -> Option<u32> {
        let c = wc_of(e, src)?;
        let mut on_tile = e.s.units.iter().filter(|u| u.owner == me && u.coordinates == c);
        let first = on_tile.clone().next()?;
        on_tile.find(|u| e.unlocked_upgrades_for(me, u).contains(&i32::from(arg))).or(Some(first)).map(|u| u.id)
    };
    let player = || slots.id_of(arg as usize).filter(|_| arg >= 0);
    Some(match kind {
        K_MOVE => Move { unit: unit()?, to: wc_of(e, dst)? },
        K_ATTACK => Attack { unit: unit()?, target: wc_of(e, dst)? },
        K_CAPTURE => Capture { unit: unit()?, at: wc_of(e, dst)? },
        K_PROMOTE => Promote { unit: unit()? },
        K_RECOVER => Recover { unit: unit()? },
        K_HEAL_OTHERS => HealOthers { unit: unit()? },
        K_EXPLODE => Explode { unit: unit()? },
        K_DISBAND => Disband { unit: unit()? },
        K_EXAMINE_RUINS => ExamineRuins { unit: unit()? },
        K_STAY => Stay { unit: unit()? },
        K_SWARM => Swarm { unit: unit()? },
        K_BOOST => Boost { unit: unit()? },
        K_DISEMBARK => Disembark { unit: unit()? },
        K_UPGRADE => Upgrade { unit: upgrader()?, to: i32::from(arg) },
        K_TRAIN => Train { city: wc_of(e, src)?, unit_type: i32::from(arg) },
        K_BUILD => Build { at: wc_of(e, src)?, improvement: i32::from(arg) },
        K_DESTROY => Destroy { at: wc_of(e, src)? },
        K_DECOMPOSE => Decompose { at: wc_of(e, src)? },
        K_CLEAR_TILE_EFFECT => ClearTileEffect { at: wc_of(e, src)?, effect: arg as u16 },
        K_CITY_REWARD => CityReward { city: wc_of(e, src)?, reward: reward_from_id(arg)? },
        K_RESEARCH => Research { tech: i32::from(arg) },
        K_BREAK_PEACE => BreakPeace { with: player()? },
        K_PEACE_TREATY => PeaceTreaty { with: player()? },
        K_ESTABLISH_EMBASSY => EstablishEmbassy { with: slots.id_of(arg as usize).filter(|_| arg >= 0)? },
        K_PEACE_RESPONSE => {
            if arg < 0 {
                return None;
            }
            PeaceResponse { with: slots.id_of((arg / 2) as usize)?, accept: arg % 2 == 1 }
        }
        K_END_TURN => EndTurn,
        _ => return None,
    })
}

/// The kind id of a command.
pub fn kind_of(cmd: &Command) -> i16 {
    use Command::*;
    match cmd {
        Move { .. } => K_MOVE,
        Attack { .. } => K_ATTACK,
        Capture { .. } => K_CAPTURE,
        Promote { .. } => K_PROMOTE,
        Recover { .. } => K_RECOVER,
        HealOthers { .. } => K_HEAL_OTHERS,
        Explode { .. } => K_EXPLODE,
        Disband { .. } => K_DISBAND,
        ExamineRuins { .. } => K_EXAMINE_RUINS,
        Stay { .. } => K_STAY,
        Swarm { .. } => K_SWARM,
        Boost { .. } => K_BOOST,
        Disembark { .. } => K_DISEMBARK,
        Upgrade { .. } => K_UPGRADE,
        Train { .. } => K_TRAIN,
        Build { .. } => K_BUILD,
        Destroy { .. } => K_DESTROY,
        Decompose { .. } => K_DECOMPOSE,
        ClearTileEffect { .. } => K_CLEAR_TILE_EFFECT,
        CityReward { .. } => K_CITY_REWARD,
        Research { .. } => K_RESEARCH,
        BreakPeace { .. } => K_BREAK_PEACE,
        PeaceTreaty { .. } => K_PEACE_TREATY,
        EstablishEmbassy { .. } => K_ESTABLISH_EMBASSY,
        PeaceResponse { .. } => K_PEACE_RESPONSE,
        EndTurn => K_END_TURN,
    }
}
