//! Compact game records: setup, every command, outcome, engine version. Replayable.

use crate::setup::{GameSetup, Seat};
use polytopia_sim::engine::{Command, Engine, Wc, NATURE};
use polytopia_sim::CityReward;

pub const RECORD_FORMAT: u8 = 1;
const MAGIC: &[u8; 4] = b"PREC";

/// The commit of the engine checkout this crate was built against (`-dirty` with local changes).
pub const ENGINE_COMMIT: &str = env!("ENGINE_COMMIT");

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    pub over: bool,
    pub turns: u32,
    /// Engine player ids.
    pub winners: Vec<u8>,
    /// Final `Engine::score` per player id (Nature excluded).
    pub scores: Vec<(u8, i32)>,
    /// The game ended at the turn limit with more than one player alive.
    pub adjudicated: bool,
}

impl Outcome {
    pub fn of(e: &Engine) -> Self {
        let scores = e.s.players.iter().filter(|p| p.id != NATURE).map(|p| (p.id, e.score(p.id))).collect();
        let alive = e.living_players().len();
        let limit = e.s.settings.turn_limit;
        Outcome {
            over: e.is_over(),
            turns: e.s.current_turn,
            winners: e.winners.clone(),
            scores,
            adjudicated: e.is_over() && alive > 1 && limit > 0 && e.s.current_turn as i32 >= limit,
        }
    }

    pub fn score_of(&self, id: u8) -> i32 {
        self.scores.iter().find(|(p, _)| *p == id).map_or(0, |(_, s)| *s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameRecord {
    pub engine_version: i32,
    pub engine_commit: String,
    pub setup: GameSetup,
    pub commands: Vec<Command>,
    pub outcome: Outcome,
}

// ---------------------------------------------------------------------------------------------------------------
// Binary codec: LEB128 varints, zigzag for signed values.

struct W(Vec<u8>);

impl W {
    fn u(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                return;
            }
            self.0.push(b | 0x80);
        }
    }
    fn i(&mut self, v: i64) {
        self.u(((v << 1) ^ (v >> 63)) as u64);
    }
    fn wc(&mut self, c: Wc) {
        self.i(i64::from(c.x));
        self.i(i64::from(c.y));
    }
    fn s(&mut self, s: &str) {
        self.u(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl R<'_> {
    fn u(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.b.get(self.p).ok_or("record truncated")?;
            self.p += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too long".into());
            }
        }
    }
    fn i(&mut self) -> Result<i64, String> {
        let u = self.u()?;
        Ok(((u >> 1) as i64) ^ -((u & 1) as i64))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.i()? as i32)
    }
    fn wc(&mut self) -> Result<Wc, String> {
        Ok(Wc { x: self.i32()?, y: self.i32()? })
    }
    fn s(&mut self) -> Result<String, String> {
        let n = self.u()? as usize;
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("record truncated")?;
        let s = String::from_utf8_lossy(&self.b[self.p..end]).into_owned();
        self.p = end;
        Ok(s)
    }
}

fn reward_to_u(r: CityReward) -> u64 {
    r as i32 as u64
}

fn reward_from_u(v: u64) -> Result<CityReward, String> {
    Ok(match v {
        0 => CityReward::None,
        1 => CityReward::CityWall,
        2 => CityReward::Park,
        3 => CityReward::Workshop,
        4 => CityReward::Explorer,
        5 => CityReward::BorderGrowth,
        6 => CityReward::SuperUnit,
        7 => CityReward::Resources,
        8 => CityReward::PopulationGrowth,
        11 => CityReward::Rebellion,
        13 => CityReward::Poison,
        100 => CityReward::TutorialExplorer,
        _ => return Err(format!("bad city reward {v}")),
    })
}

fn put_command(w: &mut W, c: &Command) {
    use Command::*;
    match *c {
        Move { unit, to } => {
            w.u(0);
            w.u(u64::from(unit));
            w.wc(to);
        }
        Attack { unit, target } => {
            w.u(1);
            w.u(u64::from(unit));
            w.wc(target);
        }
        Capture { unit, at } => {
            w.u(2);
            w.u(u64::from(unit));
            w.wc(at);
        }
        Train { city, unit_type } => {
            w.u(3);
            w.wc(city);
            w.i(i64::from(unit_type));
        }
        Build { at, improvement } => {
            w.u(4);
            w.wc(at);
            w.i(i64::from(improvement));
        }
        Research { tech } => {
            w.u(5);
            w.i(i64::from(tech));
        }
        CityReward { city, reward } => {
            w.u(6);
            w.wc(city);
            w.u(reward_to_u(reward));
        }
        Promote { unit } => unit_only(w, 7, unit),
        Recover { unit } => unit_only(w, 8, unit),
        HealOthers { unit } => unit_only(w, 9, unit),
        Explode { unit } => unit_only(w, 10, unit),
        Disband { unit } => unit_only(w, 11, unit),
        Destroy { at } => {
            w.u(12);
            w.wc(at);
        }
        Decompose { at } => {
            w.u(13);
            w.wc(at);
        }
        Upgrade { unit, to } => {
            w.u(14);
            w.u(u64::from(unit));
            w.i(i64::from(to));
        }
        ExamineRuins { unit } => unit_only(w, 15, unit),
        Stay { unit } => unit_only(w, 16, unit),
        Swarm { unit } => unit_only(w, 17, unit),
        Boost { unit } => unit_only(w, 18, unit),
        Disembark { unit } => unit_only(w, 19, unit),
        ClearTileEffect { at, effect } => {
            w.u(20);
            w.wc(at);
            w.u(u64::from(effect));
        }
        BreakPeace { with } => {
            w.u(21);
            w.u(u64::from(with));
        }
        PeaceTreaty { with } => {
            w.u(22);
            w.u(u64::from(with));
        }
        PeaceResponse { with, accept } => {
            w.u(23);
            w.u(u64::from(with) * 2 + u64::from(accept));
        }
        EstablishEmbassy { with } => {
            w.u(24);
            w.u(u64::from(with));
        }
        EndTurn => w.u(25),
    }
}

fn unit_only(w: &mut W, tag: u64, unit: u32) {
    w.u(tag);
    w.u(u64::from(unit));
}

fn get_command(r: &mut R) -> Result<Command, String> {
    use Command::*;
    let unit = |r: &mut R| -> Result<u32, String> { Ok(r.u()? as u32) };
    Ok(match r.u()? {
        0 => Move { unit: unit(r)?, to: r.wc()? },
        1 => Attack { unit: unit(r)?, target: r.wc()? },
        2 => Capture { unit: unit(r)?, at: r.wc()? },
        3 => Train { city: r.wc()?, unit_type: r.i32()? },
        4 => Build { at: r.wc()?, improvement: r.i32()? },
        5 => Research { tech: r.i32()? },
        6 => CityReward { city: r.wc()?, reward: reward_from_u(r.u()?)? },
        7 => Promote { unit: unit(r)? },
        8 => Recover { unit: unit(r)? },
        9 => HealOthers { unit: unit(r)? },
        10 => Explode { unit: unit(r)? },
        11 => Disband { unit: unit(r)? },
        12 => Destroy { at: r.wc()? },
        13 => Decompose { at: r.wc()? },
        14 => Upgrade { unit: unit(r)?, to: r.i32()? },
        15 => ExamineRuins { unit: unit(r)? },
        16 => Stay { unit: unit(r)? },
        17 => Swarm { unit: unit(r)? },
        18 => Boost { unit: unit(r)? },
        19 => Disembark { unit: unit(r)? },
        20 => ClearTileEffect { at: r.wc()?, effect: r.u()? as u16 },
        21 => BreakPeace { with: r.u()? as u8 },
        22 => PeaceTreaty { with: r.u()? as u8 },
        23 => {
            let v = r.u()?;
            PeaceResponse { with: (v / 2) as u8, accept: v % 2 == 1 }
        }
        24 => EstablishEmbassy { with: r.u()? as u8 },
        25 => EndTurn,
        t => return Err(format!("bad command tag {t}")),
    })
}

impl GameRecord {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = W(Vec::with_capacity(64 + self.commands.len() * 6));
        w.0.extend_from_slice(MAGIC);
        w.u(u64::from(RECORD_FORMAT));
        w.i(i64::from(self.engine_version));
        w.s(&self.engine_commit);
        let s = &self.setup;
        w.i(i64::from(s.seed));
        w.u(u64::from(s.size));
        w.u(u64::from(s.preset));
        w.i(i64::from(s.mode));
        w.i(i64::from(s.turn_limit));
        w.u(s.seats.len() as u64);
        for seat in &s.seats {
            w.i(i64::from(seat.tribe));
            w.u(u64::from(seat.difficulty));
        }
        w.u(self.commands.len() as u64);
        for c in &self.commands {
            put_command(&mut w, c);
        }
        let o = &self.outcome;
        w.u(u64::from(o.over) | u64::from(o.adjudicated) << 1);
        w.u(u64::from(o.turns));
        w.u(o.winners.len() as u64);
        for &x in &o.winners {
            w.u(u64::from(x));
        }
        w.u(o.scores.len() as u64);
        for &(id, score) in &o.scores {
            w.u(u64::from(id));
            w.i(i64::from(score));
        }
        w.0
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 4 || &b[..4] != MAGIC {
            return Err("not a game record".into());
        }
        let mut r = R { b, p: 4 };
        let format = r.u()?;
        if format != u64::from(RECORD_FORMAT) {
            return Err(format!("record format {format} not supported"));
        }
        let engine_version = r.i32()?;
        let engine_commit = r.s()?;
        let seed = r.i32()?;
        let size = r.u()? as u8;
        let preset = r.u()? as u8;
        let mode = r.i32()?;
        let turn_limit = r.i32()?;
        let n_seats = r.u()? as usize;
        let mut seats = Vec::with_capacity(n_seats);
        for _ in 0..n_seats {
            seats.push(Seat { tribe: r.i32()?, difficulty: r.u()? as u8 });
        }
        let n = r.u()? as usize;
        let mut commands = Vec::with_capacity(n);
        for _ in 0..n {
            commands.push(get_command(&mut r)?);
        }
        let flags = r.u()?;
        let turns = r.u()? as u32;
        let nw = r.u()? as usize;
        let mut winners = Vec::with_capacity(nw);
        for _ in 0..nw {
            winners.push(r.u()? as u8);
        }
        let ns = r.u()? as usize;
        let mut scores = Vec::with_capacity(ns);
        for _ in 0..ns {
            scores.push((r.u()? as u8, r.i32()?));
        }
        Ok(GameRecord {
            engine_version,
            engine_commit,
            setup: GameSetup { seed, size, preset, mode, turn_limit, seats },
            commands,
            outcome: Outcome { over: flags & 1 != 0, adjudicated: flags & 2 != 0, turns, winners, scores },
        })
    }

    /// Rebuild the game and play every command. Fails if a command is illegal when `check_legal` is set, or if the
    /// final outcome differs from the recorded one.
    pub fn replay(&self, check_legal: bool) -> Result<Engine, String> {
        let mut e = self.setup.build();
        for (i, c) in self.commands.iter().enumerate() {
            if check_legal {
                let legal = crate::encoding::legal_actions(&e);
                if !legal.contains(c) {
                    return Err(format!("command {i} {c:?} is not legal on replay"));
                }
            }
            e.apply_unchecked(*c);
        }
        let got = Outcome::of(&e);
        if got != self.outcome {
            return Err(format!("outcome differs on replay: recorded {:?}, replayed {:?}", self.outcome, got));
        }
        Ok(e)
    }
}

/// Append a length-prefixed record to a stream of records (`u32` little endian length, then the record).
pub fn write_record(out: &mut impl std::io::Write, rec: &GameRecord) -> std::io::Result<()> {
    let b = rec.to_bytes();
    out.write_all(&(b.len() as u32).to_le_bytes())?;
    out.write_all(&b)
}

/// Read every record of a stream written by [`write_record`].
pub fn read_records(mut data: &[u8]) -> Result<Vec<GameRecord>, String> {
    let mut out = Vec::new();
    while !data.is_empty() {
        if data.len() < 4 {
            return Err("record stream truncated".into());
        }
        let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + n {
            return Err("record stream truncated".into());
        }
        out.push(GameRecord::from_bytes(&data[4..4 + n])?);
        data = &data[4 + n..];
    }
    Ok(out)
}
