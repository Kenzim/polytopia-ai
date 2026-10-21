//! Evaluation arena: agents, a two-player match runner and JSONL results (see `docs/arena.md`).
//!
//! An [`Agent`] picks the next command for the seat to move. Retail AI, random play and MCTS are built in; a model
//! agent (Python or ONNX) plugs in later by implementing the same trait and adding a variant to [`AgentSpec`].

pub mod mcts;

use mcts::MctsParams;
use polyai_core::game::{ai_pick, MAX_COMMANDS, MAX_PER_TURN};
use polyai_core::{legal_actions, GameSetup, Outcome, Rng, Seat, TurnCounter, DEFAULT_TURN_LIMIT, DOMINATION, ENGINE_COMMIT, PRESETS, SIZES, TRIBES};
use polytopia_sim::engine::{Command, Engine, NATURE};

use std::time::Instant;

/// Evaluation seeds start here; lower seeds are training data.
pub const EVAL_SEED_MIN: i32 = polyai_core::EVAL_SEED_MIN;

/// What an agent is told besides the engine.
#[derive(Debug, Clone, Copy)]
pub struct Ctx {
    /// 0-based seat in the game (engine player id minus one).
    pub seat: usize,
    /// Engine player id of the seat to move.
    pub me: u8,
    /// Commands this player has already played this turn.
    pub per_turn: u32,
}

/// Picks the command for the current player. The engine is `&mut` because the retail AI keeps caches in it; an agent
/// must not change the game state.
pub trait Agent: Send {
    fn name(&self) -> String;
    fn pick(&mut self, e: &mut Engine, ctx: &Ctx) -> Command;
    /// True if `pick` always returns a command the engine accepts, so the runner can skip its legality check.
    fn trusted(&self) -> bool {
        false
    }
}

/// The retail AI at the handicap the setup gave this seat.
pub struct RetailAgent {
    difficulty: u8,
}

impl Agent for RetailAgent {
    fn name(&self) -> String {
        format!("retail:{}", DIFFICULTY_NAMES[usize::from(self.difficulty)])
    }
    fn pick(&mut self, e: &mut Engine, ctx: &Ctx) -> Command {
        ai_pick(e, ctx.per_turn)
    }
    fn trusted(&self) -> bool {
        true
    }
}

/// Uniform over `polyai_core::legal_actions` (the engine's legal list plus `Stay`), seeded.
pub struct RandomAgent {
    rng: Rng,
}

impl Agent for RandomAgent {
    fn name(&self) -> String {
        "random".into()
    }
    fn pick(&mut self, e: &mut Engine, _ctx: &Ctx) -> Command {
        let legal = legal_actions(e);
        if legal.is_empty() {
            return Command::EndTurn;
        }
        legal[self.rng.below(legal.len())]
    }
}

/// MCTS from weekly's generic UCT. Full information: it searches clones of the true engine state, so it sees through
/// fog.
pub struct MctsAgent {
    params: MctsParams,
    seed: u64,
    calls: u64,
}

impl Agent for MctsAgent {
    fn name(&self) -> String {
        AgentSpec::Mcts(self.params).to_string()
    }
    fn pick(&mut self, e: &mut Engine, ctx: &Ctx) -> Command {
        self.calls += 1;
        let seed = self.seed ^ self.calls.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        mcts::choose(e, &self.params, ctx.per_turn, seed)
    }
}

pub const DIFFICULTY_NAMES: [&str; 4] = ["easy", "normal", "hard", "crazy"];

/// A parsed agent name: `retail:easy|normal|hard|crazy`, `random`, `mcts:<iters>[:depth=D][:topk=K]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentSpec {
    Retail(u8),
    Random,
    Mcts(MctsParams),
}

impl std::str::FromStr for AgentSpec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let mut parts = s.split(':');
        match parts.next().unwrap_or("") {
            "random" if parts.next().is_none() => Ok(AgentSpec::Random),
            "retail" => {
                let d = parts.next().ok_or("retail needs a difficulty: retail:easy|normal|hard|crazy")?;
                let i = DIFFICULTY_NAMES.iter().position(|n| *n == d).ok_or(format!("unknown difficulty {d}"))?;
                Ok(AgentSpec::Retail(i as u8))
            }
            "mcts" => {
                let iters = parts.next().ok_or("mcts needs an iteration count: mcts:200")?;
                let mut p = MctsParams::new(iters.parse().map_err(|_| format!("bad iteration count {iters}"))?);
                for opt in parts {
                    match opt.split_once('=') {
                        Some(("topk", v)) => p.topk = Some(v.parse().map_err(|_| format!("bad topk {v}"))?),
                        Some(("depth", v)) => p.rollout_depth = v.parse().map_err(|_| format!("bad depth {v}"))?,
                        Some(("rollout", "random")) => {}
                        _ => return Err(format!("unknown mcts option {opt}")),
                    }
                }
                Ok(AgentSpec::Mcts(p))
            }
            "model" => Err("model agents are not built into polyai-arena yet (implement Agent and add a variant to AgentSpec)".into()),
            _ => Err(format!("unknown agent {s}")),
        }
    }
}

impl std::fmt::Display for AgentSpec {
    /// The canonical name, which `from_str` reads back.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentSpec::Retail(d) => write!(f, "retail:{}", DIFFICULTY_NAMES[usize::from(*d)]),
            AgentSpec::Random => write!(f, "random"),
            AgentSpec::Mcts(p) => {
                write!(f, "mcts:{}", p.iterations)?;
                if p.rollout_depth != MctsParams::new(0).rollout_depth {
                    write!(f, ":depth={}", p.rollout_depth)?;
                }
                if let Some(k) = p.topk {
                    write!(f, ":topk={k}")?;
                }
                Ok(())
            }
        }
    }
}

impl AgentSpec {
    /// The engine handicap the seat is created with (retail difficulty; Normal for the others, which only matters to
    /// the `topk` MCTS variant whose candidates come from the seat's retail scoring).
    pub fn difficulty(&self) -> u8 {
        match self {
            AgentSpec::Retail(d) => *d,
            _ => 1,
        }
    }

    pub fn build(&self, seed: i32, seat: usize) -> Box<dyn Agent> {
        let mix = (u64::from(seed as u32) << 8) | seat as u64;
        match self {
            AgentSpec::Retail(d) => Box::new(RetailAgent { difficulty: *d }),
            AgentSpec::Random => Box::new(RandomAgent { rng: Rng::new(mix ^ 0x52_414E_44) }),
            AgentSpec::Mcts(p) => Box::new(MctsAgent { params: *p, seed: Rng::new(mix ^ 0x4D43_5453).next(), calls: 0 }),
        }
    }
}

/// The 1v1 evaluation setup for a seed: Domination, a map size (Tiny, Small or Normal), a preset (five) and two
/// different tribes from the 13, all drawn from the seed, with the compute cap `DEFAULT_TURN_LIMIT` (a game still
/// running then is a draw). Both games of a seed share it; only the agents change seats.
pub fn eval_setup(seed: i32, seat_difficulty: [u8; 2]) -> GameSetup {
    let mut rng = Rng::new(u64::from(seed as u32) | 0xA2E4_0000_0000);
    let mut pool: Vec<i32> = TRIBES.to_vec();
    let t0 = pool.remove(rng.below(pool.len()));
    let t1 = pool.remove(rng.below(pool.len()));
    let size = SIZES[rng.below(SIZES.len())] as u8;
    let preset = PRESETS[rng.below(PRESETS.len())] as u8;
    GameSetup {
        seed,
        size,
        preset,
        mode: DOMINATION,
        turn_limit: DEFAULT_TURN_LIMIT,
        seats: vec![Seat { tribe: t0, difficulty: seat_difficulty[0] }, Seat { tribe: t1, difficulty: seat_difficulty[1] }],
    }
}

/// One finished game.
#[derive(Debug, Clone)]
pub struct GameResult {
    pub seed: i32,
    /// 0: first agent of the pair in seat 0; 1: seats swapped.
    pub swap: u8,
    pub agents: [String; 2],
    pub setup: GameSetup,
    /// Winning seat by elimination; `None` for a draw (the game reached the turn cap, or was cut off).
    pub winner: Option<usize>,
    pub scores: [i32; 2],
    pub turns: u32,
    /// The game reached the turn cap with both players alive: a draw. Never decided on score.
    pub capped: bool,
    pub over: bool,
    pub commands: usize,
    /// Commands an agent proposed that the engine rejected (replaced by EndTurn).
    pub illegal: [u32; 2],
    pub wall_ms: f64,
    /// Time spent inside `Agent::pick` per seat.
    pub think_ms: [f64; 2],
}

/// Play one game with the given agents in seats 0 and 1 (the setup's seat order).
pub fn play_game(setup: &GameSetup, agents: &mut [Box<dyn Agent>; 2]) -> (GameResult, Engine) {
    let started = Instant::now();
    let mut e = setup.build();
    let order: Vec<u8> = e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect();
    let mut counter = TurnCounter::default();
    let mut commands = 0usize;
    let mut illegal = [0u32; 2];
    let mut think = [0f64; 2];
    while !e.is_over() && commands < MAX_COMMANDS {
        let me = e.current_player();
        let (turn, player) = (e.s.current_turn, me);
        let cmd = match order.iter().position(|&id| id == me) {
            None => ai_pick(&mut e, counter.0),
            Some(seat) => {
                let t = Instant::now();
                let mut cmd = agents[seat].pick(&mut e, &Ctx { seat, me, per_turn: counter.0 });
                think[seat] += t.elapsed().as_secs_f64() * 1000.0;
                if counter.0 + 1 > MAX_PER_TURN {
                    cmd = Command::EndTurn;
                } else if !agents[seat].trusted() && !matches!(cmd, Command::EndTurn) && !is_legal(&e, &cmd) {
                    illegal[seat] += 1;
                    cmd = Command::EndTurn;
                }
                cmd
            }
        };
        e.apply_unchecked(cmd);
        commands += 1;
        counter.after(&e, &cmd, turn, player);
    }
    let o = Outcome::of(&e);
    let winner = o.winners.first().and_then(|w| order.iter().position(|id| id == w));
    let result = GameResult {
        seed: setup.seed,
        swap: 0,
        agents: [agents[0].name(), agents[1].name()],
        setup: setup.clone(),
        winner,
        scores: [o.score_of(order[0]), o.score_of(order[1])],
        turns: o.turns,
        capped: o.is_draw(),
        over: o.over,
        commands,
        illegal,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
        think_ms: think,
    };
    (result, e)
}

fn is_legal(e: &Engine, cmd: &Command) -> bool {
    matches!(cmd, Command::Stay { .. }) && legal_actions(e).contains(cmd) || e.legal_commands().contains(cmd)
}

/// Play `a` against `b` on `seed`; `swap` puts `b` in seat 0 (same setup, same tribes in the same seats).
pub fn play_pair(a: &AgentSpec, b: &AgentSpec, seed: i32, swap: u8) -> GameResult {
    let (s0, s1) = if swap == 0 { (a, b) } else { (b, a) };
    let setup = eval_setup(seed, [s0.difficulty(), s1.difficulty()]);
    let mut agents = [s0.build(seed, 0), s1.build(seed, 1)];
    let (mut r, _) = play_game(&setup, &mut agents);
    r.swap = swap;
    r
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl GameResult {
    /// One JSON object on one line.
    pub fn to_json(&self) -> String {
        let s = &self.setup;
        format!(
            "{{\"seed\":{},\"swap\":{},\"agents\":[{},{}],\"tribes\":[{},{}],\"setup\":{{\"size\":{},\"preset\":{},\"mode\":{},\"turn_limit\":{}}},\
             \"winner\":{},\"scores\":[{},{}],\"turns\":{},\"capped\":{},\"over\":{},\"commands\":{},\"illegal\":[{},{}],\
             \"wall_ms\":{:.1},\"think_ms\":[{:.1},{:.1}],\"engine\":{},\"game_version\":{}}}",
            self.seed,
            self.swap,
            json_str(&self.agents[0]),
            json_str(&self.agents[1]),
            s.seats[0].tribe,
            s.seats[1].tribe,
            s.size,
            s.preset,
            s.mode,
            s.turn_limit,
            self.winner.map_or(-1, |w| w as i32),
            self.scores[0],
            self.scores[1],
            self.turns,
            self.capped,
            self.over,
            self.commands,
            self.illegal[0],
            self.illegal[1],
            self.wall_ms,
            self.think_ms[0],
            self.think_ms[1],
            json_str(ENGINE_COMMIT),
            polytopia_sim::GAME_VERSION,
        )
    }
}

/// The key a result is stored under, for resuming a run: `(seed, swap, first agent, second agent)` in pair order.
pub fn job_key(seed: i32, swap: u8, a: &str, b: &str) -> String {
    format!("{seed}|{swap}|{a}|{b}")
}

/// The key of a JSONL line written by [`GameResult::to_json`] (pair order restored from the swap flag).
pub fn line_key(line: &str) -> Option<String> {
    let num = |key: &str| -> Option<i64> {
        let i = line.find(key)? + key.len();
        let end = line[i..].find(|c: char| !c.is_ascii_digit() && c != '-')? + i;
        line[i..end].parse().ok()
    };
    let seed = num("\"seed\":")?;
    let swap = num("\"swap\":")?;
    let i = line.find("\"agents\":[\"")? + "\"agents\":[\"".len();
    let rest = &line[i..];
    let (s0, rest) = rest.split_once("\",\"")?;
    let (s1, _) = rest.split_once('"')?;
    let (a, b) = if swap == 0 { (s0, s1) } else { (s1, s0) };
    Some(job_key(seed as i32, swap as u8, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_round_trip() {
        for s in ["random", "retail:easy", "retail:crazy", "mcts:200", "mcts:50:topk=5", "mcts:20:depth=30:topk=3"] {
            let spec: AgentSpec = s.parse().unwrap();
            assert_eq!(spec.to_string(), s);
        }
        assert!("retail:ultra".parse::<AgentSpec>().is_err());
        assert!("mcts".parse::<AgentSpec>().is_err());
        assert!("model:x".parse::<AgentSpec>().is_err());
        assert_eq!("mcts:5:rollout=random".parse::<AgentSpec>().unwrap().to_string(), "mcts:5");
    }

    #[test]
    fn eval_setup_is_fixed_by_seed() {
        let a = eval_setup(1_000_003, [0, 1]);
        let b = eval_setup(1_000_003, [3, 3]);
        assert_eq!(a.seats[0].tribe, b.seats[0].tribe);
        assert_ne!(a.seats[0].tribe, a.seats[1].tribe);
        assert!(!a.is_training_seed());
        assert_eq!((a.turn_limit, a.mode), (DEFAULT_TURN_LIMIT, 2));
        assert_eq!((a.size, a.preset), (b.size, b.preset));
    }

    #[test]
    fn retail_game_is_deterministic_and_swap_changes_seats() {
        let (a, b): (AgentSpec, AgentSpec) = ("retail:easy".parse().unwrap(), "retail:hard".parse().unwrap());
        let r1 = play_pair(&a, &b, 1_000_001, 0);
        let r2 = play_pair(&a, &b, 1_000_001, 0);
        assert!(r1.over);
        assert_eq!((r1.winner, r1.scores, r1.turns, r1.commands), (r2.winner, r2.scores, r2.turns, r2.commands));
        assert_eq!(r1.illegal, [0, 0]);
        let s = play_pair(&a, &b, 1_000_001, 1);
        assert_eq!(s.agents, ["retail:hard".to_string(), "retail:easy".to_string()]);
        assert_eq!(s.setup.seats[0].tribe, r1.setup.seats[0].tribe);
    }

    #[test]
    fn random_and_mcts_play_legal_games() {
        let rnd: AgentSpec = "random".parse().unwrap();
        let easy: AgentSpec = "retail:easy".parse().unwrap();
        let r = play_pair(&rnd, &easy, 1_000_002, 0);
        assert!(r.over, "random game finished");
        assert_eq!(r.illegal, [0, 0]);
        let m: AgentSpec = "mcts:4:depth=8:topk=3".parse().unwrap();
        let mut setup = eval_setup(1_000_002, [1, 0]);
        setup.turn_limit = 3;
        let mut agents = [m.build(1_000_002, 0), easy.build(1_000_002, 1)];
        let (r, _) = play_game(&setup, &mut agents);
        assert!(r.over);
        assert_eq!(r.illegal, [0, 0]);
        assert!(r.turns <= 4);
    }

    #[test]
    fn a_capped_game_is_a_draw_whatever_the_scores() {
        let easy: AgentSpec = "retail:easy".parse().unwrap();
        let crazy: AgentSpec = "retail:crazy".parse().unwrap();
        let mut setup = eval_setup(1_000_005, [0, 3]);
        setup.turn_limit = 4;
        let mut agents = [easy.build(1_000_005, 0), crazy.build(1_000_005, 1)];
        let (r, e) = play_game(&setup, &mut agents);
        assert!(r.over && r.capped && r.winner.is_none());
        assert_ne!(r.scores[0], r.scores[1], "scores differ, yet nobody wins");
        assert!(!e.winners.is_empty(), "the engine's own top-score winner is ignored");
        let line = r.to_json();
        assert!(line.contains("\"winner\":-1,") && line.contains("\"capped\":true"));
    }

    #[test]
    fn eval_setups_vary_size_preset_and_tribes() {
        let setups: Vec<_> = (1_000_000..1_000_060).map(|s| eval_setup(s, [1, 1])).collect();
        let sizes: std::collections::HashSet<u8> = setups.iter().map(|s| s.size).collect();
        let presets: std::collections::HashSet<u8> = setups.iter().map(|s| s.preset).collect();
        assert_eq!((sizes.len(), presets.len()), (3, 5));
        assert!(setups.iter().all(|s| s.seats.len() == 2 && s.seats[0].tribe != s.seats[1].tribe));
    }

    #[test]
    fn json_line_key_matches_job_key() {
        let a: AgentSpec = "retail:easy".parse().unwrap();
        let b: AgentSpec = "mcts:3".parse().unwrap();
        let mut setup = eval_setup(1_000_004, [0, 1]);
        setup.turn_limit = 2;
        let mut agents = [b.build(1_000_004, 0), a.build(1_000_004, 1)];
        let (mut r, _) = play_game(&setup, &mut agents);
        r.swap = 1;
        let line = r.to_json();
        assert!(!line.contains('\n'));
        assert_eq!(line_key(&line).unwrap(), job_key(1_000_004, 1, "retail:easy", "mcts:3"));
    }
}
