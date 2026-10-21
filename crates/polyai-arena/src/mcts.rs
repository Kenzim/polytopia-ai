//! Adapter from the engine to weekly's generic UCT (`polytopia_weekly::mcts::Game`).
//!
//! The search tree is over single engine commands, as in the engine. Everything here is full information: the
//! engine state is cloned whole, so the search sees through fog. Rollouts are uniformly random over the legal
//! command list, cut at a depth limit and scored by the score ratio.

use polyai_core::game::{ai_pick, MAX_PER_TURN};
use polytopia_sim::engine::{Command, Engine, NATURE};
use polytopia_weekly::mcts::Game;

/// An engine position as a `Game`. Players are named by engine player id.
#[derive(Clone)]
pub struct PolyGame {
    e: Engine,
    /// When set, the legal list at this node (used to prune the root to the retail AI's top candidates). Cleared by
    /// the first `apply`.
    restrict: Option<Vec<Command>>,
}

impl PolyGame {
    pub fn new(e: &Engine, restrict: Option<Vec<Command>>) -> Self {
        let mut g = PolyGame { e: e.clone(), restrict };
        g.skip_nature();
        g
    }

    /// Nature never gets a decision: it ends its turn at once.
    fn skip_nature(&mut self) {
        let mut guard = 0;
        while !self.e.is_over() && self.e.current_player() == NATURE && guard < 16 {
            let c = ai_pick(&mut self.e, 0);
            self.e.apply_unchecked(c);
            guard += 1;
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.e
    }
}

impl Game for PolyGame {
    type Action = Command;

    fn legal_actions(&self) -> Vec<Command> {
        if let Some(r) = &self.restrict {
            return r.clone();
        }
        self.e.legal_commands()
    }

    fn apply(&mut self, action: &Command) {
        self.restrict = None;
        self.e.apply_unchecked(*action);
        self.skip_nature();
    }

    fn is_terminal(&self) -> bool {
        self.e.is_over()
    }

    /// 1 for a win by elimination, 0.5 for a draw (the turn cap with several players alive; the engine's top-score
    /// winner is not used), 0 for a loss. Before the end: the share of the score held against the best
    /// opponent still alive, so a cut-off rollout still says who is ahead.
    fn reward(&self, perspective: usize) -> f64 {
        let me = perspective as u8;
        if self.e.is_over() {
            if self.e.living_players().len() > 1 {
                return 0.5;
            }
            return match self.e.winners.as_slice() {
                [] => 0.5,
                w if w.contains(&me) => 1.0,
                _ => 0.0,
            };
        }
        let mine = f64::from(self.e.score(me).max(0));
        let best = self
            .e
            .s
            .players
            .iter()
            .filter(|p| p.id != NATURE && p.id != me)
            .map(|p| f64::from(self.e.score(p.id).max(0)))
            .fold(0.0, f64::max);
        if mine + best <= 0.0 {
            0.5
        } else {
            mine / (mine + best)
        }
    }

    fn current_player(&self) -> usize {
        usize::from(self.e.current_player())
    }
}

/// Settings of the search agent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MctsParams {
    pub iterations: u32,
    /// Rollout length in commands.
    pub rollout_depth: u32,
    /// Prune the root to the retail AI's `k` best candidates (the retail difficulty is the seat's handicap).
    pub topk: Option<usize>,
    pub exploration: f64,
}

impl MctsParams {
    pub fn new(iterations: u32) -> Self {
        Self { iterations, rollout_depth: 60, topk: None, exploration: 1.0 }
    }
}

/// The retail AI's top `k` candidates that are legal now (always keeps at least one command).
pub fn top_candidates(e: &mut Engine, k: usize) -> Vec<Command> {
    let legal = e.legal_commands();
    let mut out: Vec<Command> = Vec::new();
    for (c, _) in e.ai_candidates() {
        if out.len() >= k {
            break;
        }
        if legal.contains(&c) && !out.contains(&c) {
            out.push(c);
        }
    }
    if out.is_empty() {
        out = legal;
    }
    out
}

/// One MCTS decision for the player to move. `seed` varies per call.
pub fn choose(e: &mut Engine, p: &MctsParams, per_turn: u32, seed: u64) -> Command {
    if per_turn + 1 > MAX_PER_TURN {
        return Command::EndTurn;
    }
    let legal = e.legal_commands();
    if legal.len() <= 1 {
        return legal.first().copied().unwrap_or(Command::EndTurn);
    }
    let restrict = p.topk.map(|k| top_candidates(e, k));
    if let Some(r) = &restrict {
        if r.len() == 1 {
            return r[0];
        }
    }
    let root = PolyGame::new(e, restrict);
    let cfg = polytopia_weekly::mcts::Config {
        iterations: p.iterations,
        exploration: p.exploration,
        max_rollout_depth: p.rollout_depth,
        seed,
    };
    match polytopia_weekly::mcts::search(&root, cfg) {
        Some(d) => d.action,
        None => Command::EndTurn,
    }
}
