//! Playing a setup out with the retail AI, recording every command.

use crate::record::{GameRecord, Outcome, ENGINE_COMMIT};
use crate::setup::GameSetup;
use polytopia_sim::engine::{Command, Engine};

/// A safety net against a game that never ends: well above any game that reaches the turn cap (150 turns is about
/// 10k commands).
pub const MAX_COMMANDS: usize = 200_000;
/// The retail AI ends its turn after this many commands (same cap as the engine's `step_ai_cmd`).
pub const MAX_PER_TURN: u32 = 400;

/// The command the retail AI plays next, made safe to apply: after [`MAX_PER_TURN`] commands, or when the pick is not
/// legal, it ends the turn (the same rules as `Engine::step_ai_cmd`, but without applying it).
pub fn ai_pick(e: &mut Engine, per_turn: u32) -> Command {
    let cmd = e.ai_get_move();
    if per_turn + 1 > MAX_PER_TURN {
        return Command::EndTurn;
    }
    if !matches!(cmd, Command::EndTurn | Command::Stay { .. }) && !e.is_legal(&cmd) {
        return Command::EndTurn;
    }
    cmd
}

/// Tracks the per-turn command count the way `Engine::step_ai_cmd` does.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurnCounter(pub u32);

impl TurnCounter {
    /// Call after applying `cmd`, with the turn and player from before it.
    pub fn after(&mut self, e: &Engine, cmd: &Command, turn_before: u32, player_before: u8) {
        if matches!(cmd, Command::EndTurn) || e.s.current_turn != turn_before || e.current_player() != player_before {
            self.0 = 0;
        } else {
            self.0 += 1;
        }
    }
}

/// Play the setup to the end with every seat on the retail AI. `on_decision(engine, command)` is called with the
/// position before each command (Nature's included) and the command about to be played; the engine may be used
/// mutably (for `ai_candidates`) as long as the game state is not changed.
pub fn play_ai_game(setup: &GameSetup, mut on_decision: impl FnMut(&mut Engine, &Command)) -> (GameRecord, Engine) {
    let mut e = setup.build();
    let mut commands = Vec::new();
    let mut counter = TurnCounter::default();
    let cap = setup.own_cap();
    while !crate::setup::finished(&e, cap) && commands.len() < MAX_COMMANDS {
        let cmd = ai_pick(&mut e, counter.0);
        on_decision(&mut e, &cmd);
        let (turn, player) = (e.s.current_turn, e.current_player());
        e.apply_unchecked(cmd);
        commands.push(cmd);
        counter.after(&e, &cmd, turn, player);
    }
    let record = GameRecord {
        engine_version: polytopia_sim::GAME_VERSION,
        engine_commit: ENGINE_COMMIT.to_string(),
        setup: setup.clone(),
        commands,
        outcome: Outcome::with_cap(&e, cap),
    };
    (record, e)
}
