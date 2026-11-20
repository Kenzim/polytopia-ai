//! Value shards: one row per player per sampled position, each encoded from that player's own fogged view, for
//! training and checking a win estimate that every seat can read at any moment of the game (not only when it is
//! to move). They carry no legal list and no move, so they train a value head only.
//!
//! | key | dtype | shape | meaning |
//! |---|---|---|---|
//! | tiles | int16 | [N, H, W, F_TILE] | the row's player's view (encoding.md section 2) |
//! | globals | float32 | [N, G_DIM] | the same view's globals |
//! | winner | int8 | [N] | the winner's slot relative to the row's player (0: this player won), -1 when undecided |
//! | seat | int8 | [N] | engine player id of the row's player |
//! | mover | int8 | [N] | 1 when the row's player is the one to move |
//! | turn | int16 | [N] | game turn |
//! | n_players | int8 | [N] | players in the game (Nature left out) |
//! | alive | int8 | [N] | players still alive at this position |
//! | mode | int8 | [N] | the game's rules mode (1 Perfection, 2 Domination, 3 Glory, 4 Might, 5 Custom) |
//! | slot_kind | int8 | [N, 8] | who plays each player slot of the row's player (`polyai_core::slot_kinds`): 0 none, 1 a person, 2 + d a bot of difficulty d |
//! | game_key | str | [N] | `<prefix>:<game id>`, so a game is never split between training and checking |
//! | version | int32 | [1] | `ENCODING_VERSION` |

use crate::npz::Npz;
use polyai_core::{Obs, ENCODING_VERSION, F_TILE, G_DIM};
use std::path::Path;

/// The labels and facts of one row besides its encoding.
pub struct ValueRow<'a> {
    pub obs: Obs,
    pub winner: i8,
    pub seat: u8,
    pub mover: bool,
    pub turn: i16,
    pub n_players: u8,
    pub alive: u8,
    pub mode: i8,
    pub slot_kind: [i8; 8],
    pub game_key: &'a str,
}

/// The rows of one map size, written as one `.npz`.
pub struct ValueBuf {
    h: usize,
    w: usize,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    winner: Vec<i8>,
    seat: Vec<i8>,
    mover: Vec<i8>,
    turn: Vec<i16>,
    n_players: Vec<i8>,
    alive: Vec<i8>,
    mode: Vec<i8>,
    slot_kind: Vec<i8>,
    game_key: Vec<String>,
}

impl ValueBuf {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            tiles: vec![],
            globals: vec![],
            winner: vec![],
            seat: vec![],
            mover: vec![],
            turn: vec![],
            n_players: vec![],
            alive: vec![],
            mode: vec![],
            slot_kind: vec![],
            game_key: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.winner.len()
    }

    pub fn push(&mut self, r: ValueRow<'_>) {
        assert_eq!((r.obs.h, r.obs.w), (self.h, self.w), "a value row of another map size");
        self.tiles.extend_from_slice(&r.obs.tiles);
        self.globals.extend_from_slice(&r.obs.globals);
        self.winner.push(r.winner);
        self.seat.push(r.seat as i8);
        self.mover.push(i8::from(r.mover));
        self.turn.push(r.turn);
        self.n_players.push(r.n_players as i8);
        self.alive.push(r.alive as i8);
        self.mode.push(r.mode);
        self.slot_kind.extend_from_slice(&r.slot_kind);
        self.game_key.push(r.game_key.to_string());
    }

    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        let n = self.len();
        let mut z = Npz::new(std::fs::File::create(path)?);
        z.add("tiles", &[n, self.h, self.w, F_TILE], &self.tiles)?;
        z.add("globals", &[n, G_DIM], &self.globals)?;
        z.add("winner", &[n], &self.winner)?;
        z.add("seat", &[n], &self.seat)?;
        z.add("mover", &[n], &self.mover)?;
        z.add("turn", &[n], &self.turn)?;
        z.add("n_players", &[n], &self.n_players)?;
        z.add("alive", &[n], &self.alive)?;
        z.add("mode", &[n], &self.mode)?;
        z.add("slot_kind", &[n, 8], &self.slot_kind)?;
        z.add_strs("game_key", &self.game_key)?;
        z.add("version", &[1], &[ENCODING_VERSION])?;
        z.finish()?;
        Ok(())
    }
}
