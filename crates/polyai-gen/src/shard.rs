//! Position rows, their outcome labels and the columnar shard buffer (docs/encoding.md sections 5 and 6).

use crate::npz::Npz;
use polyai_core::{ENCODING_VERSION, F_TILE, G_DIM};
use std::path::Path;

pub const K_CAND: usize = 16;
pub const MAX_SLOTS: usize = 8;

/// One recorded decision, before its outcome is known.
pub struct Row {
    pub turn: i16,
    pub me: u8,
    pub cmd_index: i32,
    pub difficulty: i8,
    pub tiles: Vec<i16>,
    pub globals: Vec<f32>,
    pub actions: Vec<[i16; 4]>,
    pub chosen: i16,
    pub cand_idx: [i16; K_CAND],
    pub cand_score: [f32; K_CAND],
}

/// The labels of a row, from the finished game.
pub struct Labels {
    pub winner: i8,
    pub final_scores: [f32; MAX_SLOTS],
    pub capped: i8,
}

/// Positions of one map size, column by column.
pub struct ShardBuf {
    pub h: usize,
    pub w: usize,
    n: usize,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    act_off: Vec<i32>,
    actions: Vec<i16>,
    chosen: Vec<i16>,
    cand_idx: Vec<i16>,
    cand_score: Vec<f32>,
    winner: Vec<i8>,
    final_scores: Vec<f32>,
    capped: Vec<i8>,
    turn: Vec<i16>,
    seed: Vec<i32>,
    cmd_index: Vec<i32>,
    difficulty: Vec<i8>,
}

impl ShardBuf {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            h,
            w,
            n: 0,
            tiles: vec![],
            globals: vec![],
            act_off: vec![0],
            actions: vec![],
            chosen: vec![],
            cand_idx: vec![],
            cand_score: vec![],
            winner: vec![],
            final_scores: vec![],
            capped: vec![],
            turn: vec![],
            seed: vec![],
            cmd_index: vec![],
            difficulty: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn push(&mut self, seed: i32, r: &Row, l: &Labels) {
        debug_assert_eq!(r.tiles.len(), self.h * self.w * F_TILE);
        debug_assert_eq!(r.globals.len(), G_DIM);
        self.n += 1;
        self.tiles.extend_from_slice(&r.tiles);
        self.globals.extend_from_slice(&r.globals);
        self.actions.extend(r.actions.iter().flatten());
        self.act_off.push(self.actions.len() as i32 / 4);
        self.chosen.push(r.chosen);
        self.cand_idx.extend_from_slice(&r.cand_idx);
        self.cand_score.extend_from_slice(&r.cand_score);
        self.winner.push(l.winner);
        self.final_scores.extend_from_slice(&l.final_scores);
        self.capped.push(l.capped);
        self.turn.push(r.turn);
        self.seed.push(seed);
        self.cmd_index.push(r.cmd_index);
        self.difficulty.push(r.difficulty);
    }

    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        let n = self.n;
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut z = Npz::new(file);
        z.add("tiles", &[n, self.h, self.w, F_TILE], &self.tiles)?;
        z.add("globals", &[n, G_DIM], &self.globals)?;
        z.add("act_off", &[n + 1], &self.act_off)?;
        z.add("actions", &[self.actions.len() / 4, 4], &self.actions)?;
        z.add("chosen", &[n], &self.chosen)?;
        z.add("cand_idx", &[n, K_CAND], &self.cand_idx)?;
        z.add("cand_score", &[n, K_CAND], &self.cand_score)?;
        z.add("winner", &[n], &self.winner)?;
        z.add("final_scores", &[n, MAX_SLOTS], &self.final_scores)?;
        z.add("capped", &[n], &self.capped)?;
        z.add("turn", &[n], &self.turn)?;
        z.add("seed", &[n], &self.seed)?;
        z.add("cmd_index", &[n], &self.cmd_index)?;
        z.add("difficulty", &[n], &self.difficulty)?;
        z.add("version", &[1], &[ENCODING_VERSION])?;
        z.finish()?.into_inner().map_err(|e| e.into_error())?.sync_all()
    }
}
