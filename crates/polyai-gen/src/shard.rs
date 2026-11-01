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

/// Where a position came from (docs/interfaces.md section 2.2). The generator's positions have [`Origin::generated`].
pub struct Origin<'a> {
    /// 0 retail generator, 1 self-play, 2 live, 3 human replay.
    pub source: i8,
    /// 1 when the mover is one of our accounts.
    pub ours: i8,
    /// `logp` of the policy that chose the move; NaN for retail, human and fallback moves.
    pub behaviour_logp: f32,
    /// The `policy_id` of that policy, or `None`.
    pub policy_id: Option<&'a str>,
    /// Identifies the game, so a game is never split across train and validation.
    pub game_key: &'a str,
}

impl Origin<'static> {
    /// A retail-generator position: `source = 0`, not ours, no behaviour policy. The caller supplies the key.
    pub fn generated(game_key: &str) -> Origin<'_> {
        Origin {
            source: 0,
            ours: 0,
            behaviour_logp: f32::NAN,
            policy_id: None,
            game_key,
        }
    }
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
    source: Vec<i8>,
    ours: Vec<i8>,
    behaviour_logp: Vec<f32>,
    policy: Vec<i16>,
    policy_ids: Vec<String>,
    game_keys: Vec<String>,
    game_key: Vec<u32>,
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
            source: vec![],
            ours: vec![],
            behaviour_logp: vec![],
            policy: vec![],
            policy_ids: vec![],
            game_keys: vec![],
            game_key: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    /// Add a retail-generator position; its game key is `gen:<seed>`.
    pub fn push(&mut self, seed: i32, r: &Row, l: &Labels) {
        self.push_with(seed, r, l, &Origin::generated(&format!("gen:{seed}")));
    }

    /// Add a position with its provenance. Games are expected to arrive one after the other, so a key equal to the
    /// previous row's is stored once.
    pub fn push_with(&mut self, seed: i32, r: &Row, l: &Labels, o: &Origin<'_>) {
        let policy = match o.policy_id {
            None => -1,
            Some(id) => match self.policy_ids.iter().position(|p| p == id) {
                Some(i) => i as i16,
                None => {
                    assert!(
                        self.policy_ids.len() < i16::MAX as usize,
                        "too many policy ids in one shard"
                    );
                    self.policy_ids.push(id.to_string());
                    (self.policy_ids.len() - 1) as i16
                }
            },
        };
        if self.game_keys.last().map(String::as_str) != Some(o.game_key) {
            self.game_keys.push(o.game_key.to_string());
        }
        self.game_key.push((self.game_keys.len() - 1) as u32);
        self.source.push(o.source);
        self.ours.push(o.ours);
        self.behaviour_logp.push(o.behaviour_logp);
        self.policy.push(policy);
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
        z.add("source", &[n], &self.source)?;
        z.add("ours", &[n], &self.ours)?;
        z.add("behaviour_logp", &[n], &self.behaviour_logp)?;
        z.add("policy", &[n], &self.policy)?;
        z.add_strs("policy_ids", &self.policy_ids)?;
        let keys: Vec<&str> = self
            .game_key
            .iter()
            .map(|&k| self.game_keys[k as usize].as_str())
            .collect();
        z.add_strs("game_key", &keys)?;
        z.add("version", &[1], &[ENCODING_VERSION])?;
        z.finish()?
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn row(h: usize, w: usize, n_actions: usize) -> Row {
        Row {
            turn: 3,
            me: 0,
            cmd_index: 7,
            difficulty: 2,
            tiles: vec![0; h * w * F_TILE],
            globals: vec![0.0; G_DIM],
            actions: vec![[1, 2, 3, 4]; n_actions],
            chosen: 0,
            cand_idx: [-1; K_CAND],
            cand_score: [0.0; K_CAND],
        }
    }

    fn labels() -> Labels {
        Labels {
            winner: 0,
            final_scores: [0.0; MAX_SLOTS],
            capped: 0,
        }
    }

    /// The `.npy` entries of an npz as (descr, shape text, payload bytes).
    fn read_npz(path: &Path) -> std::collections::HashMap<String, (String, String, Vec<u8>)> {
        let mut z = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
        let mut out = std::collections::HashMap::new();
        for i in 0..z.len() {
            let mut f = z.by_index(i).unwrap();
            let name = f.name().trim_end_matches(".npy").to_string();
            let mut b = Vec::new();
            f.read_to_end(&mut b).unwrap();
            assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
            let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
            assert_eq!((10 + hl) % 64, 0, "{name} header is not 64-aligned");
            let header = String::from_utf8(b[10..10 + hl].to_vec()).unwrap();
            let grab = |key: &str, close: char| -> String {
                let at = header.find(key).unwrap() + key.len();
                header[at..].split(close).next().unwrap().to_string()
            };
            out.insert(
                name,
                (
                    grab("'descr': '", '\''),
                    grab("'shape': (", ')'),
                    b[10 + hl..].to_vec(),
                ),
            );
        }
        out
    }

    fn ucs4(payload: &[u8], width: usize) -> Vec<String> {
        payload
            .chunks(width * 4)
            .map(|c| {
                c.chunks(4)
                    .map(|u| u32::from_le_bytes([u[0], u[1], u[2], u[3]]))
                    .take_while(|&u| u != 0)
                    .map(|u| char::from_u32(u).unwrap())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn generated_rows_carry_the_default_provenance() {
        let mut b = ShardBuf::new(2, 2);
        b.push(41, &row(2, 2, 3), &labels());
        b.push(41, &row(2, 2, 2), &labels());
        b.push(42, &row(2, 2, 1), &labels());
        let path = std::env::temp_dir().join(format!("shard_gen_{}.npz", std::process::id()));
        b.write(&path).unwrap();
        let z = read_npz(&path);
        std::fs::remove_file(&path).unwrap();
        for k in [
            "tiles",
            "globals",
            "act_off",
            "actions",
            "chosen",
            "cand_idx",
            "cand_score",
            "winner",
            "final_scores",
            "capped",
            "turn",
            "seed",
            "cmd_index",
            "difficulty",
            "version",
        ] {
            assert!(z.contains_key(k), "{k}");
        }
        assert_eq!(z["version"].2, (ENCODING_VERSION).to_le_bytes());
        assert_eq!(
            (
                z["source"].0.as_str(),
                z["source"].1.as_str(),
                z["source"].2.clone()
            ),
            ("|i1", "3,", vec![0, 0, 0])
        );
        assert_eq!(z["ours"].2, vec![0, 0, 0]);
        assert_eq!(
            (z["behaviour_logp"].0.as_str(), z["behaviour_logp"].2.len()),
            ("<f4", 12)
        );
        assert!(z["behaviour_logp"]
            .2
            .chunks(4)
            .all(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_nan()));
        assert_eq!(
            (z["policy"].0.as_str(), z["policy"].2.clone()),
            ("<i2", vec![0xff; 6])
        );
        assert_eq!(
            (
                z["policy_ids"].0.as_str(),
                z["policy_ids"].1.as_str(),
                z["policy_ids"].2.len()
            ),
            ("<U1", "0,", 0)
        );
        assert_eq!(z["game_key"].0, "<U6");
        assert_eq!(ucs4(&z["game_key"].2, 6), ["gen:41", "gen:41", "gen:42"]);
    }

    #[test]
    fn live_rows_carry_policy_ids_and_keys() {
        let mut b = ShardBuf::new(2, 2);
        let ours = |id: Option<&'static str>, logp: f32| Origin {
            source: 2,
            ours: 1,
            behaviour_logp: logp,
            policy_id: id,
            game_key: "live:0123-abc",
        };
        b.push_with(1, &row(2, 2, 2), &labels(), &ours(Some("run_a@100"), -0.5));
        b.push_with(
            1,
            &row(2, 2, 2),
            &labels(),
            &Origin {
                source: 2,
                ours: 0,
                behaviour_logp: f32::NAN,
                policy_id: None,
                game_key: "live:0123-abc",
            },
        );
        b.push_with(1, &row(2, 2, 2), &labels(), &ours(Some("run_b@7"), -1.5));
        b.push_with(
            2,
            &row(2, 2, 2),
            &labels(),
            &Origin {
                game_key: "live:ünï",
                ..ours(Some("run_a@100"), -2.0)
            },
        );
        assert_eq!(b.len(), 4);
        let path = std::env::temp_dir().join(format!("shard_live_{}.npz", std::process::id()));
        b.write(&path).unwrap();
        let z = read_npz(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(z["source"].2, vec![2, 2, 2, 2]);
        assert_eq!(z["ours"].2, vec![1, 0, 1, 1]);
        let lp: Vec<f32> = z["behaviour_logp"]
            .2
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!((lp[0], lp[2], lp[3]), (-0.5, -1.5, -2.0));
        assert!(lp[1].is_nan());
        let pol: Vec<i16> = z["policy"]
            .2
            .chunks(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(pol, [0, -1, 1, 0]);
        assert_eq!(z["policy_ids"].0, "<U9");
        assert_eq!(ucs4(&z["policy_ids"].2, 9), ["run_a@100", "run_b@7"]);
        assert_eq!(z["game_key"].0, "<U13");
        assert_eq!(
            ucs4(&z["game_key"].2, 13),
            [
                "live:0123-abc",
                "live:0123-abc",
                "live:0123-abc",
                "live:ünï"
            ]
        );
    }
}
