//! The optional engine-feature keys of a shard (`--features`, docs/encoding.md section 12).
//!
//! Like the truth keys they are appended to the finished shard file as more `.npy` entries, so `shard.rs` and `npz.rs`
//! (which the client's `shardify` includes by path) stay as they are, and a shard written without `--features` is
//! unchanged.
//!
//! | key | dtype | shape |
//! |---|---|---|
//! | f_tiles | int16 | [N, H, W, T_F] |
//! | f_globals | float32 | [N, G_F] |
//! | f_actions | float32 | [A, A_F], row `i` belongs to row `i` of `actions` |

use polyai_core::features::{Features, A_F, G_F, T_F};
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

/// The features of the positions of one shard buffer, in the same order as its rows.
pub struct FeatBuf {
    h: usize,
    w: usize,
    n: usize,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    actions: Vec<f32>,
}

impl FeatBuf {
    pub fn new(h: usize, w: usize) -> Self {
        Self { h, w, n: 0, tiles: vec![], globals: vec![], actions: vec![] }
    }

    pub fn push(&mut self, f: &Features) {
        debug_assert_eq!((f.h, f.w), (self.h, self.w));
        self.n += 1;
        self.tiles.extend_from_slice(&f.tiles);
        self.globals.extend_from_slice(&f.globals);
        self.actions.extend_from_slice(&f.actions);
    }

    /// Add `f_tiles`, `f_globals` and `f_actions` to the finished shard at `path`.
    pub fn append_to(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
        let mut zip = ZipWriter::new_append(file).map_err(std::io::Error::other)?;
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(3)).large_file(true);
        let (n, h, w) = (self.n, self.h, self.w);
        let tiles: Vec<u8> = self.tiles.iter().flat_map(|v| v.to_le_bytes()).collect();
        let globals: Vec<u8> = self.globals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let actions: Vec<u8> = self.actions.iter().flat_map(|v| v.to_le_bytes()).collect();
        for (name, descr, shape, body) in [
            ("f_tiles", "<i2", vec![n, h, w, T_F], tiles),
            ("f_globals", "<f4", vec![n, G_F], globals),
            ("f_actions", "<f4", vec![self.actions.len() / A_F, A_F], actions),
        ] {
            zip.start_file(format!("{name}.npy"), opts).map_err(std::io::Error::other)?;
            zip.write_all(&npy(descr, &shape, &body))?;
        }
        zip.finish().map_err(std::io::Error::other)?.sync_all()
    }
}

/// A `.npy` version 1.0 array: little endian, C order, header padded to a multiple of 64 bytes.
fn npy(descr: &str, shape: &[usize], body: &[u8]) -> Vec<u8> {
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({dims}), }}").into_bytes();
    let total = 10 + header.len() + 1;
    header.extend(std::iter::repeat(b' ').take((64 - total % 64) % 64));
    header.push(b'\n');
    let mut out = Vec::with_capacity(10 + header.len() + body.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::{Labels, Row, ShardBuf, K_CAND, MAX_SLOTS};
    use polyai_core::{F_TILE, G_DIM};
    use std::io::Read;

    #[test]
    fn appended_keys_sit_next_to_the_shard_keys() {
        let (h, w) = (2, 3);
        let mut shard = ShardBuf::new(h, w);
        let mut fb = FeatBuf::new(h, w);
        for i in 0..2usize {
            let n_actions = 1 + i;
            let row = Row {
                turn: 1,
                me: 1,
                cmd_index: i as i32,
                difficulty: 0,
                tiles: vec![0; h * w * F_TILE],
                globals: vec![0.0; G_DIM],
                actions: vec![[26, 0, -1, -1]; n_actions],
                chosen: 0,
                cand_idx: [-1; K_CAND],
                cand_score: [0.0; K_CAND],
            };
            shard.push(5, &row, &Labels { winner: 0, final_scores: [0.0; MAX_SLOTS], capped: 0 });
            let mut f = Features { h, w, actions: vec![0.0; n_actions * A_F], tiles: vec![0; h * w * T_F], globals: vec![0.0; G_F] };
            f.actions[(n_actions - 1) * A_F + 2] = 0.5 + i as f32;
            f.tiles[T_F + 1] = 10 + i as i16;
            f.globals[3] = 7.0 + i as f32;
            fb.push(&f);
        }
        let path = std::env::temp_dir().join(format!("feat_shard_{}.npz", std::process::id()));
        shard.write(&path).unwrap();
        fb.append_to(&path).unwrap();
        let mut z = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let names: Vec<String> = z.file_names().map(String::from).collect();
        for k in ["tiles.npy", "actions.npy", "act_off.npy", "f_tiles.npy", "f_globals.npy", "f_actions.npy"] {
            assert!(names.iter().any(|n| n == k), "{k} missing from {names:?}");
        }
        let mut read = |name: &str| {
            let mut b = Vec::new();
            z.by_name(name).unwrap().read_to_end(&mut b).unwrap();
            let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
            assert_eq!((10 + hl) % 64, 0);
            (String::from_utf8(b[10..10 + hl].to_vec()).unwrap(), b[10 + hl..].to_vec())
        };
        let (header, body) = read("f_tiles.npy");
        assert!(header.contains("'<i2'") && header.contains(&format!("(2, 2, 3, {T_F})")), "{header}");
        let per_pos = h * w * T_F * 2;
        assert_eq!(i16::from_le_bytes([body[(T_F + 1) * 2], body[(T_F + 1) * 2 + 1]]), 10);
        assert_eq!(i16::from_le_bytes([body[per_pos + (T_F + 1) * 2], body[per_pos + (T_F + 1) * 2 + 1]]), 11);
        let (header, body) = read("f_globals.npy");
        assert!(header.contains("'<f4'") && header.contains(&format!("(2, {G_F})")), "{header}");
        let at = |i: usize| f32::from_le_bytes([body[i * 4], body[i * 4 + 1], body[i * 4 + 2], body[i * 4 + 3]]);
        assert_eq!((at(3), at(G_F + 3)), (7.0, 8.0));
        let (header, body) = read("f_actions.npy");
        assert!(header.contains(&format!("(3, {A_F})")), "{header}");
        let at = |i: usize| f32::from_le_bytes([body[i * 4], body[i * 4 + 1], body[i * 4 + 2], body[i * 4 + 3]]);
        assert_eq!((at(2), at(2 * A_F + 2)), (0.5, 1.5));
        std::fs::remove_file(&path).unwrap();
    }
}
