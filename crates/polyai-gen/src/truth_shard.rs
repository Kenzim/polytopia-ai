//! The optional truth keys of a shard (`--truth`): per-tile labels of the fog and the mask of hidden tiles.
//!
//! They are appended to the finished shard file as two more `.npy` entries, so `shard.rs` and `npz.rs` (which the
//! client's `shardify` includes by path) stay as they are, and a shard written without `--truth` is unchanged.
//!
//! | key | dtype | shape |
//! |---|---|---|
//! | truth | int16 | [N, H, W, TRUTH_K] |
//! | truth_mask | uint8 | [N, H, W] |

use polyai_core::truth::{Truth, TRUTH_K};
use std::io::Write;
use std::path::Path;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

/// The truth labels of the positions of one shard buffer, in the same order as its rows.
pub struct TruthBuf {
    h: usize,
    w: usize,
    n: usize,
    labels: Vec<i16>,
    mask: Vec<u8>,
}

impl TruthBuf {
    pub fn new(h: usize, w: usize) -> Self {
        Self { h, w, n: 0, labels: vec![], mask: vec![] }
    }

    pub fn push(&mut self, t: &Truth) {
        debug_assert_eq!((t.h, t.w), (self.h, self.w));
        self.n += 1;
        self.labels.extend_from_slice(&t.labels);
        self.mask.extend_from_slice(&t.mask);
    }

    /// Add `truth` and `truth_mask` to the finished shard at `path`.
    pub fn append_to(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
        let mut zip = ZipWriter::new_append(file).map_err(std::io::Error::other)?;
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).compression_level(Some(3)).large_file(true);
        let labels: Vec<u8> = self.labels.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (n, h, w) = (self.n, self.h, self.w);
        for (name, descr, shape, body) in [
            ("truth", "<i2", vec![n, h, w, TRUTH_K], labels),
            ("truth_mask", "|u1", vec![n, h, w], self.mask.clone()),
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
        let mut tb = TruthBuf::new(h, w);
        for i in 0..2i16 {
            let row = Row {
                turn: 1,
                me: 1,
                cmd_index: i32::from(i),
                difficulty: 0,
                tiles: vec![0; h * w * F_TILE],
                globals: vec![0.0; G_DIM],
                actions: vec![[26, 0, -1, -1]],
                chosen: 0,
                cand_idx: [-1; K_CAND],
                cand_score: [0.0; K_CAND],
            };
            shard.push(5, &row, &Labels { winner: 0, final_scores: [0.0; MAX_SLOTS], capped: 0 });
            let mut labels = vec![0i16; h * w * TRUTH_K];
            labels[TRUTH_K + 2] = 100 + i;
            let mut mask = vec![0u8; h * w];
            mask[1] = 1;
            tb.push(&Truth { h, w, labels, mask });
        }
        let path = std::env::temp_dir().join(format!("truth_shard_{}.npz", std::process::id()));
        shard.write(&path).unwrap();
        tb.append_to(&path).unwrap();
        let mut z = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let names: Vec<String> = z.file_names().map(String::from).collect();
        for k in ["tiles.npy", "winner.npy", "truth.npy", "truth_mask.npy"] {
            assert!(names.iter().any(|n| n == k), "{k} missing from {names:?}");
        }
        let mut read = |name: &str| {
            let mut b = Vec::new();
            z.by_name(name).unwrap().read_to_end(&mut b).unwrap();
            let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
            assert_eq!((10 + hl) % 64, 0);
            (String::from_utf8(b[10..10 + hl].to_vec()).unwrap(), b[10 + hl..].to_vec())
        };
        let (header, body) = read("truth.npy");
        assert!(header.contains("'<i2'") && header.contains("(2, 2, 3, 6)"), "{header}");
        let at = (TRUTH_K + 2) * 2;
        assert_eq!(i16::from_le_bytes([body[at], body[at + 1]]), 100);
        let per_pos = h * w * TRUTH_K * 2;
        assert_eq!(i16::from_le_bytes([body[per_pos + at], body[per_pos + at + 1]]), 101);
        let (header, body) = read("truth_mask.npy");
        assert!(header.contains("'|u1'") && header.contains("(2, 2, 3)"), "{header}");
        assert_eq!(body, vec![0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        std::fs::remove_file(&path).unwrap();
    }
}
