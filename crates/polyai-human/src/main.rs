//! `polyai-human --raw DIR --out DIR [--meta FILE] [--keep K] [--shard-size P]`: replay finished human games through
//! the engine and write them as training shards with `source = 3` and `game_key = human:<game id>`.
//!
//! Input: `DIR` holds `<id>_init.bin` and `<id>_cur.bin` per game, the server's initial and current game state blobs
//! (what `re/tools/share_grab.sh` writes; `SaveFile::parse_server` reads them). Every game gets one JSON line in the
//! meta file (status, mode, size, seats, tribes, command counts, illegal commands, outcome, rows), so skipped games and
//! the reason are on record.
//!
//! Rows: before every command of a human seat (a seat the game does not mark `auto_play`) the position is encoded
//! exactly as `polyai-gen` encodes its own games, with the engine's legal action list and the engine AI's candidate
//! scores. A command the engine cannot map, or one outside its legal list, is counted (`unmapped`, `illegal`) and left
//! out. `difficulty` is -1 (a human), `ours` is 0 and there is no behaviour policy (NaN).
//!
//! Labels (docs/encoding.md section 5): the winner only by elimination (or capitals for Might), otherwise `winner = -1`
//! and `capped = 1`; points never decide a game. `final_scores` are logged as the engine computes them. With
//! `--resign-wins`, a two-seat game the engine leaves undecided in which exactly one seat resigned (or was kicked) is
//! labelled won by the other seat.
//!
//! `--truth` adds the keys `truth`, `truth_mask` and `truth_g` (what lies under the mover's fog, docs/belief.md) to
//! every shard. `--part I/N` converts only every N-th game, starting at the I-th (0-based) of the sorted ids, so N
//! runs with their own `--out` cover the games in parallel.
//!
//! Scope: the 12 normal tribes plus Cymanti (`polyai_core::TRIBES`). A game with any other tribe is skipped and counted.

#[path = "../../polyai-gen/src/npz.rs"]
#[allow(dead_code)]
mod npz;
#[path = "../../polyai-gen/src/shard.rs"]
#[allow(dead_code)]
mod shard;
#[path = "../../polyai-gen/src/truth_shard.rs"]
mod truth_shard;

use polyai_core::truth::{truth, truth_globals, Truth, TRUTH_G};
use polyai_core::*;
use polytopia_sim::engine::replay::SaveFile;
use polytopia_sim::engine::{Engine, NATURE};
use shard::{Labels, Origin, Row, ShardBuf, K_CAND, MAX_SLOTS};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use truth_shard::TruthBuf;

/// The command kind of the end-match command (not a decision).
const END_MATCH: u16 = 22;
/// Kind 35 is the resign command (resigning player, kicker, kicked flag). It is not applied to the engine here (no
/// effect on the replays the engine is verified on); it is counted, not treated as an unmapped command, and its player
/// is recorded in `resigned`.
const NO_COMMAND: u16 = 35;

const USAGE: &str = "usage: polyai-human --raw DIR --out DIR [--meta FILE] [--keep K] [--shard-size P] [--limit N] \
                     [--seats FILE] [--key-prefix P] [--resign-wins] [--truth] [--part I/N]";

/// `--seats FILE`: one game per line, `<id> <player>[,<player>...]`; only listed games are replayed and only the
/// listed players' moves are recorded (for games where every seat is a person to the server, such as our own).
fn read_seats(path: &Path) -> HashMap<String, Vec<u8>> {
    std::fs::read_to_string(path)
        .expect("read the seats file")
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let id = it.next()?.to_string();
            let players = it.next()?.split(',').map(|p| p.parse().expect("player id")).collect();
            Some((id, players))
        })
        .collect()
}

#[derive(Default)]
struct Meta {
    id: String,
    status: String,
    reason: String,
    mode: i32,
    height: usize,
    width: usize,
    players: usize,
    humans: usize,
    tribes: Vec<i32>,
    turn_limit: i32,
    turns: u32,
    over: bool,
    capped: bool,
    winners: Vec<u8>,
    commands: usize,
    unmapped: usize,
    illegal: usize,
    off_turn: usize,
    end_match: usize,
    no_command: usize,
    after_end: usize,
    resigned: Vec<u8>,
    rows: usize,
    left_out: Vec<String>,
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn list<T: ToString>(v: &[T]) -> String {
    format!("[{}]", v.iter().map(T::to_string).collect::<Vec<_>>().join(","))
}

impl Meta {
    fn json(&self) -> String {
        let left: Vec<String> = self.left_out.iter().map(|s| esc(s)).collect();
        format!(
            "{{\"id\":{},\"status\":{},\"reason\":{},\"mode\":{},\"height\":{},\"width\":{},\"players\":{},\"humans\":{},\
             \"tribes\":{},\"turn_limit\":{},\"turns\":{},\"over\":{},\"capped\":{},\"winners\":{},\"commands\":{},\
             \"unmapped\":{},\"illegal\":{},\"off_turn\":{},\"end_match\":{},\"no_command\":{},\"after_end\":{},\"resigned\":{},\"rows\":{},\"left_out\":[{}]}}",
            esc(&self.id),
            esc(&self.status),
            esc(&self.reason),
            self.mode,
            self.height,
            self.width,
            self.players,
            self.humans,
            list(&self.tribes),
            self.turn_limit,
            self.turns,
            self.over,
            self.capped,
            list(&self.winners),
            self.commands,
            self.unmapped,
            self.illegal,
            self.off_turn,
            self.end_match,
            self.no_command,
            self.after_end,
            list(&self.resigned),
            self.rows,
            left.join(",")
        )
    }
}

fn seed_of(id: &str) -> i32 {
    let mut h: u32 = 2166136261;
    for b in id.bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(16777619);
    }
    (h & 0x7fff_ffff) as i32
}

/// One position in `keep` is written (1: all), chosen by a hash of the game and the command index.
fn kept(seed: i32, index: usize, keep: u32) -> bool {
    if keep <= 1 {
        return true;
    }
    let mut h: u32 = 2166136261 ^ seed as u32;
    for b in (index as u32).to_le_bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(16777619);
    }
    h % keep == 0
}

type Rows = Vec<(Row, Labels)>;
/// The truth labels and scalars of the rows, in the same order; empty without `--truth`.
type Truths = Vec<(Truth, [f32; TRUTH_G])>;

/// Replay one game. `Err` is a skip with its reason; the meta is filled either way as far as it got.
/// `only`: when given, the recorded seats are these player ids instead of every seat not marked `auto_play`.
/// `resign_wins`, `with_truth`: see `--resign-wins` and `--truth` in the module documentation.
#[allow(clippy::too_many_arguments)]
fn replay(
    id: &str,
    initial: &[u8],
    current: &[u8],
    keep: u32,
    only: Option<&[u8]>,
    resign_wins: bool,
    with_truth: bool,
    m: &mut Meta,
) -> Result<(Rows, Truths, i32), String> {
    let save = SaveFile::parse_server(initial, current).map_err(|e| format!("cannot decode the blobs: {e}"))?;
    let mut e = Engine::from_state(save.initial.clone());
    let s = &e.s.settings;
    m.mode = s.base_game_mode;
    m.turn_limit = s.turn_limit;
    m.height = e.height() as usize;
    m.width = e.width() as usize;
    let seats: Vec<(u8, i32, bool)> = e
        .s
        .players
        .iter()
        .filter(|p| p.id != NATURE)
        .map(|p| (p.id, p.tribe, only.map_or(!p.auto_play, |o| o.contains(&p.id))))
        .collect();
    m.players = seats.len();
    m.humans = seats.iter().filter(|s| s.2).count();
    m.tribes = seats.iter().map(|s| s.1).collect();
    m.commands = save.commands.len();
    if let Some(t) = m.tribes.iter().find(|t| !TRIBES.contains(t)) {
        return Err(format!("tribe {t} is outside the supported tribes"));
    }
    if seats.len() > MAX_SLOTS {
        return Err(format!("{} players, the encoding holds {MAX_SLOTS}", seats.len()));
    }
    if m.humans == 0 {
        return Err("no human seat".into());
    }
    if save.commands.iter().any(|c| c.kind == 20) {
        e.start_match();
    }
    let order: Vec<u8> = seats.iter().map(|s| s.0).collect();
    let human = |p: u8| seats.iter().any(|s| s.0 == p && s.2);
    let seed = seed_of(id);
    let mut rows: Vec<Row> = Vec::new();
    let mut truths: Truths = Vec::new();
    for (i, raw) in save.commands.iter().enumerate() {
        if raw.kind == 20 || raw.player == NATURE {
            continue;
        }
        let Some(cmd) = e.command_from_raw(raw) else {
            e.s.current_command += 1;
            if raw.kind == END_MATCH {
                m.end_match += 1;
            } else if raw.kind == NO_COMMAND {
                m.no_command += 1;
                let p = raw.n as u8;
                if seats.iter().any(|s| s.0 == p) && !m.resigned.contains(&p) {
                    m.resigned.push(p);
                }
            } else {
                m.unmapped += 1;
                m.left_out.push(format!("{i} kind {} player {} (unmapped)", raw.kind, raw.player));
            }
            continue;
        };
        let me = e.current_player();
        if me != raw.player {
            m.off_turn += 1;
        } else if human(me) {
            // The legal list is needed to count an illegal command, so it is built for every human command; only
            // the encoding and the AI candidates (the expensive part) wait for the kept positions.
            let slots = SlotMap::new(&e, me);
            let legal = legal_actions(&mut e);
            match legal.iter().position(|c| *c == cmd) {
                None if e.is_over() => m.after_end += 1,
                None => {
                    m.illegal += 1;
                    m.left_out.push(format!("{i} kind {} player {} (illegal)", raw.kind, raw.player));
                }
                Some(chosen) if kept(seed, i, keep) => {
                    let obs = encode(&e, me, &slots, &legal);
                    if with_truth {
                        truths.push((truth(&e, me, &slots), truth_globals(&e, me)));
                    }
                    let actions = legal
                        .iter()
                        .map(|c| encode_command(&e, &slots, c).expect("legal command encodes"))
                        .collect();
                    let mut cand_idx = [-1i16; K_CAND];
                    let mut cand_score = [0f32; K_CAND];
                    let mut n = 0;
                    for (c, score) in e.ai_candidates() {
                        if n == K_CAND {
                            break;
                        }
                        if let Some(k) = legal.iter().position(|l| *l == c) {
                            if !cand_idx[..n].contains(&(k as i16)) {
                                cand_idx[n] = k as i16;
                                cand_score[n] = score;
                                n += 1;
                            }
                        }
                    }
                    rows.push(Row {
                        turn: e.s.current_turn as i16,
                        me,
                        cmd_index: i as i32,
                        difficulty: -1,
                        tiles: obs.tiles,
                        globals: obs.globals,
                        actions,
                        chosen: chosen as i16,
                        cand_idx,
                        cand_score,
                    });
                }
                Some(_) => {}
            }
        }
        e.apply_unchecked(cmd);
    }
    let o = Outcome::of(&e);
    m.turns = o.turns;
    m.over = o.over;
    m.capped = o.capped || o.is_draw();
    m.winners = if m.capped { Vec::new() } else { o.winners.clone() };
    if let Some(w) = resign_winner(resign_wins, &order, &m.winners, &m.resigned) {
        m.capped = false;
        m.winners = vec![w];
    }
    let out: Rows = rows
        .into_iter()
        .map(|r| {
            let ids = slot_ids(&order, r.me);
            let mut final_scores = [0f32; MAX_SLOTS];
            for k in 0..MAX_SLOTS {
                if ids[k] != 0 {
                    final_scores[k] = o.score_of(ids[k]) as f32;
                }
            }
            let winner = m
                .winners
                .iter()
                .find_map(|w| ids[..MAX_SLOTS].iter().position(|&x| x == *w && x != 0))
                .map_or(-1, |k| k as i8);
            let l = Labels {
                winner,
                final_scores,
                capped: i8::from(winner < 0),
            };
            (r, l)
        })
        .collect();
    // A command the engine rejects or cannot map means its state may have drifted from the game's, so every later
    // position is doubtful: the game is counted and left out of the shards.
    if m.illegal > 0 || m.unmapped > 0 {
        return Err(format!("not a clean replay: {} illegal, {} unmapped commands", m.illegal, m.unmapped));
    }
    m.rows = out.len();
    Ok((out, truths, seed))
}

/// The winner `--resign-wins` gives: the other seat of a two-seat game with no winner where exactly one seat resigned.
fn resign_winner(on: bool, seats: &[u8], winners: &[u8], resigned: &[u8]) -> Option<u8> {
    if !on || seats.len() != 2 || !winners.is_empty() || resigned.len() != 1 {
        return None;
    }
    seats.iter().copied().find(|&s| s != resigned[0])
}

fn flush(
    bufs: &mut HashMap<(usize, usize), ShardBuf>,
    tbufs: &mut HashMap<(usize, usize), TruthBuf>,
    key: (usize, usize),
    out: &Path,
    n: &mut usize,
) {
    let t = tbufs.remove(&key);
    if let Some(b) = bufs.remove(&key).filter(|b| b.len() > 0) {
        let path = out.join(format!("human-{:05}-{}x{}.npz", *n, key.0, key.1));
        b.write(&path).expect("write the shard");
        if let Some(t) = t {
            t.append_to(&path).expect("write the truth keys");
        }
        eprintln!("wrote {} ({} positions)", path.display(), b.len());
        *n += 1;
    }
}

fn main() {
    let (mut raw, mut out, mut meta): (Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) = (None, None, None);
    let (mut keep, mut shard_size, mut limit) = (1u32, 50_000usize, usize::MAX);
    let (mut seats, mut prefix): (Option<HashMap<String, Vec<u8>>>, String) = (None, "human".into());
    let (mut resign_wins, mut with_truth, mut part) = (false, false, (0usize, 1usize));
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--resign-wins" => resign_wins = true,
            "--truth" => with_truth = true,
            "--part" => {
                let v = val();
                let (i, n) = v.split_once('/').expect("--part I/N");
                part = (i.parse().expect("--part I/N"), n.parse().expect("--part I/N"));
                assert!(part.0 < part.1, "--part I/N needs I < N");
            }
            "--raw" => raw = Some(val().into()),
            "--out" => out = Some(val().into()),
            "--meta" => meta = Some(val().into()),
            "--keep" => keep = val().parse().expect("--keep K"),
            "--shard-size" => shard_size = val().parse().expect("--shard-size P"),
            "--limit" => limit = val().parse().expect("--limit N"),
            "--seats" => seats = Some(read_seats(Path::new(&val()))),
            "--key-prefix" => prefix = val(),
            _ => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
        }
    }
    let (Some(raw), Some(out)) = (raw, out) else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let meta = meta.unwrap_or_else(|| out.join("meta.jsonl"));
    std::fs::create_dir_all(&out).expect("create the output directory");
    for f in std::fs::read_dir(&out).expect("read the output directory").flatten() {
        let name = f.file_name().to_string_lossy().into_owned();
        if name.starts_with("human-") && name.ends_with(".npz") {
            std::fs::remove_file(f.path()).expect("remove a stale shard");
        }
    }
    let mut ids: Vec<String> = std::fs::read_dir(&raw)
        .expect("read the raw directory")
        .flatten()
        .filter_map(|f| f.file_name().to_string_lossy().strip_suffix("_init.bin").map(str::to_string))
        .filter(|id| raw.join(format!("{id}_cur.bin")).exists())
        .filter(|id| seats.as_ref().is_none_or(|s| s.contains_key(id)))
        .collect();
    ids.sort();
    let mut ids: Vec<String> = ids.into_iter().skip(part.0).step_by(part.1).collect();
    ids.truncate(limit);
    let mut meta_out = std::io::BufWriter::new(std::fs::File::create(&meta).expect("create the meta file"));
    let mut bufs: HashMap<(usize, usize), ShardBuf> = HashMap::new();
    let mut tbufs: HashMap<(usize, usize), TruthBuf> = HashMap::new();
    let mut shards = 0;
    // A panic inside the engine on a strange game is a skip, not the end of the run.
    std::panic::set_hook(Box::new(|_| {}));
    for id in &ids {
        let mut m = Meta { id: id.clone(), ..Meta::default() };
        let read = |suffix: &str| std::fs::read(raw.join(format!("{id}_{suffix}.bin")));
        let result = match (read("init"), read("cur")) {
            (Ok(i), Ok(c)) => {
                let mm = &mut m;
                let only = seats.as_ref().and_then(|s| s.get(id)).map(Vec::as_slice);
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| replay(id, &i, &c, keep, only, resign_wins, with_truth, mm)))
                    .unwrap_or_else(|p| {
                        let why = p
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| p.downcast_ref::<&str>().copied())
                            .unwrap_or("?");
                        Err(format!("the engine panicked: {why}"))
                    })
            }
            _ => Err("cannot read the blobs".into()),
        };
        match result {
            Ok((rows, truths, seed)) => {
                m.status = "ok".into();
                let key = (m.height, m.width);
                let game_key = format!("{prefix}:{id}");
                let buf = bufs.entry(key).or_insert_with(|| ShardBuf::new(key.0, key.1));
                for (r, l) in &rows {
                    let origin = Origin {
                        source: 3,
                        ours: 0,
                        behaviour_logp: f32::NAN,
                        policy_id: None,
                        game_key: &game_key,
                    };
                    buf.push_with(seed, r, l, &origin);
                }
                if with_truth {
                    let tb = tbufs.entry(key).or_insert_with(|| TruthBuf::new(key.0, key.1));
                    for (t, g) in &truths {
                        tb.push(t);
                        tb.push_scalars(g);
                    }
                }
                if buf.len() >= shard_size {
                    flush(&mut bufs, &mut tbufs, key, &out, &mut shards);
                }
            }
            Err(reason) => {
                m.status = if reason.contains("supported tribes") {
                    "unsupported_tribe"
                } else if reason.contains("the encoding holds") {
                    "too_many_players"
                } else if reason.starts_with("not a clean") {
                    "dirty"
                } else {
                    "skipped"
                }
                .into();
                m.reason = reason;
            }
        }
        eprintln!(
            "{id}: {} (mode {}, {}x{}, {} players, tribes {:?}, {} commands, {} illegal, {} unmapped, {} rows, over {}, winners {:?}) {}",
            m.status, m.mode, m.height, m.width, m.players, m.tribes, m.commands, m.illegal, m.unmapped, m.rows,
            m.over, m.winners, m.reason
        );
        writeln!(meta_out, "{}", m.json()).expect("write the meta line");
    }
    let keys: Vec<_> = bufs.keys().copied().collect();
    for key in keys {
        flush(&mut bufs, &mut tbufs, key, &out, &mut shards);
    }
    meta_out.flush().expect("flush the meta file");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_one_keeps_everything_and_keep_k_keeps_about_one_in_k() {
        assert!((0..100).all(|i| kept(7, i, 1)));
        let n = (0..4000).filter(|&i| kept(7, i, 4)).count();
        assert!((800..1200).contains(&n), "{n}");
    }

    #[test]
    fn resign_wins_only_labels_an_undecided_two_seat_game_with_one_resigner() {
        assert_eq!(resign_winner(true, &[1, 2], &[], &[2]), Some(1));
        assert_eq!(resign_winner(false, &[1, 2], &[], &[2]), None);
        assert_eq!(resign_winner(true, &[1, 2], &[2], &[2]), None);
        assert_eq!(resign_winner(true, &[1, 2], &[], &[1, 2]), None);
        assert_eq!(resign_winner(true, &[1, 2, 3], &[], &[2]), None);
    }

    #[test]
    fn a_meta_line_is_one_line_of_json_with_escaped_text() {
        let m = Meta {
            id: "a\"b".into(),
            status: "dirty".into(),
            tribes: vec![2, 17],
            left_out: vec!["1 kind 2\n(x)".into()],
            ..Meta::default()
        };
        let j = m.json();
        assert!(!j.contains('\n'));
        assert!(j.contains("\"id\":\"a\\\"b\"") && j.contains("\"tribes\":[2,17]"));
    }
}
