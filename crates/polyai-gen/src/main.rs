//! Plays retail-AI games on N threads and writes game records and `.npz` training shards.
//!
//! `polyai-gen --out DIR --games N [--seed-start S] [--threads T] [--keep K] [--shard-size P] [--difficulty D]
//!  [--turn-limit L] [--players 2:55,3:25,4:20] [--mode domination|might]`
//!
//! `--players` is a weighted list of player counts (`2:55,3:25,4:20` is the default) or a plain list (`2,3`, uniform).
//! `--turn-limit` is the compute cap (default 150): a game still running then is a draw (`capped`), never adjudicated.
//!
//! Seeds `S..S+N` must lie in the training range 1..=100000 (evaluation seeds start at 1,000,000).

mod npz;
mod shard;

use polyai_core::*;
use polytopia_sim::engine::{Command, Engine, NATURE};
use shard::{Labels, Row, ShardBuf, K_CAND, MAX_SLOTS};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const MAX_THREADS: usize = 24;

struct Args {
    out: PathBuf,
    games: usize,
    seed_start: i32,
    threads: usize,
    keep: u64,
    shard_size: usize,
    difficulty: Option<u8>,
    turn_limit: Option<i32>,
    players: Option<String>,
    mode: i32,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { out: PathBuf::new(), games: 0, seed_start: 1, threads: MAX_THREADS, keep: 4, shard_size: 50_000, difficulty: None, turn_limit: None, players: None, mode: DOMINATION };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        let bad = |f: &str| format!("bad value for {f}");
        match flag.as_str() {
            "--out" => a.out = PathBuf::from(val()?),
            "--games" => a.games = val()?.parse().map_err(|_| bad(&flag))?,
            "--seed-start" => a.seed_start = val()?.parse().map_err(|_| bad(&flag))?,
            "--threads" => a.threads = val()?.parse().map_err(|_| bad(&flag))?,
            "--keep" => a.keep = val()?.parse().map_err(|_| bad(&flag))?,
            "--shard-size" => a.shard_size = val()?.parse().map_err(|_| bad(&flag))?,
            "--difficulty" => a.difficulty = Some(val()?.parse().map_err(|_| bad(&flag))?),
            "--turn-limit" => a.turn_limit = Some(val()?.parse().map_err(|_| bad(&flag))?),
            "--players" => a.players = Some(val()?),
            "--mode" => {
                a.mode = match val()?.as_str() {
                    "domination" => DOMINATION,
                    "might" => MIGHT,
                    _ => return Err(bad(&flag)),
                }
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if a.out.as_os_str().is_empty() || a.games == 0 {
        return Err("usage: polyai-gen --out DIR --games N [--seed-start S] [--threads T<=24] [--keep K] [--shard-size P] [--difficulty 0..3] [--turn-limit L] [--players 2:55,3:25,4:20] [--mode domination|might]".into());
    }
    a.threads = a.threads.clamp(1, MAX_THREADS);
    a.keep = a.keep.max(1);
    let last = i64::from(a.seed_start) + a.games as i64 - 1;
    if a.seed_start < 1 || last > i64::from(TRAIN_SEED_MAX) {
        return Err(format!("seeds {}..={} are outside the training range 1..={}", a.seed_start, last, TRAIN_SEED_MAX));
    }
    if a.turn_limit.is_some_and(|l| l < 1) {
        return Err("--turn-limit must be at least 1 (it is the compute cap)".into());
    }
    if a.difficulty.is_some_and(|d| d > 3) {
        return Err("--difficulty is 0 (easy) to 3 (crazy)".into());
    }
    Ok(a)
}

fn mix(seed: i32, cmd_index: u32) -> u64 {
    let mut x = (seed as u32 as u64) << 32 | u64::from(cmd_index);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One finished game: its record and the labelled rows.
struct Played {
    record: GameRecord,
    rows: Vec<Row>,
    labels: Vec<Labels>,
    misses: usize,
    h: usize,
    w: usize,
}

fn play_one(setup: &GameSetup, keep: u64) -> Played {
    let order: Vec<u8> = setup.build().s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut misses = 0usize;
    let mut cmd_index = 0u32;
    let (record, last) = play_ai_game(setup, |e: &mut Engine, cmd: &Command| {
        let index = cmd_index;
        cmd_index += 1;
        let me = e.current_player();
        if me == NATURE || mix(setup.seed, index) % keep != 0 {
            return;
        }
        let slots = SlotMap::new(e, me);
        let legal = legal_actions(e);
        let Some(chosen) = legal.iter().position(|c| c == cmd) else {
            misses += 1;
            return;
        };
        let obs = encode(e, me, &slots, &legal);
        let actions: Vec<[i16; 4]> = legal.iter().map(|c| encode_command(e, &slots, c).expect("legal command encodes")).collect();
        let mut cand_idx = [-1i16; K_CAND];
        let mut cand_score = [0f32; K_CAND];
        let mut n = 0;
        for (c, score) in e.ai_candidates() {
            if n == K_CAND {
                break;
            }
            if let Some(i) = legal.iter().position(|l| *l == c) {
                if !cand_idx[..n].contains(&(i as i16)) {
                    cand_idx[n] = i as i16;
                    cand_score[n] = score;
                    n += 1;
                }
            }
        }
        let seat = order.iter().position(|&id| id == me).unwrap_or(0);
        rows.push(Row {
            turn: e.s.current_turn as i16,
            me,
            cmd_index: index as i32,
            difficulty: setup.seats[seat].difficulty as i8,
            tiles: obs.tiles,
            globals: obs.globals,
            actions,
            chosen: chosen as i16,
            cand_idx,
            cand_score,
        });
    });

    let o = &record.outcome;
    let labels = rows
        .iter()
        .map(|r| {
            let ids = slot_ids(&order, r.me);
            let mut final_scores = [0f32; MAX_SLOTS];
            for k in 0..MAX_SLOTS {
                if ids[k] != 0 {
                    final_scores[k] = o.score_of(ids[k]) as f32;
                }
            }
            let winner = o.winners.iter().find_map(|w| ids[..MAX_SLOTS].iter().position(|&x| x == *w && x != 0)).map_or(-1, |k| k as i8);
            Labels { winner, final_scores, capped: i8::from(o.is_draw()) }
        })
        .collect();
    Played { record, rows, labels, misses, h: last.height() as usize, w: last.width() as usize }
}

struct Shared {
    bufs: Mutex<HashMap<(usize, usize), ShardBuf>>,
    next_shard: AtomicUsize,
    next_game: AtomicUsize,
    games: AtomicUsize,
    positions: AtomicU64,
    commands: AtomicU64,
    misses: AtomicU64,
    capped: AtomicUsize,
    by_players: Mutex<BTreeMap<usize, usize>>,
    shards: Mutex<Vec<(PathBuf, usize)>>,
}

fn write_shard(sh: &Shared, dir: &Path, buf: ShardBuf) {
    let n = sh.next_shard.fetch_add(1, Ordering::SeqCst);
    let path = dir.join(format!("shard-{n:05}-{}x{}.npz", buf.h, buf.w));
    buf.write(&path).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    sh.shards.lock().unwrap().push((path, buf.len()));
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    std::fs::create_dir_all(&args.out).expect("create output directory");
    let sh = Shared {
        bufs: Mutex::new(HashMap::new()),
        next_shard: AtomicUsize::new(0),
        next_game: AtomicUsize::new(0),
        games: AtomicUsize::new(0),
        positions: AtomicU64::new(0),
        commands: AtomicU64::new(0),
        misses: AtomicU64::new(0),
        capped: AtomicUsize::new(0),
        by_players: Mutex::new(BTreeMap::new()),
        shards: Mutex::new(Vec::new()),
    };
    let mut spec = SetupSpec { mode: args.mode, ..SetupSpec::default() };
    if let Some(l) = args.turn_limit {
        spec.turn_limit = l;
    }
    if let Some(p) = &args.players {
        spec = spec.with_players_str(p).unwrap_or_else(|e| {
            eprintln!("--players: {e}");
            std::process::exit(2);
        });
    }
    let started = Instant::now();
    println!("polyai-gen: {} games from seed {} on {} threads, engine {}", args.games, args.seed_start, args.threads, ENGINE_COMMIT);
    std::thread::scope(|scope| {
        for worker in 0..args.threads {
            let (sh, args, spec) = (&sh, &args, &spec);
            scope.spawn(move || {
                let rec_path = args.out.join(format!("records-{worker:02}.polyrec"));
                let mut rec_out = std::io::BufWriter::new(std::fs::File::create(rec_path).expect("create record file"));
                loop {
                    let i = sh.next_game.fetch_add(1, Ordering::SeqCst);
                    if i >= args.games {
                        break;
                    }
                    let seed = args.seed_start + i as i32;
                    let mut setup = GameSetup::sample(seed, spec);
                    if let Some(d) = args.difficulty {
                        setup = setup.with_difficulty(d);
                    }
                    let p = play_one(&setup, args.keep);
                    write_record(&mut rec_out, &p.record).expect("write record");
                    let (h, w) = (p.h, p.w);
                    let mut full = None;
                    {
                        let mut bufs = sh.bufs.lock().unwrap();
                        let buf = bufs.entry((h, w)).or_insert_with(|| ShardBuf::new(h, w));
                        for (r, l) in p.rows.iter().zip(&p.labels) {
                            buf.push(seed, r, l);
                        }
                        if buf.len() >= args.shard_size {
                            full = bufs.remove(&(h, w));
                        }
                    }
                    if let Some(buf) = full {
                        write_shard(sh, &args.out, buf);
                    }
                    sh.positions.fetch_add(p.rows.len() as u64, Ordering::Relaxed);
                    sh.commands.fetch_add(p.record.commands.len() as u64, Ordering::Relaxed);
                    sh.misses.fetch_add(p.misses as u64, Ordering::Relaxed);
                    sh.capped.fetch_add(usize::from(p.record.outcome.is_draw()), Ordering::Relaxed);
                    *sh.by_players.lock().unwrap().entry(p.record.setup.seats.len()).or_default() += 1;
                    sh.games.fetch_add(1, Ordering::Relaxed);
                }
                rec_out.flush().expect("flush records");
            });
        }
        // Progress from the main thread while the workers run.
        while sh.games.load(Ordering::Relaxed) < args.games {
            std::thread::sleep(Duration::from_secs(5));
            let t = started.elapsed().as_secs_f64();
            let (g, p) = (sh.games.load(Ordering::Relaxed), sh.positions.load(Ordering::Relaxed));
            println!("{g}/{} games, {p} positions, {:.1} games/s, {:.0} positions/s", args.games, g as f64 / t, p as f64 / t);
        }
    });
    for (_, buf) in std::mem::take(&mut *sh.bufs.lock().unwrap()) {
        if buf.len() > 0 {
            write_shard(&sh, &args.out, buf);
        }
    }
    let t = started.elapsed().as_secs_f64();
    let (g, p, c) = (sh.games.load(Ordering::Relaxed), sh.positions.load(Ordering::Relaxed), sh.commands.load(Ordering::Relaxed));
    let shards = sh.shards.lock().unwrap();
    println!(
        "done: {g} games, {c} commands, {p} positions in {t:.1}s: {:.2} games/s, {:.0} positions/s, {:.0} commands/s",
        g as f64 / t,
        p as f64 / t,
        c as f64 / t
    );
    println!(
        "{} shards, {} games capped at the turn limit (draws), {} chosen commands missing from the legal list",
        shards.len(),
        sh.capped.load(Ordering::Relaxed),
        sh.misses.load(Ordering::Relaxed)
    );
    let mix: Vec<String> = sh.by_players.lock().unwrap().iter().map(|(n, c)| format!("{n} players: {c} ({:.0}%)", 100.0 * *c as f64 / g.max(1) as f64)).collect();
    println!("player counts: {}", mix.join(", "));
    for (path, n) in shards.iter() {
        println!("  {} ({n} positions)", path.display());
    }
}
