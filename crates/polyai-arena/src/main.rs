//! Round-robin match runner.
//!
//! `polyai-arena --agents A,B,C --seeds N [--seed-start 1000000] [--out FILE.jsonl] [--threads T<=24]`
//!
//! Every pair of agents plays every seed twice (seats swapped, same map and tribes). Seeds start at 1,000,000: lower
//! seeds are training data. One JSON line per game goes to `--out` (appended; games already in the file are
//! skipped, so an interrupted run resumes). Run long jobs under `nice -n 10`.

use polyai_arena::{job_key, line_key, play_pair, AgentSpec, EVAL_SEED_MIN};
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const MAX_THREADS: usize = 24;
const USAGE: &str = "usage: polyai-arena --agents A,B[,C...] --seeds N [--seed-start 1000000] [--out FILE.jsonl] [--threads T<=24]\n\
    agents: retail:easy|normal|hard|crazy, random, mcts:<iterations>[:depth=D][:topk=K]";

struct Args {
    agents: Vec<AgentSpec>,
    seeds: usize,
    seed_start: i32,
    out: Option<PathBuf>,
    threads: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { agents: Vec::new(), seeds: 0, seed_start: EVAL_SEED_MIN, out: None, threads: MAX_THREADS };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or(format!("{flag} needs a value"));
        let bad = |f: &str| format!("bad value for {f}");
        match flag.as_str() {
            "--agents" => a.agents = val()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--seeds" => a.seeds = val()?.parse().map_err(|_| bad(&flag))?,
            "--seed-start" => a.seed_start = val()?.parse().map_err(|_| bad(&flag))?,
            "--out" => a.out = Some(PathBuf::from(val()?)),
            "--threads" => a.threads = val()?.parse().map_err(|_| bad(&flag))?,
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    if a.agents.len() < 2 || a.seeds == 0 {
        return Err(USAGE.into());
    }
    if a.seed_start < EVAL_SEED_MIN {
        return Err(format!("seeds below {EVAL_SEED_MIN} are training data; got --seed-start {}", a.seed_start));
    }
    a.threads = a.threads.clamp(1, MAX_THREADS);
    Ok(a)
}

/// Tells the progress thread to stop when the workers finish or unwind from a panic.
struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Default, Clone, Copy)]
struct Tally {
    wins: [u32; 2],
    draws: u32,
    capped: u32,
    games: u32,
    wall_ms: f64,
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let names: Vec<String> = args.agents.iter().map(ToString::to_string).collect();
    let mut jobs: Vec<(usize, usize, i32, u8)> = Vec::new();
    for k in 0..args.seeds {
        let seed = args.seed_start + k as i32;
        for i in 0..names.len() {
            for j in i + 1..names.len() {
                for swap in 0..2u8 {
                    jobs.push((i, j, seed, swap));
                }
            }
        }
    }
    let mut done: HashSet<String> = HashSet::new();
    if let Some(path) = &args.out {
        if let Ok(text) = std::fs::read_to_string(path) {
            done = text.lines().filter_map(line_key).collect();
        }
    }
    let total = jobs.len();
    jobs.retain(|&(i, j, seed, swap)| !done.contains(&job_key(seed, swap, &names[i], &names[j])));
    let out = args.out.as_ref().map(|p| {
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).expect("create output directory");
        }
        Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).expect("open output file"))
    });
    eprintln!(
        "polyai-arena: {} agents, {} games ({} already done) on {} threads, seeds {}..{}",
        names.len(),
        total,
        total - jobs.len(),
        args.threads,
        args.seed_start,
        args.seed_start + args.seeds as i32 - 1
    );
    let started = Instant::now();
    let finished = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let tallies: Mutex<Vec<Tally>> = Mutex::new(vec![Tally::default(); names.len() * names.len()]);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().expect("thread pool");
    let n_jobs = jobs.len();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut last = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(30) {
                    last = Instant::now();
                    eprintln!("{}/{n_jobs} games, {:.1}s", finished.load(Ordering::Relaxed), started.elapsed().as_secs_f64());
                }
            }
        });
        let _stop = StopOnDrop(&stop);
        pool.install(|| {
            jobs.par_iter().for_each(|&(i, j, seed, swap)| {
                let r = play_pair(&args.agents[i], &args.agents[j], seed, swap);
                if let Some(out) = &out {
                    let mut f = out.lock().unwrap();
                    writeln!(f, "{}", r.to_json()).expect("write result");
                }
                // Seat of agent i: 0 without swap, 1 with.
                let (si, sj) = if swap == 0 { (0, 1) } else { (1, 0) };
                let mut t = tallies.lock().unwrap();
                let c = &mut t[i * names.len() + j];
                c.games += 1;
                c.wall_ms += r.wall_ms;
                c.capped += u32::from(r.capped);
                match r.winner {
                    Some(w) if w == si => c.wins[0] += 1,
                    Some(w) if w == sj => c.wins[1] += 1,
                    _ => c.draws += 1,
                }
                drop(t);
                finished.fetch_add(1, Ordering::Relaxed);
            });
        });
    });
    eprintln!("done: {n_jobs} games in {:.1}s", started.elapsed().as_secs_f64());
    let t = tallies.lock().unwrap();
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let c = t[i * names.len() + j];
            if c.games > 0 {
                println!(
                    "{} vs {}: {}-{}-{} (W-L-D for the first), {} capped (draws), {:.1}s per game",
                    names[i],
                    names[j],
                    c.wins[0],
                    c.wins[1],
                    c.draws,
                    c.capped,
                    c.wall_ms / f64::from(c.games) / 1000.0
                );
            }
        }
    }
    println!("tables: /kvm/polytopia/.venv/bin/python ai/tools/elo.py <file.jsonl>");
}
