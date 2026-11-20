//! Single-thread cost of what the rollout collector does per model decision, and a checksum of everything it produces.
//!
//!   cargo run --release -p polyai-core --example bench_decision [GAMES] [TURNS] [PLAYERS]
//!
//! Plays GAMES Might games (seeds 1.., standard size and preset mix, PLAYERS as `SetupSpec::with_players_str` takes
//! them, default `2:55,3:25,4:20`) to TURNS turns (default 30) with every seat on the retail AI. Before every command
//! of a player it does the collector's work for that position as if a model were to move: slot map, legal list,
//! observation, action encoding, engine features, truth labels, then a copy of the game. Each part is timed and its
//! output goes into an FNV hash, so a change that only makes them faster leaves the checksums as they were.
//! The retail AI's pick is timed apart (`retail_ai`) and is not part of the per-decision total.
//!
//! With the environment variable `PART` set to one of the part names, it also keeps every fourth position and runs
//! that part alone over them `REPEAT` times (default 20) at the end: the loop to point a profiler at.

use polyai_core::features::features;
use polyai_core::truth::{truth, truth_globals};
use polyai_core::{ai_pick, encode, encode_command, finished, legal_actions, GameSetup, SetupSpec, SlotMap, TurnCounter, MIGHT};
use polytopia_sim::engine::{Engine, NATURE};
use std::time::{Duration, Instant};

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn i16s(&mut self, v: &[i16]) {
        for x in v {
            self.bytes(&x.to_le_bytes());
        }
    }

    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
}

const PARTS: [&str; 9] = ["slots", "legal", "encode", "actions", "features", "truth", "clone", "apply", "retail_ai"];

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let games: i32 = arg(1).map_or(20, |a| a.parse().expect("GAMES"));
    let turns: i32 = arg(2).map_or(30, |a| a.parse().expect("TURNS"));
    let players = arg(3).unwrap_or_else(|| "2:55,3:25,4:20".into());
    let spec = SetupSpec { mode: MIGHT, ..SetupSpec::default() }.with_players_str(&players).expect("PLAYERS");
    let mut t = [Duration::ZERO; PARTS.len()];
    let mut sums: Vec<Fnv> = (0..6).map(|_| Fnv::new()).collect();
    let (mut decisions, mut legal_total, mut commands) = (0u64, 0u64, 0u64);
    let part = std::env::var("PART").ok();
    let mut kept: Vec<Engine> = Vec::new();
    for seed in 1..=games {
        let mut setup = GameSetup::sample(seed, &spec);
        setup.turn_limit = turns;
        let mut e = setup.build();
        let mut counter = TurnCounter::default();
        let cap = setup.own_cap();
        while !finished(&e, cap) {
            let t0 = Instant::now();
            let cmd = ai_pick(&mut e, counter.0);
            t[8] += t0.elapsed();
            let me = e.current_player();
            if me != NATURE {
                let t0 = Instant::now();
                let slots = SlotMap::new(&e, me);
                let t1 = Instant::now();
                let legal = legal_actions(&e);
                let t2 = Instant::now();
                let obs = encode(&e, me, &slots, &legal);
                let t3 = Instant::now();
                let actions: Vec<[i16; 4]> = legal.iter().map(|c| encode_command(&e, &slots, c).expect("legal command encodes")).collect();
                let t4 = Instant::now();
                let f = features(&e, me, &slots, &legal);
                let t5 = Instant::now();
                let tr = (truth(&e, me, &slots), truth_globals(&e, me));
                let t6 = Instant::now();
                let copy = e.clone();
                let t7 = Instant::now();
                for (k, (a, b)) in [(t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t5, t6), (t6, t7)].into_iter().enumerate() {
                    t[k] += b - a;
                }
                std::hint::black_box(&copy);
                if part.is_some() && decisions % 4 == 0 {
                    kept.push(copy);
                }
                sums[0].bytes(format!("{legal:?}").as_bytes());
                sums[1].i16s(&obs.tiles);
                sums[1].f32s(&obs.globals);
                for a in &actions {
                    sums[2].i16s(a);
                }
                sums[3].f32s(&f.actions);
                sums[3].i16s(&f.tiles);
                sums[3].f32s(&f.globals);
                sums[4].i16s(&tr.0.labels);
                sums[4].bytes(&tr.0.mask);
                sums[4].f32s(&tr.1);
                decisions += 1;
                legal_total += legal.len() as u64;
            }
            let (turn, player) = (e.s.current_turn, e.current_player());
            let t0 = Instant::now();
            e.apply_unchecked(cmd);
            t[7] += t0.elapsed();
            commands += 1;
            counter.after(&e, &cmd, turn, player);
        }
        sums[5].bytes(format!("{:?}", e.s).as_bytes());
    }
    let us = |d: Duration, n: u64| d.as_secs_f64() * 1e6 / n.max(1) as f64;
    println!("{games} games to turn {turns}, players {players}: {decisions} decisions, {commands} commands, {:.1} legal moves on average", legal_total as f64 / decisions.max(1) as f64);
    let total: Duration = t[..7].iter().sum();
    for (k, name) in PARTS.iter().enumerate() {
        let n = if k >= 7 { commands } else { decisions };
        println!("  {name:9}{:8.2} us", us(t[k], n));
    }
    println!("  per decision, slots to clone {:.2} us", us(total, decisions));
    let names = ["legal", "obs", "actions", "features", "truth", "state"];
    let line: Vec<String> = names.iter().zip(&sums).map(|(n, s)| format!("{n} {:016x}", s.0)).collect();
    println!("checksums: {}", line.join(" "));
    if let Some(part) = part {
        let repeat: u32 = std::env::var("REPEAT").ok().map_or(20, |r| r.parse().expect("REPEAT"));
        let t0 = Instant::now();
        for _ in 0..repeat {
            for e in &kept {
                run_part(&part, e);
            }
        }
        println!("{part} alone: {:.2} us over {} positions x {repeat}", us(t0.elapsed(), kept.len() as u64 * u64::from(repeat)), kept.len());
    }
}

/// One part of the per-decision work on its own (the legal list it needs is not timed apart).
fn run_part(part: &str, e: &Engine) {
    let me = e.current_player();
    let slots = SlotMap::new(e, me);
    match part {
        "slots" => {
            std::hint::black_box(&slots);
        }
        "legal" => {
            std::hint::black_box(legal_actions(e));
        }
        "clone" => {
            std::hint::black_box(e.clone());
        }
        "truth" => {
            std::hint::black_box((truth(e, me, &slots), truth_globals(e, me)));
        }
        "retail_ai" => {
            let mut x = e.clone();
            std::hint::black_box(ai_pick(&mut x, 0));
        }
        "apply" => {
            let mut x = e.clone();
            let legal = x.legal_commands();
            x.apply_unchecked(legal[legal.len() / 2]);
            std::hint::black_box(x);
        }
        _ => {
            let legal = legal_actions(e);
            match part {
                "encode" => {
                    std::hint::black_box(encode(e, me, &slots, &legal));
                }
                "actions" => {
                    std::hint::black_box(legal.iter().map(|c| encode_command(e, &slots, c)).collect::<Vec<_>>());
                }
                "features" => {
                    std::hint::black_box(features(e, me, &slots, &legal));
                }
                _ => panic!("PART is one of {PARTS:?}"),
            }
        }
    }
}
