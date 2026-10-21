//! Plays retail-vs-retail games with a high turn limit and prints one CSV line per game, to measure how long games
//! last when nothing adjudicates them.
//!
//! `cargo run --release -p polyai-core --example game_lengths -- --games N [--seed-start S] [--threads T<=16]
//!  [--limit L] [--players 2:55,3:25,4:20] [--difficulty D]`
//!
//! Columns: `seed,players,size,preset,turns,capped,commands,wall_ms,difficulties`. Run under `nice -n 10`.

use polyai_core::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

fn main() {
    let (mut games, mut seed_start, mut threads, mut limit) = (100usize, 1i32, 8usize, 300i32);
    let (mut players, mut difficulty): (Option<String>, Option<u8>) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--games" => games = v.parse().unwrap(),
            "--seed-start" => seed_start = v.parse().unwrap(),
            "--threads" => threads = v.parse().unwrap(),
            "--limit" => limit = v.parse().unwrap(),
            "--players" => players = Some(v),
            "--difficulty" => difficulty = Some(v.parse().unwrap()),
            _ => panic!("unknown argument {flag}"),
        }
    }
    let mut spec = SetupSpec { turn_limit: limit, ..Default::default() };
    if let Some(p) = &players {
        spec = spec.with_players_str(p).unwrap();
    }
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::<String>::new());
    println!("seed,players,size,preset,turns,capped,commands,wall_ms,difficulties");
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, 16) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= games {
                    break;
                }
                let mut setup = GameSetup::sample(seed_start + i as i32, &spec);
                if let Some(d) = difficulty {
                    setup = setup.with_difficulty(d);
                }
                let t = Instant::now();
                let (rec, _) = play_ai_game(&setup, |_, _| {});
                let o = &rec.outcome;
                let diffs: Vec<String> = setup.seats.iter().map(|s| s.difficulty.to_string()).collect();
                let line = format!(
                    "{},{},{},{},{},{},{},{:.0},{}",
                    setup.seed,
                    setup.seats.len(),
                    setup.size,
                    setup.preset,
                    o.turns,
                    u8::from(o.is_draw()),
                    rec.commands.len(),
                    t.elapsed().as_secs_f64() * 1000.0,
                    diffs.join("")
                );
                out.lock().unwrap().push(line);
            });
        }
    });
    for l in out.lock().unwrap().iter() {
        println!("{l}");
    }
    eprintln!("{games} games in {:.1}s", started.elapsed().as_secs_f64());
}
