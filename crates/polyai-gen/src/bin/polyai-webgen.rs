//! Games of the kinds the website hosts, played out by the retail AI, written as value shards
//! (`polyai-gen/src/value_shard.rs`) for training and checking the analysis win chance.
//!
//! `polyai-webgen --out DIR --games N [--seed-start S] [--threads T] [--keep K] [--max-turns M] [--shard-size P]
//!  [--players 2:20,3:25,4:25,5:20,6:10] [--modes perfection:50,domination:30,might:20]`
//!
//! `--players` and `--modes` are weighted lists (the defaults shown); seeds give the same games only with the same
//! lists.
//!
//! A game is set up as the web's engine service sets up a generated game (`Engine::try_new_game` with one seat or
//! more marked human, bot handicaps by level, `settings.difficulty` the top bot level): mode Perfection (30 turns,
//! the top score wins), Domination (no turn limit, the last player left wins) or Might (the holder of every capital
//! wins); 2 to 6 players; tiny to large maps of every type; bots from easy to crazy. The person's seat is played by
//! the retail AI at a level drawn per game (easy to crazy, or with no handicap at all), switched to auto-play only
//! while it picks its command, so the game keeps the person's rules (who counts as human for the end of the game).
//!
//! Before one command in `K` (default 32, by a hash of the game and the command index), every living player's own
//! view is written, labelled with the winner by the game's own rule (`Engine::winners`). A game still running at
//! `M` turns (default 200) is undecided: its rows are written with `winner = -1`. `meta.jsonl` holds one line per
//! game (setup, the person's stand-in level, turns, winners, rows).

#[path = "../npz.rs"]
#[allow(dead_code)]
mod npz;
#[path = "../value_shard.rs"]
mod value_shard;

use polyai_core::*;
use polytopia_sim::ai::decide::handicap_from_difficulty_value;
use polytopia_sim::engine::{Command, Engine, GameOptions, PlayerSetup, NATURE};
use polytopia_sim::{BotDifficulty, MapPreset, MapSize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use value_shard::{ValueBuf, ValueRow};

const PERFECTION: i32 = 1;
const PERFECTION_TURNS: i32 = 30;
/// The person's stand-in level that plays with no handicap (not on auto-play), stronger than crazy's stupidity rule.
const UNHANDICAPPED: i32 = -1;

struct Args {
    out: PathBuf,
    games: usize,
    seed_start: i32,
    threads: usize,
    keep: u32,
    max_turns: u32,
    shard_size: usize,
    players: Vec<(usize, u32)>,
    modes: Vec<(i32, u32)>,
}

/// `a:w,b:w` -> [(a, w)].
fn weighted<T>(s: &str, parse: impl Fn(&str) -> Option<T>) -> Option<Vec<(T, u32)>> {
    s.split(',').map(|part| {
        let (k, w) = part.split_once(':')?;
        Some((parse(k.trim())?, w.trim().parse().ok()?))
    }).collect::<Option<Vec<_>>>().filter(|v| !v.is_empty() && v.iter().any(|x| x.1 > 0))
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        out: PathBuf::new(),
        games: 0,
        seed_start: 1,
        threads: 16,
        keep: 32,
        max_turns: 200,
        shard_size: 50_000,
        players: vec![(2, 20), (3, 25), (4, 25), (5, 20), (6, 10)],
        modes: vec![(PERFECTION, 50), (DOMINATION, 30), (MIGHT, 20)],
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().ok_or(format!("{flag} needs a value"))?;
        let bad = |_| format!("bad value for {flag}");
        match flag.as_str() {
            "--out" => a.out = PathBuf::from(v),
            "--games" => a.games = v.parse().map_err(bad)?,
            "--seed-start" => a.seed_start = v.parse().map_err(bad)?,
            "--threads" => a.threads = v.parse().map_err(bad)?,
            "--keep" => a.keep = v.parse().map_err(bad)?,
            "--max-turns" => a.max_turns = v.parse().map_err(bad)?,
            "--shard-size" => a.shard_size = v.parse().map_err(bad)?,
            "--players" => {
                a.players = weighted(&v, |k| k.parse().ok().filter(|n| (2..=8).contains(n))).ok_or(format!("bad value for {flag}"))?
            }
            "--modes" => {
                let mode = |k: &str| match k {
                    "perfection" => Some(PERFECTION),
                    "domination" => Some(DOMINATION),
                    "might" => Some(MIGHT),
                    _ => None,
                };
                a.modes = weighted(&v, mode).ok_or(format!("bad value for {flag}"))?
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if a.out.as_os_str().is_empty() || a.games == 0 {
        return Err("usage: polyai-webgen --out DIR --games N [--seed-start S] [--threads T] [--keep K] [--max-turns M] [--shard-size P]".into());
    }
    a.keep = a.keep.max(1);
    a.threads = a.threads.max(1);
    Ok(a)
}

/// One sampled game.
struct WebSetup {
    seed: i32,
    mode: i32,
    size: MapSize,
    preset: MapPreset,
    tribes: Vec<i32>,
    /// Per seat: `None` for a person, else the bot's difficulty (0 easy to 3 crazy).
    levels: Vec<Option<i32>>,
    /// The difficulty the retail AI plays a person's seat at, or [`UNHANDICAPPED`].
    stand_in: i32,
}

fn sample(seed: i32, a: &Args) -> WebSetup {
    let mut r = Rng::new(seed as u32 as u64 | 0x7EB0_0000_0000);
    let mode = a.modes[r.weighted(&a.modes.iter().map(|m| m.1).collect::<Vec<_>>())].0;
    let n = a.players[r.weighted(&a.players.iter().map(|m| m.1).collect::<Vec<_>>())].0;
    let sizes: Vec<MapSize> = [MapSize::Tiny, MapSize::Small, MapSize::Normal, MapSize::Large].into_iter().filter(|s| s.max_players() >= n).collect();
    let size = sizes[r.below(sizes.len())];
    let preset = [MapPreset::Dryland, MapPreset::Lakes, MapPreset::Continents, MapPreset::Archipelago, MapPreset::WaterWorld, MapPreset::Pangea][r.below(6)];
    let mut pool: Vec<i32> = TRIBES.to_vec();
    let tribes: Vec<i32> = (0..n).map(|_| pool.remove(r.below(pool.len()))).collect();
    let people = if n >= 3 && r.below(10) == 0 { 2 } else { 1 };
    let one_level = r.below(10) < 7;
    let game_level = r.below(4) as i32;
    let mut levels: Vec<Option<i32>> = (0..n).map(|_| Some(if one_level { game_level } else { r.below(4) as i32 })).collect();
    let mut seats: Vec<usize> = (0..n).collect();
    for _ in 0..people {
        let k = seats.remove(r.below(seats.len()));
        levels[k] = None;
    }
    let stand_in = [0, 1, 2, 3, UNHANDICAPPED][r.below(5)];
    WebSetup { seed, mode, size, preset, tribes, levels, stand_in }
}

fn build(s: &WebSetup) -> Result<Engine, String> {
    let opts = GameOptions {
        version: polytopia_sim::GAME_VERSION,
        seed: s.seed,
        size: s.size,
        preset: s.preset,
        players: s.tribes.iter().zip(&s.levels).map(|(&tribe, l)| PlayerSetup { tribe, human: l.is_none() }).collect(),
        difficulty: BotDifficulty::Normal,
        mode: s.mode,
        turn_limit: if s.mode == PERFECTION { PERFECTION_TURNS } else { 0 },
        score_limit: 0,
    };
    let mut e = Engine::try_new_game(&opts).map_err(|e| e.to_string())?;
    for (k, l) in s.levels.iter().enumerate() {
        if let (Some(p), Some(d)) = (e.player_mut(k as u8 + 1), l) {
            p.handicap = handicap_from_difficulty_value(*d);
        }
    }
    let top = s.levels.iter().flatten().copied().max().unwrap_or(1);
    e.s.settings.difficulty = handicap_from_difficulty_value(top);
    e.s.seed = s.seed;
    Ok(e)
}

/// The retail AI's command for the player to move; a person's seat is played at the setup's stand-in level.
fn pick(e: &mut Engine, s: &WebSetup, per_turn: u32) -> Command {
    let me = e.current_player();
    let person = me != NATURE && e.player(me).is_some_and(|p| !p.auto_play);
    if !person || s.stand_in == UNHANDICAPPED {
        return ai_pick(e, per_turn);
    }
    let saved = e.player(me).map(|p| p.handicap).unwrap_or(1);
    if let Some(p) = e.player_mut(me) {
        p.auto_play = true;
        p.handicap = handicap_from_difficulty_value(s.stand_in);
    }
    let cmd = ai_pick(e, per_turn);
    if let Some(p) = e.player_mut(me) {
        p.auto_play = false;
        p.handicap = saved;
    }
    cmd
}

fn kept(seed: i32, index: usize, keep: u32) -> bool {
    let mut h: u32 = 2166136261 ^ seed as u32;
    for b in (index as u32).to_le_bytes() {
        h = (h ^ u32::from(b)).wrapping_mul(16777619);
    }
    h % keep == 0
}

struct View {
    obs: Obs,
    seat: u8,
    mover: bool,
    turn: i16,
    alive: u8,
    slot_kind: [i8; 8],
}

struct Played {
    setup: WebSetup,
    engine: Option<Engine>,
    views: Vec<View>,
    winners: Vec<u8>,
    turns: u32,
    error: Option<String>,
}

fn play(seed: i32, a: &Args) -> Played {
    let (keep, max_turns) = (a.keep, a.max_turns);
    let setup = sample(seed, a);
    let mut e = match build(&setup) {
        Ok(e) => e,
        Err(err) => return Played { setup, engine: None, views: vec![], winners: vec![], turns: 0, error: Some(err) },
    };
    let mut views = Vec::new();
    let mut counter = TurnCounter::default();
    let mut index = 0usize;
    while !e.is_over() && e.s.current_turn <= max_turns && index < MAX_COMMANDS {
        let me = e.current_player();
        if me != NATURE && kept(seed, index, keep) {
            let legal = legal_actions(&e);
            let alive = e.living_players();
            for &p in &alive {
                let slots = SlotMap::new(&e, p);
                let obs = encode(&e, p, &slots, if p == me { &legal } else { &[] });
                views.push(View { obs, seat: p, mover: p == me, turn: e.s.current_turn as i16, alive: alive.len() as u8, slot_kind: slot_kinds(&e, p) });
            }
        }
        let cmd = pick(&mut e, &setup, counter.0);
        let (turn, player) = (e.s.current_turn, e.current_player());
        e.apply_unchecked(cmd);
        counter.after(&e, &cmd, turn, player);
        index += 1;
    }
    let winners = if e.is_over() { e.winners.clone() } else { Vec::new() };
    let turns = e.s.current_turn;
    Played { setup, engine: Some(e), views, winners, turns, error: None }
}

fn meta_line(p: &Played, rows: usize) -> String {
    let s = &p.setup;
    let levels: Vec<String> = s.levels.iter().map(|l| l.map_or("null".to_string(), |d| d.to_string())).collect();
    let tribes: Vec<String> = s.tribes.iter().map(i32::to_string).collect();
    let winners: Vec<String> = p.winners.iter().map(u8::to_string).collect();
    let scores: Vec<String> = p.engine.as_ref().map_or(vec![], |e| {
        e.s.players.iter().filter(|q| q.id != NATURE).map(|q| format!("[{},{}]", q.id, e.score(q.id))).collect()
    });
    format!(
        "{{\"id\":\"web:{}\",\"mode\":{},\"size\":{},\"preset\":{},\"players\":{},\"tribes\":[{}],\"levels\":[{}],\"stand_in\":{},\"turns\":{},\"winners\":[{}],\"scores\":[{}],\"rows\":{},\"error\":{}}}",
        s.seed,
        s.mode,
        s.size as i32,
        s.preset as i32,
        s.tribes.len(),
        tribes.join(","),
        levels.join(","),
        s.stand_in,
        p.turns,
        winners.join(","),
        scores.join(","),
        rows,
        p.error.as_ref().map_or("null".to_string(), |e| format!("{e:?}"))
    )
}

fn main() {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    std::fs::create_dir_all(&args.out).expect("create the output directory");
    let next = AtomicUsize::new(0);
    let shards = AtomicUsize::new(0);
    let bufs: Mutex<HashMap<(usize, usize), ValueBuf>> = Mutex::new(HashMap::new());
    let meta = Mutex::new(std::io::BufWriter::new(std::fs::File::create(args.out.join("meta.jsonl")).expect("create meta.jsonl")));
    let write = |buf: ValueBuf, key: (usize, usize)| {
        let n = shards.fetch_add(1, Ordering::Relaxed);
        let path = args.out.join(format!("value-{n:05}-{}x{}.npz", key.0, key.1));
        buf.write(&path).expect("write a value shard");
        eprintln!("wrote {} ({} rows)", path.display(), buf.len());
    };
    std::panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= args.games {
                    return;
                }
                let seed = args.seed_start + k as i32;
                let played = std::panic::catch_unwind(|| play(seed, &args)).unwrap_or_else(|_| Played {
                    setup: sample(seed, &args),
                    engine: None,
                    views: vec![],
                    winners: vec![],
                    turns: 0,
                    error: Some("the engine panicked".into()),
                });
                let key_name = format!("web:{seed}");
                let order: Vec<u8> = played.engine.as_ref().map_or(vec![], |e| e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect());
                let n_players = order.len() as u8;
                let rows = played.views.len();
                let mut full = Vec::new();
                {
                    let mut b = bufs.lock().unwrap();
                    for v in &played.views {
                        let ids = slot_ids(&order, v.seat);
                        let winner = played.winners.iter().find_map(|w| ids[..8].iter().position(|&x| x == *w && x != 0)).map_or(-1, |k| k as i8);
                        let key = (v.obs.h, v.obs.w);
                        let buf = b.entry(key).or_insert_with(|| ValueBuf::new(key.0, key.1));
                        buf.push(ValueRow {
                            obs: v.obs.clone(),
                            winner,
                            seat: v.seat,
                            mover: v.mover,
                            turn: v.turn,
                            n_players,
                            alive: v.alive,
                            mode: played.setup.mode as i8,
                            slot_kind: v.slot_kind,
                            game_key: &key_name,
                        });
                        if buf.len() >= args.shard_size {
                            full.push(key);
                        }
                    }
                    for key in &full {
                        if let Some(buf) = b.remove(key) {
                            drop(b);
                            write(buf, *key);
                            b = bufs.lock().unwrap();
                        }
                    }
                }
                writeln!(meta.lock().unwrap(), "{}", meta_line(&played, rows)).expect("write meta");
            });
        }
    });
    let left: Vec<((usize, usize), ValueBuf)> = bufs.into_inner().unwrap().into_iter().collect();
    for (key, buf) in left {
        if buf.len() > 0 {
            write(buf, key);
        }
    }
    meta.into_inner().unwrap().flush().expect("flush meta");
}
