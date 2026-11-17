//! Replay a headless-client game record (the server's two state blobs) and print per-turn statistics for every
//! player as JSON lines, for `tools/loss_autopsy.py`.
//!
//! ```text
//! cargo run --release -p polyai-core --example replay_stats -- <game>_init.bin <game>_cur.bin
//! ```
//!
//! Lines on stdout:
//! - `{"t":"turn",...}` once per player turn, taken just before that player's end-turn command: score, stars in hand
//!   at the start and end of the turn, income, cities, city levels, population, villages and enemy cities captured so
//!   far, units, army value (sum of unit costs), techs, kills, losses, territory, explored tiles, and the number of
//!   moves, attacks, trains, builds and researches of the turn.
//! - `{"t":"ev",...}` for every research, train, build, capture, city reward and ruin, with the name of what was
//!   chosen.
//! - `{"t":"end",...}` once: the final turn, whether the engine thinks the game is over, the winners, who holds each
//!   original capital, each player's tribe, and how many commands the engine could not read.
use polytopia_sim::engine::replay::SaveFile;
use polytopia_sim::{Command, Engine, NATURE};
use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Default, Clone)]
struct Tally {
    start_stars: i32,
    villages: u32,
    enemy_cities: u32,
    moves: u32,
    attacks: u32,
    trains: u32,
    builds: u32,
    researches: u32,
}

fn name(names: &[Option<String>], idx: i32) -> String {
    usize::try_from(idx).ok().and_then(|i| names.get(i).cloned().flatten()).unwrap_or_else(|| idx.to_string())
}

fn unit_name(e: &Engine, idx: i32) -> String {
    usize::try_from(idx).ok().and_then(|i| e.t.units.get(i)).and_then(|u| u.as_ref()).map_or_else(|| idx.to_string(), |u| u.data.id.clone())
}

fn improvement_name(e: &Engine, idx: i32) -> String {
    usize::try_from(idx).ok().and_then(|i| e.t.improvements.get(i)).and_then(|u| u.as_ref()).map_or_else(|| idx.to_string(), |u| u.data.id.clone())
}

fn unit_cost(e: &Engine, idx: i32) -> i32 {
    usize::try_from(idx).ok().and_then(|i| e.t.units.get(i)).and_then(|u| u.as_ref()).map_or(0, |u| u.data.cost)
}

fn turn_line(e: &Engine, p: u8, tally: &Tally) -> String {
    let s = &e.s;
    let (mut units, mut army) = (0, 0);
    for u in s.units.iter().filter(|u| u.owner == p) {
        units += 1;
        army += unit_cost(e, u.passenger.as_ref().map_or(u.type_idx, |q| q.type_idx));
    }
    let (mut levels, mut pop) = (0, 0);
    for imp in &s.improvements {
        let is_city = usize::try_from(imp.tile).ok().and_then(|i| s.map.tiles.get(i)).is_some_and(|t| t.is_city && t.owner == p);
        if is_city {
            levels += i32::from(imp.level);
            pop += i32::from(imp.population);
        }
    }
    let territory = s.map.tiles.iter().filter(|t| t.owner == p).count();
    let explored = s.fog.iter().filter(|f| f.explorers.contains(&p)).count();
    let rt = s.player_runtime.iter().find(|r| r.player == p);
    let stars = e.player(p).map_or(0, |x| x.currency);
    let mut out = String::new();
    write!(
        out,
        "{{\"t\":\"turn\",\"turn\":{},\"player\":{p},\"score\":{},\"start_stars\":{},\"stars\":{stars},\"income\":{},\
         \"cities\":{},\"city_levels\":{levels},\"pop\":{pop},\"villages\":{},\"enemy_cities\":{},\"units\":{units},\
         \"army\":{army},\"techs\":{},\"kills\":{},\"losses\":{},\"territory\":{territory},\"explored\":{explored},\
         \"moves\":{},\"attacks\":{},\"trains\":{},\"builds\":{},\"researches\":{}}}",
        s.current_turn,
        e.score(p),
        tally.start_stars,
        e.income(p),
        e.city_count(p),
        tally.villages,
        tally.enemy_cities,
        e.techs(p).len(),
        rt.map_or(0, |r| r.kills),
        rt.map_or(0, |r| r.casualties),
        tally.moves,
        tally.attacks,
        tally.trains,
        tally.builds,
        tally.researches,
    )
    .unwrap();
    out
}

fn event(turn: u32, p: u8, kind: &str, what: &str) {
    println!("{{\"t\":\"ev\",\"turn\":{turn},\"player\":{p},\"kind\":\"{kind}\",\"what\":\"{what}\"}}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [init, cur] = args.as_slice() else {
        eprintln!("usage: replay_stats <init.bin> <cur.bin>");
        std::process::exit(2);
    };
    let read = |p: &str| std::fs::read(p).unwrap_or_else(|e| panic!("cannot read {p}: {e}"));
    let save = SaveFile::parse_server(&read(init), &read(cur)).unwrap_or_else(|e| panic!("cannot decode: {e}"));
    let mut e = Engine::from_state(save.initial.clone());
    if save.commands.iter().any(|c| c.kind == 20) {
        e.start_match();
    }
    let players: Vec<u8> = e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect();
    let mut tally: BTreeMap<u8, Tally> = players.iter().map(|&p| (p, Tally::default())).collect();
    let first = e.current_player();
    if let Some(t) = tally.get_mut(&first) {
        t.start_stars = e.player(first).map_or(0, |x| x.currency);
    }
    let mut unsupported = 0;
    for c in &save.commands {
        if c.kind == 20 || c.player == NATURE {
            continue;
        }
        let Some(cmd) = e.command_from_raw(c) else {
            e.s.current_command += 1;
            unsupported += 1;
            continue;
        };
        let (turn, p) = (e.s.current_turn, c.player);
        let t = tally.entry(p).or_default();
        match cmd {
            Command::Move { .. } => t.moves += 1,
            Command::Attack { .. } => t.attacks += 1,
            Command::Train { unit_type, .. } => {
                t.trains += 1;
                event(turn, p, "train", &unit_name(&e, unit_type));
            }
            Command::Build { improvement, .. } => {
                t.builds += 1;
                event(turn, p, "build", &improvement_name(&e, improvement));
            }
            Command::Research { tech } => {
                t.researches += 1;
                event(turn, p, "research", &name(&e.t.tech_name, tech));
            }
            Command::Capture { at, .. } => {
                let owner = e.tile(at).map_or(0, |x| x.owner);
                let what = if owner == 0 { "village" } else { "city" };
                if owner == 0 {
                    t.villages += 1;
                } else {
                    t.enemy_cities += 1;
                }
                event(turn, p, "capture", what);
            }
            Command::CityReward { reward, .. } => event(turn, p, "reward", &format!("{reward:?}")),
            Command::ExamineRuins { .. } => event(turn, p, "ruin", ""),
            Command::EndTurn => println!("{}", turn_line(&e, p, t)),
            _ => {}
        }
        e.apply_unchecked(cmd);
        if matches!(cmd, Command::EndTurn) {
            let t = tally.entry(p).or_default();
            *t = Tally { villages: t.villages, enemy_cities: t.enemy_cities, ..Tally::default() };
            let next = e.current_player();
            if let Some(n) = tally.get_mut(&next) {
                n.start_stars = e.player(next).map_or(0, |x| x.currency);
            }
        }
    }
    let mut caps = String::new();
    for &p in &players {
        let holder = e.s.map.tiles.iter().find(|t| t.capital_of == p).map_or(0, |t| t.owner);
        let _ = write!(caps, "{}\"{p}\":{holder}", if caps.is_empty() { "" } else { "," });
    }
    let winners: Vec<String> = e.winners.iter().map(u8::to_string).collect();
    let tribes: Vec<String> = e.s.players.iter().filter(|p| p.id != NATURE).map(|p| format!("\"{}\":\"{}\"", p.id, name(&e.t.tribe_name, p.tribe))).collect();
    println!(
        "{{\"t\":\"end\",\"turn\":{},\"over\":{},\"winners\":[{}],\"capital_holder\":{{{caps}}},\"unsupported\":{unsupported},\"tribes\":{{{}}},\"final_scores\":{{{}}}}}",
        e.s.current_turn,
        e.is_over(),
        winners.join(","),
        tribes.join(","),
        players.iter().map(|&p| format!("\"{p}\":{}", e.score(p))).collect::<Vec<_>>().join(","),
    );
}
