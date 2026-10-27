//! Action round trip over positions from retail-AI games (all 13 tribes), and record replay.

use polyai_core::*;
use polytopia_sim::engine::{Command, NATURE};
use std::collections::{HashMap, HashSet};

fn setup_for(seed: i32, tribes: &[i32]) -> GameSetup {
    let mut s = GameSetup::sample(seed, &SetupSpec { n_players: vec![tribes.len()], sizes: vec![polytopia_sim::MapSize::Tiny], ..Default::default() });
    for (seat, &t) in s.seats.iter_mut().zip(tribes) {
        seat.tribe = t;
    }
    s.turn_limit = 30;
    s
}

#[test]
fn action_round_trip_all_tribes() {
    // Every tribe plays in at least two games, against a rotating opponent.
    let mut seen_tribes = HashSet::new();
    let mut positions = 0usize;
    let mut kinds: HashMap<i16, usize> = HashMap::new();
    for (g, &tribe) in TRIBES.iter().cycle().take(TRIBES.len() * 2).enumerate() {
        let other = TRIBES[(g + 5) % TRIBES.len()];
        let other = if other == tribe { TRIBES[(g + 6) % TRIBES.len()] } else { other };
        let setup = setup_for(1 + g as i32, &[tribe, other]);
        seen_tribes.insert(tribe);
        let mut n = 0usize;
        play_ai_game(&setup, |e, cmd| {
            let me = e.current_player();
            if me == NATURE {
                return;
            }
            n += 1;
            // The full check on a sample, a cheaper one elsewhere: every position's chosen command, every 7th list.
            let slots = SlotMap::new(e, me);
            let legal = legal_actions(e);
            let idx = legal.iter().position(|c| c == cmd);
            assert!(idx.is_some(), "game {g}: chosen {cmd:?} not in the legal list");
            *kinds.entry(kind_of(cmd)).or_default() += 1;
            let check_all = n % 7 == 0;
            let cands: Vec<&Command> = if check_all { legal.iter().collect() } else { vec![cmd] };
            let mut seen = HashSet::new();
            for c in cands {
                let a = encode_command(e, &slots, c).unwrap_or_else(|| panic!("game {g}: cannot encode {c:?}"));
                let back = decode_command(e, &slots, a).unwrap_or_else(|| panic!("game {g}: cannot decode {a:?} from {c:?}"));
                assert_eq!(&back, c, "game {g}: round trip of {c:?} via {a:?}");
                if check_all {
                    assert!(seen.insert(a), "game {g}: two commands share the tuple {a:?}");
                }
            }
            if check_all {
                let obs = encode(e, me, &slots, &legal);
                assert_eq!(obs.tiles.len(), obs.h * obs.w * F_TILE);
                assert_eq!(obs.globals.len(), G_DIM);
            }
        });
        positions += n;
    }
    assert_eq!(seen_tribes.len(), 13);
    assert!(positions > 5000, "only {positions} positions");
    eprintln!("{positions} positions, kinds used: {kinds:?}");
}

#[test]
fn record_replays_and_survives_bytes() {
    for seed in [3, 4, 5] {
        let setup = GameSetup::sample(seed, &SetupSpec { sizes: vec![polytopia_sim::MapSize::Tiny], turn_limit: 20, ..Default::default() });
        let (rec, e) = play_ai_game(&setup, |_, _| {});
        assert!(rec.outcome.over);
        let bytes = rec.to_bytes();
        let back = GameRecord::from_bytes(&bytes).unwrap();
        assert_eq!(back, rec);
        let replayed = back.replay(true).expect("replay");
        assert_eq!(replayed.winners, e.winners);
        for p in &e.s.players {
            assert_eq!(replayed.score(p.id), e.score(p.id));
        }
        let mut stream = Vec::new();
        write_record(&mut stream, &rec).unwrap();
        write_record(&mut stream, &rec).unwrap();
        assert_eq!(read_records(&stream).unwrap(), vec![rec.clone(), rec]);
    }
}

#[test]
fn setup_sampler_is_seeded_and_in_range() {
    let a = GameSetup::sample(77, &SetupSpec::default());
    assert_eq!(a, GameSetup::sample(77, &SetupSpec::default()));
    let mut tribes = HashSet::new();
    let mut players = HashSet::new();
    for seed in 1..400 {
        let s = GameSetup::sample(seed, &SetupSpec::default());
        assert!((2..=4).contains(&s.seats.len()));
        assert_eq!(s.turn_limit, DEFAULT_TURN_LIMIT);
        assert_eq!(s.mode, 2);
        let distinct: HashSet<i32> = s.seats.iter().map(|x| x.tribe).collect();
        assert_eq!(distinct.len(), s.seats.len());
        tribes.extend(distinct);
        players.insert(s.seats.len());
    }
    assert_eq!(tribes.len(), 13);
    assert_eq!(players.len(), 3);
}

#[test]
fn player_counts_follow_the_weights_and_explicit_lists_stay_uniform() {
    let count = |spec: &SetupSpec, n: i32| -> Vec<usize> {
        let mut c = vec![0usize; 5];
        for seed in 1..=n {
            c[GameSetup::sample(seed, spec).seats.len()] += 1;
        }
        c
    };
    let c = count(&SetupSpec::default(), 4000);
    let share = |k: usize| c[k] as f64 / 4000.0;
    assert!((share(2) - 0.55).abs() < 0.03 && (share(3) - 0.25).abs() < 0.03 && (share(4) - 0.20).abs() < 0.03, "{c:?}");
    let uniform = SetupSpec { n_players: vec![2, 3], n_player_weights: vec![], ..Default::default() };
    let c = count(&uniform, 2000);
    assert!(c[4] == 0 && (c[2] as f64 / 2000.0 - 0.5).abs() < 0.04, "{c:?}");
    let spec = SetupSpec::default().with_players_str("2:1,4:3").unwrap();
    let c = count(&spec, 2000);
    assert!(c[3] == 0 && (c[4] as f64 / 2000.0 - 0.75).abs() < 0.04, "{c:?}");
    assert!(SetupSpec::default().with_players_str("2:1,3").is_err());
    assert!(SetupSpec::default().with_players_str("1").is_err());
    assert_eq!(SetupSpec::default().with_players_str("2").unwrap().n_players, vec![2]);
}

#[test]
fn a_game_cut_off_at_the_cap_is_a_draw_and_the_model_never_sees_the_cap() {
    // Two players at a 4-turn cap: nobody is eliminated, so the engine's top-score winner must not be reported.
    let mut setup = setup_for(41, &[TRIBES[0], TRIBES[1]]);
    setup.turn_limit = 4;
    let mut left = Vec::new();
    let (rec, e) = play_ai_game(&setup, |e, _| {
        if e.current_player() != NATURE && e.s.current_command % 40 == 0 {
            let slots = SlotMap::new(e, e.current_player());
            let legal = legal_actions(e);
            left.push(encode(e, e.current_player(), &slots, &legal).globals[G_SCALARS + 4..G_SCALARS + 6].to_vec());
        }
    });
    assert!(rec.outcome.over && rec.outcome.capped && rec.outcome.is_draw());
    assert!(rec.outcome.winners.is_empty());
    assert!(!e.winners.is_empty(), "the engine itself still names a top-score winner");
    assert!(!left.is_empty() && left.iter().all(|v| v == &[0.0, 0.0]), "turns left leaks the cap: {left:?}");
    let back = GameRecord::from_bytes(&rec.to_bytes()).unwrap();
    assert_eq!(back, rec);
    back.replay(true).expect("replay");
}

#[test]
fn might_games_end_on_capitals_and_stop_at_our_own_cap() {
    let spec = SetupSpec { mode: MIGHT, n_players: vec![2], n_player_weights: vec![], ..Default::default() };
    // The engine gets no turn limit (its capital rule only runs without one); the cap is ours.
    let s0 = GameSetup::sample(7, &spec);
    assert_eq!((s0.mode, s0.turn_limit, s0.own_cap()), (MIGHT, DEFAULT_TURN_LIMIT, DEFAULT_TURN_LIMIT));
    let e0 = s0.build();
    assert!(e0.s.settings.win_by_capital && e0.s.settings.turn_limit == 0);
    let mut by_capital = 0;
    for seed in 1..=12 {
        let setup = GameSetup::sample(seed, &spec);
        let (rec, e) = play_ai_game(&setup, |_, _| {});
        assert!(rec.outcome.over, "seed {seed} did not finish");
        if !rec.outcome.capped {
            let w = rec.outcome.winners[0];
            assert_eq!(e.capital_holder(), Some(w), "seed {seed}: the winner holds every capital");
            by_capital += usize::from(e.living_players().len() > 1);
        }
        GameRecord::from_bytes(&rec.to_bytes()).unwrap().replay(false).expect("replay");
    }
    assert!(by_capital > 0, "no 1v1 Might game ended with both players alive (a capital win)");
    // A short cap of our own stops the game as a draw.
    let mut setup = GameSetup::sample(3, &spec);
    setup.turn_limit = 4;
    let (rec, e) = play_ai_game(&setup, |_, _| {});
    assert!(rec.outcome.over && rec.outcome.capped && rec.outcome.winners.is_empty() && !e.is_over());
    assert_eq!(e.s.current_turn, 4);
    GameRecord::from_bytes(&rec.to_bytes()).unwrap().replay(true).expect("replay");
}

#[test]
fn observation_honours_fog_and_slots() {
    let setup = setup_for(11, &[TRIBES[0], TRIBES[1], TRIBES[2]]);
    let mut checked = 0;
    play_ai_game(&setup, |e, _| {
        let me = e.current_player();
        if me == NATURE || e.s.current_command % 25 != 0 {
            return;
        }
        let slots = SlotMap::new(e, me);
        assert_eq!(slots.slot_of(me), Some(0));
        let legal = legal_actions(e);
        let obs = encode(e, me, &slots, &legal);
        for k in 0..obs.h * obs.w {
            let row = &obs.tiles[k * F_TILE..(k + 1) * F_TILE];
            if !e.is_explored(me, k) {
                assert_eq!(row[C_UNEXPLORED], 1);
                assert!(row.iter().enumerate().all(|(c, &v)| c == C_UNEXPLORED || v == 0), "fog leaks at tile {k}");
            } else {
                assert_eq!(row[C_UNEXPLORED], 0);
                assert!(row[C_TERRAIN] > 0);
            }
            for c in 0..N_CAT {
                assert!(row[c] >= 0 && row[c] < CAT_VOCABS[c], "col {c} = {} out of vocabulary", row[c]);
            }
        }
        assert!(obs.globals.iter().all(|x| x.is_finite()));
        assert_eq!(obs.globals[G_TRIBE..G_TRIBE + N_TRIBES].iter().sum::<f32>(), 1.0);
        checked += 1;
    });
    assert!(checked > 20);
}

#[test]
fn diplomacy_commands_round_trip() {
    let setup = setup_for(21, &[TRIBES[0], TRIBES[1], TRIBES[2]]);
    let e = setup.build();
    for me in e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE) {
        let slots = SlotMap::new(&e, me);
        for other in e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE && id != me) {
            for c in [
                Command::BreakPeace { with: other },
                Command::PeaceTreaty { with: other },
                Command::EstablishEmbassy { with: other },
                Command::PeaceResponse { with: other, accept: true },
                Command::PeaceResponse { with: other, accept: false },
            ] {
                let a = encode_command(&e, &slots, &c).unwrap();
                assert_eq!(decode_command(&e, &slots, a), Some(c));
            }
        }
    }
}

#[test]
fn ai_candidates_do_not_change_the_game() {
    for seed in [31, 32] {
        let setup = setup_for(seed, &[TRIBES[3], TRIBES[8], TRIBES[12]]);
        let (plain, _) = play_ai_game(&setup, |_, _| {});
        let (probed, _) = play_ai_game(&setup, |e, _| {
            if e.current_player() != NATURE {
                let _ = e.ai_candidates();
            }
        });
        assert_eq!(plain.commands, probed.commands, "seed {seed}");
    }
}
