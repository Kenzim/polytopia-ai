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
        assert_eq!(s.turn_limit, 50);
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
