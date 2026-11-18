//! The model is never offered a build on a tile its player has not explored (the engine lists road builds there).

use polyai_core::*;
use polytopia_sim::engine::{Command, NATURE};

#[test]
fn no_builds_on_unexplored_tiles() {
    let mut hidden = 0usize;
    for seed in 1..=6 {
        let mut setup = GameSetup::sample(seed, &SetupSpec { n_players: vec![2], sizes: vec![polytopia_sim::MapSize::Tiny], ..Default::default() });
        setup.turn_limit = 30;
        play_ai_game(&setup, |e, _| {
            let me = e.current_player();
            if me == NATURE {
                return;
            }
            let unexplored = |c: &Command| matches!(*c, Command::Build { at, .. } if !e.is_explored_at(me, at));
            hidden += engine_actions(e).iter().filter(|c| unexplored(c)).count();
            assert!(!legal_actions(e).iter().any(unexplored), "seed {seed}: build on an unexplored tile offered");
        });
    }
    assert!(hidden > 0, "no unexplored build was ever listed, so the test checks nothing");
}
