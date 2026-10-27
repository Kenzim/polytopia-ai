//! Game setups: a seeded sampler and the conversion to an engine game.

use crate::encoding::TRIBES;
use polytopia_sim::engine::{Engine, GameOptions, PlayerSetup};
use polytopia_sim::{BotDifficulty, MapPreset, MapSize};

/// Training games use seeds 1..=TRAIN_SEED_MAX; evaluation seeds start at EVAL_SEED_MIN.
pub const TRAIN_SEED_MAX: i32 = 100_000;
pub const EVAL_SEED_MIN: i32 = 1_000_000;

pub const DOMINATION: i32 = 2;
/// Might: the game ends when one player holds every capital (or is the last one left). The engine applies that rule
/// only to a game without a turn limit, so a Might game is built with none and the compute cap is checked here
/// ([`GameSetup::cap_reached`]).
pub const MIGHT: i32 = polytopia_sim::GAME_MODE_MIGHT;
/// The turn at which a game is stopped for compute only. A game still running with more than one player alive at
/// this turn is a draw (`Outcome::capped`); the engine's top-score rule is never used as a result.
pub const DEFAULT_TURN_LIMIT: i32 = 150;
/// Default player-count mix of the sampler: 1v1 about 55%, three players 25%, four 20% (design.md v1.1).
pub const DEFAULT_PLAYER_WEIGHTS: [(usize, u32); 3] = [(2, 55), (3, 25), (4, 20)];

pub const SIZES: [MapSize; 3] = [MapSize::Tiny, MapSize::Small, MapSize::Normal];
pub const PRESETS: [MapPreset; 5] = [MapPreset::Dryland, MapPreset::Lakes, MapPreset::Continents, MapPreset::Pangea, MapPreset::Archipelago];

pub fn size_from_u8(v: u8) -> Option<MapSize> {
    Some(match v {
        1 => MapSize::Tiny,
        2 => MapSize::Small,
        3 => MapSize::Normal,
        4 => MapSize::Large,
        5 => MapSize::Huge,
        6 => MapSize::Massive,
        _ => return None,
    })
}

pub fn preset_from_u8(v: u8) -> Option<MapPreset> {
    Some(match v {
        1 => MapPreset::Dryland,
        2 => MapPreset::Lakes,
        3 => MapPreset::Continents,
        4 => MapPreset::Archipelago,
        5 => MapPreset::WaterWorld,
        6 => MapPreset::Pangea,
        _ => return None,
    })
}

/// splitmix64: a small seeded generator, enough for sampling setups.
#[derive(Debug, Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        r.next();
        r
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Index drawn with the given weights.
    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        let mut x = (self.next() % u64::from(total.max(1))) as u32;
        for (i, &w) in weights.iter().enumerate() {
            if x < w {
                return i;
            }
            x -= w;
        }
        weights.len() - 1
    }
}

/// One seat: engine tribe idx and bot difficulty (0 easy, 1 normal, 2 hard, 3 crazy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seat {
    pub tribe: i32,
    pub difficulty: u8,
}

/// Everything needed to rebuild a game from nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameSetup {
    pub seed: i32,
    pub size: u8,
    pub preset: u8,
    pub mode: i32,
    pub turn_limit: i32,
    pub seats: Vec<Seat>,
}

/// What `GameSetup::sample` draws from. The default is the imitation-data distribution of design.md section 6.
#[derive(Debug, Clone)]
pub struct SetupSpec {
    pub n_players: Vec<usize>,
    /// Weight of each entry of `n_players`; empty means uniform (the old explicit-list behaviour).
    pub n_player_weights: Vec<u32>,
    pub sizes: Vec<MapSize>,
    pub presets: Vec<MapPreset>,
    /// Weights for difficulty 0..3.
    pub difficulty_weights: [u32; 4],
    pub turn_limit: i32,
    /// Game mode of every sampled game: [`DOMINATION`] or [`MIGHT`].
    pub mode: i32,
}

impl Default for SetupSpec {
    fn default() -> Self {
        Self {
            n_players: DEFAULT_PLAYER_WEIGHTS.iter().map(|w| w.0).collect(),
            n_player_weights: DEFAULT_PLAYER_WEIGHTS.iter().map(|w| w.1).collect(),
            sizes: SIZES.to_vec(),
            presets: PRESETS.to_vec(),
            difficulty_weights: [10, 20, 35, 35],
            turn_limit: DEFAULT_TURN_LIMIT,
            mode: DOMINATION,
        }
    }
}

impl SetupSpec {
    /// Player counts with weights. Fails on an empty list, a count outside 2..=13 or a weight list of another length
    /// or with a zero total.
    pub fn with_players(mut self, counts: Vec<usize>, weights: Vec<u32>) -> Result<Self, String> {
        if counts.is_empty() || counts.iter().any(|n| !(2..=TRIBES.len()).contains(n)) {
            return Err("player counts must be a non-empty list of values from 2 to 13".into());
        }
        if !weights.is_empty() && (weights.len() != counts.len() || weights.iter().all(|&w| w == 0)) {
            return Err("need one weight per player count, and at least one above zero".into());
        }
        self.n_players = counts;
        self.n_player_weights = weights;
        Ok(self)
    }

    /// Parse `2:55,3:25,4:20` (weighted) or `2,3` (uniform).
    pub fn with_players_str(self, s: &str) -> Result<Self, String> {
        let (mut counts, mut weights) = (Vec::new(), Vec::new());
        for part in s.split(',') {
            let (n, w) = part.split_once(':').map_or((part, None), |(n, w)| (n, Some(w)));
            counts.push(n.trim().parse().map_err(|_| format!("bad player count '{n}'"))?);
            if let Some(w) = w {
                weights.push(w.trim().parse().map_err(|_| format!("bad weight '{w}'"))?);
            }
        }
        if !weights.is_empty() && weights.len() != counts.len() {
            return Err("give a weight for every player count or for none".into());
        }
        self.with_players(counts, weights)
    }
}

impl GameSetup {
    /// A setup drawn deterministically from `seed`.
    pub fn sample(seed: i32, spec: &SetupSpec) -> Self {
        let mut rng = Rng::new(seed as u32 as u64 | 0x5E7_0000_0000);
        let n = if spec.n_player_weights.len() == spec.n_players.len() {
            spec.n_players[rng.weighted(&spec.n_player_weights)]
        } else {
            spec.n_players[rng.below(spec.n_players.len())]
        };
        let size = spec.sizes[rng.below(spec.sizes.len())] as u8;
        let preset = spec.presets[rng.below(spec.presets.len())] as u8;
        let mut pool: Vec<i32> = TRIBES.to_vec();
        let mut seats = Vec::with_capacity(n);
        for _ in 0..n {
            let tribe = pool.remove(rng.below(pool.len()));
            let difficulty = rng.weighted(&spec.difficulty_weights) as u8;
            seats.push(Seat { tribe, difficulty });
        }
        Self { seed, size, preset, mode: spec.mode, turn_limit: spec.turn_limit, seats }
    }

    /// The compute cap this crate enforces itself: the turn limit of a Might game (built without an engine limit, see
    /// [`MIGHT`]); 0 for every other mode, where the engine ends the game at its turn limit.
    pub fn own_cap(&self) -> i32 {
        if self.mode == MIGHT {
            self.turn_limit
        } else {
            0
        }
    }

    /// The game has reached the cap of [`Self::own_cap`] without ending.
    pub fn cap_reached(&self, e: &Engine) -> bool {
        cap_reached(e, self.own_cap())
    }

    /// The setup with every seat's difficulty replaced.
    pub fn with_difficulty(mut self, difficulty: u8) -> Self {
        for s in &mut self.seats {
            s.difficulty = difficulty;
        }
        self
    }

    pub fn options(&self) -> GameOptions {
        GameOptions {
            version: polytopia_sim::GAME_VERSION,
            seed: self.seed,
            size: size_from_u8(self.size).unwrap_or(MapSize::Small),
            preset: preset_from_u8(self.preset).unwrap_or(MapPreset::Continents),
            players: self.seats.iter().map(|s| PlayerSetup { tribe: s.tribe, human: false }).collect(),
            difficulty: BotDifficulty::Normal,
            mode: self.mode,
            turn_limit: if self.mode == MIGHT { 0 } else { self.turn_limit },
            score_limit: 0,
        }
    }

    /// Build the game: engine map and players, then each seat's own handicap.
    pub fn build(&self) -> Engine {
        let mut e = Engine::new_game(&self.options());
        for (i, seat) in self.seats.iter().enumerate() {
            e.s.players[i].handicap = polytopia_sim::ai::decide::handicap_from_difficulty_value(i32::from(seat.difficulty));
        }
        e
    }

    pub fn is_training_seed(&self) -> bool {
        (1..=TRAIN_SEED_MAX).contains(&self.seed)
    }
}

/// A running game has reached `cap` (0: no cap of our own). See [`GameSetup::own_cap`].
pub fn cap_reached(e: &Engine, cap: i32) -> bool {
    cap > 0 && !e.is_over() && e.s.current_turn as i32 >= cap
}

/// The game is over, or stopped at our own cap.
pub fn finished(e: &Engine, cap: i32) -> bool {
    e.is_over() || cap_reached(e, cap)
}
