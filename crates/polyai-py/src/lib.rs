//! Python module `polyai_engine`: single games (`Game`) and many games stepped in parallel (`VecEnv`).
//!
//! Layouts (tiles, globals, actions) are those of `docs/encoding.md`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use polyai_core::*;
use polytopia_sim::engine::{Command, Engine, NATURE};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

const MAX_THREADS: usize = 24;

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

// ---------------------------------------------------------------------------------------------------------------
// Setups

fn setup_to_dict<'py>(py: Python<'py>, s: &GameSetup) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("seed", s.seed)?;
    d.set_item("size", s.size)?;
    d.set_item("preset", s.preset)?;
    d.set_item("mode", s.mode)?;
    d.set_item("turn_limit", s.turn_limit)?;
    d.set_item("seats", s.seats.iter().map(|x| (x.tribe, x.difficulty)).collect::<Vec<_>>())?;
    Ok(d)
}

fn setup_from_dict(d: &Bound<'_, PyDict>) -> PyResult<GameSetup> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> { d.get_item(k)?.ok_or_else(|| value_err(format!("setup has no '{k}'"))) };
    let seats: Vec<(i32, u8)> = get("seats")?.extract()?;
    if !(2..=8).contains(&seats.len()) {
        return Err(value_err("a game needs 2 to 8 seats"));
    }
    let s = GameSetup {
        seed: get("seed")?.extract()?,
        size: get("size")?.extract()?,
        preset: get("preset")?.extract()?,
        mode: get("mode")?.extract()?,
        turn_limit: get("turn_limit")?.extract()?,
        seats: seats.into_iter().map(|(tribe, difficulty)| Seat { tribe, difficulty }).collect(),
    };
    if size_from_u8(s.size).is_none() || preset_from_u8(s.preset).is_none() {
        return Err(value_err("unknown map size or preset id"));
    }
    if s.seats.iter().any(|x| tribe_index(x.tribe).is_none()) {
        return Err(value_err("tribe is not one of the 13 supported (see TRIBES)"));
    }
    Ok(s)
}

/// Player counts as given from Python: a list (uniform) or a dict `{count: weight}`.
#[derive(FromPyObject)]
enum PlayerCounts {
    List(Vec<usize>),
    Weighted(BTreeMap<usize, u32>),
}

fn make_spec(n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, turn_limit: i32) -> PyResult<SetupSpec> {
    let mut spec = SetupSpec { turn_limit, ..Default::default() };
    if let Some(n) = n_players {
        let (counts, weights) = match n {
            PlayerCounts::List(v) => (v, Vec::new()),
            PlayerCounts::Weighted(m) => (m.keys().copied().collect(), m.values().copied().collect()),
        };
        spec = spec.with_players(counts, weights).map_err(value_err)?;
    }
    if let Some(v) = sizes {
        spec.sizes = v.into_iter().map(|x| size_from_u8(x).ok_or_else(|| value_err("size ids: 1 tiny, 2 small, 3 normal"))).collect::<PyResult<_>>()?;
        if spec.sizes.is_empty() {
            return Err(value_err("sizes is empty"));
        }
    }
    if let Some(v) = presets {
        spec.presets = v.into_iter().map(|x| preset_from_u8(x).ok_or_else(|| value_err("unknown preset id"))).collect::<PyResult<_>>()?;
        if spec.presets.is_empty() {
            return Err(value_err("presets is empty"));
        }
    }
    Ok(spec)
}

// ---------------------------------------------------------------------------------------------------------------
// Array helpers

fn tiles_array<'py>(py: Python<'py>, v: Vec<i16>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_vec(py, v).reshape(shape)?.into_any())
}

fn actions_array<'py>(py: Python<'py>, a: &[[i16; 4]]) -> PyResult<Bound<'py, PyAny>> {
    let flat: Vec<i16> = a.iter().flatten().copied().collect();
    Ok(PyArray1::from_vec(py, flat).reshape([a.len(), 4])?.into_any())
}

fn seat_of(order: &[u8], id: u8) -> Option<usize> {
    order.iter().position(|&x| x == id)
}

fn player_order(e: &Engine) -> Vec<u8> {
    e.s.players.iter().map(|p| p.id).filter(|&id| id != NATURE).collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Game

/// One game. Seats are indices into the setup's seat list; Nature has no seat.
#[pyclass(name = "Game")]
struct PyGame(Mutex<GameInner>);

struct GameInner {
    e: Engine,
    setup: GameSetup,
    order: Vec<u8>,
    commands: Vec<Command>,
    counter: TurnCounter,
    legal: Option<Vec<Command>>,
}

impl GameInner {
    fn from_parts(setup: GameSetup) -> Self {
        let e = setup.build();
        let order = player_order(&e);
        Self { e, setup, order, commands: Vec::new(), counter: TurnCounter::default(), legal: None }
    }

    fn legal(&mut self) -> &Vec<Command> {
        if self.legal.is_none() {
            self.legal = Some(legal_actions(&self.e));
        }
        self.legal.as_ref().unwrap()
    }

    fn apply(&mut self, cmd: Command) {
        let (turn, player) = (self.e.s.current_turn, self.e.current_player());
        self.e.apply_unchecked(cmd);
        self.commands.push(cmd);
        self.counter.after(&self.e, &cmd, turn, player);
        self.legal = None;
    }

    fn me(&self) -> PyResult<u8> {
        let me = self.e.current_player();
        if me == NATURE {
            return Err(value_err("it is Nature's turn: call play_ai_move()"));
        }
        Ok(me)
    }
}

#[pymethods]
impl PyGame {
    /// Build a game from a setup dict (as returned by `setup()` or `sample_setup`).
    #[staticmethod]
    fn from_setup(setup: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self(Mutex::new(GameInner::from_parts(setup_from_dict(setup)?))))
    }

    /// A game drawn from `seed` with the imitation-data distribution, optionally narrowed.
    #[staticmethod]
    #[pyo3(signature = (seed, n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT))]
    fn sample(seed: i32, n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, difficulty: Option<u8>, turn_limit: i32) -> PyResult<Self> {
        let mut setup = GameSetup::sample(seed, &make_spec(n_players, sizes, presets, turn_limit)?);
        if let Some(d) = difficulty {
            setup = setup.with_difficulty(d);
        }
        Ok(Self(Mutex::new(GameInner::from_parts(setup))))
    }

    fn setup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self.0.lock().unwrap();
        setup_to_dict(py, &g.setup)
    }

    fn clone(&self) -> Self {
        let g = self.0.lock().unwrap();
        Self(Mutex::new(GameInner { e: g.e.clone(), setup: g.setup.clone(), order: g.order.clone(), commands: g.commands.clone(), counter: g.counter, legal: g.legal.clone() }))
    }

    fn is_over(&self) -> bool {
        let g = self.0.lock().unwrap();
        g.e.is_over()
    }

    #[getter]
    fn turn(&self) -> u32 {
        let g = self.0.lock().unwrap();
        g.e.s.current_turn
    }

    #[getter]
    fn n_players(&self) -> usize {
        let g = self.0.lock().unwrap();
        g.order.len()
    }

    /// (height, width) of the map.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        let g = self.0.lock().unwrap();
        (g.e.height() as usize, g.e.width() as usize)
    }

    /// Seat of the player to move, `None` on Nature's turn.
    fn current_seat(&self) -> Option<usize> {
        let g = self.0.lock().unwrap();
        seat_of(&g.order, g.e.current_player())
    }

    /// Score per seat.
    fn scores(&self) -> Vec<i32> {
        let g = self.0.lock().unwrap();
        g.order.iter().map(|&id| g.e.score(id)).collect()
    }

    /// Winning seats by elimination (empty while the game is running and for a draw at the turn cap).
    fn winners(&self) -> Vec<usize> {
        let g = self.0.lock().unwrap();
        Outcome::of(&g.e).winners.iter().filter_map(|&w| seat_of(&g.order, w)).collect()
    }

    /// True when the game was stopped at the turn cap with more than one player alive (a draw).
    fn capped(&self) -> bool {
        let g = self.0.lock().unwrap();
        Outcome::of(&g.e).capped
    }

    /// Alias of `capped()`, kept for one version.
    fn adjudicated(&self) -> bool {
        self.capped()
    }

    /// The legal list of the player to move as int16[N, 4] `(kind, arg, src_tile, dst_tile)`.
    fn legal_actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut g = self.0.lock().unwrap();
        let me = g.me().or_else(|_| Ok::<u8, PyErr>(NATURE))?;
        let slots = SlotMap::new(&g.e, me);
        let legal = g.legal().clone();
        let tuples: Vec<[i16; 4]> = legal.iter().map(|c| encode_command(&g.e, &slots, c).ok_or_else(|| value_err("cannot encode a legal command"))).collect::<PyResult<_>>()?;
        actions_array(py, &tuples)
    }

    /// `(tiles int16[H, W, 41], globals float32[247])` for the player to move.
    fn encode<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let mut g = self.0.lock().unwrap();
        let me = g.me()?;
        let slots = SlotMap::new(&g.e, me);
        let legal = g.legal().clone();
        let obs = encode(&g.e, me, &slots, &legal);
        Ok((tiles_array(py, obs.tiles, &[obs.h, obs.w, F_TILE])?, PyArray1::from_vec(py, obs.globals).into_any()))
    }

    /// Play the command at `index` of the current legal list.
    fn step(&self, index: usize) -> PyResult<()> {
        let mut g = self.0.lock().unwrap();
        let cmd = *g.legal().get(index).ok_or_else(|| PyIndexError::new_err("action index outside the legal list"))?;
        g.apply(cmd);
        Ok(())
    }

    /// Play an action tuple; it must be one of the legal ones.
    fn step_action(&self, action: [i16; 4]) -> PyResult<()> {
        let mut g = self.0.lock().unwrap();
        let me = g.e.current_player();
        let slots = SlotMap::new(&g.e, me);
        let cmd = decode_command(&g.e, &slots, action).ok_or_else(|| value_err("action does not decode"))?;
        if !g.legal().contains(&cmd) {
            return Err(value_err("action is not legal"));
        }
        g.apply(cmd);
        Ok(())
    }

    /// The retail AI's own pick as an index into the legal list (nothing is played).
    fn ai_move(&self) -> usize {
        let mut g = self.0.lock().unwrap();
        let c = g.counter.0;
        let cmd = ai_pick(&mut g.e, c);
        g.legal().iter().position(|c| *c == cmd).unwrap_or(usize::MAX)
    }

    /// Let the retail AI (or Nature) play one command; returns its legal-list index.
    fn play_ai_move(&self) -> usize {
        let mut g = self.0.lock().unwrap();
        let c = g.counter.0;
        let cmd = ai_pick(&mut g.e, c);
        let idx = g.legal().iter().position(|c| *c == cmd).unwrap_or(usize::MAX);
        g.apply(cmd);
        idx
    }

    /// The AI's best candidates (noise free) as `(legal indices int16[K], scores float32[K])`, best first.
    #[pyo3(signature = (k=16))]
    fn ai_candidates<'py>(&self, py: Python<'py>, k: usize) -> (Bound<'py, PyArray1<i16>>, Bound<'py, PyArray1<f32>>) {
        let mut g = self.0.lock().unwrap();
        let legal = g.legal().clone();
        let mut idx = Vec::new();
        let mut scores = Vec::new();
        for (c, s) in g.e.ai_candidates() {
            if idx.len() == k {
                break;
            }
            if let Some(i) = legal.iter().position(|l| *l == c) {
                if !idx.contains(&(i as i16)) {
                    idx.push(i as i16);
                    scores.push(s);
                }
            }
        }
        (PyArray1::from_vec(py, idx), PyArray1::from_vec(py, scores))
    }

    /// The game so far as a compact record (bytes; see `docs/encoding.md` section 7).
    fn record<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let g = self.0.lock().unwrap();
        let rec = GameRecord { engine_version: polytopia_sim::GAME_VERSION, engine_commit: ENGINE_COMMIT.to_string(), setup: g.setup.clone(), commands: g.commands.clone(), outcome: Outcome::of(&g.e) };
        PyBytes::new(py, &rec.to_bytes())
    }
}

// ---------------------------------------------------------------------------------------------------------------
// VecEnv

#[derive(Clone)]
enum ModelSeats {
    /// One model seat per game, `seed % n_players`.
    Rotate,
    All,
    Fixed(Vec<usize>),
}

struct Pending {
    legal_len: usize,
    obs: Obs,
    actions: Vec<[i16; 4]>,
    seat: i8,
    turn: i16,
}

struct Slot {
    setup: GameSetup,
    e: Engine,
    order: Vec<u8>,
    is_model: Vec<bool>,
    counter: TurnCounter,
    commands: usize,
    decisions: u32,
    legal: Vec<Command>,
    pending: Option<Pending>,
}

impl Slot {
    fn new(setup: GameSetup, model: &ModelSeats) -> Self {
        let e = setup.build();
        let order = player_order(&e);
        let n = order.len();
        let is_model = match model {
            ModelSeats::Rotate => (0..n).map(|s| s == setup.seed.rem_euclid(n as i32) as usize).collect(),
            ModelSeats::All => vec![true; n],
            ModelSeats::Fixed(v) => (0..n).map(|s| v.contains(&s)).collect(),
        };
        let mut slot = Slot { setup, e, order, is_model, counter: TurnCounter::default(), commands: 0, decisions: 0, legal: Vec::new(), pending: None };
        slot.advance();
        slot
    }

    fn done(&self) -> bool {
        self.e.is_over() || self.commands >= MAX_COMMANDS
    }

    fn apply(&mut self, cmd: Command) {
        let (turn, player) = (self.e.s.current_turn, self.e.current_player());
        self.e.apply_unchecked(cmd);
        self.commands += 1;
        self.counter.after(&self.e, &cmd, turn, player);
    }

    /// Play every non-model seat (and Nature) until a model seat is to move or the game ends.
    fn advance(&mut self) {
        self.pending = None;
        while !self.done() {
            let me = self.e.current_player();
            if let Some(seat) = seat_of(&self.order, me).filter(|&s| self.is_model[s]) {
                let slots = SlotMap::new(&self.e, me);
                let legal = legal_actions(&self.e);
                let obs = encode(&self.e, me, &slots, &legal);
                let actions = legal.iter().map(|c| encode_command(&self.e, &slots, c).expect("legal command encodes")).collect();
                self.pending = Some(Pending { legal_len: legal.len(), obs, actions, seat: seat as i8, turn: self.e.s.current_turn as i16 });
                self.legal = legal;
                return;
            }
            let cmd = ai_pick(&mut self.e, self.counter.0);
            self.apply(cmd);
        }
    }

    fn choose(&mut self, index: usize) {
        let cmd = self.legal[index];
        self.decisions += 1;
        self.apply(cmd);
        self.advance();
    }
}

/// What a finished game reports.
struct Finished {
    game: usize,
    setup: GameSetup,
    is_model: Vec<bool>,
    winners: Vec<usize>,
    scores: Vec<i32>,
    turns: u32,
    capped: bool,
    decisions: u32,
    commands: usize,
}

impl Finished {
    fn of(game: usize, s: &Slot) -> Self {
        let o = Outcome::of(&s.e);
        Finished {
            game,
            setup: s.setup.clone(),
            is_model: s.is_model.clone(),
            winners: o.winners.iter().filter_map(|&w| seat_of(&s.order, w)).collect(),
            scores: s.order.iter().map(|&id| o.score_of(id)).collect(),
            turns: o.turns,
            capped: o.is_draw(),
            decisions: s.decisions,
            commands: s.commands,
        }
    }

    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("game", self.game)?;
        d.set_item("seed", self.setup.seed)?;
        d.set_item("n_players", self.setup.seats.len())?;
        d.set_item("tribes", self.setup.seats.iter().map(|s| s.tribe).collect::<Vec<_>>())?;
        d.set_item("difficulties", self.setup.seats.iter().map(|s| s.difficulty).collect::<Vec<_>>())?;
        d.set_item("model_seats", &self.is_model)?;
        d.set_item("winners", &self.winners)?;
        d.set_item("scores", &self.scores)?;
        d.set_item("turns", self.turns)?;
        d.set_item("capped", self.capped)?;
        d.set_item("adjudicated", self.capped)?; // alias of `capped`, kept for one version
        d.set_item("decisions", self.decisions)?;
        d.set_item("commands", self.commands)?;
        Ok(d)
    }
}

/// The games awaiting a model decision for one map size, as plain vectors.
#[derive(Default)]
struct Group {
    h: usize,
    w: usize,
    game: Vec<i32>,
    seat: Vec<i8>,
    n_players: Vec<i8>,
    turn: Vec<i16>,
    seed: Vec<i32>,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    act_off: Vec<i32>,
    actions: Vec<[i16; 4]>,
}

impl Group {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let b = self.game.len();
        d.set_item("game", PyArray1::from_vec(py, self.game.clone()))?;
        d.set_item("seat", PyArray1::from_vec(py, self.seat.clone()))?;
        d.set_item("n_players", PyArray1::from_vec(py, self.n_players.clone()))?;
        d.set_item("turn", PyArray1::from_vec(py, self.turn.clone()))?;
        d.set_item("seed", PyArray1::from_vec(py, self.seed.clone()))?;
        d.set_item("tiles", tiles_array(py, self.tiles.clone(), &[b, self.h, self.w, F_TILE])?)?;
        d.set_item("globals", PyArray1::from_vec(py, self.globals.clone()).reshape([b, G_DIM])?)?;
        d.set_item("act_off", PyArray1::from_vec(py, self.act_off.clone()))?;
        d.set_item("actions", actions_array(py, &self.actions)?)?;
        Ok(d)
    }
}

fn groups_of(slots: &[Slot], which: &[usize]) -> Vec<Group> {
    let mut groups: BTreeMap<(usize, usize), Group> = BTreeMap::new();
    for &i in which {
        let s = &slots[i];
        let Some(p) = s.pending.as_ref() else { continue };
        let g = groups.entry((p.obs.h, p.obs.w)).or_insert_with(|| Group { h: p.obs.h, w: p.obs.w, act_off: vec![0], ..Default::default() });
        g.game.push(i as i32);
        g.seat.push(p.seat);
        g.n_players.push(s.order.len() as i8);
        g.turn.push(p.turn);
        g.seed.push(s.setup.seed);
        g.tiles.extend_from_slice(&p.obs.tiles);
        g.globals.extend_from_slice(&p.obs.globals);
        g.actions.extend_from_slice(&p.actions);
        g.act_off.push(g.actions.len() as i32);
    }
    groups.into_values().collect()
}

/// Many games stepped in parallel. Seats the model does not control are played by the retail AI at their handicap.
#[pyclass(name = "VecEnv")]
struct PyVecEnv(Mutex<EnvInner>);

struct EnvInner {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    spec: SetupSpec,
    model: ModelSeats,
    difficulty: Option<u8>,
    seat_difficulties: Option<Vec<u8>>,
    auto_reset: bool,
    eval: bool,
    seed_start: i64,
    drawn: i64,
}

impl EnvInner {
    fn next_setup(&mut self) -> GameSetup {
        let k = self.drawn;
        self.drawn += 1;
        let seed = if self.eval { self.seed_start + k } else { 1 + (self.seed_start - 1 + k).rem_euclid(i64::from(TRAIN_SEED_MAX)) } as i32;
        let mut setup = GameSetup::sample(seed, &self.spec);
        if let Some(d) = self.difficulty {
            setup = setup.with_difficulty(d);
        }
        if let Some(ds) = &self.seat_difficulties {
            for (seat, &d) in setup.seats.iter_mut().zip(ds) {
                seat.difficulty = d;
            }
        }
        setup
    }

    fn fill(&mut self, which: &[usize]) {
        let setups: Vec<GameSetup> = which.iter().map(|_| self.next_setup()).collect();
        let model = self.model.clone();
        let built: Vec<Slot> = self.pool.install(|| setups.into_par_iter().map(|s| Slot::new(s, &model)).collect());
        for (&i, slot) in which.iter().zip(built) {
            if i < self.slots.len() {
                self.slots[i] = slot;
            } else {
                self.slots.push(slot);
            }
        }
    }
}

#[pymethods]
impl PyVecEnv {
    /// `model_seats`: `"rotate"` (one model seat per game, `seed % n_players`), `"all"` (self-play, the model plays every
    /// seat) or a list of seat indices. `split`: `"train"` draws seeds from 1..=100000 (wrapping), `"eval"` needs
    /// `seed_start >= 1000000`. `seat_difficulties` sets each seat's handicap by seat index (applied after
    /// `difficulty`; every value 0..=3, one per seat of the largest player count).
    #[new]
    #[pyo3(signature = (n_games, seed_start=1, model_seats=None, split="train", n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT, threads=24, auto_reset=true, seat_difficulties=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n_games: usize,
        seed_start: i32,
        model_seats: Option<&Bound<'_, PyAny>>,
        split: &str,
        n_players: Option<PlayerCounts>,
        sizes: Option<Vec<u8>>,
        presets: Option<Vec<u8>>,
        difficulty: Option<u8>,
        turn_limit: i32,
        threads: usize,
        auto_reset: bool,
        seat_difficulties: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        if let Some(ds) = &seat_difficulties {
            if ds.iter().any(|&d| d > 3) {
                return Err(value_err("seat_difficulties values are 0..=3"));
            }
        }
        let eval = match split {
            "train" => false,
            "eval" => true,
            _ => return Err(value_err("split is 'train' or 'eval'")),
        };
        if eval && seed_start < EVAL_SEED_MIN {
            return Err(value_err(format!("eval seeds start at {EVAL_SEED_MIN}")));
        }
        if !eval && !(1..=TRAIN_SEED_MAX).contains(&seed_start) {
            return Err(value_err(format!("train seeds are 1..={TRAIN_SEED_MAX}")));
        }
        if n_games == 0 {
            return Err(value_err("n_games must be positive"));
        }
        let model = match model_seats {
            None => ModelSeats::Rotate,
            Some(o) => match o.extract::<String>() {
                Ok(s) if s == "rotate" => ModelSeats::Rotate,
                Ok(s) if s == "all" => ModelSeats::All,
                Ok(_) => return Err(value_err("model_seats is 'rotate', 'all' or a list of seats")),
                Err(_) => ModelSeats::Fixed(o.extract::<Vec<usize>>()?),
            },
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.clamp(1, MAX_THREADS)).build().map_err(|e| value_err(e.to_string()))?;
        let mut env = EnvInner {
            slots: Vec::new(),
            pool,
            spec: make_spec(n_players, sizes, presets, turn_limit)?,
            model,
            difficulty,
            seat_difficulties,
            auto_reset,
            eval,
            seed_start: i64::from(seed_start),
            drawn: 0,
        };
        let which: Vec<usize> = (0..n_games).collect();
        py.allow_threads(|| env.fill(&which));
        Ok(PyVecEnv(Mutex::new(env)))
    }

    #[getter]
    fn n_games(&self) -> usize {
        self.0.lock().unwrap().slots.len()
    }

    /// Restart every game from fresh seeds and return the games awaiting a decision.
    fn reset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        py.allow_threads(|| {
            let mut env = self.0.lock().unwrap();
            let which: Vec<usize> = (0..env.slots.len()).collect();
            env.fill(&which);
        });
        self.observe(py)
    }

    /// All games awaiting a model decision, one dict per map size (keys: game, seat, n_players, turn, seed, tiles,
    /// globals, act_off, actions; the legal list of the i-th game is `actions[act_off[i]:act_off[i + 1]]`).
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let env = self.0.lock().unwrap();
        let all: Vec<usize> = (0..env.slots.len()).collect();
        let groups = groups_of(&env.slots, &all);
        drop(env);
        PyList::new(py, groups.iter().map(|g| g.to_py(py)).collect::<PyResult<Vec<_>>>()?)
    }

    /// Play `choices[i]` (an index into game `games[i]`'s legal list), then auto-play every other seat until a model
    /// decision or the end. Returns `(observations of the games just stepped, finished games)`. With `auto_reset` a
    /// finished game is replaced by a new seed at once and appears in the observations.
    fn step<'py>(&self, py: Python<'py>, games: PyReadonlyArray1<i32>, choices: PyReadonlyArray1<i32>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let (games, choices) = (games.as_slice()?.to_vec(), choices.as_slice()?.to_vec());
        if games.len() != choices.len() {
            return Err(value_err("games and choices differ in length"));
        }
        let mut env = self.0.lock().unwrap();
        let env = &mut *env;
        let mut chosen: Vec<Option<usize>> = vec![None; env.slots.len()];
        for (&g, &c) in games.iter().zip(&choices) {
            let g = usize::try_from(g).ok().filter(|&g| g < env.slots.len()).ok_or_else(|| PyIndexError::new_err(format!("no game {g}")))?;
            let pending = env.slots[g].pending.as_ref().ok_or_else(|| value_err(format!("game {g} is not awaiting a decision")))?;
            if chosen[g].is_some() {
                return Err(value_err(format!("game {g} appears twice")));
            }
            let c = usize::try_from(c).ok().filter(|&c| c < pending.legal_len).ok_or_else(|| PyIndexError::new_err(format!("choice {c} is outside the legal list of game {g}")))?;
            chosen[g] = Some(c);
        }
        let (groups, finished) = py.allow_threads(|| {
            let slots = &mut env.slots;
            env.pool.install(|| {
                slots.par_iter_mut().zip(chosen.par_iter()).for_each(|(slot, c)| {
                    if let Some(c) = *c {
                        slot.choose(c);
                    }
                })
            });
            let mut touched: Vec<usize> = chosen.iter().enumerate().filter_map(|(i, c)| c.map(|_| i)).collect();
            let over: Vec<usize> = touched.iter().copied().filter(|&i| env.slots[i].done()).collect();
            let finished: Vec<Finished> = over.iter().map(|&i| Finished::of(i, &env.slots[i])).collect();
            if env.auto_reset {
                env.fill(&over);
            }
            touched.retain(|&i| env.slots[i].pending.is_some());
            (groups_of(&env.slots, &touched), finished)
        });
        let groups = PyList::new(py, groups.iter().map(|g| g.to_py(py)).collect::<PyResult<Vec<_>>>()?)?;
        let finished = PyList::new(py, finished.iter().map(|f| f.to_py(py)).collect::<PyResult<Vec<_>>>()?)?;
        Ok((groups, finished))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Module

/// The setup `seed` draws, as a dict (see `Game.from_setup`).
#[pyfunction]
#[pyo3(signature = (seed, n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT))]
fn sample_setup<'py>(py: Python<'py>, seed: i32, n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, difficulty: Option<u8>, turn_limit: i32) -> PyResult<Bound<'py, PyDict>> {
    let mut setup = GameSetup::sample(seed, &make_spec(n_players, sizes, presets, turn_limit)?);
    if let Some(d) = difficulty {
        setup = setup.with_difficulty(d);
    }
    setup_to_dict(py, &setup)
}

/// `(tiles, globals)` of the position the game is in, for the player to move.
#[pyfunction]
fn encode_game<'py>(py: Python<'py>, game: &PyGame) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    game.encode(py)
}

#[pymodule]
fn polyai_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(sample_setup, m)?)?;
    m.add_function(wrap_pyfunction!(encode_game, m)?)?;
    m.add("ENCODING_VERSION", ENCODING_VERSION)?;
    m.add("F_TILE", F_TILE)?;
    m.add("G_DIM", G_DIM)?;
    m.add("N_KINDS", N_KINDS)?;
    m.add("TRIBES", TRIBES.to_vec())?;
    m.add("KIND_NAMES", KIND_NAMES.to_vec())?;
    m.add("CAT_VOCABS", CAT_VOCABS.to_vec())?;
    m.add("ENGINE_COMMIT", ENGINE_COMMIT)?;
    Ok(())
}
