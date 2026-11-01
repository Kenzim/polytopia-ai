//! Python module `polyai_engine`: single games (`Game`) and many games stepped in parallel (`VecEnv`).
//!
//! Layouts (tiles, globals, actions) are those of `docs/encoding.md`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use polyai_core::features::{self, Features};
use polyai_core::*;
use polytopia_sim::engine::replay::{RawCmd, SaveFile};
use polytopia_sim::engine::{Command, Engine, NATURE};
use polytopia_sim::GameState;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

const MAX_THREADS: usize = 24;
/// Players the encoding has slots for (the mover and 7 others).
const MAX_PLAYERS: usize = 8;

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

fn make_spec(n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, turn_limit: i32, mode: i32) -> PyResult<SetupSpec> {
    if mode != DOMINATION && mode != MIGHT {
        return Err(value_err(format!("mode is {DOMINATION} (Domination) or {MIGHT} (Might)")));
    }
    let mut spec = SetupSpec { turn_limit, mode, ..Default::default() };
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

fn no_setup() -> PyErr {
    value_err("this game was rebuilt from a server state and has no setup")
}

/// Differences between the replayed engine and the server's own current state (the checks of the client's `play.rs`).
fn server_diffs(e: &Engine, cur: &GameState, current: &[u8]) -> Vec<String> {
    let mut d = Vec::new();
    if e.s.current_turn != cur.current_turn {
        d.push(format!("turn engine {} server {}", e.s.current_turn, cur.current_turn));
    }
    if e.current_player() != cur.current_player_id().unwrap_or(0) {
        d.push(format!("current player engine {} server {}", e.current_player(), cur.current_player_id().unwrap_or(0)));
    }
    // While the server's action stack is paused on a trigger (a city reward) its score actions are still queued, so its
    // stored scores lag; the engine has already recomputed them.
    let paused = !e.deferred.is_empty() || e.pending_trigger(e.current_player()).is_some();
    for p in cur.players.iter().filter(|p| p.id != NATURE) {
        let mine = e.player(p.id).map(|q| (q.currency, if paused { p.score } else { q.score }));
        if mine != Some((p.currency, p.score)) {
            d.push(format!("p{} stars/score engine {:?} server ({}, {})", p.id, mine, p.currency, p.score));
        }
    }
    for u in &cur.units {
        match e.unit(u.id) {
            None => d.push(format!("unit {} missing in engine", u.id)),
            Some(m) if m.coordinates != u.coordinates || m.health != u.health || m.owner != u.owner || m.type_idx != u.type_idx => {
                d.push(format!("unit {} engine {:?} hp {} server {:?} hp {}", u.id, m.coordinates, m.health, u.coordinates, u.health))
            }
            _ => {}
        }
    }
    for u in &e.s.units {
        if !cur.units.iter().any(|f| f.id == u.id) {
            d.push(format!("unit {} extra in engine", u.id));
        }
    }
    let server_trig: Vec<(u8, i32, i32)> =
        SaveFile::live_triggers(current).unwrap_or_default().into_iter().filter(|t| t.1 == 1).map(|(p, _, w)| (p, w.x, w.y)).collect();
    let engine_trig: Vec<(u8, i32, i32)> = e.triggers.iter().filter(|t| t.peace_from.is_none()).map(|t| (t.player, t.city.x, t.city.y)).collect();
    if server_trig != engine_trig {
        d.push(format!("city triggers engine {engine_trig:?} server {server_trig:?}"));
    }
    d
}

/// The engine at the end of the server's command stack, checked against the server's state, with `seat` to move.
fn rebuild_server(initial: &[u8], current: &[u8], seat: u8) -> Result<Engine, String> {
    if initial.is_empty() || current.is_empty() {
        return Err("the game holds no state data".into());
    }
    let save = SaveFile::parse_server(initial, current).map_err(|e| format!("cannot decode the game state: {e}"))?;
    let mut e = Engine::from_state(save.initial.clone());
    if save.commands.iter().any(|c| c.kind == 20) {
        e.start_match();
    }
    let mut unsupported = 0;
    for c in &save.commands {
        if c.kind == 20 || c.player == NATURE {
            continue;
        }
        match e.command_from_raw(c) {
            Some(cmd) => e.apply_unchecked(cmd),
            None => {
                e.s.current_command += 1;
                unsupported += 1;
            }
        }
    }
    let d = server_diffs(&e, &save.current, current);
    if !d.is_empty() {
        let shown: Vec<_> = d.iter().take(5).cloned().collect();
        return Err(format!("the engine does not agree with the server's state ({} differences, {unsupported} commands unsupported): {}", d.len(), shown.join("; ")));
    }
    if let Some(p) = e.s.players.iter().find(|p| p.id != NATURE && tribe_index(p.tribe).is_none()) {
        return Err(format!("player {} has tribe {}, which is not one of the 13 supported tribes", p.id, p.tribe));
    }
    let n_players = e.s.players.iter().filter(|p| p.id != NATURE).count();
    if n_players > MAX_PLAYERS {
        return Err(format!("the game has {n_players} players; the encoding holds at most {MAX_PLAYERS}"));
    }
    if e.is_over() {
        return Err("the game is over".into());
    }
    if !e.s.players.iter().any(|p| p.id == seat && p.id != NATURE) {
        return Err(format!("seat {seat} is not a player of this game"));
    }
    if e.current_player() != seat {
        return Err(format!("it is player {}'s turn, not seat {seat}'s", e.current_player()));
    }
    Ok(e)
}

/// One game. Seats are indices into the setup's seat list; Nature has no seat.
#[pyclass(name = "Game")]
struct PyGame(Mutex<GameInner>);

struct GameInner {
    e: Engine,
    /// `None` for a game rebuilt from a server's state blobs (`Game.from_server`), which has no setup.
    setup: Option<GameSetup>,
    order: Vec<u8>,
    commands: Vec<Command>,
    counter: TurnCounter,
    legal: Option<Vec<Command>>,
}

impl GameInner {
    fn from_parts(setup: GameSetup) -> Self {
        let e = setup.build();
        let order = player_order(&e);
        Self { e, setup: Some(setup), order, commands: Vec::new(), counter: TurnCounter::default(), legal: None }
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

    /// Our own cap of a Might game; 0 for other modes and for server games, which have no setup.
    fn own_cap(&self) -> i32 {
        self.setup.as_ref().map_or(0, GameSetup::own_cap)
    }

    fn outcome(&self) -> Outcome {
        Outcome::with_cap(&self.e, self.own_cap())
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

    /// Rebuild a live game from the two state blobs the server holds (as `SaveFile::parse_server` reads them): the
    /// initial state and the current state with its command stack. The commands are replayed on the engine and the
    /// result is checked against the server's current state (turn, mover, scores, units, pending city rewards), as the
    /// headless client does before it asks the retail AI. `seat` is the engine player id of our account; it must be
    /// the player to move. Raises `ValueError` with the reason when the state cannot be used (undecodable, the engine
    /// disagrees with the server, a tribe outside the 13 supported, the game is over, or it is not `seat`'s turn).
    /// Such a game has no setup (`setup()` and `record()` raise) and its seats are player-list indices.
    #[staticmethod]
    fn from_server(py: Python<'_>, initial: &[u8], current: &[u8], seat: u8) -> PyResult<Self> {
        let built = py.allow_threads(|| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rebuild_server(initial, current, seat)))
                .unwrap_or_else(|_| Err("the engine failed while replaying the game state".to_string()))
        });
        let e = built.map_err(value_err)?;
        let order = player_order(&e);
        Ok(Self(Mutex::new(GameInner { e, setup: None, order, commands: Vec::new(), counter: TurnCounter::default(), legal: None })))
    }

    /// Engine player id of the player to move (255 is Nature). For a server game this is the id `seat` names.
    #[getter]
    fn mover_id(&self) -> u8 {
        let g = self.0.lock().unwrap();
        g.e.current_player()
    }

    /// The command at `index` of the current legal list as the bytes a retail command stack holds
    /// (`RawCmd::to_bytes`, no version prefix). A `Stay` is encoded like any other command. Raises `ValueError` when
    /// the engine has no retail encoding for the command.
    fn raw_command<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyBytes>> {
        let mut g = self.0.lock().unwrap();
        let cmd = *g.legal().get(index).ok_or_else(|| PyIndexError::new_err("action index outside the legal list"))?;
        let raw = g.e.command_to_raw(&cmd).ok_or_else(|| value_err(format!("the engine has no retail encoding for {cmd:?}")))?;
        Ok(PyBytes::new(py, &raw.to_bytes()))
    }

    /// The inverse of `raw_command`: the legal-list index of a serialised command. Raises `ValueError` when the bytes
    /// do not parse, the engine has no such command, or it is not in the current legal list.
    fn index_of_raw(&self, raw: &[u8]) -> PyResult<usize> {
        let mut g = self.0.lock().unwrap();
        let raw = RawCmd::from_bytes(raw).map_err(|e| value_err(format!("cannot read a command: {e}")))?;
        let cmd = g.e.command_from_raw(&raw).ok_or_else(|| value_err("the engine has no command for these bytes"))?;
        g.legal().iter().position(|c| *c == cmd).ok_or_else(|| value_err(format!("{cmd:?} is not in the legal list")))
    }

    /// `format!("{cmd:?}")` of the command at `index` of the current legal list, for logs.
    fn describe(&self, index: usize) -> PyResult<String> {
        let mut g = self.0.lock().unwrap();
        let cmd = *g.legal().get(index).ok_or_else(|| PyIndexError::new_err("action index outside the legal list"))?;
        Ok(format!("{cmd:?}"))
    }

    /// A game drawn from `seed` with the imitation-data distribution, optionally narrowed.
    #[staticmethod]
    #[pyo3(signature = (seed, n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT, mode=DOMINATION))]
    #[allow(clippy::too_many_arguments)]
    fn sample(seed: i32, n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, difficulty: Option<u8>, turn_limit: i32, mode: i32) -> PyResult<Self> {
        let mut setup = GameSetup::sample(seed, &make_spec(n_players, sizes, presets, turn_limit, mode)?);
        if let Some(d) = difficulty {
            setup = setup.with_difficulty(d);
        }
        Ok(Self(Mutex::new(GameInner::from_parts(setup))))
    }

    fn setup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self.0.lock().unwrap();
        setup_to_dict(py, g.setup.as_ref().ok_or_else(no_setup)?)
    }

    fn clone(&self) -> Self {
        let g = self.0.lock().unwrap();
        Self(Mutex::new(GameInner { e: g.e.clone(), setup: g.setup.clone(), order: g.order.clone(), commands: g.commands.clone(), counter: g.counter, legal: g.legal.clone() }))
    }

    /// The game ended, or a Might game reached its turn cap (see `polyai_core::setup::MIGHT`).
    fn is_over(&self) -> bool {
        let g = self.0.lock().unwrap();
        finished(&g.e, g.own_cap())
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
        g.outcome().winners.iter().filter_map(|&w| seat_of(&g.order, w)).collect()
    }

    /// True when the game was stopped at the turn cap with more than one player alive (a draw).
    fn capped(&self) -> bool {
        let g = self.0.lock().unwrap();
        g.outcome().capped
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
        let tuples: Vec<[i16; 4]> = legal.iter().map(|c| encode_command(&g.e, &slots, c).ok_or_else(|| value_err(format!("cannot encode the legal command {c:?}")))).collect::<PyResult<_>>()?;
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

    /// The engine features (docs/encoding.md section 12) for the player to move:
    /// `(f_tiles int16[H, W, T_F], f_globals float32[G_F], f_actions float32[N, A_F])`, the last aligned with
    /// `legal_actions()`. They use only what the mover can see.
    fn features<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let mut g = self.0.lock().unwrap();
        let me = g.me()?;
        let slots = SlotMap::new(&g.e, me);
        let legal = g.legal().clone();
        let f = features::features(&g.e, me, &slots, &legal);
        Ok((
            tiles_array(py, f.tiles, &[f.h, f.w, features::T_F])?,
            PyArray1::from_vec(py, f.globals).into_any(),
            PyArray1::from_vec(py, f.actions).reshape([legal.len(), features::A_F])?.into_any(),
        ))
    }

    /// The truth labels for the player to move (docs/belief.md): `(truth int16[H, W, TRUTH_K], truth_mask uint8[H, W],
    /// truth_g float32[TRUTH_G])`. Labels are zero on explored tiles; they are never part of `encode()`.
    fn truth<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let g = self.0.lock().unwrap();
        let me = g.me()?;
        let slots = SlotMap::new(&g.e, me);
        let t = truth::truth(&g.e, me, &slots);
        Ok((
            tiles_array(py, t.labels, &[t.h, t.w, truth::TRUTH_K])?,
            PyArray1::from_vec(py, t.mask).reshape([t.h, t.w])?.into_any(),
            PyArray1::from_vec(py, truth::truth_globals(&g.e, me).to_vec()).into_any(),
        ))
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
    fn record<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let g = self.0.lock().unwrap();
        let setup = g.setup.clone().ok_or_else(no_setup)?;
        let outcome = g.outcome();
        let rec = GameRecord { engine_version: polytopia_sim::GAME_VERSION, engine_commit: ENGINE_COMMIT.to_string(), setup, commands: g.commands.clone(), outcome };
        Ok(PyBytes::new(py, &rec.to_bytes()))
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
    feats: Option<Features>,
    truth: Option<(truth::Truth, [f32; truth::TRUTH_G])>,
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
    /// Compute the engine features of every pending decision.
    with_features: bool,
    /// Compute the truth labels (what lies under the fog, docs/belief.md) of every pending decision.
    with_truth: bool,
}

impl Slot {
    fn new(setup: GameSetup, model: &ModelSeats, with_features: bool, with_truth: bool) -> Self {
        let e = setup.build();
        let order = player_order(&e);
        let n = order.len();
        let is_model = match model {
            ModelSeats::Rotate => (0..n).map(|s| s == setup.seed.rem_euclid(n as i32) as usize).collect(),
            ModelSeats::All => vec![true; n],
            ModelSeats::Fixed(v) => (0..n).map(|s| v.contains(&s)).collect(),
        };
        let mut slot = Slot { setup, e, order, is_model, counter: TurnCounter::default(), commands: 0, decisions: 0, legal: Vec::new(), pending: None, with_features, with_truth };
        slot.advance();
        slot
    }

    fn done(&self) -> bool {
        finished(&self.e, self.setup.own_cap()) || self.commands >= MAX_COMMANDS
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
                let feats = self.with_features.then(|| features::features(&self.e, me, &slots, &legal));
                let truth = self.with_truth.then(|| (truth::truth(&self.e, me, &slots), truth::truth_globals(&self.e, me)));
                self.pending = Some(Pending { legal_len: legal.len(), obs, actions, feats, truth, seat: seat as i8, turn: self.e.s.current_turn as i16 });
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
        let o = Outcome::with_cap(&s.e, s.setup.own_cap());
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
    /// Engine features, only when the env was built with `features=True`.
    with_features: bool,
    f_tiles: Vec<i16>,
    f_globals: Vec<f32>,
    f_actions: Vec<f32>,
    /// Truth labels, only when the env was built with `truth=True`: per tile labels and hidden mask, per position scalars.
    with_truth: bool,
    truth: Vec<i16>,
    truth_mask: Vec<u8>,
    truth_g: Vec<f32>,
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
        if self.with_features {
            d.set_item("f_tiles", tiles_array(py, self.f_tiles.clone(), &[b, self.h, self.w, features::T_F])?)?;
            d.set_item("f_globals", PyArray1::from_vec(py, self.f_globals.clone()).reshape([b, features::G_F])?)?;
            d.set_item("f_actions", PyArray1::from_vec(py, self.f_actions.clone()).reshape([self.actions.len(), features::A_F])?)?;
        }
        if self.with_truth {
            d.set_item("truth", tiles_array(py, self.truth.clone(), &[b, self.h, self.w, truth::TRUTH_K])?)?;
            d.set_item("truth_mask", PyArray1::from_vec(py, self.truth_mask.clone()).reshape([b, self.h, self.w])?)?;
            d.set_item("truth_g", PyArray1::from_vec(py, self.truth_g.clone()).reshape([b, truth::TRUTH_G])?)?;
        }
        Ok(d)
    }
}

fn groups_of(slots: &[Slot], which: &[usize]) -> Vec<Group> {
    let mut groups: BTreeMap<(usize, usize), Group> = BTreeMap::new();
    for &i in which {
        let s = &slots[i];
        let Some(p) = s.pending.as_ref() else { continue };
        let g = groups.entry((p.obs.h, p.obs.w)).or_insert_with(|| Group { h: p.obs.h, w: p.obs.w, act_off: vec![0], with_features: p.feats.is_some(), with_truth: p.truth.is_some(), ..Default::default() });
        g.game.push(i as i32);
        g.seat.push(p.seat);
        g.n_players.push(s.order.len() as i8);
        g.turn.push(p.turn);
        g.seed.push(s.setup.seed);
        g.tiles.extend_from_slice(&p.obs.tiles);
        g.globals.extend_from_slice(&p.obs.globals);
        g.actions.extend_from_slice(&p.actions);
        if let Some(f) = &p.feats {
            g.f_tiles.extend_from_slice(&f.tiles);
            g.f_globals.extend_from_slice(&f.globals);
            g.f_actions.extend_from_slice(&f.actions);
        }
        if let Some((t, tg)) = &p.truth {
            g.truth.extend_from_slice(&t.labels);
            g.truth_mask.extend_from_slice(&t.mask);
            g.truth_g.extend_from_slice(tg);
        }
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
    features: bool,
    truth: bool,
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
        let (features, with_truth) = (self.features, self.truth);
        let built: Vec<Slot> = self.pool.install(|| setups.into_par_iter().map(|s| Slot::new(s, &model, features, with_truth)).collect());
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
    #[pyo3(signature = (n_games, seed_start=1, model_seats=None, split="train", n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT, threads=24, auto_reset=true, seat_difficulties=None, mode=DOMINATION, features=false, truth=false))]
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
        mode: i32,
        features: bool,
        truth: bool,
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
            spec: make_spec(n_players, sizes, presets, turn_limit, mode)?,
            model,
            difficulty,
            seat_difficulties,
            auto_reset,
            eval,
            seed_start: i64::from(seed_start),
            drawn: 0,
            features,
            truth,
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
    /// globals, act_off, actions; the legal list of the i-th game is `actions[act_off[i]:act_off[i + 1]]`). Built with
    /// `features=True` each dict also holds f_tiles, f_globals and f_actions (docs/encoding.md section 12), aligned
    /// with tiles, globals and actions. Built with `truth=True` each dict also holds `truth` int16[B, H, W, TRUTH_K],
    /// `truth_mask` uint8[B, H, W] and `truth_g` float32[B, TRUTH_G] (docs/belief.md): labels for the training loss,
    /// never model inputs.
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
#[pyo3(signature = (seed, n_players=None, sizes=None, presets=None, difficulty=None, turn_limit=DEFAULT_TURN_LIMIT, mode=DOMINATION))]
#[allow(clippy::too_many_arguments)]
fn sample_setup<'py>(py: Python<'py>, seed: i32, n_players: Option<PlayerCounts>, sizes: Option<Vec<u8>>, presets: Option<Vec<u8>>, difficulty: Option<u8>, turn_limit: i32, mode: i32) -> PyResult<Bound<'py, PyDict>> {
    let mut setup = GameSetup::sample(seed, &make_spec(n_players, sizes, presets, turn_limit, mode)?);
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
    m.add("FEAT_T", features::T_F)?;
    m.add("FEAT_G", features::G_F)?;
    m.add("FEAT_A", features::A_F)?;
    m.add("FEAT_T_NAMES", features::TF_NAMES.to_vec())?;
    m.add("FEAT_G_NAMES", features::GF_NAMES.to_vec())?;
    m.add("FEAT_A_NAMES", features::A_NAMES.to_vec())?;
    m.add("FEAT_T_DIVISORS", features::TF_DIVISORS.to_vec())?;
    m.add("TRUTH_K", truth::TRUTH_K)?;
    m.add("TRUTH_G", truth::TRUTH_G)?;
    m.add("TRUTH_NAMES", truth::TRUTH_NAMES.to_vec())?;
    m.add("TRUTH_VOCABS", truth::TRUTH_VOCABS.to_vec())?;
    m.add("TRUTH_G_NAMES", truth::TRUTH_G_NAMES.to_vec())?;
    m.add("TRIBES", TRIBES.to_vec())?;
    m.add("KIND_NAMES", KIND_NAMES.to_vec())?;
    m.add("CAT_VOCABS", CAT_VOCABS.to_vec())?;
    m.add("ENGINE_COMMIT", ENGINE_COMMIT)?;
    m.add("DOMINATION", DOMINATION)?;
    m.add("MIGHT", MIGHT)?;
    Ok(())
}
