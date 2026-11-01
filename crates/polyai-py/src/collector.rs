//! The rollout collector's data plane (docs/collector.md): arm scheduling, inference batches, the learner's rows and GAE.
//!
//! `RolloutCore` owns no randomness and no network. Python draws the roles of fresh games and runs the policies;
//! per inference round it makes one `next_round` call (events and batches), one `choose` per batch, and one
//! `finish_round` call (choices in, engine steps started).

use super::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::MutexGuard;
use std::time::{Duration, Instant};

type StepOut = (Vec<usize>, Vec<Finished>);

/// Threads that run engine steps. Not a rayon pool: a step calls `install` on its own env's pool, and a rayon worker
/// waiting in a nested `install` runs other queued jobs on top of it, so one slow step would hold up the arm below it.
struct Workers {
    jobs: Sender<Box<dyn FnOnce() + Send>>,
}

impl Workers {
    fn new(n: usize) -> Self {
        let (jobs, rx) = channel::<Box<dyn FnOnce() + Send>>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..n {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let job = rx.lock().unwrap().recv();
                match job {
                    Ok(job) => job(),
                    Err(_) => break,
                }
            });
        }
        Workers { jobs }
    }

    fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        let _ = self.jobs.send(Box::new(job));
    }
}

/// Generalised advantage estimation of one trajectory (`polyai.rl.compute_gae`, same arithmetic in `f64`).
fn gae(values: &[f64], final_reward: f64, terminal: bool, gamma: f64, lam: f64) -> (Vec<f32>, Vec<f32>) {
    let mut n = values.len();
    let mut reward: Vec<f64>;
    let v_next: Vec<f64>;
    let v: &[f64];
    if terminal {
        reward = vec![0.0; n];
        reward[n - 1] = final_reward;
        v_next = values[1..].iter().copied().chain(std::iter::once(0.0)).collect();
        v = values;
    } else {
        if n <= 1 {
            return (Vec::new(), Vec::new());
        }
        n -= 1;
        reward = vec![0.0; n];
        v_next = values[1..].to_vec();
        v = &values[..n];
    }
    let mut adv = vec![0.0f64; n];
    let mut last = 0.0f64;
    for t in (0..n).rev() {
        let delta = reward[t] + gamma * v_next[t] - v[t];
        last = delta + gamma * lam * last;
        adv[t] = last;
    }
    (adv.iter().map(|&a| a as f32).collect(), adv.iter().zip(v).map(|(&a, &x)| (a + x) as f32).collect())
}

/// The learner's decisions on boards of one size (the columns of `polyai.rl.SizeStore`).
#[derive(Default)]
struct Store {
    h: usize,
    w: usize,
    n: usize,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    lens: Vec<i64>,
    actions: Vec<[i16; 4]>,
    with_features: bool,
    f_tiles: Vec<i16>,
    f_globals: Vec<f32>,
    f_actions: Vec<f32>,
    with_truth: bool,
    truth: Vec<i16>,
    truth_mask: Vec<u8>,
    truth_g: Vec<f32>,
    chosen: Vec<i32>,
    logp: Vec<f32>,
    value: Vec<f32>,
    adv: Vec<f32>,
    ret: Vec<f32>,
    train: Vec<bool>,
}

impl Store {
    fn new(h: usize, w: usize, p: &Pending) -> Self {
        Store { h, w, with_features: p.feats.is_some(), with_truth: p.truth.is_some(), ..Default::default() }
    }

    fn push(&mut self, p: &Pending, chosen: i32, logp: f32, value: f32) -> u32 {
        let id = self.n as u32;
        self.n += 1;
        self.tiles.extend_from_slice(&p.obs.tiles);
        self.globals.extend_from_slice(&p.obs.globals);
        self.lens.push(p.actions.len() as i64);
        self.actions.extend_from_slice(&p.actions);
        if let Some(f) = &p.feats {
            self.f_tiles.extend_from_slice(&f.tiles);
            self.f_globals.extend_from_slice(&f.globals);
            self.f_actions.extend_from_slice(&f.actions);
        }
        if let Some((t, tg)) = &p.truth {
            self.truth.extend_from_slice(&t.labels);
            self.truth_mask.extend_from_slice(&t.mask);
            self.truth_g.extend_from_slice(tg);
        }
        self.chosen.push(chosen);
        self.logp.push(logp);
        self.value.push(value);
        self.adv.push(0.0);
        self.ret.push(0.0);
        self.train.push(false);
        id
    }

    /// The keys, dtypes and shapes of `SizeStore.finalize()`.
    fn into_dict<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let (n, h, w) = (self.n, self.h, self.w);
        let mut off = Vec::with_capacity(n + 1);
        off.push(0i64);
        for l in &self.lens {
            off.push(off.last().unwrap() + l);
        }
        d.set_item("tiles", tiles_array(py, self.tiles, &[n, h, w, F_TILE])?)?;
        d.set_item("globals", PyArray1::from_vec(py, self.globals).reshape([n, G_DIM])?)?;
        d.set_item("act_off", PyArray1::from_vec(py, off))?;
        d.set_item("actions", actions_array(py, &self.actions)?)?;
        if self.with_features {
            let na = self.actions.len();
            d.set_item("f_tiles", tiles_array(py, self.f_tiles, &[n, h, w, features::T_F])?)?;
            d.set_item("f_globals", PyArray1::from_vec(py, self.f_globals).reshape([n, features::G_F])?)?;
            d.set_item("f_actions", PyArray1::from_vec(py, self.f_actions).reshape([na, features::A_F])?)?;
        }
        if self.with_truth {
            d.set_item("truth", tiles_array(py, self.truth, &[n, h, w, truth::TRUTH_K])?)?;
            d.set_item("truth_mask", PyArray1::from_vec(py, self.truth_mask).reshape([n, h, w])?)?;
            d.set_item("truth_g", PyArray1::from_vec(py, self.truth_g).reshape([n, truth::TRUTH_G])?)?;
        }
        let train: Vec<i64> = self.train.iter().enumerate().filter_map(|(i, &t)| t.then_some(i as i64)).collect();
        d.set_item("chosen", PyArray1::from_vec(py, self.chosen))?;
        d.set_item("logp", PyArray1::from_vec(py, self.logp))?;
        d.set_item("value", PyArray1::from_vec(py, self.value))?;
        d.set_item("adv", PyArray1::from_vec(py, self.adv))?;
        d.set_item("ret", PyArray1::from_vec(py, self.ret))?;
        d.set_item("train", PyArray1::from_vec(py, train))?;
        Ok(d)
    }
}

struct ArmState {
    env: Arc<Mutex<EnvInner>>,
    /// The learner's seat in a retail arm; `None` for a model arm (the seat is drawn per game).
    retail_seat: Option<usize>,
    active: Vec<bool>,
    n_active: usize,
    quota: usize,
    learner: Vec<i64>,
    opp: Vec<usize>,
    /// Groups (one board size each, as `groups_of` makes them) of active games awaiting a decision.
    ready: Vec<Vec<usize>>,
    in_flight: bool,
    steps: u64,
    traj: Vec<Vec<u32>>,
    /// Index in `Core::stores` of the store each slot's trajectory lives in.
    traj_store: Vec<usize>,
}

impl ArmState {
    fn ready_rows(&self) -> usize {
        self.ready.iter().map(Vec::len).sum()
    }
}

/// What a harvest brings back for Python, in the order the Python collector handles them: per arm, its finished games,
/// then the fresh games that need roles.
enum Event {
    Finished(usize, Finished),
    /// (arm, slot, players): a fresh game of a model arm, which needs its roles before it is scheduled.
    Role(usize, usize, usize),
}

#[derive(Default)]
struct Events {
    list: Vec<Event>,
    roles: bool,
}

/// The rows of one inference batch, as plain vectors (the layout of `rl_pool.merge_padded`).
struct BatchData {
    key: usize,
    h: usize,
    w: usize,
    padded: bool,
    game: Vec<i32>,
    seat: Vec<i8>,
    n_players: Vec<i8>,
    turn: Vec<i16>,
    seed: Vec<i32>,
    tiles: Vec<i16>,
    globals: Vec<f32>,
    act_off: Vec<i32>,
    actions: Vec<[i16; 4]>,
    board_mask: Vec<bool>,
    features: Option<(Vec<i16>, Vec<f32>, Vec<f32>)>,
}

impl BatchData {
    fn into_dict<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = self.game.len();
        let (h, w) = (self.h, self.w);
        let o = PyDict::new(py);
        o.set_item("game", PyArray1::from_vec(py, self.game))?;
        o.set_item("seat", PyArray1::from_vec(py, self.seat))?;
        o.set_item("n_players", PyArray1::from_vec(py, self.n_players))?;
        o.set_item("turn", PyArray1::from_vec(py, self.turn))?;
        o.set_item("seed", PyArray1::from_vec(py, self.seed))?;
        o.set_item("tiles", tiles_array(py, self.tiles, &[b, h, w, F_TILE])?)?;
        o.set_item("globals", PyArray1::from_vec(py, self.globals).reshape([b, G_DIM])?)?;
        if let Some((ft, fg, fa)) = self.features {
            let na = self.actions.len();
            o.set_item("f_tiles", tiles_array(py, ft, &[b, h, w, features::T_F])?)?;
            o.set_item("f_globals", PyArray1::from_vec(py, fg).reshape([b, features::G_F])?)?;
            o.set_item("f_actions", PyArray1::from_vec(py, fa).reshape([na, features::A_F])?)?;
        }
        if self.padded {
            o.set_item("board_mask", PyArray1::from_vec(py, self.board_mask).reshape([b, h, w])?)?;
        }
        o.set_item("act_off", PyArray1::from_vec(py, self.act_off))?;
        o.set_item("actions", actions_array(py, &self.actions)?)?;
        let d = PyDict::new(py);
        d.set_item("key", self.key)?;
        d.set_item("obs", o)?;
        Ok(d)
    }
}

/// One policy call of the round: the policy, the board bucket and the rows (arm, slot) in batch order.
struct Part {
    key: usize,
    bucket: (usize, usize),
    rows: Vec<(usize, usize)>,
}

#[derive(Clone, Copy)]
enum Wait {
    No,
    Block,
    Timeout(Duration),
}

struct Core {
    arms: Vec<ArmState>,
    names: Vec<String>,
    workers: Workers,
    tx: Sender<(usize, StepOut)>,
    rx: Receiver<(usize, StepOut)>,
    max_lead: u64,
    min_infer: usize,
    gather: Duration,
    pad_sizes: bool,
    gamma: f64,
    lam: f64,
    drop_capped: bool,
    /// In order of first use, as the Python collector's dict of `SizeStore`s.
    stores: Vec<Store>,
    n_learner: usize,
    n_opp: usize,
    t_wait: f64,
    go: Vec<usize>,
    parts: Vec<Part>,
}

/// Close the learner's trajectory in `slot` of `arm`: GAE over its stored values, marking the rows as trained.
fn flush(arm: &mut ArmState, stores: &mut [Store], slot: usize, outcome: f64, terminal: bool, gamma: f64, lam: f64) {
    let ids = std::mem::take(&mut arm.traj[slot]);
    if ids.is_empty() {
        return;
    }
    let st = &mut stores[arm.traj_store[slot]];
    let values: Vec<f64> = ids.iter().map(|&i| f64::from(st.value[i as usize])).collect();
    let (adv, ret) = gae(&values, outcome, terminal, gamma, lam);
    for (k, &id) in ids.iter().take(adv.len()).enumerate() {
        let id = id as usize;
        st.adv[id] = adv[k];
        st.ret[id] = ret[k];
        st.train[id] = true;
    }
}

impl Core {
    fn harvest(&mut self, wait: Wait, ev: &mut Events) {
        let mut done: Vec<(usize, StepOut)> = Vec::new();
        let t0 = Instant::now();
        match wait {
            Wait::Block if self.arms.iter().any(|a| a.in_flight) => {
                if let Ok(x) = self.rx.recv() {
                    done.push(x);
                }
            }
            Wait::Timeout(d) => match self.rx.recv_timeout(d) {
                Ok(x) => done.push(x),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {}
            },
            _ => {}
        }
        if !matches!(wait, Wait::No) {
            self.t_wait += t0.elapsed().as_secs_f64();
        }
        while let Ok(x) = self.rx.try_recv() {
            done.push(x);
        }
        done.sort_by_key(|(i, _)| *i);
        for (i, out) in done {
            self.after_step(i, out, ev);
        }
    }

    /// Take in a finished engine step of arm `ai` (`Collector._after_step`).
    fn after_step(&mut self, ai: usize, out: StepOut, ev: &mut Events) {
        let (touched, finished) = out;
        let (gamma, lam, drop_capped) = (self.gamma, self.lam, self.drop_capped);
        let arm = &mut self.arms[ai];
        arm.in_flight = false;
        let mut reset = vec![false; arm.active.len()];
        for f in finished {
            let slot = f.game;
            let seat = arm.retail_seat.map_or(arm.learner[slot], |s| s as i64);
            let outcome = if f.capped || f.winners.is_empty() {
                0.5
            } else if f.winners.iter().any(|&w| w as i64 == seat) {
                1.0
            } else {
                0.0
            };
            if f.capped && drop_capped {
                arm.traj[slot].clear();
            } else {
                flush(arm, &mut self.stores, slot, outcome, true, gamma, lam);
            }
            reset[slot] = true;
            if arm.n_active > arm.quota {
                arm.active[slot] = false;
                arm.n_active -= 1;
            }
            ev.list.push(Event::Finished(ai, f));
        }
        let mut by_size: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        let handle = arm.env.clone();
        let env = handle.lock().unwrap();
        for &i in &touched {
            let p = env.slots[i].pending.as_ref().expect("a stepped slot awaits a decision");
            by_size.entry((p.obs.h, p.obs.w)).or_default().push(i);
        }
        for (_, slots) in by_size {
            let keep: Vec<usize> = slots.into_iter().filter(|&i| arm.active[i]).collect();
            if keep.is_empty() {
                continue;
            }
            if arm.retail_seat.is_none() {
                for &i in keep.iter().filter(|&&i| reset[i]) {
                    ev.list.push(Event::Role(ai, i, env.slots[i].order.len()));
                    ev.roles = true;
                }
            }
            arm.ready.push(keep);
        }
    }

    /// Arms with games awaiting a decision, no step in flight and no more than `max_lead` steps ahead of the slowest.
    fn may_step(&self) -> Result<Vec<usize>, String> {
        let live: Vec<usize> = (0..self.arms.len()).filter(|&i| self.arms[i].in_flight || !self.arms[i].ready.is_empty()).collect();
        if live.is_empty() {
            return Err("no game is waiting for a decision".into());
        }
        let floor = live.iter().map(|&i| self.arms[i].steps).min().unwrap();
        Ok(live.into_iter().filter(|&i| !self.arms[i].in_flight && !self.arms[i].ready.is_empty() && self.arms[i].steps <= floor + self.max_lead).collect())
    }

    fn bucket_of(&self, h: usize, w: usize) -> (usize, usize) {
        if self.pad_sizes {
            let side = h.max(w);
            let b = if side <= 16 { 16 } else if side <= 20 { 20 } else { side };
            (b, b)
        } else {
            (h, w)
        }
    }

    /// Sort the rows of the arms in `go` into one part per (policy, board bucket), in the order the Python collector does.
    fn plan_parts(&self, go: &[usize]) -> Vec<Part> {
        let mut parts: Vec<Part> = Vec::new();
        for &ai in go {
            let arm = &self.arms[ai];
            let env = arm.env.lock().unwrap();
            for group in &arm.ready {
                let p0 = env.slots[group[0]].pending.as_ref().unwrap();
                let bucket = self.bucket_of(p0.obs.h, p0.obs.w);
                let mut keyed: Vec<(usize, usize)> = Vec::with_capacity(group.len()); // (key, slot)
                for &slot in group {
                    let key = if arm.retail_seat.is_some() {
                        0
                    } else {
                        let p = env.slots[slot].pending.as_ref().unwrap();
                        if i64::from(p.seat) == arm.learner[slot] { 0 } else { arm.opp[slot] }
                    };
                    keyed.push((key, slot));
                }
                let mut keys: Vec<usize> = keyed.iter().map(|k| k.0).collect();
                keys.sort_by(|a, b| self.names[*a].cmp(&self.names[*b]));
                keys.dedup();
                for key in keys {
                    let pos = match parts.iter().position(|p| p.key == key && p.bucket == bucket) {
                        Some(i) => i,
                        None => {
                            parts.push(Part { key, bucket, rows: Vec::new() });
                            parts.len() - 1
                        }
                    };
                    parts[pos].rows.extend(keyed.iter().filter(|k| k.0 == key).map(|k| (ai, k.1)));
                }
            }
        }
        parts
    }

    /// The padded (or, when all boards have one size, plain) batch of `rows`.
    fn build_batch(&self, key: usize, rows: &[(usize, usize)], guards: &[Option<MutexGuard<'_, EnvInner>>]) -> BatchData {
        let pend = |&(ai, slot): &(usize, usize)| -> (&Slot, &Pending) {
            let s = &guards[ai].as_ref().unwrap().slots[slot];
            (s, s.pending.as_ref().unwrap())
        };
        let (mut hp, mut wp) = (0, 0);
        let mut padded = false;
        let first = pend(&rows[0]).1;
        for r in rows {
            let p = pend(r).1;
            hp = hp.max(p.obs.h);
            wp = wp.max(p.obs.w);
            padded |= p.obs.h != first.obs.h || p.obs.w != first.obs.w;
        }
        let n = rows.len();
        let with_feats = first.feats.is_some();
        let mut b = BatchData {
            key,
            h: hp,
            w: wp,
            padded,
            game: Vec::with_capacity(n),
            seat: Vec::with_capacity(n),
            n_players: Vec::with_capacity(n),
            turn: Vec::with_capacity(n),
            seed: Vec::with_capacity(n),
            tiles: if padded { vec![0; n * hp * wp * F_TILE] } else { Vec::with_capacity(n * hp * wp * F_TILE) },
            globals: Vec::with_capacity(n * G_DIM),
            act_off: Vec::with_capacity(n + 1),
            actions: Vec::new(),
            board_mask: if padded { vec![false; n * hp * wp] } else { Vec::new() },
            features: with_feats.then(|| {
                (
                    if padded { vec![0; n * hp * wp * features::T_F] } else { Vec::with_capacity(n * hp * wp * features::T_F) },
                    Vec::with_capacity(n * features::G_F),
                    Vec::new(),
                )
            }),
        };
        b.act_off.push(0);
        for (i, r) in rows.iter().enumerate() {
            let (s, p) = pend(r);
            b.game.push(r.1 as i32);
            b.seat.push(p.seat);
            b.n_players.push(s.order.len() as i8);
            b.turn.push(p.turn);
            b.seed.push(s.setup.seed);
            b.globals.extend_from_slice(&p.obs.globals);
            let (h, w) = (p.obs.h, p.obs.w);
            if padded {
                for y in 0..h {
                    let src = &p.obs.tiles[y * w * F_TILE..(y + 1) * w * F_TILE];
                    let at = ((i * hp + y) * wp) * F_TILE;
                    b.tiles[at..at + w * F_TILE].copy_from_slice(src);
                    let mask_at = (i * hp + y) * wp;
                    b.board_mask[mask_at..mask_at + w].fill(true);
                }
                for a in &p.actions {
                    let mut a = *a;
                    for c in [2, 3] {
                        let idx = i32::from(a[c]);
                        if idx >= 0 {
                            a[c] = (idx / w as i32 * wp as i32 + idx % w as i32) as i16;
                        }
                    }
                    b.actions.push(a);
                }
            } else {
                b.tiles.extend_from_slice(&p.obs.tiles);
                b.actions.extend_from_slice(&p.actions);
            }
            b.act_off.push(b.actions.len() as i32);
            if let (Some((ft, fg, fa)), Some(f)) = (b.features.as_mut(), &p.feats) {
                if padded {
                    for y in 0..h {
                        let src = &f.tiles[y * w * features::T_F..(y + 1) * w * features::T_F];
                        let at = ((i * hp + y) * wp) * features::T_F;
                        ft[at..at + w * features::T_F].copy_from_slice(src);
                    }
                } else {
                    ft.extend_from_slice(&f.tiles);
                }
                fg.extend_from_slice(&f.globals);
                fa.extend_from_slice(&f.actions);
            }
        }
        b
    }

    /// Harvest, schedule and build the batches (`Collector.tick` up to the policy calls). Events that need Python (roles of
    /// fresh games) end the round early, without batches.
    fn next_round(&mut self) -> Result<(Events, Vec<BatchData>), String> {
        let mut ev = Events::default();
        self.parts.clear();
        self.go.clear();
        self.harvest(Wait::No, &mut ev);
        if ev.roles {
            return Ok((ev, Vec::new()));
        }
        let mut go = self.may_step()?;
        if self.min_infer > 0 && !go.is_empty() {
            let deadline = Instant::now() + self.gather;
            while go.iter().map(|&i| self.arms[i].ready_rows()).sum::<usize>() < self.min_infer {
                let left = deadline.saturating_duration_since(Instant::now());
                if !self.arms.iter().any(|a| a.in_flight) || left.is_zero() {
                    break;
                }
                self.harvest(Wait::Timeout(left), &mut ev);
                if ev.roles {
                    return Ok((ev, Vec::new()));
                }
                go = self.may_step()?;
            }
        }
        if go.is_empty() {
            self.harvest(Wait::Block, &mut ev);
            return Ok((ev, Vec::new()));
        }
        let parts = self.plan_parts(&go);
        let handles: Vec<Option<Arc<Mutex<EnvInner>>>> = (0..self.arms.len()).map(|i| go.contains(&i).then(|| self.arms[i].env.clone())).collect();
        let guards: Vec<Option<MutexGuard<'_, EnvInner>>> = handles.iter().map(|h| h.as_ref().map(|h| h.lock().unwrap())).collect();
        let batches: Vec<BatchData> = parts.iter().map(|p| self.build_batch(p.key, &p.rows, &guards)).collect();
        drop(guards);
        self.parts = parts;
        self.go = go;
        Ok((ev, batches))
    }

    /// Record the policies' answers: the learner's rows into the stores, the choices into engine steps, started on the
    /// workers (`Collector.tick` after the policy calls).
    fn finish_round(&mut self, results: Vec<(Vec<i32>, Vec<f32>, Vec<f32>)>) -> Result<(), String> {
        if results.len() != self.parts.len() {
            return Err(format!("{} results for {} batches", results.len(), self.parts.len()));
        }
        let parts = std::mem::take(&mut self.parts);
        let go = std::mem::take(&mut self.go);
        let mut chosen: Vec<Option<Vec<Option<usize>>>> = vec![None; self.arms.len()];
        {
            let handles: Vec<Option<Arc<Mutex<EnvInner>>>> = (0..self.arms.len()).map(|i| go.contains(&i).then(|| self.arms[i].env.clone())).collect();
            let guards: Vec<Option<MutexGuard<'_, EnvInner>>> = handles.iter().map(|h| h.as_ref().map(|h| h.lock().unwrap())).collect();
            for &ai in &go {
                chosen[ai] = Some(vec![None; guards[ai].as_ref().unwrap().slots.len()]);
            }
            for (part, (index, logp, value)) in parts.iter().zip(&results) {
                let n = part.rows.len();
                if index.len() != n || logp.len() != n || value.len() != n {
                    return Err("a result does not match the rows of its batch".into());
                }
                for (r, &(ai, slot)) in part.rows.iter().enumerate() {
                    let p = guards[ai].as_ref().unwrap().slots[slot].pending.as_ref().unwrap();
                    let c = usize::try_from(index[r]).ok().filter(|&c| c < p.legal_len).ok_or_else(|| format!("choice {} is outside the legal list of game {slot}", index[r]))?;
                    chosen[ai].as_mut().unwrap()[slot] = Some(c);
                    if part.key == 0 {
                        let hw = (p.obs.h, p.obs.w);
                        let si = match self.stores.iter().position(|s| (s.h, s.w) == hw) {
                            Some(i) => i,
                            None => {
                                self.stores.push(Store::new(hw.0, hw.1, p));
                                self.stores.len() - 1
                            }
                        };
                        let id = self.stores[si].push(p, index[r], logp[r], value[r]);
                        self.arms[ai].traj[slot].push(id);
                        self.arms[ai].traj_store[slot] = si;
                        self.n_learner += 1;
                    } else {
                        self.n_opp += 1;
                    }
                }
            }
        }
        for &ai in &go {
            let arm = &mut self.arms[ai];
            arm.ready.clear();
            arm.steps += 1;
            arm.in_flight = true;
            let (env, tx, ch) = (arm.env.clone(), self.tx.clone(), chosen[ai].take().unwrap());
            self.workers.spawn(move || {
                let out = env.lock().unwrap().step_slots(&ch);
                let _ = tx.send((ai, out));
            });
        }
        Ok(())
    }

    /// Settle every step in flight, arm by arm in order (`Collector.collect`, after the loop).
    fn drain(&mut self) -> Events {
        let mut ev = Events::default();
        let mut done: Vec<(usize, StepOut)> = Vec::new();
        let t0 = Instant::now();
        while self.arms.iter().filter(|a| a.in_flight).count() > done.len() {
            match self.rx.recv() {
                Ok(x) => done.push(x),
                Err(_) => break,
            }
        }
        self.t_wait += t0.elapsed().as_secs_f64();
        done.sort_by_key(|(i, _)| *i);
        for (i, out) in done {
            self.after_step(i, out, &mut ev);
        }
        ev
    }

    fn cut(&mut self) {
        let (gamma, lam) = (self.gamma, self.lam);
        for arm in &mut self.arms {
            for slot in 0..arm.traj.len() {
                flush(arm, &mut self.stores, slot, 0.0, false, gamma, lam);
            }
        }
    }
}

fn events_to_py<'py>(py: Python<'py>, ev: Events) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for e in ev.list {
        match e {
            Event::Finished(arm, f) => out.append(("fin", arm, f.to_py(py)?))?,
            Event::Role(arm, slot, n) => out.append(("role", arm, slot, n))?,
        }
    }
    Ok(out)
}

/// The collector's data plane for a set of arms; see `polyai.rl_collect.RustCollector`.
#[pyclass(name = "RolloutCore")]
pub struct PyCore(Mutex<Core>);

#[pymethods]
impl PyCore {
    #[new]
    #[pyo3(signature = (workers, max_lead, min_infer, gather_ms, pad_sizes, gamma, lam, drop_capped))]
    #[allow(clippy::too_many_arguments)]
    fn new(workers: usize, max_lead: u64, min_infer: usize, gather_ms: f64, pad_sizes: bool, gamma: f64, lam: f64, drop_capped: bool) -> PyResult<Self> {
        let (tx, rx) = channel();
        let workers = Workers::new(workers.clamp(1, MAX_THREADS));
        Ok(PyCore(Mutex::new(Core {
            arms: Vec::new(),
            names: vec!["learner".to_string()],
            workers,
            tx,
            rx,
            max_lead,
            min_infer,
            gather: Duration::from_secs_f64(gather_ms.max(0.0) / 1000.0),
            pad_sizes,
            gamma,
            lam,
            drop_capped,
            stores: Vec::new(),
            n_learner: 0,
            n_opp: 0,
            t_wait: 0.0,
            go: Vec::new(),
            parts: Vec::new(),
        })))
    }

    /// Add an arm (a `VecEnv`); `retail_seat` is the learner's seat of a retail arm, `None` for a model arm. Returns its index.
    #[pyo3(signature = (env, retail_seat=None))]
    fn add_arm(&self, env: &PyVecEnv, retail_seat: Option<usize>) -> usize {
        let mut c = self.0.lock().unwrap();
        let cap = env.0.lock().unwrap().slots.len();
        c.arms.push(ArmState {
            env: env.0.clone(),
            retail_seat,
            active: vec![false; cap],
            n_active: 0,
            quota: 0,
            learner: vec![0; cap],
            opp: vec![0; cap],
            ready: Vec::new(),
            in_flight: false,
            steps: 0,
            traj: vec![Vec::new(); cap],
            traj_store: vec![0; cap],
        });
        c.arms.len() - 1
    }

    /// The index of policy `name` in the table that orders the parts of a round (registered once; "learner" is 0).
    fn register(&self, name: &str) -> usize {
        let mut c = self.0.lock().unwrap();
        match c.names.iter().position(|n| n == name) {
            Some(i) => i,
            None => {
                c.names.push(name.to_string());
                c.names.len() - 1
            }
        }
    }

    fn set_quota(&self, arm: usize, quota: usize) {
        self.0.lock().unwrap().arms[arm].quota = quota;
    }

    fn n_active(&self, arm: usize) -> usize {
        self.0.lock().unwrap().arms[arm].n_active
    }

    fn active_mask<'py>(&self, py: Python<'py>, arm: usize) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.0.lock().unwrap().arms[arm].active.clone())
    }

    /// Every game of the arm awaiting a decision, one `(slots, n_players)` pair per board size, in the order
    /// `VecEnv.observe` lists them (active or not: see `active_mask`).
    fn observe_slots<'py>(&self, py: Python<'py>, arm: usize) -> PyResult<Bound<'py, PyList>> {
        let handle = self.0.lock().unwrap().arms[arm].env.clone();
        let env = handle.lock().unwrap();
        let mut by: BTreeMap<(usize, usize), (Vec<i32>, Vec<i8>)> = BTreeMap::new();
        for (i, s) in env.slots.iter().enumerate() {
            if let Some(p) = s.pending.as_ref() {
                let e = by.entry((p.obs.h, p.obs.w)).or_default();
                e.0.push(i as i32);
                e.1.push(s.order.len() as i8);
            }
        }
        PyList::new(py, by.into_values().map(|(s, n)| (PyArray1::from_vec(py, s), PyArray1::from_vec(py, n))))
    }

    /// Make the games `slots` (all idle, awaiting a decision, of one board size) active and ready.
    fn activate(&self, arm: usize, slots: Vec<usize>) -> PyResult<()> {
        let mut c = self.0.lock().unwrap();
        let a = &mut c.arms[arm];
        if a.in_flight {
            return Err(value_err("activate while a step is in flight"));
        }
        for &s in &slots {
            if a.active[s] {
                return Err(value_err(format!("game {s} is already active")));
            }
            a.active[s] = true;
        }
        a.n_active += slots.len();
        a.ready.push(slots);
        Ok(())
    }

    /// Roles of games: the learner's seat and the opponent policy index (a `register` result) per `(arm, slot)`.
    fn set_roles(&self, arms: Vec<usize>, slots: Vec<usize>, learner: Vec<i64>, opp: Vec<usize>) -> PyResult<()> {
        if arms.len() != slots.len() || arms.len() != learner.len() || arms.len() != opp.len() {
            return Err(value_err("set_roles needs arrays of one length"));
        }
        let mut c = self.0.lock().unwrap();
        for (((&a, &s), &l), &o) in arms.iter().zip(&slots).zip(&learner).zip(&opp) {
            c.arms[a].learner[s] = l;
            c.arms[a].opp[s] = o;
        }
        Ok(())
    }

    /// Start a rollout: clear the stores, counters and step counts (steps in flight must have been drained).
    fn begin_rollout(&self) {
        let mut c = self.0.lock().unwrap();
        c.stores.clear();
        c.n_learner = 0;
        c.n_opp = 0;
        c.t_wait = 0.0;
        for a in &mut c.arms {
            a.steps = 0;
        }
    }

    /// `(learner decisions, opponent decisions, seconds spent waiting for engine steps)` of the rollout so far.
    fn counters(&self) -> (usize, usize, f64) {
        let c = self.0.lock().unwrap();
        (c.n_learner, c.n_opp, c.t_wait)
    }

    /// One round: `(events, batches)`. Events, in the order to handle them: `("fin", arm, dict)` for a finished game,
    /// `("role", arm, slot, players)` for a fresh game of a model arm (then there are no batches: answer with
    /// `set_roles` and call again). Batches are `{"key": policy index, "obs": dict}` in the order of the Python
    /// collector's parts.
    fn next_round<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let (ev, batches) = py.allow_threads(|| self.0.lock().unwrap().next_round()).map_err(PyRuntimeError::new_err)?;
        let events = events_to_py(py, ev)?;
        let batches = PyList::new(py, batches.into_iter().map(|b| b.into_dict(py)).collect::<PyResult<Vec<_>>>()?)?;
        Ok((events, batches))
    }

    /// The policies' answers to the batches of the round, `(index, logp, value)` each, in batch order. Returns the
    /// learner and opponent decisions of the rollout so far.
    fn finish_round(&self, py: Python<'_>, results: Vec<(PyReadonlyArray1<i32>, PyReadonlyArray1<f32>, PyReadonlyArray1<f32>)>) -> PyResult<(usize, usize)> {
        let results = results
            .iter()
            .map(|(i, l, v)| Ok((i.as_slice()?.to_vec(), l.as_slice()?.to_vec(), v.as_slice()?.to_vec())))
            .collect::<PyResult<Vec<_>>>()?;
        py.allow_threads(|| {
            let mut c = self.0.lock().unwrap();
            c.finish_round(results).map_err(value_err)?;
            Ok((c.n_learner, c.n_opp))
        })
    }

    /// Settle the steps in flight; returns events as `next_round` does.
    fn drain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let ev = py.allow_threads(|| self.0.lock().unwrap().drain());
        events_to_py(py, ev)
    }

    /// End of a rollout: close every unfinished trajectory as truncated.
    fn cut(&self) {
        self.0.lock().unwrap().cut();
    }

    /// The rollout's rows per board size, as `SizeStore.finalize()` returns them, in order of first use (emptied).
    fn take_stores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let stores = std::mem::take(&mut self.0.lock().unwrap().stores);
        PyList::new(py, stores.into_iter().map(|st| Ok(((st.h, st.w), st.into_dict(py)?))).collect::<PyResult<Vec<_>>>()?)
    }
}

/// `polyai.rl.compute_gae` in Rust, exposed for the tests.
#[pyfunction]
pub fn compute_gae_rust<'py>(py: Python<'py>, values: Vec<f64>, final_reward: f64, terminal: bool, gamma: f64, lam: f64) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>) {
    let (a, r) = gae(&values, final_reward, terminal, gamma, lam);
    (PyArray1::from_vec(py, a), PyArray1::from_vec(py, r))
}
