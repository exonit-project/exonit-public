//! Observer of the cross-platform v3 FPP gate (plan 2026-10-15-fpp-native-fix-plan.md, decision (t) point 2): the
//! per-scenario per-draw quantities that `scripts/fpp/gen_v3_gate_dumps.py` records from the Python reference, recorded
//! from the same places of this port. It is OFF unless a [`GateRecorder`] lives on the thread (only the `fpp_gate_dump`
//! driver starts one); then it never changes a computed bit, it only copies.
//!
//! What is recorded, per scenario reduction of a pass (Python `_lnZ_from_logw` call), in reduction order:
//! * `lnl`: the full per-draw lnL array (all N draws, -inf where not evaluated), BEFORE the prior is added;
//! * production pass: `state` (int8 per draw: -1 not evaluated by a production fold kernel, else the kernel's
//!   exposure-order status) and the contact roots (`fpp_native._contact_times_nb` = [`super::contact::contact_times`]
//!   with the k the kernel used) of every evaluated draw whose lnL is finite and within `window` nat of the maximum
//!   finite lnL of that evaluation (Python `Recorder._contacts`, window over the whole evaluation, never per batch);
//! * parity pass: `state` and contacts are absent (`null` in the dump).
//!
//! The mechanics mirror the generator's wrappers: the production kernels report their rows (parameters, lnL, status;
//! [`kernel_rows`]); the scenario code then attaches them to the full draw indices of its evaluation ([`note`], Python
//! `_note_exposure_stats`), and stages the full lnL array just before its reduction ([`stage`]); the reduction appends
//! the staged payload in lockstep with the step-7 diagnostics sink ([`push_reduction`]), so the records map to scenario
//! names exactly as the draw-weight diagnostics do (`calc_v3::take_records`).

use std::cell::RefCell;
use std::collections::HashMap;

/// One scenario evaluation of the gate dump.
#[derive(Clone, Debug, Default)]
pub struct GateRecord {
    pub lnl: Vec<f64>,
    pub state: Option<Vec<i8>>,
    /// (draw indices ascending, roots row-major [n, 5])
    pub contacts: Option<(Vec<i64>, Vec<f64>)>,
}

#[derive(Clone, Copy, Debug)]
struct KernelRow {
    period: f64,
    a_rs: f64,
    inc_rad: f64,
    ecc: f64,
    w_rad: f64,
    k: f64,
    lnl: f64,
    status: i64,
}

struct Note {
    slot: &'static str,
    selected: Vec<usize>,
    rows: Vec<KernelRow>,
}

struct GateState {
    production: bool,
    window: f64,
    rows: Vec<KernelRow>,
    notes: Vec<Note>,
    staged: Option<GateRecord>,
    records: Vec<Option<GateRecord>>,
    named: HashMap<String, Option<GateRecord>>,
    error: Option<String>,
}

thread_local! {
    static GATE: RefCell<Option<GateState>> = const { RefCell::new(None) };
}

/// RAII guard: while it lives, the gate observer records on this thread.
pub struct GateRecorder {
    _private: (),
}

impl GateRecorder {
    /// `production`: the pass runs the production fold (state + contacts recorded); `window`: contact window (nat).
    pub fn start(production: bool, window: f64) -> Self {
        GATE.with(|g| {
            *g.borrow_mut() = Some(GateState {
                production,
                window,
                rows: Vec::new(),
                notes: Vec::new(),
                staged: None,
                records: Vec::new(),
                named: HashMap::new(),
                error: None,
            })
        });
        GateRecorder { _private: () }
    }

    /// The first inconsistency seen (a batch that does not land at its draw indices, unmatched rows), if any.
    pub fn error(&self) -> Option<String> {
        GATE.with(|g| g.borrow().as_ref().and_then(|s| s.error.clone()))
    }
}

impl Drop for GateRecorder {
    fn drop(&mut self) {
        GATE.with(|g| *g.borrow_mut() = None);
    }
}

pub(crate) fn active() -> bool {
    GATE.with(|g| g.borrow().is_some())
}

fn with_state(f: impl FnOnce(&mut GateState)) {
    GATE.with(|g| {
        if let Some(s) = g.borrow_mut().as_mut() {
            f(s);
        }
    });
}

fn fail(s: &mut GateState, msg: String) {
    if s.error.is_none() {
        s.error = Some(msg);
    }
}

/// A production kernel's rows, in kernel row order (called by `batch_*_lnl_prod` when the observer is active).
#[allow(clippy::too_many_arguments)]
pub(crate) fn kernel_rows(
    periods: &[f64],
    a_rs: &[f64],
    incs_rad: &[f64],
    eccs: &[f64],
    ws_rad: &[f64],
    ks_used: &[f64],
    lnl: &[f64],
    status: &[i64],
) {
    with_state(|s| {
        for j in 0..lnl.len() {
            s.rows.push(KernelRow {
                period: periods[j],
                a_rs: a_rs[j],
                inc_rad: incs_rad[j],
                ecc: eccs[j],
                w_rad: ws_rad[j],
                k: ks_used[j],
                lnl: lnl[j],
                status: status[j],
            });
        }
    });
}

/// Python `_note_exposure_stats`: the kernel rows reported since the last note belong to the evaluation whose full
/// draw indices are `selected` (row order = kernel row order).
pub(crate) fn note(slot: &'static str, selected: &[usize]) {
    with_state(|s| {
        let rows = std::mem::take(&mut s.rows);
        if rows.len() != selected.len() {
            fail(s, format!("gate note {slot}: {} kernel rows for {} selected draws", rows.len(), selected.len()));
        }
        s.notes.retain(|n| n.slot != slot);
        s.notes.push(Note { slot, selected: selected.to_vec(), rows });
    });
}

/// Stage the full lnL array (before the prior) of the reduction that follows; attaches the evaluation noted under
/// `slot`, if any.
pub(crate) fn stage(slot: &'static str, lnl_full: &[f64]) {
    with_state(|s| {
        let n = lnl_full.len();
        let note = s
            .notes
            .iter()
            .position(|x| x.slot == slot)
            .map(|i| s.notes.remove(i));
        let (state, contacts) = match note {
            Some(note) => {
                let mut state = vec![-1i8; n];
                if note.rows.len() != note.selected.len() {
                    fail(s, format!("gate stage {slot}: rows and selection differ"));
                    return;
                }
                for (j, &i) in note.selected.iter().enumerate() {
                    let r = note.rows[j];
                    if i >= n || r.lnl.to_bits() != lnl_full[i].to_bits() {
                        fail(s, format!("gate stage {slot}: batch lnL does not land at draw {i}"));
                        return;
                    }
                    state[i] = r.status as i8;
                }
                let floor = note
                    .rows
                    .iter()
                    .map(|r| r.lnl)
                    .filter(|x| x.is_finite())
                    .fold(f64::NEG_INFINITY, f64::max)
                    - s.window;
                let mut keep: Vec<(i64, [f64; 5])> = Vec::new();
                // window = inf (gate protocol 2026-10-25, generator 853609f): EVERY evaluated draw, finite lnL or not
                let every = s.window.is_infinite();
                if every || floor.is_finite() {
                    for (j, &i) in note.selected.iter().enumerate() {
                        let r = note.rows[j];
                        if every || (r.lnl.is_finite() && r.lnl >= floor) {
                            keep.push((
                                i as i64,
                                super::contact::contact_times(r.period, r.a_rs, r.inc_rad, r.ecc, r.w_rad, r.k),
                            ));
                        }
                    }
                }
                keep.sort_by_key(|x| x.0); // stable, as Python's argsort(kind="stable")
                let idx: Vec<i64> = keep.iter().map(|x| x.0).collect();
                let roots: Vec<f64> = keep.iter().flat_map(|x| x.1).collect();
                (Some(state), Some((idx, roots)))
            }
            None if s.production => (Some(vec![-1i8; n]), Some((Vec::new(), Vec::new()))),
            None => (None, None),
        };
        s.staged = Some(GateRecord {
            lnl: lnl_full.to_vec(),
            state,
            contacts,
        });
    });
}

/// Called by every RECORDED reduction (in lockstep with the step-7 sink): appends the staged payload (or none).
pub(crate) fn push_reduction() {
    with_state(|s| {
        let rec = s.staged.take();
        s.records.push(rec);
    });
}

/// Start of a Monte-Carlo pass: drop the previous pass's records (they are aligned with a fresh sink).
pub(crate) fn begin_pass() {
    with_state(|s| {
        s.records.clear();
        s.named.clear();
        s.rows.clear();
        s.notes.clear();
        s.staged = None;
    });
}

/// `calc_v3::take_records`: the records since `n0` belong to `names` (same rule as the draw-weight diagnostics).
pub(crate) fn assign(n0: usize, names: &[String]) {
    with_state(|s| {
        let got: Vec<Option<GateRecord>> = if n0 <= s.records.len() {
            s.records[n0..].iter_mut().map(Option::take).collect()
        } else {
            Vec::new()
        };
        if !got.is_empty() && got.len() != names.len() {
            fail(s, format!("gate: {} records for {} scenarios", got.len(), names.len()));
        }
        let mut got = got.into_iter();
        for name in names {
            s.named.insert(name.clone(), got.next().flatten());
        }
    });
}

/// End of a pass: the per-scenario records of this pass (scenarios without a reduction are absent).
pub(crate) fn take_named() -> HashMap<String, Option<GateRecord>> {
    let mut out = HashMap::new();
    with_state(|s| out = std::mem::take(&mut s.named));
    out
}

