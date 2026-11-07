//! Step 7, decision C: the MC-noise policy of `worker/fpp_native.py` section 9b
//! (fpp-integration-r2 03a4bcc, `:4955-5177`), ported literally.
//!
//! * the per-scenario draw-weight diagnostics sink of `_lnZ_from_logw` ([`DrawDiag`], [`DiagSink`]);
//! * `stage_seed`, `_mc_stage_ladder`, `_delta_se`, `_mc_heavy_scenarios`, `_mc_summarise`, `_mc_assess`,
//!   `_mc_futile`, `_mc_stage_record`, `_mc_run_stages` and `mc_validated_3sigma`.
//!
//! Operation order is the Python order: numpy array `** 2` is `x * x`, every `np.sum` is the numpy pairwise
//! sum (also over the masked subsets), `MC_TAIL_SIGMAS * MC_EV_SD` is formed before it multiplies `beta`, and
//! `np.clip` follows the subtraction.  Nothing here reads a clock, the environment or the host
//! (`decision_functions_reference_no_clock` checks the source).

use crate::fpp::npsum::pairwise_sum;
use crate::fpp_policy::{CANDIDATE_FPP, VALIDATED_FPP, VALIDATED_NFPP};
use std::cell::RefCell;
use std::collections::HashMap;

// ------------------------------------------------------------------ constants (verbatim, `:4967-4991`)
pub const MC_SIGMA: f64 = 3.0;
pub const MC_GATES_FPP: [f64; 2] = [VALIDATED_FPP, CANDIDATE_FPP];
pub const MC_GATE_NFPP: f64 = VALIDATED_NFPP;
pub const MC_STAGE_FACTORS: [usize; 3] = [1, 10, 100];
pub const MC_N_CAP: usize = 1_000_000;
pub const MC_MIN_REFINE_N: usize = 100_000;
pub const MC_ESS_MIN: f64 = 10.0;
pub const MC_HEAVY_P_MIN: f64 = 1e-4;
pub const MC_TAIL_K_TOP: usize = 10;
pub const MC_TAIL_SIGMAS: f64 = 3.0;
pub const MC_ABS_TOL: f64 = 1e-9;
pub const MC_FUTILE_ESS_DOM: f64 = 25.0;

/// `MC_EV_SD = math.pi / math.sqrt(6.0)` (sqrt and `/` are correctly rounded, so this is portable).
pub fn mc_ev_sd() -> f64 {
    std::f64::consts::PI / 6.0f64.sqrt()
}

/// `f"{g:g}"` of the three gates, as Python formats them in the reason strings.
fn gate_text(g: f64) -> &'static str {
    if g.to_bits() == VALIDATED_FPP.to_bits() {
        "0.015"
    } else if g.to_bits() == CANDIDATE_FPP.to_bits() {
        "0.5"
    } else if g.to_bits() == VALIDATED_NFPP.to_bits() {
        "0.001"
    } else {
        unreachable!("only the three registered gates are formatted")
    }
}

// ------------------------------------------------------------------ draw-weight diagnostics sink

/// One `_DRAW_DIAG` record: ESS, number of draws, share of the heaviest draw, extreme-value tail scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawDiag {
    pub ess: f64,
    pub n: usize,
    pub max_share: f64,
    pub tail_scale: f64,
}

thread_local! {
    static SINK: RefCell<Option<Vec<DrawDiag>>> = const { RefCell::new(None) };
}

/// RAII guard of the per-pass diagnostics sink (`_DRAW_DIAG = sink` ... `finally: _DRAW_DIAG = None`).
/// While it lives, every `lnz_from_logw` on this thread appends one record.
pub struct DiagSink {
    _private: (),
}

impl DiagSink {
    pub fn start() -> Self {
        SINK.with(|s| *s.borrow_mut() = Some(Vec::new()));
        DiagSink { _private: () }
    }

    /// Number of records so far.
    pub fn len(&self) -> usize {
        SINK.with(|s| s.borrow().as_ref().map_or(0, Vec::len))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The records appended since index `n0`.
    pub fn since(&self, n0: usize) -> Vec<DrawDiag> {
        SINK.with(|s| s.borrow().as_ref().map_or_else(Vec::new, |v| v[n0..].to_vec()))
    }
}

impl Drop for DiagSink {
    fn drop(&mut self) {
        SINK.with(|s| *s.borrow_mut() = None);
    }
}

pub(crate) fn sink_active() -> bool {
    SINK.with(|s| s.borrow().is_some())
}

pub(crate) fn sink_push(d: DrawDiag) {
    SINK.with(|s| {
        if let Some(v) = s.borrow_mut().as_mut() {
            v.push(d);
        }
    });
}

/// `_tail_scale(w)`: (max - min of the top k finite log-weights) / H_(k-1), k = min(MC_TAIL_K_TOP, #finite);
/// NaN below 3 finite draws.  `h = sum(1.0 / i for i in range(1, kk))` is a left-to-right Python sum.
pub fn tail_scale(w: &[f64]) -> f64 {
    let mut fin: Vec<f64> = w.iter().copied().filter(|x| x.is_finite()).collect();
    let kk = MC_TAIL_K_TOP.min(fin.len());
    if kk < 3 {
        return f64::NAN;
    }
    // only the max and the kk-th largest of the top kk are used, so any selection order gives the same bits
    fin.sort_by(|a, b| b.total_cmp(a));
    let top_max = fin[0];
    let top_min = fin[kk - 1];
    (top_max - top_min) / cpython_float_sum((1..kk).map(|i| 1.0 / i as f64))
}

/// CPython >= 3.12 built-in `sum()` of floats (the reference venv is 3.12.13): Neumaier-compensated, the compensation
/// added at the end when it is non-zero and finite (`Python/bltinmodule.c`, `builtin_sum_impl`). A plain left-to-right
/// sum differs in the last bits (2 ULP on the tail scale of the E2E fixtures).
pub fn cpython_float_sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let (mut f, mut c) = (0.0f64, 0.0f64);
    for x in items {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

// ------------------------------------------------------------------ seeds and ladder

/// `stage_seed(tic_id, period, t0, stage, seed0)`: FNV-1a 64 of
/// `f"{tic_id}:{period:.10e}:{t0:.10e}:stage{stage}:{seed0}"`, masked to 32 bits.  `seed0 = None` uses `tic_seed`.
pub fn stage_seed_key(tic_id: &str, period: f64, t0: f64, stage: usize, seed0: Option<u64>) -> String {
    assert!(stage >= 1, "stage_seed is for refinement stages >= 1; stage 0 uses tic_seed()");
    let seed0 = seed0.unwrap_or_else(|| u64::from(crate::fpp::rng::tic_seed(tic_id, period, t0)));
    format!("{}:stage{stage}:{seed0}", crate::fpp::rng::tic_seed_key(tic_id, period, t0))
}

pub fn stage_seed(tic_id: &str, period: f64, t0: f64, stage: usize, seed0: Option<u64>) -> u32 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for byte in stage_seed_key(tic_id, period, t0, stage, seed0).bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (h & 0xFFFF_FFFF) as u32
}

/// `_mc_stage_ladder(N, refine, n_cap)`.
pub fn stage_ladder(n: usize, refine: bool, n_cap: Option<usize>) -> Vec<usize> {
    let cap = n_cap.unwrap_or(MC_N_CAP);
    if !refine || n < MC_MIN_REFINE_N {
        return vec![n];
    }
    let mut ladder = vec![n];
    for &f in &MC_STAGE_FACTORS[1..] {
        if n * f <= cap {
            ladder.push(n * f);
        }
    }
    ladder
}

// ------------------------------------------------------------------ summary

/// `_mc_summarise` output.
#[derive(Clone, Debug, PartialEq)]
pub struct McSummary {
    pub names: Vec<String>,
    pub probs: Vec<f64>,
    pub fpp: f64,
    pub nfpp: f64,
    pub fpp_se: f64,
    pub nfpp_se: f64,
    pub heavy: Vec<String>,
    pub min_ess: f64,
    pub min_ess_scenario: String,
    pub n: usize,
    pub ess: Vec<f64>,
    pub rel_var: Vec<f64>,
    pub tail_box_fpp: (f64, f64),
    pub tail_box_nfpp: (f64, f64),
    pub n_low_ess: usize,
    pub tail_scale: Vec<f64>,
}

fn np_sum_masked(values: &[f64], mask: &[bool]) -> f64 {
    let sub: Vec<f64> = values.iter().zip(mask).filter(|(_, &m)| m).map(|(&v, _)| v).collect();
    pairwise_sum(&sub)
}

/// `np.maximum(a, b)`: `a` if `a >= b` or `a` is NaN, else `b`.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// `_delta_se(probs, in_set, rel_var)`.
pub fn delta_se(probs: &[f64], in_set: &[bool], rel_var: &[f64]) -> f64 {
    let f = np_sum_masked(probs, in_set);
    let terms: Vec<f64> = (0..probs.len())
        .map(|i| {
            let ind = if in_set[i] { 1.0 } else { 0.0 };
            let d = ind - f;
            ((d * d) * (probs[i] * probs[i])) * rel_var[i]
        })
        .collect();
    pairwise_sum(&terms).sqrt()
}

/// `_mc_heavy_scenarios(names, probs, ess)`: `(not e >= MC_ESS_MIN) and p >= MC_HEAVY_P_MIN`.
pub fn heavy_scenarios(names: &[String], probs: &[f64], ess: &[f64]) -> Vec<String> {
    names
        .iter()
        .zip(probs.iter().zip(ess))
        .filter(|(_, (&p, &e))| !(e >= MC_ESS_MIN) && p >= MC_HEAVY_P_MIN)
        .map(|(n, _)| n.clone())
        .collect()
}

fn fractions(lnz: &[f64], false_set: &[bool], nearby_set: &[bool]) -> (f64, f64) {
    let m = lnz
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    let q: Vec<f64> = lnz.iter().map(|&x| (x - m).exp()).collect();
    let total = pairwise_sum(&q);
    let q: Vec<f64> = q.iter().map(|&x| x / total).collect();
    (np_sum_masked(&q, false_set), np_sum_masked(&q, nearby_set))
}

/// `_mc_summarise(lnZ, nearby_scen, diag, N)`. `lnz` and `nearby` are in dict insertion order; `None` when no
/// lnZ is finite or the probability total is not positive.
pub fn summarise(
    lnz: &[(String, f64)],
    nearby: &[(String, f64)],
    diag: &HashMap<String, DrawDiag>,
    n: usize,
) -> Option<McSummary> {
    // all_scen = dict(lnZ, **nearby_scen)
    let mut names: Vec<String> = Vec::with_capacity(lnz.len() + nearby.len());
    let mut values: Vec<f64> = Vec::with_capacity(lnz.len() + nearby.len());
    for (name, v) in lnz.iter().chain(nearby) {
        if let Some(i) = names.iter().position(|x| x == name) {
            values[i] = *v;
        } else {
            names.push(name.clone());
            values.push(*v);
        }
    }
    let n_target = lnz.len();
    let all_lnz: Vec<f64> = values
        .iter()
        .map(|&x| if x.is_finite() { x } else { f64::NEG_INFINITY })
        .collect();
    if !all_lnz.iter().any(|&x| x > f64::NEG_INFINITY) {
        return None;
    }
    let max_lnz = all_lnz
        .iter()
        .copied()
        .filter(|&x| x > f64::NEG_INFINITY)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = all_lnz.iter().map(|&x| (x - max_lnz).exp()).collect();
    let total = pairwise_sum(&probs);
    if total <= 0.0 {
        return None;
    }
    for p in &mut probs {
        *p /= total;
    }
    let target_prob = |key: &str| -> f64 {
        names[..n_target]
            .iter()
            .position(|x| x == key)
            .map_or(0.0, |i| probs[i])
    };
    let fpp = 1.0 - target_prob("TP") - target_prob("PTP") - target_prob("DTP");
    let nfpp = pairwise_sum(&probs[n_target..]);
    let fpp = fpp.clamp(0.0, 1.0);
    let nfpp = nfpp.clamp(0.0, 1.0);

    let ess: Vec<f64> = names
        .iter()
        .zip(&values)
        .map(|(name, &v)| match diag.get(name) {
            Some(d) => d.ess,
            None => {
                if v.is_finite() {
                    f64::NAN
                } else {
                    0.0
                }
            }
        })
        .collect();
    let n_draws: Vec<f64> = names
        .iter()
        .map(|name| diag.get(name).map_or(n as f64, |d| d.n as f64))
        .collect();
    let rel_var: Vec<f64> = (0..names.len())
        .map(|i| {
            let rv = if ess[i] > 0.0 {
                np_maximum(1.0 / ess[i] - 1.0 / n_draws[i], 0.0)
            } else {
                0.0
            };
            if probs[i] > 0.0 {
                rv
            } else {
                0.0
            }
        })
        .collect();
    let false_set: Vec<bool> = names
        .iter()
        .map(|n| !matches!(n.as_str(), "TP" | "PTP" | "DTP"))
        .collect();
    let nearby_set: Vec<bool> = (0..names.len()).map(|i| i >= n_target).collect();
    let fpp_se = delta_se(&probs, &false_set, &rel_var);
    let nfpp_se = delta_se(&probs, &nearby_set, &rel_var);

    let beta: Vec<f64> = names
        .iter()
        .map(|name| diag.get(name).map_or(f64::NAN, |d| d.tail_scale))
        .collect();
    let finite_lnz: Vec<bool> = all_lnz.iter().map(|x| x.is_finite()).collect();
    let low: Vec<bool> = (0..names.len())
        .map(|i| finite_lnz[i] && !(ess[i] >= MC_ESS_MIN))
        .collect();
    let known: Vec<bool> = (0..names.len()).map(|i| low[i] && beta[i].is_finite()).collect();
    let sel: Vec<usize> = (0..names.len()).filter(|&i| low[i] && !known[i]).collect();
    let heavy = heavy_scenarios(
        &sel.iter().map(|&i| names[i].clone()).collect::<Vec<_>>(),
        &sel.iter().map(|&i| probs[i]).collect::<Vec<_>>(),
        &sel.iter().map(|&i| ess[i]).collect::<Vec<_>>(),
    );
    let k = MC_TAIL_SIGMAS * mc_ev_sd();
    let half: Vec<f64> = (0..names.len())
        .map(|i| if known[i] { k * beta[i] } else { 0.0 })
        .collect();
    let up: Vec<f64> = (0..names.len())
        .map(|i| if false_set[i] { half[i] } else { -half[i] })
        .collect();
    let up_n: Vec<f64> = (0..names.len())
        .map(|i| if nearby_set[i] { half[i] } else { -half[i] })
        .collect();
    let shifted = |s: &[f64], plus: bool| -> Vec<f64> {
        all_lnz
            .iter()
            .zip(s)
            .map(|(&x, &d)| if plus { x + d } else { x - d })
            .collect()
    };
    let (fpp_hi, _) = fractions(&shifted(&up, true), &false_set, &nearby_set);
    let (fpp_lo, _) = fractions(&shifted(&up, false), &false_set, &nearby_set);
    let (_, nfpp_hi) = fractions(&shifted(&up_n, true), &false_set, &nearby_set);
    let (_, nfpp_lo) = fractions(&shifted(&up_n, false), &false_set, &nearby_set);
    let ess_live: Vec<f64> = (0..names.len())
        .map(|i| {
            if probs[i] > 0.0 && !ess[i].is_nan() {
                ess[i]
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let any_finite = ess_live.iter().any(|x| x.is_finite());
    let i_min = if any_finite {
        let mut best = 0usize;
        for i in 1..ess_live.len() {
            if ess_live[i] < ess_live[best] {
                best = i;
            }
        }
        best
    } else {
        0
    };
    Some(McSummary {
        min_ess: if any_finite { ess[i_min] } else { 0.0 },
        min_ess_scenario: names[i_min].clone(),
        names,
        probs,
        fpp,
        nfpp,
        fpp_se,
        nfpp_se,
        heavy,
        n,
        ess,
        rel_var,
        tail_box_fpp: (fpp_lo, fpp_hi),
        tail_box_nfpp: (nfpp_lo, nfpp_hi),
        n_low_ess: low.iter().filter(|&&x| x).count(),
        tail_scale: beta,
    })
}

// ------------------------------------------------------------------ assessment, futility, verdict rule

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McAssess {
    pub resolved: bool,
    pub reasons: Vec<String>,
}

/// `_mc_assess(summ, prev)`.
pub fn assess(summ: &McSummary, prev: Option<&McSummary>) -> McAssess {
    let mut reasons = Vec::new();
    for g in MC_GATES_FPP {
        if (summ.fpp - g).abs() < MC_SIGMA * summ.fpp_se {
            reasons.push(format!("fpp_gate_{}", gate_text(g)));
        }
    }
    if (summ.nfpp - MC_GATE_NFPP).abs() < MC_SIGMA * summ.nfpp_se {
        reasons.push(format!("nfpp_gate_{}", gate_text(MC_GATE_NFPP)));
    }
    for n in &summ.heavy {
        reasons.push(format!("heavy_tail:{n}"));
    }
    let (lo, hi) = summ.tail_box_fpp;
    for g in MC_GATES_FPP {
        if lo < g && g < hi {
            reasons.push(format!("tail_box_fpp_{}", gate_text(g)));
        }
    }
    let (lo, hi) = summ.tail_box_nfpp;
    if lo < MC_GATE_NFPP && MC_GATE_NFPP < hi {
        reasons.push(format!("tail_box_nfpp_{}", gate_text(MC_GATE_NFPP)));
    }
    if let Some(prev) = prev {
        for (key, now, before, se, prev_se) in [
            ("fpp", summ.fpp, prev.fpp, summ.fpp_se, prev.fpp_se),
            ("nfpp", summ.nfpp, prev.nfpp, summ.nfpp_se, prev.nfpp_se),
        ] {
            let tol = MC_SIGMA * se.hypot(prev_se) + MC_ABS_TOL;
            if (now - before).abs() > tol {
                reasons.push(format!("stage_disagreement_{key}"));
            }
        }
    }
    McAssess {
        resolved: reasons.is_empty(),
        reasons,
    }
}

/// `(index, ESS)` of the dominant scenario of a stage summary: the first of maximal probability (names order).
/// The ESS is the Kish ESS stored in the summary (NaN if unknown). Shared by futility and `fpp_unstable`.
pub fn dominant_ess(summ: &McSummary) -> (usize, f64) {
    let mut i = 0usize;
    for j in 1..summ.probs.len() {
        if summ.probs[j] > summ.probs[i] {
            i = j;
        }
    }
    (i, summ.ess[i])
}

/// `(name, ESS)` of the dominant scenario.
pub fn dominant(summ: &McSummary) -> (&str, f64) {
    let (i, ess) = dominant_ess(summ);
    (summ.names[i].as_str(), ess)
}

/// `mc_is_unstable(summ)`: dominant scenario Kish ESS < 25.0 (strict; NaN fails closed as unstable).
pub fn is_unstable(summ: &McSummary) -> bool {
    let (_, ess) = dominant_ess(summ);
    !(ess >= MC_FUTILE_ESS_DOM)
}

/// `_mc_futile(summ)`: the first scenario of maximal probability has `not ess >= MC_FUTILE_ESS_DOM`.
pub fn futile(summ: &McSummary) -> bool {
    is_unstable(summ)
}

/// `mc_validated_3sigma(fpp, fpp_se, nfpp, nfpp_se)`.
pub fn validated_3sigma(fpp: f64, fpp_se: f64, nfpp: f64, nfpp_se: f64) -> bool {
    fpp + MC_SIGMA * fpp_se < VALIDATED_FPP && nfpp + MC_SIGMA * nfpp_se < VALIDATED_NFPP
}

// ------------------------------------------------------------------ stage record and stage loop

#[derive(Clone, Debug, PartialEq)]
pub struct TopScenario {
    pub scenario: String,
    pub p: f64,
    pub ess: f64,
    pub rel_se: f64,
}

/// `_mc_stage_record` (one entry of `refinement_stages`).
#[derive(Clone, Debug, PartialEq)]
pub struct StageRecord {
    pub stage: usize,
    pub n: usize,
    pub seed: u64,
    pub fpp: f64,
    pub nfpp: f64,
    pub fpp_mc_se: f64,
    pub nfpp_mc_se: f64,
    pub min_ess: f64,
    pub min_ess_scenario: String,
    pub heavy_scenarios: Vec<String>,
    pub n_low_ess: usize,
    pub tail_box_fpp: [f64; 2],
    pub tail_box_nfpp: [f64; 2],
    pub resolved: bool,
    pub reasons: Vec<String>,
    pub top_scenarios: Vec<TopScenario>,
}

pub fn stage_record(stage: usize, n: usize, seed: u64, summ: &McSummary, assess: &McAssess) -> StageRecord {
    let neg: Vec<f64> = summ.probs.iter().map(|&p| -p).collect();
    let mut order: Vec<usize> = (0..neg.len()).collect();
    // np.argsort(-probs, kind="stable"): a stable ascending sort of -p (no NaN in probs)
    order.sort_by(|&a, &b| neg[a].partial_cmp(&neg[b]).expect("probabilities are not NaN"));
    let top = order
        .into_iter()
        .take(5)
        .map(|i| TopScenario {
            scenario: summ.names[i].clone(),
            p: summ.probs[i],
            ess: summ.ess[i],
            rel_se: summ.rel_var[i].sqrt(),
        })
        .collect();
    StageRecord {
        stage,
        n,
        seed,
        fpp: summ.fpp,
        nfpp: summ.nfpp,
        fpp_mc_se: summ.fpp_se,
        nfpp_mc_se: summ.nfpp_se,
        min_ess: summ.min_ess,
        min_ess_scenario: summ.min_ess_scenario.clone(),
        heavy_scenarios: summ.heavy.clone(),
        n_low_ess: summ.n_low_ess,
        tail_box_fpp: [summ.tail_box_fpp.0, summ.tail_box_fpp.1],
        tail_box_nfpp: [summ.tail_box_nfpp.0, summ.tail_box_nfpp.1],
        resolved: assess.resolved,
        reasons: assess.reasons.clone(),
        top_scenarios: top,
    }
}

/// One Monte-Carlo pass as the stage loop sees it: `(lnZ, nearby_scen, diag)` plus the caller's payload.
pub struct StagePass<P> {
    pub lnz: Vec<(String, f64)>,
    pub nearby: Vec<(String, f64)>,
    pub diag: HashMap<String, DrawDiag>,
    pub payload: P,
}

/// The LAST stage run by [`run_stages`].
pub struct StagedOutcome<P> {
    pub pass: StagePass<P>,
    pub summary: McSummary,
    pub assess: McAssess,
    pub log: Vec<StageRecord>,
}

/// `_mc_run_stages(pass_fn, ladder, seed_for_stage, futility)`.  `Ok(None)` when a pass has no finite
/// evidence (Python returns None); an error of `pass_fn` propagates unchanged.
pub fn run_stages<P, E>(
    mut pass_fn: impl FnMut(usize, u64) -> Result<StagePass<P>, E>,
    ladder: &[usize],
    seed_for_stage: impl Fn(usize) -> u64,
    futility: bool,
) -> Result<Option<StagedOutcome<P>>, E> {
    let mut log = Vec::new();
    let mut prev: Option<McSummary> = None;
    let mut last: Option<(StagePass<P>, McSummary, McAssess)> = None;
    for (idx, &n_k) in ladder.iter().enumerate() {
        let seed_k = seed_for_stage(idx);
        let out = pass_fn(n_k, seed_k)?;
        let Some(summ) = summarise(&out.lnz, &out.nearby, &out.diag, n_k) else {
            return Ok(None);
        };
        let mut a = assess(&summ, prev.as_ref());
        let is_futile = futility && idx == 0 && !a.resolved && idx + 1 < ladder.len() && futile(&summ);
        if is_futile {
            a.reasons.push("futility_stop".into());
        }
        log.push(stage_record(idx, n_k, seed_k, &summ, &a));
        let stop = a.resolved || is_futile;
        prev = Some(summ.clone());
        last = Some((out, summ, a));
        if stop {
            break;
        }
    }
    let (pass, summary, assess) = last.expect("a ladder has at least one stage");
    Ok(Some(StagedOutcome {
        pass,
        summary,
        assess,
        log,
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn decision_functions_reference_no_clock() {
        // Source-level guard (mirror of Python test_decision_functions_reference_no_clock): nothing the stage
        // logic is made of may read a clock, the environment or the host.
        let src = include_str!("mc.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        for banned in [
            "std::time", "Instant", "SystemTime", "std::env", "env::var", "available_parallelism",
            "num_cpus", "thread::sleep", "getloadavg", "hostname",
        ] {
            assert!(!body.contains(banned), "mc.rs decision code references {banned}");
        }
    }
}
