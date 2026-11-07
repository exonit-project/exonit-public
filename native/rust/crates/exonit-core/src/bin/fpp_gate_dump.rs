//! v3 FPP gate DUMP driver (Rust port): reads one staged gate-input case (`scripts/fpp/v3_gate_format.py`, schema
//! `fpp-v3-gate-input/1`), runs `calc_fpp_by_source_v3` exactly as the Python reference generator
//! (`scripts/fpp/gen_v3_gate_dumps.py`) runs `calc_fpp_by_source`: production switch set (`FPP_EPHEM_REFINE` off in
//! this port), fixed N (`adaptive_n = False`: one Monte-Carlo stage), the staged per-candidate seed, the staged star
//! list (no TIC query), and writes the dump directory `<out>/<tic>/<switch_set>/N<N>/` in the generator's format:
//! per scenario the per-draw lnL of all N draws, the int8 exposure-order state, the contact roots of the
//! weight-carrying production draws (window 41 nat over each evaluation), and the run / reported FPP and NFPP.
//!
//! BLIND RULE: a case whose `set` is not `dev` is refused unless `--gate-run` is given (the pre-registered gate run).
//!
//!   fpp_gate_dump --case STAGE/cases/<tic>.json --out DUMPS --N 100000 [--switch-set production|parity]
//!                 [--contacts-window 41] [--gate-run]
//! Exit: 0 = dump written (any status), 2 = usage / unreadable input, 3 = observer inconsistency.

use exonit_core::fpp::error::FppError;
use exonit_core::fpp::provenance;
use exonit_core::fpp::v2::calc_v3::{calc_fpp_by_source_v3_with, CalcV3Inputs, CalcV3Result, McOptions};
use exonit_core::fpp::v2::gate::{GateRecord, GateRecorder};
use serde_json::{json, Map, Number, Value};
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "fpp-v3-gate-input/1";
const DUMP_SCHEMA: &str = "fpp-v3-gate-dump/1";

fn hx(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn unhx(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().expect("hex float"), 16).expect("hex float"))
}

fn is_hex16(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = hmac_sha256::Hash::new();
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A staged star dict -> the nearby-star record the port parses (hex floats decoded; NaN = JSON null).
fn decode_star(s: &Value) -> Value {
    let mut out = Map::new();
    for (k, v) in s.as_object().expect("star dict") {
        let decoded = match v {
            Value::String(t) if k != "ID" && is_hex16(t) => {
                let x = f64::from_bits(u64::from_str_radix(t, 16).unwrap());
                Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
            }
            other => other.clone(),
        };
        out.insert(k.clone(), decoded);
    }
    Value::Object(out)
}

struct Writer {
    dir: PathBuf,
}

impl Writer {
    fn array(&self, rel: &str, raw: Vec<u8>, n: usize) -> Value {
        let path = self.dir.join("bin").join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir bin");
        std::fs::write(&path, &raw).expect("write array");
        json!({"file": format!("bin/{rel}"), "sha256": sha256_hex(&raw), "n": n})
    }
    fn f64s(&self, rel: &str, a: &[f64]) -> Value {
        self.array(rel, a.iter().flat_map(|x| x.to_le_bytes()).collect(), a.len())
    }
    fn i8s(&self, rel: &str, a: &[i8]) -> Value {
        self.array(rel, a.iter().map(|&x| x as u8).collect(), a.len())
    }
    fn i64s(&self, rel: &str, a: &[i64]) -> Value {
        self.array(rel, a.iter().flat_map(|x| x.to_le_bytes()).collect(), a.len())
    }
}

/// Python's result dict holds a non-finite lnZ as None (JSON null): `-inf` (never reached / zero viability) and NaN.
fn window_json(window: f64) -> Value {
    if window.is_infinite() { json!("inf") } else { json!(window) }
}

fn scenario_entry(w: &Writer, base: &str, name: &str, lnz: f64, prob: f64, rec: &Option<GateRecord>, window: f64) -> Value {
    let mut e = json!({"name": name, "lnZ": if !lnz.is_finite() { Value::Null } else { Value::String(hx(lnz)) },
                       "prob": hx(prob), "lnl": null, "state": null, "contacts": null});
    if let Some(r) = rec {
        e["lnl"] = w.f64s(&format!("{base}.lnl.f64"), &r.lnl);
        if let Some(st) = &r.state {
            e["state"] = w.i8s(&format!("{base}.state.i8"), st);
        }
        if let Some((idx, roots)) = &r.contacts {
            if idx.is_empty() {
                e["contacts"] = json!({"idx_file": null, "roots_file": null, "n": 0, "window_nat": window_json(window)});
            } else {
                let ie = w.i64s(&format!("{base}.cidx.i64"), idx);
                let re = w.f64s(&format!("{base}.croots.f64"), roots);
                e["contacts"] = json!({"idx_file": ie, "roots_file": re, "n": idx.len(), "window_nat": window_json(window)});
            }
        }
    }
    e
}

fn result_json(r: &CalcV3Result) -> Value {
    let mut scen = Map::new();
    for run in &r.gate_runs {
        if run.dilution_corrected == r.dilution_corrected {
            for (name, z, p, _) in &run.scenarios {
                scen.insert(name.clone(), json!({"lnZ": if !z.is_finite() { Value::Null } else { Value::String(hx(*z)) },
                                                  "prob": hx(*p)}));
            }
        }
    }
    json!({"fpp": hx(r.fpp), "nfpp": hx(r.nfpp), "verdict": r.verdict, "fpp_flags": r.flags,
           "fpp_scen": scen, "dilution_corrected": r.dilution_corrected, "fpp_seed": r.seed,
           "fpp_mc_se": hx(r.fpp_mc_se), "nfpp_mc_se": hx(r.nfpp_mc_se), "mc_policy": r.mc_policy,
           "borderline": r.borderline, "borderline_reasons": r.borderline_reasons,
           "n_draws_final": r.n_draws_final, "n_draws_total": r.n_draws_total})
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(case_path), Some(out_root)) = (arg(&args, "--case"), arg(&args, "--out")) else {
        eprintln!("usage: fpp_gate_dump --case CASE.json --out DUMPS --N 100000 [--switch-set production] [--gate-run]");
        std::process::exit(2);
    };
    let n_draws: usize = arg(&args, "--N").map_or(100_000, |s| s.parse().expect("--N integer"));
    let switch_set = arg(&args, "--switch-set").unwrap_or_else(|| "production".into());
    // 41 = the binding 40 nat + 1 nat margin; "inf" = every evaluated draw (the held-out gate reference, 853609f)
    let window: f64 = arg(&args, "--contacts-window").map_or(41.0, |s| s.parse().expect("--contacts-window"));
    let gate_run = args.iter().any(|a| a == "--gate-run");
    let production = match switch_set.as_str() {
        "production" => true,
        "parity" => false,
        other => {
            eprintln!("unknown switch set {other}");
            std::process::exit(2);
        }
    };
    let raw = std::fs::read(&case_path).unwrap_or_else(|e| {
        eprintln!("{case_path}: {e}");
        std::process::exit(2)
    });
    let case: Value = serde_json::from_slice(&raw).expect("case JSON");
    if case["schema"] != INPUT_SCHEMA {
        eprintln!("{case_path}: not a gate input case");
        std::process::exit(2);
    }
    if case["set"] != "dev" && !gate_run {
        eprintln!("TIC {}: held-out case refused (blind rule; --gate-run is for the gate run only)", case["tic"]);
        std::process::exit(2);
    }
    if case["open"].as_array().is_some_and(|a| !a.is_empty()) {
        eprintln!("TIC {}: OPEN inputs {} - not run", case["tic"], case["open"]);
        std::process::exit(2);
    }
    let tic = case["tic"].as_u64().map(|t| t.to_string()).unwrap_or_else(|| case["tic"].as_str().unwrap().into());
    let inp = &case["inputs"];
    let time: Vec<f64> = inp["time"].as_array().unwrap().iter().map(unhx).collect();
    let flux: Vec<f64> = inp["flux"].as_array().unwrap().iter().map(unhx).collect();
    let stars: Vec<Value> = inp["stars"].as_array().unwrap().iter().map(decode_star).collect();
    let nearby = json!({"stars": stars, "gal_b": unhx(&case["gal_b"])});
    let lc_source = case["lc_source"].as_str().map(str::to_string);
    let cadence_s = if case["cadence_s"].is_null() { None } else { Some(unhx(&case["cadence_s"])) };
    let seed = case["seed"].as_u64().expect("seed") as u32;
    let population = if inp["population"].is_null() { None } else { Some(&inp["population"]) };
    let crosscheck = if inp["population_crosscheck"].is_null() { None } else { Some(&inp["population_crosscheck"]) };
    let geometry = if inp["geometry"].is_null() { None } else { Some(&inp["geometry"]) };
    let input = CalcV3Inputs {
        tic_id: &tic,
        period: unhx(&inp["period"]),
        t0: unhx(&inp["t0"]),
        duration: unhx(&inp["duration_d"]),
        depth_ppm: unhx(&inp["depth_ppm"]),
        time: &time,
        flux: &flux,
        nearby_raw: &nearby,
        geometry,
        population,
        population_sha256: inp["population_sha256"].as_str(),
        population_version: inp["population_version"].as_str(),
        population_crosscheck: crosscheck,
        cadence_s,
        lc_source: lc_source.as_deref(),
        tglc_near_edge: case["tglc_near_edge"].as_bool(),
        seed_ephemeris: None,
    };
    let dir = Path::new(&out_root).join(&tic).join(&switch_set).join(format!("N{n_draws}"));
    if dir.join("bin").exists() {
        std::fs::remove_dir_all(dir.join("bin")).expect("clear old bin");
    }
    std::fs::create_dir_all(&dir).expect("mkdir dump");
    let started = std::time::Instant::now();
    let recorder = GateRecorder::start(production, window);
    let outcome = calc_fpp_by_source_v3_with(
        &input,
        n_draws,
        seed,
        production,
        McOptions { adaptive_n: Some(false), n_cap: None, futility_stop: None, unstable_flag: None },
    );
    let observer_error = recorder.error();
    drop(recorder);
    let runtime = started.elapsed().as_secs_f64();
    if let Some(e) = observer_error {
        eprintln!("TIC {tic}: gate observer inconsistency: {e}");
        std::process::exit(3);
    }
    let w = Writer { dir: dir.clone() };
    let (status, skip_reason, result, gates, runs) = match &outcome {
        Ok(r) => {
            let mut runs = Vec::new();
            for (ri, run) in r.gate_runs.iter().enumerate() {
                let scen: Vec<Value> = run
                    .scenarios
                    .iter()
                    .enumerate()
                    .map(|(si, (name, z, p, rec))| scenario_entry(&w, &format!("run{ri}/s{si:03}"), name, *z, *p, rec, window))
                    .collect();
                runs.push(json!({"dilution_corrected": run.dilution_corrected, "fpp": hx(run.fpp),
                                 "nfpp": hx(run.nfpp), "scenarios": scen}));
            }
            let gates = json!({"fpp_lt_0.015": r.fpp < 0.015, "fpp_lt_0.5": r.fpp < 0.5, "nfpp_lt_0.001": r.nfpp < 0.001});
            ("ok", Value::Null, result_json(r), gates, runs)
        }
        Err(FppError::Unavailable { reason, .. }) => {
            ("unavailable", Value::String(reason.as_str().to_string()), Value::Null, Value::Null, Vec::new())
        }
        Err(e) => {
            let sr = e.skip_reason().map_or(Value::Null, |r| Value::String(r.as_str().to_string()));
            ("skipped", sr, Value::Null, Value::Null, Vec::new())
        }
    };
    let producer = json!({"impl": "rust", "git_sha": option_env!("FPP_GATE_GIT_SHA").unwrap_or("unknown"),
                          "platform": provenance::platform(), "libm": provenance::libm(),
                          "build_profile": provenance::build_profile(), "fpp_backend": "cpu-rust",
                          "error": match &outcome { Err(e) => Value::String(e.to_string()), Ok(_) => Value::Null }});
    let switches = exonit_core::task_processor::fpp_v3_switches(production);
    let dump = json!({
        "schema": DUMP_SCHEMA, "producer": producer,
        "case": {"tic": case["tic"], "toi": case["toi"], "set": case["set"], "input_sha256": sha256_hex(&raw)},
        "switch_set": switch_set, "production_profile": if production { json!("port-ephem-off") } else { Value::Null },
        "switches": switches, "N": n_draws, "seed": seed, "status": status, "skip_reason": skip_reason,
        "result": result, "gates": gates, "runs": runs});
    let text = serde_json::to_string(&dump).unwrap() + "\n";
    let tmp = dir.join("dump.json.tmp");
    std::fs::write(&tmp, text).expect("write dump");
    std::fs::rename(&tmp, dir.join("dump.json")).expect("rename dump");
    let meta = json!({"runtime_s": runtime, "impl": "rust", "build_profile": provenance::build_profile()});
    std::fs::write(dir.join("run_meta.json"), serde_json::to_string(&meta).unwrap() + "\n").expect("write meta");
    println!("TIC {tic} {switch_set} N={n_draws}: {status} {} {runtime:.1}s", skip_reason.as_str().unwrap_or(""));
}
