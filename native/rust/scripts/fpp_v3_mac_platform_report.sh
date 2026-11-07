#!/bin/bash
set +e

cd ~/fpp-compute-rust/ext-fpp-v3-rust-stage-9cffe4e || exit 2
report_dir=/tmp/ext-fpp-v3-rust-platform-report
mkdir -p "$report_dir"

run_case() {
    profile=$1
    label=$2
    target=$3
    test_name=$4
    report_mode=$5
    log="$report_dir/${profile}-${label}.log"

    if [ "$profile" = release ]; then
        if [ "$report_mode" = 1 ]; then
            FPP_PLATFORM_REPORT=1 nice -n 15 cargo test --release -p exonit-core -j2 --test "$target" "$test_name" -- --exact --nocapture >"$log" 2>&1
        else
            nice -n 15 cargo test --release -p exonit-core -j2 --test "$target" "$test_name" -- --exact --nocapture >"$log" 2>&1
        fi
    else
        if [ "$report_mode" = 1 ]; then
            FPP_PLATFORM_REPORT=1 nice -n 15 cargo test -p exonit-core -j2 --test "$target" "$test_name" -- --exact --nocapture >"$log" 2>&1
        else
            nice -n 15 cargo test -p exonit-core -j2 --test "$target" "$test_name" -- --exact --nocapture >"$log" 2>&1
        fi
    fi
    status=$?
    printf '%s\n' "$status" >"$log.exit"
}

run_profile() {
    profile=$1
    run_case "$profile" e2e fpp_calc_v3_e2e corrected_calc_matches_all_python_parity_and_production_cases 0
    run_case "$profile" kepler fpp_kepler_v3_golden platform_report_kepler_aggregate 0
    run_case "$profile" geometry fpp_geometry_v3_golden platform_report_geometry_aggregate 0
    run_case "$profile" d_family fpp_d_family_e2e_golden platform_report_d_family_aggregate 1
    run_case "$profile" s_lnz fpp_v2_s_family_golden s_family_lnz_bit_exact 1
    run_case "$profile" s_per_draw fpp_v2_s_family_golden s_family_per_draw_bit_exact 1
    run_case "$profile" beb_per_draw fpp_d_reducer_probe production_beb_twin_reduction_exposes_python_numpy_intermediates 1
    run_case "$profile" nearby_per_draw fpp_nearby_neb_probe_v3 nearby_neb_production_trace_matches_python_at_raw_mt_position_168 1
}

run_profile debug
run_profile release

for profile in debug release; do
    binary_dir="target/$profile/deps"
    nm_log="$report_dir/${profile}-calc-imports.log"
    : >"$nm_log"
    for binary in "$binary_dir"/fpp_calc_v3_e2e-*; do
        if [ -x "$binary" ]; then
            shasum -a 256 "$binary" >>"$nm_log"
            nm -u "$binary" | grep -E 'sincos|_sin|_cos|_pow' >>"$nm_log"
        fi
    done
    printf '%s\n' "$?" >"$nm_log.exit"
done

printf 'report_dir=%s\n' "$report_dir"
