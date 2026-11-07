#!/usr/bin/env bash
# R1 (task 30): Debug-versus-Release identity of the PRODUCTION FPP results on
# the host it runs on (meant for macOS / Windows, where the libm differs from
# glibc and the compiler may rewrite libm calls per optimisation level).
#
# Builds and runs, in Debug and in Release, the tests that print serialized
# snapshots of production outputs (u64 bits or their sha256):
#   fpp_profile_snapshot     population_v3, stellar relations, background LDC
#   fpp_calc_v3_e2e          the full corrected calc, 8 cases (FPPV3-SNAPSHOT)
#   fpp_v2_s_family_golden   S-family helpers, lnZ and full per-draw outputs
#   fpp_d_family_e2e_golden  D/B-family lnZ
#   fpp_step7_e2e, fpp_mass_fallback_v3 (task 33) the step-7 MC fields of the production ladder and of the
#                            mass-fallback cases (FPPV3-MC-SNAPSHOT, every float by its bits); the 8 corrected E2E
#                            cases also print one FPPV3-MC-SNAPSHOT line each
# then requires the extracted snapshot lines to be byte-identical between the
# two profiles. Test pass/fail is recorded separately (.exit files): this
# script judges only the cross-profile identity.
#
# usage: nice -n 15 env CARGO_BUILD_JOBS=2 scripts/fpp_profile_identity.sh <log-dir>
set -u
out=${1:?usage: fpp_profile_identity.sh <log-dir>}
mkdir -p "$out"
cd "$(dirname "$0")/.." || exit 2
expected_min=51  # 10 profile sections + 8 E2E cases + 4 S tests + 1 D test + task 33: 8 E2E MC + 6 step-7 + 14 mass results
for profile in debug release; do
  flag=""
  [ "$profile" = release ] && flag="--release"
  log="$out/profile-$profile.log"
  cargo test -p exonit-core $flag --no-fail-fast \
    --test fpp_profile_snapshot --test fpp_calc_v3_e2e \
    --test fpp_v2_s_family_golden --test fpp_d_family_e2e_golden \
    --test fpp_step7_e2e --test fpp_mass_fallback_v3 \
    -- --nocapture --test-threads=1 > "$log" 2>&1
  echo $? > "$out/profile-$profile.exit"
  # --nocapture can put a marker after the harness's "test name ... " prefix:
  # keep each line from its marker on.
  grep -oE '(FPP-PROFILE-SNAPSHOT |FPPV3-SNAPSHOT |FPPV3-MC-SNAPSHOT |FPP-PLATFORM [A-Za-z0-9_-]+: .*output_sha256_le=).*' "$log" \
    | sort > "$out/profile-$profile.snapshot"
done
n_debug=$(wc -l < "$out/profile-debug.snapshot")
n_release=$(wc -l < "$out/profile-release.snapshot")
sha_debug=$(shasum -a 256 "$out/profile-debug.snapshot" 2>/dev/null || sha256sum "$out/profile-debug.snapshot")
sha_release=$(shasum -a 256 "$out/profile-release.snapshot" 2>/dev/null || sha256sum "$out/profile-release.snapshot")
echo "debug:   lines=$n_debug exit=$(cat "$out/profile-debug.exit") ${sha_debug%% *}"
echo "release: lines=$n_release exit=$(cat "$out/profile-release.exit") ${sha_release%% *}"
if [ "$n_debug" -lt "$expected_min" ] || [ "$n_release" -lt "$expected_min" ]; then
  echo "FPP-PROFILE-IDENTITY FAIL: expected at least $expected_min snapshot lines per profile"
  exit 1
fi
if cmp -s "$out/profile-debug.snapshot" "$out/profile-release.snapshot"; then
  echo "FPP-PROFILE-IDENTITY PASS: Debug and Release snapshots are byte-identical ($n_debug lines)"
  exit 0
fi
echo "FPP-PROFILE-IDENTITY FAIL: Debug and Release snapshots differ"
diff "$out/profile-debug.snapshot" "$out/profile-release.snapshot" | head -40
exit 1
