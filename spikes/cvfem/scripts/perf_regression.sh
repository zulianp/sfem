#!/usr/bin/env bash
# The performance-regression gate for this spike.
#
# What it measures is the PACKED MATRIX-FREE THROUGHPUT of cvfem_hex8_ns_upwind_bench --
# the residual and the Jacobian action at --layout packed, which reach >2000 MDOF/s on a
# Grace socket. That is deliberate and it is the point of this script: those kernels are
# the fast path, they are what the numbers in docs/ quote, and they are what the compiler
# flags in cmake/CVFEMCMakeFunctions.cmake were tuned on -- the load-bearing
# -fno-finite-math-only was justified by measuring exactly them. A slower driver is not a
# substitute. cvfem_ns_apply_bench, the semi-structured macro apply, saturates near
# 267 MDOF/s; agreement there says close to nothing about a kernel running nine times
# faster, because it is not bound by the same thing.
#
# TWO MODES, and the difference matters:
#
#   scripts/perf_regression.sh --against OLD_BIN     A/B, interleaved      <- the real gate
#   scripts/perf_regression.sh                       vs recorded baseline  <- coarse check
#   scripts/perf_regression.sh --record              write that baseline
#   scripts/perf_regression.sh --list                print the gated configurations
#
# Use --against for any change you actually want to clear. It builds nothing: point it at a
# binary built from the commit you are comparing to. Both binaries are measured in the same
# allocation with the order alternating every repetition, which is the only way to remove
# the node-to-node and run-order effects that otherwise dominate:
#
#   * Node to node. The same binary measured 2433 MDOF/s on nid006544 and 2566 on nid006545
#     for the packed sumfact residual at 8,586,756 dofs -- 5.5% -- and 2321 vs 2575 at
#     16,693,124 dofs, 11%. An absolute baseline therefore cannot be banded tightly, which
#     is why the baseline mode below is coarse and is not what clears a change.
#   * Run order. Running one binary consistently before the other made the second look
#     24-28% faster on assembly; an A-vs-A control showed a 12% pure position effect.
#     --against alternates, so position is balanced across repetitions.
#
# Interleaved, the two sides agree to within 1.8% across every configuration here, which is
# what the 5% band in --against mode is set against.
#
# Environment:
#   BIN        binary under test (default: the first build*/ that has it)
#   BASELINE   baseline csv (default: perf/baseline_<tag>.csv)
#   TAG        machine tag in the default baseline name (default: grace)
#   REPS       repetitions per configuration, median taken (default: 9)
#   THREADS    OMP threads (default: 72)
#   OUT        directory for the raw per-run csv (default: a mktemp dir)
#   CVFEM_COMMIT  commit recorded in a --record baseline when git is unavailable
#
# Exits 0 if every gated configuration is within band, 1 otherwise. A configuration that
# comes out FASTER than its band is reported but does not fail; investigate it rather than
# re-recording reflexively.

set -uo pipefail

# ---------------------------------------------------------------- the gated configurations
#
# key|operation|layout|kernel|cube_n|ab_band_pct|baseline_band_pct|extra_options
#
# ab_band_pct is the band for --against, where node effects cancel and the two sides agree
# to within 1.8% in practice.
#
# baseline_band_pct is the band for the coarse absolute mode, or "-" for a configuration
# that is RECORDED BUT NOT GATED there. Three are marked "-" and the reason is measurement,
# not indifference: their run-to-run spread on identical binaries is too wide for an
# absolute number to mean anything. `--assemble` returned spreads of 20-30% in one
# five-repetition run, and `--bsr-apply` returned 3.5% in one run and 68.7% in the next. A
# band wide enough to hold those (the naive formula produced 103%) gates nothing at all,
# and a narrow one fires every time; either way it trains people to ignore the gate. They
# still carry a recorded number, and --against still checks them, because interleaving in
# one allocation removes exactly the variation that makes them useless here.
#
# Two configurations are not measured at all. `--layout atomic --assemble` is bimodal
# rather than merely noisy -- identical binaries return either ~29 or ~38.5 MDOF/s, a 32%
# same-binary spread that looks like NUMA or pinning luck across invocations -- and
# `--layout packed --kernel sympy --assemble` shows the same bimodality less severely.
CONFIGS=(
    "residual_packed_sumfact|residual|packed|sumfact|128|5|12|"
    "residual_packed_sumfact_big|residual|packed|sumfact|160|5|12|"
    "residual_packed_sympy|residual|packed|sympy|128|5|12|"
    "residual_packed_current|residual|packed|current|128|5|12|"
    "jac_action_packed_sumfact|jac_action|packed|sumfact|128|5|12|"
    "jac_action_packed_sympy|jac_action|packed|sympy|128|5|12|"
    "jac_action_packed_current|jac_action|packed|current|128|5|12|"
    "residual_colored_sumfact|residual|colored|sumfact|128|5|12|"
    "bsr_apply_packed_sumfact|bsr_apply|packed|sumfact|128|8|-|"
    "assemble_store_sumfact|assemble|store|sumfact|128|10|-|"
    "assemble_colored_sumfact|assemble|colored|sumfact|128|10|-|"
    "assemble_packed_sumfact|assemble|packed|sumfact|128|10|-|"
    # The operator the SOLVER runs, as opposed to the element kernel in isolation. These
    # are the ones to watch when a change touches the Rhie-Chow term or the boundary
    # closure, and the four above them cannot see either. Measured on Grace at 8,586,756
    # dofs, they run at 63%, 53% and 28% of the bare kernel's 2579 MDOF/s.
    "residual_packed_rc|residual|packed|sumfact|128|5|12|--rhie-chow"
    "residual_packed_rc_bnd|residual|packed|sumfact|128|5|12|--rhie-chow --boundary"
    "residual_packed_rc_perapply|residual|packed|sumfact|128|5|12|--rhie-chow --pgrad-per-apply"
)

op_flag() {
    case "$1" in
        residual)   echo "" ;;
        jac_action) echo "--jac-action" ;;
        assemble)   echo "--assemble" ;;
        bsr_apply)  echo "--bsr-apply" ;;
        *) echo "unknown operation: $1" >&2; exit 2 ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SPIKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SPIKE_ROOT"

MODE=check
REF_BIN=""
case "${1:-}" in
    --record)  MODE=record ;;
    --against) MODE=against; REF_BIN="${2:?--against needs a path to the reference binary}" ;;
    --list)    printf '%s\n' "${CONFIGS[@]}" | cut -d'|' -f1-5 | column -t -s'|'; exit 0 ;;
    --help|-h) sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

TAG="${TAG:-grace}"
BASELINE="${BASELINE:-perf/baseline_${TAG}.csv}"
# 9, not 5: at five repetitions a single bad sample moved the recorded median for
# residual_packed_sympy from ~2050 to 1690 MDOF/s, and the next check reported it as
# 21.7% "faster than band". Nine is enough for the median to shrug that off.
REPS="${REPS:-9}"
THREADS="${THREADS:-72}"

if [ -z "${BIN:-}" ]; then
    for d in build_omp build_alps build_ab build build_cuda; do
        if [ -x "$d/cvfem_hex8_ns_upwind_bench" ]; then BIN="$d/cvfem_hex8_ns_upwind_bench"; break; fi
    done
fi
[ -n "${BIN:-}" ] && [ -x "$BIN" ] || {
    echo "error: cvfem_hex8_ns_upwind_bench not found; build it or set BIN" >&2
    echo "       e.g. source scripts/alps_env.sh && cvfem_build --target cvfem_hex8_ns_upwind_bench" >&2
    exit 2; }
if [ "$MODE" = against ]; then
    [ -x "$REF_BIN" ] || { echo "error: reference binary not executable: $REF_BIN" >&2; exit 2; }
fi

OUT="${OUT:-$(mktemp -d)}"; mkdir -p "$OUT"
CSV="$OUT/perf_regression.csv"; rm -f "$CSV"

echo "### cvfem packed-throughput regression gate  (mode: $MODE)"
echo "### binary   : $BIN"
[ "$MODE" = against ] && echo "### reference: $REF_BIN"
[ "$MODE" != against ] && echo "### baseline : $BASELINE"
echo "### threads  : $THREADS   reps: $REPS   raw csv: $CSV"
echo "### host     : $(hostname)   $(date '+%Y-%m-%d %H:%M:%S')"

measure() {  # binary tag_prefix key operation layout kernel n [extra options]
    local bin=$1 pfx=$2 key=$3 op=$4 layout=$5 kernel=$6 n=$7 extra=${8:-}
    # shellcheck disable=SC2046
    OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
        stdbuf -oL "$bin" --n "$n" --repeat 20 --warmup 3 \
            --layout "$layout" --kernel "$kernel" $(op_flag "$op") $extra \
            --csv "$CSV" --tag "${pfx}${key}" >/dev/null 2>&1 \
        || echo "### WARNING: ${pfx}${key} returned $? -- it will show as missing below"
}

# The first invocation in an allocation is systematically off -- measured at 12% here -- so
# one is thrown away before anything is recorded.
echo "### discarding one warm-up invocation"
OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BIN" --n 96 --repeat 3 --warmup 1 --layout packed --kernel sumfact >/dev/null 2>&1

# Runs every configuration REPS times, or only the keys named in $1 (newline separated).
sweep() {
    local only="${1:-}"
    for rep in $(seq 1 "$REPS"); do
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r key op layout kernel n band bband extra <<<"$cfg"
        if [ -n "$only" ] && ! printf '%s\n' "$only" | grep -qx -- "$key"; then continue; fi
        echo "### rep $rep  $key"
        if [ "$MODE" = against ]; then
            # Alternate which side goes first, so position-in-pair is balanced.
            if [ $((rep % 2)) -eq 1 ]; then
                measure "$BIN"     "new_" "$key" "$op" "$layout" "$kernel" "$n" "$extra"
                measure "$REF_BIN" "ref_" "$key" "$op" "$layout" "$kernel" "$n" "$extra"
            else
                measure "$REF_BIN" "ref_" "$key" "$op" "$layout" "$kernel" "$n" "$extra"
                measure "$BIN"     "new_" "$key" "$op" "$layout" "$kernel" "$n" "$extra"
            fi
        else
            measure "$BIN" "" "$key" "$op" "$layout" "$kernel" "$n" "$extra"
        fi
    done
    done
}

sweep

[ -s "$CSV" ] || { echo "error: no measurements were produced" >&2; exit 2; }

CONFIG_SPEC=$(printf '%s\n' "${CONFIGS[@]}")
FAILED="$OUT/failing_keys.txt"
export CONFIG_SPEC CSV BASELINE MODE THREADS REPS BIN REF_BIN FAILED

cat > "$OUT/analyse.py" <<'PY'
import csv, os, statistics as st, sys, subprocess, datetime
from pathlib import Path

configs = [l.split('|') for l in os.environ["CONFIG_SPEC"].splitlines()
           if l.strip() and not l.strip().startswith('#')]
raw, baseline_path = Path(os.environ["CSV"]), Path(os.environ["BASELINE"])
mode = os.environ["MODE"]

runs, dofs = {}, {}
with raw.open() as f:
    for row in csv.DictReader(f):
        runs.setdefault(row["tag"], []).append(float(row["MDOF_s"]))
        dofs[row["tag"]] = int(row["dofs"])

def med(tag):
    v = runs.get(tag)
    return st.median(v) if v else None
def spread(tag):
    v = runs.get(tag)
    return (max(v) - min(v)) / st.median(v) * 100 if v and st.median(v) else 0.0

# ------------------------------------------------------------------ interleaved A/B: the gate
if mode == "against":
    print()
    print(f"{'config':<28}{'ndof':>10}{'ref':>9}{'new':>9}{'new/ref':>9}{'band':>7}  verdict")
    fail = 0
    failing = []
    only = {k for k in os.environ.get("CONFIRM_ONLY", "").split() if k}
    for key, op, layout, kernel, n, band, bband, *_ in configs:
        if only and key not in only: continue
        r, m, b = med("ref_" + key), med("new_" + key), float(band)
        if r is None or m is None:
            print(f"{key:<28}{'':>10}{'':>9}{'':>9}{'':>9}{'':>7}  MISSING"); fail += 1; continue
        d = (m - r) / r * 100
        verdict = "REGRESSION" if d < -b else ("faster than band" if d > b else "ok")
        if d < -b:
            fail += 1
            failing.append(key)
        print(f"{key:<28}{dofs['new_'+key]:>10}{r:>9.1f}{m:>9.1f}{d:>8.1f}%{b:>6.0f}%  {verdict}")
    print(f"\nboth sides measured in one allocation, order alternating every rep, "
          f"medians of {os.environ['REPS']}")
    Path(os.environ["FAILED"]).write_text("\n".join(failing) + ("\n" if failing else ""))
    if fail:
        print(f"\nFAILED: {fail} configuration(s) outside band", file=sys.stderr); sys.exit(1)
    print("\nPASSED: packed throughput unchanged against the reference binary")
    sys.exit(0)

# ------------------------------------------------------------------------- absolute baseline
def provenance():
    def sh(*c):
        try: return subprocess.run(c, capture_output=True, text=True).stdout.strip() or ""
        except Exception: return ""
    return {
        "recorded": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": sh("hostname") or "unknown",
        "commit": sh("git", "rev-parse", "--short", "HEAD") or os.environ.get("CVFEM_COMMIT", "unknown"),
        "threads": os.environ["THREADS"], "reps": os.environ["REPS"], "binary": os.environ["BIN"],
    }

# The bands live in CONFIGS above. They are wide on purpose -- they must absorb the
# node-to-node variation measured at 5-11% for these kernels -- and this mode is a coarse
# "is this machine still in the right ballpark" check, not what clears a change.

if mode == "record":
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    p = provenance()
    with baseline_path.open("w") as f:
        f.write("# Reference throughput for scripts/perf_regression.sh (baseline mode).\n")
        f.write("# COARSE by construction: the bands absorb node-to-node variation, measured at\n")
        f.write("# 5-11% for these kernels between two Grace nodes. A change is cleared with\n")
        f.write("#   scripts/perf_regression.sh --against <binary from the previous commit>\n")
        f.write("# which measures both in one allocation and bands at 5%.\n")
        f.write("# Re-record only deliberately, and say in the commit message what changed and\n")
        f.write("# on what evidence -- a silently re-recorded baseline gates nothing.\n")
        for k, v in p.items():
            f.write(f"# {k}: {v}\n")
        f.write("key,operation,layout,kernel,cube_n,ndof,mdof_s,band_pct,observed_spread_pct\n")
        for key, op, layout, kernel, n, ab_band, bband, *_ in configs:
            if med(key) is None:
                print(f"  {key}: MISSING, not recorded", file=sys.stderr); continue
            f.write(f"{key},{op},{layout},{kernel},{n},{dofs[key]},{med(key):.1f},{bband},{spread(key):.1f}\n")
    print(f"\nwrote {baseline_path}")
    print(f"{'config':<28}{'ndof':>10}{'MDOF/s':>10}{'spread':>9}  baseline band")
    for key, op, layout, kernel, n, ab_band, bband, *_ in configs:
        if med(key) is not None:
            b = f"{bband}%" if bband != "-" else "not gated (too noisy)"
            print(f"{key:<28}{dofs[key]:>10}{med(key):>10.1f}{spread(key):>8.1f}%  {b}")
    sys.exit(0)

if not baseline_path.exists():
    print(f"error: no baseline at {baseline_path}; run --record on the reference machine",
          file=sys.stderr)
    sys.exit(2)

ref = {}
with baseline_path.open() as f:
    for row in csv.DictReader(l for l in f if not l.startswith("#")):
        ref[row["key"]] = row

print()
print(f"{'config':<28}{'ndof':>10}{'ref':>9}{'now':>9}{'delta':>9}{'band':>7}  verdict")
fail = notes = 0
for key, op, layout, kernel, n, ab_band, bband, *_ in configs:
    if key not in ref:
        print(f"{key:<28}{'':>10}{'':>9}{'':>9}{'':>9}{'':>7}  no baseline entry"); notes += 1; continue
    if med(key) is None:
        print(f"{key:<28}{'':>10}{float(ref[key]['mdof_s']):>9.1f}{'--':>9}{'':>9}{'':>7}  MISSING"); fail += 1; continue
    r, m = float(ref[key]["mdof_s"]), med(key)
    d = (m - r) / r * 100
    if ref[key]["band_pct"] == "-":
        print(f"{key:<28}{dofs[key]:>10}{r:>9.1f}{m:>9.1f}{d:>8.1f}%{'--':>7}  info only (not gated here)")
        continue
    b = float(ref[key]["band_pct"])
    if d < -b:   verdict = "REGRESSION"; fail += 1
    elif d > b:  verdict = "faster than band"; notes += 1
    else:        verdict = "ok"
    print(f"{key:<28}{dofs[key]:>10}{r:>9.1f}{m:>9.1f}{d:>8.1f}%{b:>6.0f}%  {verdict}")

hdr = [l.strip('# \n') for l in baseline_path.open()
       if any(l.startswith(f'# {k}:') for k in ('host', 'commit', 'recorded'))]
print(f"\nbaseline: {', '.join(hdr)}")
print("this mode is coarse -- clear a change with --against, not with this")
if fail:
    print(f"\nFAILED: {fail} configuration(s) outside their band", file=sys.stderr)
    sys.exit(1)
print(f"\nPASSED{f' ({notes} note(s))' if notes else ''}")
PY

# ------------------------------------------------------------------- confirmation pass
#
# A configuration that fails is re-measured before it is believed. This is not leniency:
# residual_packed_sympy has now produced two wild low readings on this machine in a
# handful of runs -- 1690 against a usual 2050 while recording a baseline, and 1260 in an
# A/B whose immediate rerun came back at +0.1% -- so a single failing sample is not
# evidence of a regression, and reporting it as one teaches people to ignore the gate.
#
# The cost is zero unless something fails, and the discipline is the one that caught the
# run-order artifact earlier: measure it again before you believe it.
set +e
python3 "$OUT/analyse.py"
rc=$?
set -e

if [ "$rc" -ne 0 ] && [ "$MODE" = against ] && [ -s "$FAILED" ]; then
    echo
    echo "### $(wc -l < "$FAILED" | tr -d ' ') configuration(s) failed; re-measuring to confirm"
    keys=$(cat "$FAILED")
    CSV="$OUT/perf_regression_confirm.csv"; rm -f "$CSV"; export CSV
    CONFIRM_ONLY="$(tr '\n' ' ' < "$FAILED")"; export CONFIRM_ONLY
    sweep "$keys"
    echo
    echo "### confirmation pass"
    set +e
    python3 "$OUT/analyse.py"
    rc2=$?
    set -e
    if [ "$rc2" -eq 0 ]; then
        echo
        echo "NOT CONFIRMED: the regression did not reproduce when those configurations were"
        echo "measured again. Treating it as measurement noise, not a regression."
        rc=0
    else
        echo
        echo "CONFIRMED: the regression reproduced on a second measurement."
    fi
fi
exit $rc
