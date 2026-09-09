#!/usr/bin/env bash
# Run the CVFEM verification matrix and write a directory a report can be built from.
#
#   scripts/verify_report.sh                     # everything, into verification_runs/<stamp>
#   OUT=/tmp/v scripts/verify_report.sh          # somewhere specific
#   VERIFY_GROUPS="mms port" scripts/verify_report.sh   # a subset while iterating
#
# Then:
#   python3 python/cvfem_verify_report.py <OUT> -o docs/CVFEM_Verification_Report.md --html
#
# This script only produces evidence; every judgement lives in the report generator, so a
# report can be rebuilt or its thresholds changed without paying for the runs again.
#
# Each run gets its own COMPLETE unfiltered log. A grep pattern is a guess about what will
# matter, and when a case fails in an unanticipated way the explaining lines are exactly
# the ones a filter drops.
set -u

# SLURM_SUBMIT_DIR first: sbatch copies the batch script to /var/spool/slurmd/job<id>/,
# so anything derived from the running script's own path resolves somewhere useless.
for cand in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)" "$PWD"; do
    [ -n "$cand" ] && [ -f "$cand/CMakeLists.txt" ] && [ -d "$cand/drivers" ] && SPIKE_ROOT="$cand" && break
done
if [ -z "${SPIKE_ROOT:-}" ]; then
    echo "verify_report: cannot locate the spike root (tried SLURM_SUBMIT_DIR, script dir, cwd)" >&2
    exit 1
fi

BUILD=${BUILD:-$SPIKE_ROOT/build}
DRIVER=${DRIVER:-$BUILD/cvfem_hex8_ns_ssgmg}
OUT=${OUT:-$SPIKE_ROOT/verification_runs/$(date +%Y%m%d-%H%M%S)}
# VERIFY_GROUPS and not GROUPS: GROUPS is a bash built-in array holding the caller's
# group IDs, so ${GROUPS:-...} silently expands to a numeric GID and never to the
# default. That is exactly what happened -- a job ran with "groups : 33203" and
# selected nothing at all.
VERIFY_GROUPS=${VERIFY_GROUPS:-"unit mms bc port step"}
# 4/8/16/32 reproduces the dof ladder docs/CVFEM_Verification_Farrell.md records
# (500, 2916, 19652, 143748), so a rate measured here is comparable with the one there.
MMS_LADDER=${MMS_LADDER:-"4 8 16 32"}
PORT_SWEEP=${PORT_SWEEP:-"-0.16 -0.08 0 0.16 0.5 1.0 1.5 3.0"}
NL_MAX_IT=${NL_MAX_IT:-40}

if [ ! -x "$DRIVER" ]; then
    echo "verify_report: no driver at $DRIVER (set DRIVER= or build first)" >&2
    exit 1
fi
mkdir -p "$OUT"
TSV="$OUT/runs.tsv"
: > "$TSV"

# The channel geometry and fluid these cases use. The exact Poiseuille pressure is
# p(x) = G (Lx/2 - x) with G = 8 mu U / Ly^2, so the outlet value -- which is what a
# prescribed-pressure port is judged against -- is -G Lx / 2.
LX=4; LY=1; MU=0.01; U=1
P_EXACT_OUTLET=$(awk -v mu=$MU -v u=$U -v ly=$LY -v lx=$LX 'BEGIN{printf "%.10g", -(8*mu*u/(ly*ly))*lx/2}')
# The step's exact volumetric inflow is 1/9; see cvfem_ns_channel_case.hpp.
MASS_EXACT=$(awk 'BEGIN{printf "%.12g", 1.0/9.0}')

echo ">>> verification matrix  $(date +%H:%M:%S)"
echo "    spike root : $SPIKE_ROOT"
echo "    driver     : $DRIVER"
echo "    output     : $OUT"
echo "    groups     : $VERIFY_GROUPS"
echo "    p_exact(outlet) = $P_EXACT_OUTLET   exact step flux = $MASS_EXACT"

want() { case " $VERIFY_GROUPS " in *" $1 "*) return 0;; *) return 1;; esac; }

# run <group> <label> <extra tsv fields> -- <env assignments...>
run() {
    local group=$1 label=$2 extra=$3; shift 3; [ "$1" = "--" ] && shift
    local log="${group}_${label}.log"
    printf "  %-8s %-14s " "$group" "$label"
    local t0=$SECONDS
    # Only SFEM_NL_MAX_IT is set for every case. Geometry and fluid are NOT: the
    # manufactured solution lives on [0,2]^3 -- its pressure gauge constant is zero-mean
    # there and nowhere else, and the driver rejects an override for exactly that reason --
    # so forcing the channel's 4x1x1 on it silently measures a different problem. That is
    # not hypothetical: it produced u_linf of 26, 10, 8.8 and 14 down a ladder that should
    # have been converging quadratically.
    # The default goes first so a per-run SFEM_NL_MAX_IT in the argument list wins. A
    # `VAR=x run ...` prefix would not do: bash keeps such an assignment after a *function*
    # returns, so it would leak into every later case.
    ( export SFEM_NL_MAX_IT=$NL_MAX_IT
      for kv in "$@"; do export "$kv"; done
      "$DRIVER" "$OUT/out_${group}_${label}" ) > "$OUT/$log" 2>&1
    local rc=$?
    printf "exit=%-3s %4ds  %s\n" "$rc" "$((SECONDS - t0))" \
        "$(grep -m1 -E 'u_linf:|sum of continuity' "$OUT/$log" | cut -c1-60)"
    printf "%s\t%s\t%s\t%s\n" "$group" "$label" "$log" "$extra" >> "$TSV"
}

# ---- unit tests: the layer that names a line rather than a case ----
CTEST_TOTAL=0; CTEST_PASS=0; CTEST_FAIL=0; CTEST_FAILING=""
if want unit && command -v ctest > /dev/null 2>&1; then
    echo "  unit     ctest"
    ( cd "$BUILD" && ctest ) > "$OUT/ctest.log" 2>&1
    line=$(grep -m1 -E "tests passed.*out of" "$OUT/ctest.log")
    CTEST_PASS=$(echo "$line" | sed -n 's/.*, \([0-9]*\) tests failed out of \([0-9]*\).*/\2/p')
    CTEST_FAIL=$(echo "$line" | sed -n 's/.*, \([0-9]*\) tests failed out of.*/\1/p')
    CTEST_TOTAL=${CTEST_PASS:-0}
    CTEST_FAIL=${CTEST_FAIL:-0}
    CTEST_PASS=$((CTEST_TOTAL - CTEST_FAIL))
    CTEST_FAILING=$(sed -n 's/^\t*[0-9]* - \([A-Za-z0-9_]*\) (Failed).*/\1/p' "$OUT/ctest.log" | tr '\n' ',' | sed 's/,$//')
    echo "           $CTEST_PASS/$CTEST_TOTAL passed"
fi

# ---- spatial order of accuracy, manufactured solution ----
# mu = 1/Re is mandated by the manufactured pressure, so this group overrides it.
if want mms; then
    # Re = 1, which is the informative point rather than a convenient one: upwinding is
    # inactive there, so a first-order reading would mean a consistency error rather than
    # benign upwind diffusion. The manufactured pressure also mandates rho = 1, mu = 1/Re.
    for n in $MMS_LADDER; do
        run mms "n$n" "" -- SFEM_CASE=mms SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=1 \
            SFEM_MU=1 SFEM_RHO=1
    done
fi

# ---- the boundary conditions, judged against each other and against a closed form ----
if want bc; then
    run bc dirichlet "" -- SFEM_LX=$LX SFEM_LY=$LY SFEM_MU=$MU SFEM_U=$U SFEM_CASE=poiseuille SFEM_N=12 SFEM_ELEMENT_REFINE_LEVEL=1 \
        SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet
    # This one does not converge -- the do-nothing outflow is not the exact Poiseuille
    # outlet -- so it is capped rather than left to grind out 5 continuation stages of 40
    # Newton steps, which cost 405 s for a result that is an identity against traction0 and
    # holds at any iteration count, provided both runs use the same one.
    run bc natural "" -- SFEM_NL_MAX_IT=12 SFEM_LX=$LX SFEM_LY=$LY SFEM_MU=$MU SFEM_U=$U SFEM_CASE=poiseuille SFEM_N=12 SFEM_ELEMENT_REFINE_LEVEL=1 \
        SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=natural
    run bc traction0 "" -- SFEM_NL_MAX_IT=12 SFEM_LX=$LX SFEM_LY=$LY SFEM_MU=$MU SFEM_U=$U SFEM_CASE=poiseuille SFEM_N=12 SFEM_ELEMENT_REFINE_LEVEL=1 \
        SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet \
        SFEM_TRACTION_SIDESET=outlet "SFEM_TRACTION=0 0 0"
fi

# ---- a port holds the level, and nothing else ----
if want port; then
    for pb in $PORT_SWEEP; do
        run port "p$pb" "p_exact_outlet=$P_EXACT_OUTLET" -- SFEM_LX=$LX SFEM_LY=$LY SFEM_MU=$MU SFEM_U=$U \
            SFEM_CASE=poiseuille SFEM_N=12 SFEM_ELEMENT_REFINE_LEVEL=1 \
            SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet \
            SFEM_PRESSURE_SIDESET=outlet SFEM_PRESSURE=$pb
    done
fi

# ---- global mass conservation on the one non-box domain the spike has ----
if want step; then
    # Flat, Re = 20, no multigrid, and an exact linear solve.
    #
    # SFEM_PRECOND=direct builds a dense LU of the fine Jacobian, which is affordable at
    # 7,060 dofs and is the point: it takes the linear solver out of the question entirely,
    # so what is left is a statement about the discretisation. Block-Jacobi cannot solve
    # this case -- it has nothing to say about the pressure coupling and the Krylov residual
    # wanders and then diverges -- and multigrid is deliberately not used here.
    run step lshape "mass_exact=$MASS_EXACT" -- SFEM_CASE=step SFEM_BOUNDARY_MASK=1 \
        SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_MU=0.1 SFEM_GMG=0 SFEM_PRECOND=direct
fi

# ---- assemble the manifest ----
# Exported so the heredoc below can read them; a shell variable is not in its environment.
export SPIKE_ROOT CTEST_TOTAL CTEST_PASS CTEST_FAIL CTEST_FAILING
python3 - "$OUT" "$TSV" <<'PYEOF'
import json, os, subprocess, sys, platform
out, tsv = sys.argv[1], sys.argv[2]
runs = []
with open(tsv) as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        group, label, log = parts[0], parts[1], parts[2]
        rec = {"group": group, "label": label, "log": log}
        for kv in (parts[3] if len(parts) > 3 else "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    rec[k] = float(v)
                except ValueError:
                    rec[k] = v
        runs.append(rec)

def sh(cmd, default="--"):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip() or default
    except Exception:
        return default

manifest = {
    "runs": runs,
    "machine": os.environ.get("SLURM_JOB_NODELIST") or platform.node(),
    "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    "commit": sh("git -C %s rev-parse --short HEAD" % os.environ.get("SPIKE_ROOT", ".")),
}
ct_total = int(os.environ.get("CTEST_TOTAL", "0") or 0)
if ct_total:
    failing = [f for f in (os.environ.get("CTEST_FAILING", "") or "").split(",") if f]
    manifest["ctest"] = {"total": ct_total,
                         "passed": int(os.environ.get("CTEST_PASS", "0") or 0),
                         "failed": int(os.environ.get("CTEST_FAIL", "0") or 0),
                         "failing": failing}
with open(os.path.join(out, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print("    manifest   : %d run(s)" % len(runs))
PYEOF

echo "<<< done  $(date +%H:%M:%S)"
echo
echo "Build the report with:"
echo "  python3 $SPIKE_ROOT/python/cvfem_verify_report.py $OUT \\"
echo "      -o $SPIKE_ROOT/docs/CVFEM_Verification_Report.md --html"
