#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-nozzle-outflow-P37
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=nozzle_outflow_P37_%j.out
#SBATCH --error=nozzle_outflow_P37_%j.err
#
# THE DO-NOTHING OUTFLOW AGAINST THE CONVECTIVE ONE, ON THE FDA NOZZLE.
#
# jobs/outflow_ab.sbatch does this on the backward-facing step. This is the same experiment on
# the geometry the campaign is actually for, which it could not be run on until the outlet
# treatment stopped being hard-coded to `natural` for the nozzle.
#
# WHAT THIS CAN SHOW AND WHAT IT CANNOT, first, because the two are easy to conflate.
#
# It CANNOT show absence of reflection directly. That needs a vortex crossing the plane and
# nothing coming back, measured by a probe upstream of the outlet which the diagnostics do not
# have. Nothing here substitutes for it.
#
# It CAN show three things worth having on their own:
#
#   the mass balance, which is a check of the IMPLEMENTATION rather than of the physics -- the
#   convective condition enforces it by construction, so if |mdot_in + mdot_out| does not fall
#   to round-off the shift is wrong and every eps_num computed under it is unsafe;
#
#   the budget closure and the kinetic energy, which nothing here enforces, so they are an
#   independent statement. A boundary that reflects returns energy the budget already accounted
#   as leaving, and closure is where that shows;
#
#   the enstrophy, as the jet's shear layer reaches the outlet. A reflecting plane keeps
#   vorticity in the domain that should have left.
#
# THE SMOOTHER IS NOT A VARIABLE HERE, and that is the point of the predicate this job exists to
# exercise. SFEM_VANKA_MULT=0 and SFEM_VANKA_OMEGA=0.5 follow `outlet_is_open`, which covers
# natural AND convective, so both arms run the same smoother. Keyed the old way -- on whether the
# outlet is traction-free -- the convective arm would have silently taken omega back to 1 and
# re-enabled the multiplicative sweep, and by the table beside SFEM_VANKA_OMEGA in the driver
# that is worth 1,067 iterations against 502 on this geometry, and a stall at core 4. The A/B
# would then have been measuring two smoothers while appearing to measure two outflows.
#
# Same mesh, same dt, same solver, same seed, one variable.
set -u
# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path
# says nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was
# run from; SPIKE_ROOT overrides it for a manual run.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "nozzle_outflow_ab.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
D="${BIN:-${BUILD:-$SPIKE_ROOT/build_gate}/cvfem_hex8_ns_ssgmg}"
[ -x "$D" ] || { echo "nozzle outflow: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_nozzle_outflow/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
echo "binary $D"
echo

# Core 2 at level 4 is 116,212 dof -- the size the cost model measured at 1.253 s per step, so
# 60 steps of two arms fits the debug cap with room. dt 1e-3 at Re 500 is the campaign's own
# step size, and 60 steps is long enough for the throat jet to reach the outlet: the bulk speed
# in the 6 mm pipe is ~4.6 cm/s over the 80 mm from the throat to x_out.
NS=${NS:-60}; DT=${DT:-1e-3}; LV=${LV:-4}
# The axial counts are set explicitly rather than left to the driver's default of "8 6 16 32":
# the 116,212-dof figure and the 1.253 s per step this job is sized against were measured in
# jobs/dns_cost_P33.sbatch with "4 3 8 12", and inheriting a different mesh would make both the
# budget and the comparison to the cost model wrong.
BASE="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine
      SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1
      SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1
      SFEM_DT=$DT SFEM_NSTEPS=$NS SFEM_BDF_ORDER=2 SFEM_ENABLE_OUTPUT=0"
AXIAL=${AXIAL:-"4 3 8 12"}

run_one() {
    local tag=$1 mode=$2
    local d="$OUT/$tag"; mkdir -p "$d"
    local log="$d/log" csv="$d/diag.csv"
    # A real output directory, never /dev/null: the driver writes a coarse mesh there and
    # MPI_Aborts after printing its summary if it cannot.
    ( cd "$d" && stdbuf -oL -eL env $BASE "SFEM_NOZZLE_NAXIAL=$AXIAL" \
        SFEM_OUTLET=$mode SFEM_DIAG_CSV="$csv" \
        "$D" "$d/out" > "$log" 2>&1 )
    local rc=$?
    # What the driver itself says about the outlet it built. On the convective arm this is the
    # check that the new nozzle_outlet_node wiring selected the disc without its rim: the rim
    # belongs to the wall, so the prescribed count must be smaller than the disc's node count
    # and the controlled area must be positive.
    grep -m1 "convective outflow:" "$log" 2>/dev/null | sed 's/^/    /'
    grep -m1 "^nozzle: wall faces" "$log" 2>/dev/null | sed 's/^/    /'
    grep -m1 "^preconditioner:" "$log" 2>/dev/null | sed 's/^/    /'
    local ndof
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    # Summarised over every step rather than read off the last one: a boundary problem that
    # grows is invisible in a final value and obvious in a maximum.
    python3 - "$csv" "$tag" "${ndof:--}" "$rc" <<'PY'
import csv, sys
path, tag, ndof, rc = sys.argv[1:5]
try:
    rows = [r for r in csv.DictReader(open(path)) if r.get('step','').isdigit()]
except Exception:
    rows = []
if not rows:
    print('%-12s %9s %4s   no diagnostic rows' % (tag, ndof, rc)); raise SystemExit
def col(n): return [float(r[n]) for r in rows]
mi, mo = col('mdot_in'), col('mdot_out')
merr = [abs(a+b)/abs(a) if a else float('nan') for a, b in zip(mi, mo)]
clo  = [abs(x) for x in col('closure')]
ens  = col('enstrophy'); E = col('E'); dv = col('div_l2'); en = col('eps_num')
print('%-12s %9s %4s %5d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e' %
      (tag, ndof, rc, len(rows), sum(merr)/len(merr), max(merr),
       sum(clo)/len(clo), max(clo), ens[-1], E[-1], max(dv)))
PY
}

printf '%-12s %9s %4s %5s %11s %11s %11s %11s %11s %11s %11s\n' \
    arm ndof rc steps mass_mean mass_max clos_mean clos_max enstr_end E_final div_max
run_one natural    natural
run_one convective convective

echo
echo "=== eps_num over the run, both arms (the term the outflow treatment feeds)"
for tag in natural convective; do
    printf -- "--- %s\n" "$tag"
    awk -F, 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next}
             $1+0>1 {n++; s+=$(c["eps_num"]); if($(c["eps_num"])>m||n==1)m=$(c["eps_num"])}
             END{ if(n) printf "    eps_num mean %.6e  max %.6e over %d steps\n", s/n, m, n }' \
        "$OUT/$tag/diag.csv" 2>/dev/null
done

echo
echo "=== linear work per step, both arms"
#
# The driver does not print the smoother settings back, so this does not try to read them: it
# reports the quantity a smoother change would actually move. Both arms take SFEM_VANKA_MULT=0
# and SFEM_VANKA_OMEGA=0.5 from `outlet_is_open`, so linear counts in the same range are the
# evidence that the two arms differ in the outflow treatment and nothing else. Counts differing
# by a factor -- the 1,067 against 502 in the driver's own table -- would mean the predicate did
# not hold and the comparison is void.
for tag in natural convective; do
    awk -F, -v t="$tag" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i;next}
         {n++; li=$(c["lin_it"]); ni=$(c["newton_it"]); if(n>1){sl+=li-pl; sn+=ni; k++} pl=li}
         END{ if(k) printf "%-12s newton/step %.2f  linear/step %.2f over %d steps\n", t, sn/k, sl/k, k }' \
        "$OUT/$tag/diag.csv" 2>/dev/null
done

echo "=== DONE $(date)"
