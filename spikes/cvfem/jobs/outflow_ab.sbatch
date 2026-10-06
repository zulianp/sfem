#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-outflow
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=outflow_%j.out
#SBATCH --error=outflow_%j.err
#
# THE DO-NOTHING OUTFLOW AGAINST THE CONVECTIVE ONE, ON A TRANSIENT.
#
# WHAT THIS CAN SHOW AND WHAT IT CANNOT, first, because the two are easy to conflate.
#
# It CANNOT show absence of reflection directly. The proper test is a vortex crossing the
# plane and not sending anything back upstream, and measuring that needs a probe upstream of
# the outlet which the diagnostics do not have. Nothing here is a substitute for it.
#
# It CAN show two things that are worth having on their own.
#
# FIRST, the mass balance -- and this arm is a check of the implementation, not of the
# physics, because the convective condition enforces it by construction. mdot_in + mdot_out
# is zero for a flow that enters and leaves through these two planes and nowhere else. The
# budget job measured |mdot_in + mdot_out|/|mdot_in| at 0.4452, 0.1662 and 0.0563 down the
# step's mesh ladder under the do-nothing outflow, falling with h but never small. Since
# eps_num is DEFINED as the residual of the energy budget, whatever those fluxes fail to
# account for lands in it, and the convective outlet should drive the term to round-off. If
# it does not, the shift is wrong and the number it is supposed to justify is not safe.
#
# SECOND, the budget closure and the kinetic energy over the run, which are NOT enforced by
# anything here and so are an independent statement. A boundary that reflects returns energy
# the budget has already accounted as leaving, so closure is where that would show.
#
# Same mesh, same dt, same solver, same seed. The only difference is SFEM_STEP_OUTFLOW, which
# is the point: an A/B whose arms differ in two things measures neither.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "outflow: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_outflow/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

NN=${NN:-4}; LV=${LV:-4}; MU=${MU:-0.01}; DT=${DT:-0.02}; NS=${NS:-40}
BASE="SFEM_CASE=step SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0
      SFEM_N=$NN SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_MU=$MU SFEM_DT=$DT SFEM_NSTEPS=$NS
      SFEM_BDF_ORDER=2 SFEM_CONV_HO=1 SFEM_CONV_LIMITER=0"

run_one() {
    local tag=$1 mode=$2
    local log="$OUT/$tag.log" csv="$OUT/$tag.csv"
    stdbuf -oL -eL env $BASE SFEM_STEP_OUTFLOW=$mode SFEM_DIAG_CSV="$csv" \
        uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | tail -1)
    # Summarised from the CSV over every step, not read off the last one: a boundary problem
    # that grows is invisible in a final value and obvious in a maximum.
    python3 - "$csv" "$tag" "${ndof:--}" "${conv:-$rc}" <<'PY'
import csv, sys
path, tag, ndof, conv = sys.argv[1:5]
try:
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if r.get('step','').isdigit()]
except Exception:
    rows = []
if not rows:
    print('%-14s %9s %5s   no diagnostic rows' % (tag, ndof, conv)); raise SystemExit
def col(n): return [float(r[n]) for r in rows]
mi, mo = col('mdot_in'), col('mdot_out')
merr = [abs(a+b)/abs(a) if a else float('nan') for a, b in zip(mi, mo)]
clo  = [abs(float(r['closure'])) for r in rows]
E    = col('E'); dv = col('div_l2')
print('%-14s %9s %5s %5d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e' %
      (tag, ndof, conv, len(rows), sum(merr)/len(merr), max(merr),
       sum(clo)/len(clo), max(clo), E[-1], max(dv)))
PY
}
printf '%-14s %9s %5s %5s %11s %11s %11s %11s %11s %11s\n' \
    arm ndof conv steps mass_mean mass_max clos_mean clos_max E_final div_max
run_one natural    natural
run_one convective convective
echo
echo "logs in $OUT"
echo "done $(date)"
