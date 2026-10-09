#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-stprobe
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=stprobe_%j.out
#SBATCH --error=stprobe_%j.err
#
# Does the turbulent step case run, and what makes its time steps hard?
#
# A first local run converged every step but only by falling back to the Reynolds ramp on
# each one -- the direct solve failed and the retry inserted a stage. That is the continuation
# fallback doing its job, but a step that needs it is a step that is being asked something too
# hard, and on a long run it doubles the cost. This separates the candidates one at a time
# rather than guessing: the timestep, the grading, the perturbation and the Newton budget.
#
# `stages` is the diagnostic. With the ramp paid once on the first step and one solve per step
# after, a clean run is steps + (ramp - 1). Anything above that is the fallback firing.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_stprobe/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
printf "%-26s %-9s %-8s %-8s %-7s %-10s %s\n" run ndof steps stages conv lin_it t_solve

one() {  # one <tag> <env...>
    local tag=$1; shift
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && timeout 400 uenv run $U -- env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=step_turb SFEM_NX=84 SFEM_NY=16 SFEM_NZ=16 \
        SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 "$@" \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd st sg c li ts
    nd=$( grep -o 'ndof: [0-9]*'           "$r/log" | head -1 | awk '{print $2}')
    st=$( grep -cE "^=== step"             "$r/log")
    sg=$( grep -cE "^stage [0-9]+/"        "$r/log")
    c=$(  grep -o 'newton_converged: [01]' "$r/log" | head -1 | awk '{print $2}')
    li=$( grep -o 'lin_it_total: [0-9]*'   "$r/log" | head -1 | awk '{print $2}')
    ts=$( grep -o 't_solve: [0-9.]*'       "$r/log" | head -1 | awk '{print $2}')
    printf "%-26s %-9s %-8s %-8s %-7s %-10s %s\n" \
        "$tag" "${nd:-?}" "${st:-0}" "${sg:-?}" "${c:-?}" "${li:-?}" "${ts:-?}"
}

# The baseline the local run used, then one change at a time from it.
one base_dt05         SFEM_MU=0.02 SFEM_DT=0.05   SFEM_NSTEPS=8 SFEM_NL_MAX_IT=12 SFEM_STEP_GRADE=1.5 SFEM_IC_PERTURB=0.05
one dt_smaller        SFEM_MU=0.02 SFEM_DT=0.0125 SFEM_NSTEPS=8 SFEM_NL_MAX_IT=12 SFEM_STEP_GRADE=1.5 SFEM_IC_PERTURB=0.05
one no_grading        SFEM_MU=0.02 SFEM_DT=0.05   SFEM_NSTEPS=8 SFEM_NL_MAX_IT=12 SFEM_STEP_GRADE=0   SFEM_IC_PERTURB=0.05
one no_perturb        SFEM_MU=0.02 SFEM_DT=0.05   SFEM_NSTEPS=8 SFEM_NL_MAX_IT=12 SFEM_STEP_GRADE=1.5 SFEM_IC_PERTURB=0
one more_newton       SFEM_MU=0.02 SFEM_DT=0.05   SFEM_NSTEPS=8 SFEM_NL_MAX_IT=40 SFEM_STEP_GRADE=1.5 SFEM_IC_PERTURB=0.05
one dt_smaller_newton SFEM_MU=0.02 SFEM_DT=0.0125 SFEM_NSTEPS=8 SFEM_NL_MAX_IT=40 SFEM_STEP_GRADE=1.5 SFEM_IC_PERTURB=0.05

echo
echo "=== Reynolds and flux report from the baseline ==="
grep -E "^step:|^channel:|^constraints:" $OUT/base_dt05/log 2>/dev/null | head -4
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
