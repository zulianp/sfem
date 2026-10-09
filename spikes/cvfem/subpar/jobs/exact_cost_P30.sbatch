#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-exact-cost-P30
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=exact_cost_P30_%j.out
#SBATCH --error=exact_cost_P30_%j.err
# What does the operator/smoother inconsistency cost, in iterations?
#
# The Krylov operator is the exact Jacobian A_frozen + K W^-1 G; the Vanka patches are built
# from A_frozen alone, because the exact term widens the pressure coupling beyond nearest
# neighbours (see Hex8RhieChowT). Exact patches on 81 slots would close that gap, at 1.5x the
# stencil storage plus a fold map for the widened offsets -- so the question this answers,
# BEFORE any of that is built, is how much the gap costs today.
#
#   exact  SFEM_RC_EXACT_JAC=1 (default): exact operator, frozen smoother -- inconsistent
#   frozen SFEM_RC_EXACT_JAC=0          : frozen operator, frozen smoother -- consistent
#
# The comparable quantity is linear iterations PER Newton step: the two solve slightly
# different Newton systems, so total Newton counts need not match. Interleaved and repeated.
set -u
SPIKE=$SCRATCH/sfem/spikes/cvfem; B=$SPIKE/build_gate
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_exact_cost/$SLURM_JOB_ID; mkdir -p "$OUT"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
run() {
    local tag=$1 lvl=$2 secs=$3; shift 3; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 "$@" \
        "$B/cvfem_hex8_ns_ssgmg" /dev/null > out.log 2>&1 )
    echo "--- $tag exit $?"
    grep -E "ndof:|newton_converged|t_solve" "$OUT/$tag/out.log" | head -4
}
run L4_exact_1  4 300 SFEM_RC_EXACT_JAC=1
run L4_frozen_1 4 300 SFEM_RC_EXACT_JAC=0
run L4_frozen_2 4 300 SFEM_RC_EXACT_JAC=0
run L4_exact_2  4 300 SFEM_RC_EXACT_JAC=1
run L8_exact_1  8 600 SFEM_RC_EXACT_JAC=1
run L8_frozen_1 8 600 SFEM_RC_EXACT_JAC=0
echo "=== DONE $(date)"
