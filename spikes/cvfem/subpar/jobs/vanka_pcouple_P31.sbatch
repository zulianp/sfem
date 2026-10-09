#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pcouple-P31
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vanka_pcouple_P31_%j.out
#SBATCH --error=vanka_pcouple_P31_%j.err
# How sensitive is the solver to the patch pressure couplings?
#
# SFEM_VANKA_PCOUPLE_PERT scales the patch G block and the pressure-pressure entry by 1+eps --
# exactly the entries an exact-operator (81-slot) patch would correct. The exact term is 2-7%
# of the operator, so if perturbing these by that much leaves the iteration count flat, exact
# patches cannot help and their 1.5x storage plus a widened fold map is not worth building.
#
# eps = 0 is the control and must reproduce 617 linear iterations at 116,212 dof and 1153 at
# 893,924 -- the numbers every earlier run produced. A control that moves means the knob is
# not inert at its default, which invalidates the sweep rather than the operator.
set -u
SPIKE=$SCRATCH/sfem/spikes/cvfem; B=$SPIKE/build_gate
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_pcouple/$SLURM_JOB_ID; mkdir -p "$OUT"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
run() {
    local tag=$1 lvl=$2 secs=$3 eps=$4; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_VANKA_PCOUPLE_PERT=$eps "$B/cvfem_hex8_ns_ssgmg" /dev/null > out.log 2>&1 )
    printf "%-16s eps=%-6s " "$tag" "$eps"
    grep -E "newton_converged" "$OUT/$tag/out.log" | head -1
}
echo "=== L4 (116,212 dof), control must give lin_it_total 617"
run L4_e000  4 300 0
run L4_ep02  4 300 0.02
run L4_em02  4 300 -0.02
run L4_ep05  4 300 0.05
run L4_em05  4 300 -0.05
run L4_ep10  4 300 0.10
echo "=== L8 (893,924 dof), control must give lin_it_total 1153"
run L8_e000  8 600 0
run L8_ep05  8 600 0.05
run L8_em05  8 600 -0.05
echo "=== DONE $(date)"
