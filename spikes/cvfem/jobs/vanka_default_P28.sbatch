#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vanka-default-P28
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vanka_default_P28_%j.out
#SBATCH --error=vanka_default_P28_%j.err
# The stencils are now the only fine-level Vanka path, so this checks the DEFAULT rather than an
# option: the equivalence gate against a BSR built for the same state, the case with and without
# multigrid, and the unit tests. The binary is built on the login node; this job only runs it.
set -u
SPIKE=$SCRATCH/sfem/spikes/cvfem; B=$SPIKE/build_gate
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_vanka_default/$SLURM_JOB_ID; mkdir -p "$OUT"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
run() {
    local tag=$1 secs=$2; shift 2; mkdir -p "$OUT/$tag"
    ( cd "$OUT/$tag" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=4 \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 "$@" "$B/cvfem_hex8_ns_ssgmg" /dev/null > out.log 2>&1 )
    echo "--- $tag exit $?"
    grep -E "ndof:|vanka stencil vs bsr|newton_converged|t_solve" "$OUT/$tag/out.log" | head -6
}
run gate_check 300 SFEM_GMG=1 SFEM_NL_MAX_IT=3 SFEM_VANKA_CHECK=1
run with_mg    300 SFEM_GMG=1
run without_mg 600 SFEM_GMG=0
echo "== ctest"
cd "$B" && ctest --output-on-failure -j 8 2>&1 | tail -12
echo "=== DONE $(date)"
