#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-re
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_re_%j.out
#SBATCH --error=egal_re_%j.err

# Solver robustness against Reynolds number, element-wise Galerkin throughout.
#
# This one belongs on `normal`, not `debug`. A converged solve at this size costs 778 s at
# Re=100, and higher Re needs more, so six Reynolds numbers is 80 to 150 minutes -- past
# the 30 minute mark where debug is the right queue. A job array on debug is not an escape
# either: that QOS allows 2 submitted and 1 running.
#
# The run must go to completion, not be capped: the driver does Reynolds continuation
# in exactly two stages -- solve at Re=1 with rho = mu/(U Ly), then jump straight to the
# physical density -- so a capped run measures the Re=1 stage and says nothing about the
# target. That is how an earlier scoping attempt here produced six runs that all solved
# the same problem. Whether the single jump from Re=1 to the physical Re survives is
# itself the robustness question.
#
# Poiseuille keeps the exact solution parabolic at every Re, since the convective term
# vanishes at the solution, so u_linf remains a real correctness check while the
# linearised systems become convection-dominated.
#
# Re = rho * U * Ly / mu with rho = U = Ly = 1, so mu = 1/Re.

set -u
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for re in 100 200 400 800 1600 3200; do
mu=$(python3 -c "print(1.0/${re})")
OUT="$SCRATCH/egal_out_re${re}"
mkdir -p "$OUT"
echo "### Re=${re} (mu=${mu}), element-wise, 108 macro-elements, L=4, 72 threads ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
              SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
              SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
              SFEM_GMG_EGAL=1 SFEM_MU=${mu} SFEM_VERIFY_TOL=1e9 \
              ${BIN} ${OUT}" \
    2>&1 | grep -E "nnodes|ndof|Re:|^stage:|newton_converged|lin_it_total|t_solve|u_linf|verification"
    echo
done
