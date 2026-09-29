#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-nozzle-affine
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_affine_%j.out
#SBATCH --error=nozzle_affine_%j.err

# The FDA nozzle with its curved elements approximated as affine: how much does it cost in
# accuracy, and does it buy the semi-structured solvers?
#
#   sbatch jobs/nozzle_affine.sbatch            (from spikes/cvfem)
#
# Everything at throat Re = 1, where the centreline has known values: 2 u_in in the inlet and
# outlet pipes (stations -0.088 and +0.080), 2 u_t = 18 u_in in the throat (-0.020).
#
# STEP 1 -- flat, per-element centre Jacobian (SFEM_GEOM=affine) against the isoparametric
# control on the same mesh. The dense LU removes the solver from the comparison. The core 4
# pair is in jobs/nozzle_affine_perf.sbatch, because its LU alone costs 13 minutes; the
# isoparametric core 4 is already measured (job 4656808: 2.018 / 18.17 / 2.054).
#
# STEP 2 -- semi-structured, every micro cell of a macro element given that macro element's
# centre Jacobian (sscvfem_hoisted_cell). The severity is the MACRO size, not the level, so the
# sweep holds the fine lattice fixed and changes the macro size:
#
#   fine lattice                     macro mesh              L
#   core 4 / 2 / 2, axial 8 6 16 24  core 2, axial 4 3 8 12   2   (same lattice as flat core 4)
#   core 8 / 4 / 4, axial 16..48     core 4, axial 8 6 16 24  2
#   core 8 / 4 / 4, axial 16..48     core 2, axial 4 3 8 12   4
#
# Note what the macro size also sets: micro nodes are interpolated inside the macro element,
# so the wall is the macro mesh's polygon however fine the lattice. That is part of the
# approximation being measured, not a separate effect.
#
# Each semi-structured point is solved with and without multigrid: FGMRES + Vanka, and
# FGMRES + geometric multigrid. SFEM_FGMRES=1 explicitly.
set -u

SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "nozzle_affine.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
cd "$SPIKE_ROOT"

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
# Its own build tree: build_solver belongs to whatever else is running from this checkout.
BUILD=${BUILD:-$SPIKE_ROOT/build_nozzle}
DRIVER=$BUILD/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_nozzle/$SLURM_JOB_ID}
mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  driver $DRIVER  out $OUT  $(date)"

run() {  # run <tag> <seconds> <env...>
    local tag=$1 secs=$2; shift 2
    uenv run $UENV_ARGS -- env CVRUN_DIR="$OUT" CVRUN_OUT="$OUT/$tag" \
        CVRUN_FILTER='nnodes:|ndof:|^nozzle: (station|sum|inflow|SFEM_GEOM)|newton_converged|highest Re SOLVED|t_solve|lin_it_total|verification|semi_structured' \
        timeout "$secs" bash scripts/cvrun.sh "$tag" "$DRIVER" \
        SFEM_CASE=nozzle SFEM_NOZZLE_RE=1 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 "$@" || true
}

MESH_C2="SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
MESH_C4="SFEM_NOZZLE_NCORE=4 SFEM_NOZZLE_NBORE=2 SFEM_NOZZLE_NOUTER=2"
FLAT="SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_GMG=0 SFEM_PRECOND=direct SFEM_DIRECT_MAX_DOF=40000"
VANKA="SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0"
MG="SFEM_GMG=1"

# ---- step 1 ----
run flat_iso_core2    120 $FLAT $MESH_C2 "SFEM_NOZZLE_NAXIAL=8 6 16 24" SFEM_GEOM=isoparam
run flat_affine_core2 120 $FLAT $MESH_C2 "SFEM_NOZZLE_NAXIAL=8 6 16 24" SFEM_GEOM=affine

# ---- step 2 ----
for arm in vanka mg; do
    [ $arm = vanka ] && ARM="$VANKA" || ARM="$MG"
    run ss_c2L2_$arm 240 SFEM_GEOM=affine SFEM_ELEMENT_REFINE_LEVEL=2 $ARM $MESH_C2 "SFEM_NOZZLE_NAXIAL=4 3 8 12"
    run ss_c4L2_$arm 300 SFEM_GEOM=affine SFEM_ELEMENT_REFINE_LEVEL=2 $ARM $MESH_C4 "SFEM_NOZZLE_NAXIAL=8 6 16 24"
    run ss_c2L4_$arm 300 SFEM_GEOM=affine SFEM_ELEMENT_REFINE_LEVEL=4 $ARM $MESH_C2 "SFEM_NOZZLE_NAXIAL=4 3 8 12"
done

echo "=== DONE $(date) ==="
