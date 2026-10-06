#!/usr/bin/env bash
# Produce a ParaView-ready diaphragm-pump cycle: run it, fetch it, build the XDMF.
#
#   scripts/pump_visualize.sh                 # run locally, small
#   REMOTE=alps scripts/pump_visualize.sh     # run on the Grace debug partition and fetch
#
# The XDMF step needs numpy, meshio and h5py, which live in the repository venv and not in
# the Alps uenv -- so the solve happens wherever it is cheapest and the visualisation is
# always built here, from the raw fields. That split is why this is a script and not a flag.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SPIKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-${SPIKE_ROOT}/../../venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

N=${SFEM_N:-16}
NSTEPS=${SFEM_NSTEPS:-32}
OUT=${OUT:-$SPIKE_ROOT/viz/pump_cycle}

# BiCGStab, not the dense LU and not multigrid. Measured on this case, two time steps:
# 5 s at N=16 with BiCGStab against 38 s for the LU at a quarter the size, and the LU does
# not finish at all past N=8. Multigrid is unavailable -- the pressure port is a flat-mesh
# boundary condition, because the semi-structured kernels take no boundary data.
PUMP_ENV=(SFEM_CASE=pump "SFEM_N=$N" SFEM_MU=0.02 SFEM_U=1
          SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=3000
          "SFEM_DT=$(awk -v n=$NSTEPS 'BEGIN{printf "%.10g", 1.0/n}')"
          "SFEM_NSTEPS=$NSTEPS" SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2
          SFEM_WRITE_STEPS=1)

if [ "${REMOTE:-}" = "" ]; then
    rm -rf "$OUT"; mkdir -p "$OUT"
    ( export "${PUMP_ENV[@]}"; "${BUILD:-$SPIKE_ROOT/build}/cvfem_hex8_ns_ssgmg" "$OUT" ) \
        > "$OUT/run.log" 2>&1 || { echo "solve failed; see $OUT/run.log" >&2; exit 1; }
else
    echo ">>> submitting on $REMOTE"
    JOB=$(ssh "$REMOTE" "cd \$SCRATCH/cvfem_blk && sbatch --parsable pumpviz.sbatch")
    echo "    job $JOB; waiting"
    while ssh "$REMOTE" "squeue -j $JOB -h -o %T" | grep -q .; do sleep 30; done
    rm -rf "$OUT"; mkdir -p "$OUT"
    rsync -a "$REMOTE:\$SCRATCH/cvfem_pumpviz/$JOB/" "$OUT/"
fi

grep -E "^pump: chamber|ndof:|newton_converged|^pump: \|port" "$OUT/run.log" || true
echo "frames: $(ls -d "$OUT"/step_* 2>/dev/null | wc -l)"
"$PYTHON" "$SPIKE_ROOT/python/create_xdmf.py" "$OUT" "$OUT/pump_cycle.xdmf"
echo
echo "Open in ParaView:  $OUT/pump_cycle.xdmf   (keep pump_cycle.h5 beside it)"
