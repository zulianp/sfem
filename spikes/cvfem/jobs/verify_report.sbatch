#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-verify
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=verify_%j.out
#SBATCH --error=verify_%j.err

# The verification matrix on one Grace socket, and the report built from it.
#
#   sbatch jobs/verify_report.sbatch
#   sbatch --export=ALL,VERIFY_GROUPS="mms port" jobs/verify_report.sbatch
#
# Results are what the driver printed, so they carry their own dof counts; the machine and
# thread count reach the report through the manifest. --exclusive because the report
# records timings beside the correctness results and a shared node makes those meaningless.
set -u

# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path says
# nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was run from.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "verify_report.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=${OUT:-$SCRATCH/cvfem_verify/$SLURM_JOB_ID}
mkdir -p "$OUT"

echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "spike root $SPIKE_ROOT"
echo "output     $OUT"

# The driver runs inside the uenv; so does the report generator, which needs only the
# standard library and so has nothing to install.
uenv run $UENV_ARGS -- env \
    SPIKE_ROOT="$SPIKE_ROOT" \
    BUILD="${BUILD:-$SPIKE_ROOT/build_blk}" \
    DRIVER="${DRIVER:-${BUILD:-$SPIKE_ROOT/build_blk}/cvfem_hex8_ns_ssgmg}" \
    OUT="$OUT" \
    VERIFY_GROUPS="${VERIFY_GROUPS:-mms bc port step}" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    bash "$SPIKE_ROOT/scripts/verify_report.sh"

uenv run $UENV_ARGS -- python3 "$SPIKE_ROOT/python/cvfem_verify_report.py" "$OUT" \
    -o "$OUT/CVFEM_Verification_Report.md" --html

echo
echo "=== report ==="
cat "$OUT/CVFEM_Verification_Report.md"
echo "=== DONE $(date) ==="
