#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ab-refactor
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_ab_refactor_%j.out
#SBATCH --error=cvfem_ab_refactor_%j.err
#
# THE REFACTOR GATE: the packed matrix-free throughput of cvfem_hex8_ns_upwind_bench, A/B
# against a binary built from an earlier commit. Interleaved in one allocation, which is the
# only form that survives the 5-11% node-to-node variation scripts/perf_regression.sh
# documents -- two separate jobs cannot tell a regression from a different node.
#
#   CVFEM_SRC   the tree under test       (default: the parent of this script's directory)
#   CVFEM_REF   the reference TREE, built (required; its build/ must hold the bench binary)
#
#     sbatch --export=ALL,CVFEM_SRC=$PWD,CVFEM_REF=/path/to/older/tree jobs/ab_refactor.sbatch
#
# The reference tree is a separate checkout rather than a rebuild in place, because the gate
# has to run both binaries in the same allocation and `git checkout` cannot produce two at
# once. `git archive <commit> spikes/cvfem | tar -x` is enough to make one.
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script, so a path derived from the script's own location is
# wrong on the compute node.
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_REF:?set CVFEM_REF to a built reference tree}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL

REF_BIN="$CVFEM_REF/build/cvfem_hex8_ns_upwind_bench"
[ -x "$REF_BIN" ] || { echo "reference binary not executable: $REF_BIN" >&2; exit 2; }

source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"
export BIN="$CVFEM_SRC/build/cvfem_hex8_ns_upwind_bench"
export OUT="$CVFEM_SRC/ab_out"
cvfem_uenv scripts/perf_regression.sh --against "$REF_BIN"
echo "### gate exit: $?"
