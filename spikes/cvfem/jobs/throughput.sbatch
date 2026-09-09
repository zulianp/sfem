#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-throughput
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=tp_%j.out
#SBATCH --error=tp_%j.err
# Where the time goes, on one Grace socket, across problem sizes and on both operators.
#
# --exclusive because a throughput measured on a shared node is not a throughput, and the
# smallest sizes are here only to show where saturation begins -- they are below it by
# construction and must not be quoted as rates.
#
# Each run gets its own directory: the tracer writes smesh.trace.csv into the working
# directory, so runs in a shared one would overwrite each other's evidence.
set -u
BLK=$SCRATCH/cvfem_blk; B=$BLK/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_tp/$SLURM_JOB_ID}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
: > "$OUT/runs.tsv"

one() {
  local label=$1 n=$2 lvl=$3
  local d="$OUT/$label"; mkdir -p "$d"
  ( cd "$d"
    export SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl
    export SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
    export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
    uenv run $U -- $B/cvfem_hex8_ns_ssgmg "$d/out" > "$d/run.log" 2>&1 )
  local ndof=$(grep -m1 -o 'ndof: [0-9]*' "$d/run.log" | awk '{print $2}')
  if [ -n "$ndof" ] && [ -f "$d/smesh.trace.csv" ]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "$label" "$d/smesh.trace.csv" "$ndof" "$OMP_NUM_THREADS" "$(hostname)" >> "$OUT/runs.tsv"
    printf "  %-16s ndof=%-9s t_solve=%s\n" "$label" "$ndof" \
      "$(grep -m1 -o 't_solve: [0-9.]* s' "$d/run.log" | awk '{print $2}')"
  else
    printf "  %-16s FAILED (no trace or no ndof)\n" "$label"
  fi
}

echo "=== flat HEX8 ==="
for n in 16 24 32 48 64; do one "flat_N$n" $n 1; done
echo "=== semi-structured, level 2 ==="
for n in 16 24 32 48 64; do one "ss_L2_N$((n/2))" $((n/2)) 2; done
echo "=== semi-structured, level 4 ==="
for n in 16 24 32 48 64; do one "ss_L4_N$((n/4))" $((n/4)) 4; done
echo "runs: $(wc -l < $OUT/runs.tsv)"
echo "=== DONE $(date) ==="
