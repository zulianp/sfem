#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-det-layout
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=det_layout_%j.out
#SBATCH --error=det_layout_%j.err

# IS THE OPERATOR BIT-REPRODUCIBLE, PER LAYOUT? (paper table T2)
#
# The claim the packed format makes is that its result does not depend on thread timing: the
# summation order for a node is fixed by the index arrays rather than by which thread arrives
# first. The atomic layout fixes no order, so floating-point addition being non-associative
# makes it irreproducible against itself. This job measures that instead of asserting it.
#
# Two questions per (layout, operation), and they are different questions:
#   A. repeats at a FIXED thread count -- does the same binary on the same input give the same
#      bits twice? An atomic sweep can fail this on its own.
#   B. across THREAD COUNTS -- does the answer depend on how many threads ran? This is the one
#      that matters for reproducible science, and it is the stronger claim.
# Both are reported as "distinct fingerprints / runs". 1/N is reproducible; anything else is not.
#
# WHY THE FINGERPRINT AND NOT THE CHECKSUM. The driver prints both. The checksum is a running
# sum of millions of values, so a last-bit difference on a handful of nodes falls below the ULP
# of the total and rounds away -- it reported two closures as identical when a per-node
# comparison of the same operator showed hundreds of doubles differing. The fingerprint folds
# every value's bit pattern in (bench l.1654-1673), so any single changed bit changes it.
#
# FINGERPRINTS ARE COMPARABLE ONLY WITHIN ONE LAYOUT, AND THIS IS NOT A DETAIL.
# PackedMesh::create renumbers the mesh nodes IN PLACE, so a packed run and an atomic run fold
# their node arrays in different orders and will disagree even when both are perfectly
# reproducible. Comparing them would measure the renumbering, not the determinism. The
# equivalence of the two operators is a separate question, and it is already gated by
# tests/cvfem_packed_vs_atomic_test.cpp, which keys nodes BY COORDINATE for exactly this reason.
# So: rows are compared down a column (within a layout), never across.
#
# The operators carry their solver terms (Rhie-Chow and the boundary closure) rather than being
# bare element kernels, because a bare kernel is not what a solver applies and the bare
# comparison overstates the layout margin by 1.5-2x.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
OUT=${OUT:-$SCRATCH/cvfem_det_layout/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

N=${N:-100}
REPEATS=${REPEATS:-3}
THREAD_LIST=${THREAD_LIST:-"1 8 36 72"}
FIXED_THREADS=${FIXED_THREADS:-72}

echo "### host $(hostname)  n=$N  repeats=$REPEATS at ${FIXED_THREADS}t  thread sweep: $THREAD_LIST"
echo "### fingerprints are comparable only DOWN a column (within one layout): packing renumbers"
echo

# One run; echoes the fingerprint or a dash. Kept as a function so the two questions below
# cannot drift apart in how they invoke the binary.
run_fp() {
    local lbl=$1 threads=$2 layout=$3 op=$4
    local log=$OUT/${lbl}.log
    local flags="--layout $layout --rhie-chow --boundary"
    [ "$op" = jac ] && flags="$flags --jac-action"
    timeout "${RUN_TIMEOUT:-240}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=$threads OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat 1 --warmup 0 $flags" \
      > "$log" 2>&1
    local fp
    fp=$(grep -am1 'fingerprint:' "$log" | awk '{print $2}')
    echo "${fp:--}"
}

# How many distinct values a list holds. The verdict is this number, not the values.
distinct() { printf '%s\n' "$@" | sort -u | grep -avc '^-$' ; }

printf '%-9s %-5s %-28s %-28s\n' layout op "A: repeats @${FIXED_THREADS}t" "B: across threads"
for layout in atomic packed colored ecolor; do
for op in res jac; do
    fps_a=()
    for i in $(seq 1 "$REPEATS"); do
        fps_a+=("$(run_fp "${layout}_${op}_rep$i" "$FIXED_THREADS" "$layout" "$op")")
    done
    fps_b=()
    for t in $THREAD_LIST; do
        fps_b+=("$(run_fp "${layout}_${op}_t$t" "$t" "$layout" "$op")")
    done
    da=$(distinct "${fps_a[@]}"); db=$(distinct "${fps_b[@]}")
    # A missing fingerprint means a run failed; say so rather than counting it as agreement.
    na=$(printf '%s\n' "${fps_a[@]}" | grep -avc '^-$'); nb=$(printf '%s\n' "${fps_b[@]}" | grep -avc '^-$')
    printf '%-9s %-5s %-28s %-28s\n' "$layout" "$op" \
           "$da distinct / $na runs$([ "$na" -lt "$REPEATS" ] && echo ' (INCOMPLETE)')" \
           "$db distinct / $nb runs$([ "$nb" -lt "$(echo "$THREAD_LIST" | wc -w | tr -d ' ')" ] && echo ' (INCOMPLETE)')"
done
done

echo
echo "1 distinct = bit-reproducible. Compare down a column only."
echo "raw fingerprints:"
grep -aH 'fingerprint:' "$OUT"/*.log 2>/dev/null | sed "s|$OUT/||" | sort
echo "logs in $OUT"
