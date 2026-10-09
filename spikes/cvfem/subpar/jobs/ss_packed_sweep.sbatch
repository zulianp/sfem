#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-sspack
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=sspack_%j.out
#SBATCH --error=sspack_%j.err
#
# What packing the macro-element mesh is worth, against the scatter it replaces.
#
# The semi-structured scatter stages every macro-element face, edge and corner, because a node
# between two macro-elements cannot be written by either alone. Grouping them into packs makes
# such a node pack-owned, so only the pack's outer boundary is staged: measured on a 64-macro
# mesh at eight per pack, 96.3 / 78.4 / 52.9 percent at levels 2, 4 and 8 becomes
# 48.1 / 24.6 / 12.4. The gradient's cost tracks that fraction almost exactly across levels,
# which is why this is the lever and the arithmetic was not.
#
# N*L is held at 64 so every row is the same 4,343,300 dof and the same fine mesh. The flat
# row is the target; pack size 0 is the scatter path, which is the thing to beat.
#
# Pack size is capped by arithmetic: pack_idx_t is uint16_t, so a pack holds at most 65535
# nodes and a macro-element carries (L+1)^3 -- 729 at level 8, allowing 89. The operator
# clamps and says so rather than overflowing.
set -u
D=${DRIVER:?set DRIVER}
OUT=${OUT:-$SCRATCH/cvfem_sspack/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo "out  $OUT"

one() {  # one <tag> <n> <level> <sspack>
    local tag=$1 n=$2 lvl=$3 ps=$4
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r"
      export SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_SS_PACK_SIZE=$ps
      export SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
      export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
      timeout 500 uenv run $U -- "$D" "$r/out" > "$r/log" 2>&1 )
    local ndof; ndof=$(grep -m1 -o 'ndof: [0-9]*' "$r/log" | awk '{print $2}')
    printf "  %-16s ndof=%-9s %-28s %s\n" "$tag" "${ndof:-?}" \
        "$(grep -m1 '^ss: packed' "$r/log" | cut -c1-28)" \
        "$(grep -m1 -o 't_solve: [0-9.]* s' "$r/log" | awk '{print "t_solve="$2}')"
}

one flat_N64      64 1 0
for lvl in 2 4 8; do
    n=$(( 64 / lvl ))
    one "ssL${lvl}_scatter" $n $lvl 0
    for ps in 4 8 16 32; do one "ssL${lvl}_pack$ps" $n $lvl $ps; done
done
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
