#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-psl
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=psl_%j.out
#SBATCH --error=psl_%j.err
#
# Does pack size matter to a semi-structured run, and does the answer depend on the level?
#
# It cannot touch the semi-structured kernels themselves: initialize() takes the
# semi-structured branch and returns at cvfem_hex8_ns_op.cpp:417, before the packing block at
# 435, so no PackedMesh is ever built for them.
#
# But a semi-structured multigrid hierarchy derefines toward the macro mesh, and THAT level is
# flat and therefore packed. A macro mesh has few elements -- at N=8 it is 512 -- so a pack
# size of 2048 makes it one pack, and one pack is one thread. If that matters, a semi-
# structured run is carrying a serial coarse operator and the pack size is the wrong knob
# rather than an absent one.
#
# Swept across levels because the number of macro elements at fixed fine mesh falls as the
# level rises: at N*L = 64, level 2 leaves 32768 macro elements and level 8 leaves 512. So if
# the effect exists at all, it must grow with the level.
set -u
D=${DRIVER:?set DRIVER}
OUT=${OUT:-$SCRATCH/cvfem_psl/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  $(date)"
printf "%-24s %-9s %-10s %-7s %-9s %s\n" run ndof macro_elems conv lin_it t_solve

one() {  # one <tag> <n> <level> <packsize>
    local tag=$1 n=$2 lvl=$3 ps=$4
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && timeout 380 uenv run $U -- env OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=cavity_reg SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_GMG=1 \
        SFEM_PACK_SIZE=$ps SFEM_MU=0.002 SFEM_NL_MAX_IT=30 SFEM_LSOLVE_MAX_IT=3000 \
        SFEM_ENABLE_OUTPUT=0 \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd c li ts me
    nd=$( grep -o 'ndof: [0-9]*'           "$r/log" | head -1 | awk '{print $2}')
    c=$(  grep -o 'newton_converged: [01]' "$r/log" | head -1 | awk '{print $2}')
    li=$( grep -o 'lin_it_total: [0-9]*'   "$r/log" | head -1 | awk '{print $2}')
    ts=$( grep -o 't_solve: [0-9.]*'       "$r/log" | head -1 | awk '{print $2}')
    me=$(awk -v n=$n 'BEGIN{printf "%d", n*n*n}')
    printf "%-24s %-9s %-10s %-7s %-9s %s\n" "$tag" "${nd:-?}" "$me" "${c:-?}" "${li:-?}" "${ts:-?}"
}

# Level 8: 512 macro elements, so 2048 is one pack and 128 is four.
for ps in 0 128 512 2048; do one "L8_ps$ps" 8 8 $ps; done
# Level 2 at the same fine mesh: 32768 macro elements, where 2048 is 16 packs and the
# coarse level is not starved. If the effect is real it should be absent here.
for ps in 0 2048; do one "L2_ps$ps" 32 2 $ps; done
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
