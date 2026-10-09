#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-tet4-arms
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_tet4_arms_%j.out
#SBATCH --error=cvfem_tet4_arms_%j.err
#
# WHICH TET4 MICRO-KERNEL SURVIVES.
#
# DESIGN.md's correction leaves one micro-kernel per kernel, the one Grace measured fastest.
# The TET4 benchmark carries THIRTEEN arms behind its own --kernel flag -- a private enum, not
# the HEX8 one -- and `perf/` holds no TET4 row at all: every one of them is unmeasured on this
# hardware, so there is nothing on record to reduce them by. Hence this job, before anything is
# retired. The default is sympy_row_simd, which is also unmeasured.
#
# Three operations, because a residual winner need not be an assembly winner -- that is exactly
# what the HEX8 campaign found, where the generated arrangement wins the atomic assembly and
# loses the packed residual.
#
# Interleaved, best-of-three, one allocation: the reasons conv_ho_bench.sbatch records.
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script.
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"

B="$CVFEM_SRC/build/cvfem_tet4_ns_upwind_bench"
[ -x "$B" ] || { echo "bench not built: $B" >&2; exit 2; }
OUT="$CVFEM_SRC/tet4_out"; mkdir -p "$OUT"
CSV="$OUT/tet4.csv"
rm -f "$OUT/table.txt"

N=${N:-96}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### res_/asm_/jac_ are the residual, the assembly and the matrix-free action"
echo
printf '%-26s %10s %10s %14s\n' arm MELEM_s dofs checksum

for arm in "res_current|--kernel current" \
           "res_sympy|--kernel sympy" \
           "res_current_slots|--kernel current_slots" \
           "res_sympy_slots|--kernel sympy_slots" \
           "res_sympy_direct|--kernel sympy_direct" \
           "res_sympy_block|--kernel sympy_block" \
           "res_sympy_face|--kernel sympy_face" \
           "res_sympy_simd|--kernel sympy_simd" \
           "res_sympy_simd_clean|--kernel sympy_simd_clean" \
           "res_sympy_block_simd|--kernel sympy_block_simd" \
           "res_sympy_row_simd|--kernel sympy_row_simd" \
           "res_sympy_row_simd_fused|--kernel sympy_row_simd_fused" \
           "res_sympy_face_simd|--kernel sympy_face_simd" \
           "asm_current|--kernel current --assemble" \
           "asm_sympy|--kernel sympy --assemble" \
           "asm_current_slots|--kernel current_slots --assemble" \
           "asm_sympy_slots|--kernel sympy_slots --assemble" \
           "asm_sympy_direct|--kernel sympy_direct --assemble" \
           "asm_sympy_block|--kernel sympy_block --assemble" \
           "asm_sympy_face|--kernel sympy_face --assemble" \
           "asm_sympy_simd|--kernel sympy_simd --assemble" \
           "asm_sympy_simd_clean|--kernel sympy_simd_clean --assemble" \
           "asm_sympy_block_simd|--kernel sympy_block_simd --assemble" \
           "asm_sympy_row_simd|--kernel sympy_row_simd --assemble" \
           "asm_sympy_row_simd_fused|--kernel sympy_row_simd_fused --assemble" \
           "asm_sympy_face_simd|--kernel sympy_face_simd --assemble" \
           "jac_current|--kernel current --jac-action" \
           "jac_sympy|--kernel sympy --jac-action" \
           "jac_current_slots|--kernel current_slots --jac-action" \
           "jac_sympy_slots|--kernel sympy_slots --jac-action" \
           "jac_sympy_direct|--kernel sympy_direct --jac-action" \
           "jac_sympy_block|--kernel sympy_block --jac-action" \
           "jac_sympy_face|--kernel sympy_face --jac-action" \
           "jac_sympy_simd|--kernel sympy_simd --jac-action" \
           "jac_sympy_simd_clean|--kernel sympy_simd_clean --jac-action" \
           "jac_sympy_block_simd|--kernel sympy_block_simd --jac-action" \
           "jac_sympy_row_simd|--kernel sympy_row_simd --jac-action" \
           "jac_sympy_row_simd_fused|--kernel sympy_row_simd_fused --jac-action" \
           "jac_sympy_face_simd|--kernel sympy_face_simd --jac-action"; do
    IFS='|' read -r lbl extra <<<"$arm"
    best=""; dofs=""; ck=""
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_p${pass}.log
        # A PER-ARM TIMEOUT, so one wedged arm cannot take the whole wall clock with it.
        timeout "${RUN_TIMEOUT:-360}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 --layout packed $extra --csv $CSV --tag ${lbl}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MELEM/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs'); ck=$(val 'checksum')
    done
    # The rate is printed as it arrives, not at the end: these runs are long and a filter that
    # holds the table back until the last arm hides a wedged one until the job is over.
    printf '%-26s %10s %10s %14s\n' "$lbl" "${best:-FAIL}" "${dofs:-?}" "${ck:-?}" | tee -a "$OUT/table.txt"
done
echo
echo "### fastest per operation:"
awk '{split($1,q,"_"); op=q[1]; k=substr($1,length(op)+2);
      if ($2+0 > best[op]+0) { best[op]=$2; who[op]=k } }
     END { for (o in best) printf "  %-4s %-22s %s\\n", o, who[o], best[o] }' "$OUT/table.txt"
echo "### done $(date '+%F %T')"
