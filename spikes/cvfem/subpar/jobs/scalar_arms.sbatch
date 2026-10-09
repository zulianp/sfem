#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-scalar-arms
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_scalar_arms_%j.out
#SBATCH --error=cvfem_scalar_arms_%j.err
#
# WHICH ISOPARAMETRIC SCALAR MICRO-KERNEL SURVIVES, AND WHETHER `split` EARNS ITS PATH.
#
# DESIGN.md's correction leaves one micro-kernel per kernel, the one Grace measured fastest.
# Three pairs reach this driver with no head-to-head measurement anywhere, and subpar/README.md
# is explicit that the campaign which retired the affine generated residual "measured affine
# geometry only, and nothing here is evidence about" the isoparametric one:
#
#   --geom isoparam --layout atomic, residual    --kernel current  vs  --kernel sympy
#   --geom isoparam --layout atomic, assembly    --kernel current  vs  --kernel sympy
#   --kernel split                               assembly that reuses a frozen linear half,
#                                                against the full assembly it would replace
#
# `split` is included because it is an unmeasured second path to the same assembled matrix, and
# the only honest way to retire it is against a number. Its claimed advantage appears across
# REPEATED assembly -- the linear half is assembled once and reused -- so --repeat is what
# exposes it, and the arm is run at the same --repeat as the full assembly beside it.
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

B="$CVFEM_SRC/build/cvfem_hex8_ns_upwind_bench"
[ -x "$B" ] || { echo "bench not built: $B" >&2; exit 2; }
OUT="$CVFEM_SRC/scalar_out"; mkdir -p "$OUT"
CSV="$OUT/scalar.csv"
rm -f "$OUT/table.txt"

N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### iso_* are --geom isoparam --layout atomic; split_* are affine unless named iso"
echo
printf '%-26s %10s %10s %14s\n' arm MDOF_s dofs checksum

for arm in "iso_res_current|--geom isoparam --kernel current" \
           "iso_res_sympy|--geom isoparam --kernel sympy" \
           "iso_asm_current|--assemble --geom isoparam --kernel current" \
           "iso_asm_sympy|--assemble --geom isoparam --kernel sympy" \
           "aff_asm_sumfact|--assemble --kernel sumfact" \
           "aff_asm_split|--assemble --kernel split" \
           "iso_asm_split|--assemble --geom isoparam --kernel split"; do
    IFS='|' read -r lbl extra <<<"$arm"
    best=""; dofs=""; ck=""
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_p${pass}.log
        # A PER-ARM TIMEOUT, so one wedged arm cannot take the whole wall clock with it.
        timeout "${RUN_TIMEOUT:-360}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 --layout atomic $extra --csv $CSV --tag ${lbl}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs'); ck=$(val 'checksum')
    done
    # The rate is printed as it arrives, not at the end: these runs are long and a filter that
    # holds the table back until the last arm hides a wedged one until the job is over.
    printf '%-26s %10s %10s %14s\n' "$lbl" "${best:-FAIL}" "${dofs:-?}" "${ck:-?}" | tee -a "$OUT/table.txt"
done
echo
echo "### pairs (the second name is the arm that would be retired):"
awk '{r[$1]=$2} END {
  split("iso_res_current:iso_res_sympy iso_asm_current:iso_asm_sympy aff_asm_sumfact:aff_asm_split", P, " ");
  for (i in P) { split(P[i], q, ":");
    if (q[1] in r && q[2] in r && r[q[2]]+0 > 0)
      printf "  %-16s %-9s vs %-16s %-9s  ratio=%.3f\\n", q[1], r[q[1]], q[2], r[q[2]], r[q[1]]/r[q[2]] }
}' "$OUT/table.txt"
echo "### done $(date '+%F %T')"
