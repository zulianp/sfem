#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-bnd-skip
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=bnd_%j.out
#SBATCH --error=bnd_%j.err
# The boundary closure, which swept the whole mesh to touch its skin.
#
# This one cannot be measured with the benchmark: its --boundary pass already skips
# elements whose face mask is zero (cvfem_hex8_layout_common.hpp). The solver's two
# passes did not, and the solver is where the 19% figure in docs/CVFEM_Throughput.md
# came from -- so the measurement has to come from the solver's own trace.
#
# ssgmg_B is built from the parent commit. Both runs are the same case, same size, and
# the numbers compared are the tracer's own per-scope totals, not wall clock, so the
# non-determinism of block-Jacobi on the flat path does not enter: what matters is the
# time per call of apply_boundary_scs_jacobian_action, not how many calls there were.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_bnd/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"

one() {  # one <binary> <label> <n>
  local bin=$1 label=$2 n=$3
  local d="$OUT/$label"; mkdir -p "$d"
  ( cd "$d"
    export SFEM_CASE=poiseuille SFEM_N=$n SFEM_GMG=0 SFEM_PRECOND=bjacobi
    export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0 SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
    uenv run $U -- "$B/$bin" "$d/out" > run.log 2>&1 )
  local ndof; ndof=$(grep -m1 -o 'ndof: [0-9]*' "$d/run.log" | awk '{print $2}')
  printf "  %-22s ndof=%-9s " "$label" "${ndof:-?}"
  if [ -f "$d/smesh.trace.csv" ]; then
    awk -F, -v OFS=' ' '
      $1 ~ /apply_boundary_scs_jacobian_action/ { bj_c=$2; bj_t=$3 }
      $1 ~ /apply_boundary_scs_residual/        { br_c=$2; br_t=$3 }
      $1 ~ /apply_jacobian_action_packed/       { ja_c=$2; ja_t=$3 }
      $1 ~ /assemble_nodal_q_grad/              { qg_c=$2; qg_t=$3 }
      END {
        printf "bnd_jac=%.4fs/%d calls (%.1f us/call)  elem_sweep=%.4fs  qgrad=%.4fs  bnd_share=%.1f%%\n",
               bj_t, bj_c, (bj_c>0 ? 1e6*bj_t/bj_c : 0), ja_t, qg_t,
               (bj_t+ja_t+qg_t > 0 ? 100.0*bj_t/(bj_t+ja_t+qg_t) : 0)
      }' "$d/smesh.trace.csv"
  else
    echo "no trace"
  fi
}

for n in 48 96; do
  echo
  echo "=== N=$n ==="
  one ssgmg_B                "before_N$n" $n
  one cvfem_hex8_ns_ssgmg    "after_N$n"  $n
done

echo
echo "done $(date)"
