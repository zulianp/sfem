#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-paper-table
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_paper_table_%j.out
#SBATCH --error=cvfem_paper_table_%j.err
#
# THE PAPER'S KERNEL-COST TABLE, REMEASURED.
#
# docs/CVFEM_NSE.tex section "What the terms cost" gives eleven throughputs on one 72-core
# Grace socket at 4,121,204 dofs, with the bare element sweep as the reference. Those numbers
# are what a reader takes away from the paper, and nothing in the suite re-measures them: the
# throughput gate covers twenty-one configurations at 8,586,756 dofs and a different set of
# terms, and docs/CVFEM_Kernels.md is at 3,650,692. So a refactor can move a paper number
# without any gate noticing.
#
# This job measures exactly the paper's rows, at the paper's size, in one allocation. Each row
# is run REPS times and the MEDIAN reported, because a single invocation on this machine carries
# a few percent and the first one in an allocation is systematically off.
#
# Both layouts, which is the standing rule for any rate in this repository: `packed` is the
# layout the paper's figures came from and `atomic` is the standard one, and a number for one
# alone is not a result. The assembly row's range in the paper (39-87) spans layouts, so it gets
# all four.
#
# 4,121,204 dofs is --n 100: 101^3 nodes x 4 fields.
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

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
N=${N:-100}
REPEAT=${REPEAT:-20}
REPS=${REPS:-5}
DT=${DT:-0.01}

# key|paper row|paper MDOF/s|flags
#
# The residual and action rows are CUMULATIVE, as the paper's "+" rows are: each adds a term to
# the one above it.
ROWS=(
  "res_bare|residual, interior flux only|2945|"
  "res_rc|+ Rhie-Chow, reconstruction reused|1759|--rhie-chow"
  "res_rc_bnd|+ boundary closure|1559|--rhie-chow --boundary"
  "res_rc_bnd_dt|+ transient|1436|--rhie-chow --boundary --transient $DT"
  "res_rc_bnd_dt_perapply|+ reconstruction rebuilt every evaluation|667|--rhie-chow --boundary --transient $DT --pgrad-per-apply"
  "jac_bare|Jacobian action, interior flux only|2067|--jac-action"
  "jac_rc|+ exact Rhie-Chow|605|--jac-action --rhie-chow"
  "jac_rc_bnd_dt|+ boundary closure and transient|561|--jac-action --rhie-chow --boundary --transient $DT"
  "bsr_apply|matrix-vector product Jv|462|--bsr-apply"
  "assemble|assembly of J|39-87|--assemble"
  "block_diag|block diagonal D|90|--assemble-diag"
)

# The HEADLINE rate -- the first `MDOF/s:` line -- which is the one the paper quotes. Not the
# last match: the driver also prints MDOF/s_element_visits, which counts an element's nodes once
# per element it belongs to and is six to eight times larger.
rate() {
    awk '/^  MDOF\/s:/ {print $2; exit} END {}'
}

median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }

echo "### paper kernel-cost table, remeasured"
echo "### host $(hostname)  threads $OMP_NUM_THREADS  --n $N  repeat $REPEAT  reps $REPS"
echo

# One throwaway invocation: the first in an allocation is systematically off.
cvfem_uenv "$B" --n 64 --repeat 3 --warmup 1 --layout packed > /dev/null 2>&1

for layout in packed atomic colored store; do
    echo "### layout $layout"
    printf '%-24s %-46s %10s %10s %8s\n' key "paper row" "paper" "measured" "change"
    for row in "${ROWS[@]}"; do
        IFS='|' read -r key label paper flags <<<"$row"
        # colored and store only carry the assembly in the paper's range
        if [ "$layout" != packed ] && [ "$layout" != atomic ] && [ "$key" != assemble ]; then continue; fi
        # --assemble-diag is the atomic diagonal whatever the layout says, and the driver
        # refuses any other spelling rather than quietly running something else.
        if [ "$key" = block_diag ] && [ "$layout" != atomic ]; then continue; fi
        vals=()
        for _ in $(seq 1 "$REPS"); do
            # shellcheck disable=SC2086
            v=$(cvfem_uenv "$B" --n "$N" --repeat "$REPEAT" --warmup 3 --layout "$layout" $flags 2>/dev/null | rate)
            [ "$v" = "-" ] && continue
            vals+=("$v")
        done
        if [ ${#vals[@]} -eq 0 ]; then
            printf '%-24s %-46s %10s %10s %8s\n' "$key" "$label" "$paper" "refused" "-"
            continue
        fi
        m=$(median "${vals[@]}")
        if { [ "$layout" = packed ] || [ "$key" = block_diag ]; } && [ "$paper" != "39-87" ]; then
            ch=$(awk -v m="$m" -v p="$paper" 'BEGIN{printf "%+.1f%%", (m-p)/p*100}')
        else
            ch="-"
        fi
        printf '%-24s %-46s %10s %10.1f %8s\n' "$key" "$label" "$paper" "$m" "$ch"
    done
    echo
done
echo "### done"
