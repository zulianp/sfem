#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-paper-ho
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_paper_ho_%j.out
#SBATCH --error=cvfem_paper_ho_%j.err
#
# THE HIGHER-ORDER CONVECTIVE FLUX, at the size and layout the paper's cost table uses.
#
# docs/CVFEM_NSE.tex's table carries no higher-order row at all: every operator in it runs the
# first-order upwind convective flux. The deferred correction is the scheme the solver runs when
# accuracy matters, it is four to seven times the arithmetic, and it is the one part of the
# operator family the paper costs nothing for.
#
# The throughput gate does cover six of these rows, so the question "did they regress" already
# has an answer there. What it cannot give is a number at the paper's size next to the
# first-order rows it should be read against.
#
# TWO CAVEATS THAT DECIDE HOW THESE NUMBERS MAY BE USED.
#
# First, a higher-order packed rate is NOT comparable with a first-order one. The correction runs
# through the SCALAR sum-factored kernel -- the only one that accepts the nodal velocity gradient
# -- while the first-order packed residual is 16-wide SIMD. So the honest comparison for a
# limiter is against another limiter, or against the first-order rate on the same scalar kernel,
# and the table below reports the ratio only within the higher-order block.
#
# Second, the rate line the driver prints for these is MDOF/s_residual or MDOF/s_jac_action.
# Taking the headline `MDOF/s:` line works for the first-order rows and silently mismatches
# here, which is how a higher-order row can end up reporting an element-visit rate six times too
# large.
#
# A ROW MAY COME BACK `refused`, AND ON GRACE TWO OF THEM DO. That is not an unsupported
# option: it is the driver's own packed-versus-atomic oracle rejecting the answer and exiting
# non-zero. Measured and bisected (jobs 4985427/31/33/39):
#
#   * only the exact higher-order JACOBIAN ACTION on the PACKED layout, and only with the two
#     smooth limiters -- Venkatakrishnan 1.7e-04, Darwish-Moukalled 2.8e-05. Unlimited and the
#     bounded-face clip are clean at 7e-16, the residual is clean, and --lagged-ho (which drops
#     the correction's own derivative) is clean at 6.8e-16. So it is the limiter's derivative.
#   * only at cube sizes that are NOT a power of two: 72, 80, 88, 96, 100, 104 and 112 all fail;
#     64 and 128 pass.
#   * independent of thread count -- it fails at one thread -- and of the pack size and the
#     space-filling-curve ordering, which change the magnitude and not the verdict.
#   * GRACE ONLY. The development machine is clean at every one of those sizes, 5.9e-16 to
#     8.9e-16, which is why nothing local sees it.
#   * PRE-EXISTING. A binary built from caae5d606, before any of this session's work, returns
#     the identical 1.680453e-04.
#
# Nothing gated it: the ctest higher-order checks are residual-only and run at --n 8, and the
# throughput gate's jac_action_packed_ho_venk row runs at --n 128 -- both powers of two, which
# is exactly where the oracle passes.
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: Slurm copies the batch script.
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"
B="$CVFEM_SRC/build/cvfem_hex8_ns_upwind_bench"
[ -x "$B" ] || { echo "bench not built: $B" >&2; exit 2; }

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
N=${N:-100}; REPEAT=${REPEAT:-20}; REPS=${REPS:-5}; DT=${DT:-0.01}

# The operation-specific rate line, and the model flop count beside it so a row that did not
# actually carry the correction is visible rather than merely slow.
rate() { awk '/MDOF\/s_residual:|MDOF\/s_jac_action:/{r=$2} /flops_per_element.*model:/{f=$2} END{if(r=="")r="-"; if(f=="")f="-"; print r, f}'; }
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }

ROWS=(
  "res_fo|residual, first-order upwind (reference)|--rhie-chow"
  "res_ho_unlim|+ deferred correction, unlimited|--rhie-chow --conv-ho 0"
  "res_ho_clip|+ deferred correction, bounded-face clip|--rhie-chow --conv-ho 1"
  "res_ho_venk|+ deferred correction, Venkatakrishnan|--rhie-chow --conv-ho 2"
  "res_ho_dm|+ deferred correction, Darwish-Moukalled|--rhie-chow --conv-ho 3"
  "res_ho_venk_bnd_dt|+ Venkatakrishnan, boundary and transient|--rhie-chow --conv-ho 2 --boundary --transient $DT"
  "jac_fo|Jacobian action, first-order upwind (reference)|--jac-action --rhie-chow"
  "jac_ho_unlim|+ deferred correction, unlimited|--jac-action --rhie-chow --conv-ho 0"
  "jac_ho_venk|+ deferred correction, Venkatakrishnan|--jac-action --rhie-chow --conv-ho 2"
  "jac_ho_venk_bnd_dt|+ Venkatakrishnan, boundary and transient|--jac-action --rhie-chow --conv-ho 2 --boundary --transient $DT"
)

echo "### higher-order convective flux at the paper's size"
echo "### host $(hostname)  threads $OMP_NUM_THREADS  --n $N (4,121,204 dofs)  repeat $REPEAT  reps $REPS"
cvfem_uenv "$B" --n 64 --repeat 3 --warmup 1 --layout packed > /dev/null 2>&1

for layout in packed atomic; do
    echo
    echo "### layout $layout"
    printf '%-22s %-46s %10s %12s %10s\n' key row "MDOF/s" "flops/elem" "vs 1st-ord"
    ref=""
    for row in "${ROWS[@]}"; do
        IFS='|' read -r key label flags <<<"$row"
        vals=(); fl="-"
        for _ in $(seq 1 "$REPS"); do
            # shellcheck disable=SC2086
            read -r v f <<<"$(cvfem_uenv "$B" --n "$N" --repeat "$REPEAT" --warmup 3 --layout "$layout" $flags 2>/dev/null | rate)"
            [ "$v" = "-" ] && continue
            vals+=("$v"); fl="$f"
        done
        if [ ${#vals[@]} -eq 0 ]; then
            # Say WHY rather than `refused`: the driver's oracle prints the disagreement it
            # rejected, and that is the useful half of this row.
            # shellcheck disable=SC2086
            why=$(cvfem_uenv "$B" --n "$N" --repeat 1 --warmup 0 --layout "$layout" $flags 2>&1 |
                      awk '/_vs_atomic_rel:/{printf "%s %s", $1, $2}')
            printf '%-22s %-46s %10s %12s %10s  %s\n' "$key" "$label" rejected - - "${why:-no rate printed}"
            continue
        fi
        m=$(median "${vals[@]}")
        case $key in res_fo|jac_fo) ref=$m; r="1.00" ;;
                     *) r=$(awk -v a="$ref" -v b="$m" 'BEGIN{printf "%.2fx", a/b}') ;;
        esac
        printf '%-22s %-46s %10.1f %12s %10s\n' "$key" "$label" "$m" "$fl" "$r"
    done
done
echo
echo "### done"
