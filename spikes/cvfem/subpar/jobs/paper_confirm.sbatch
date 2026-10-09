#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-paper-confirm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_paper_confirm_%j.out
#SBATCH --error=cvfem_paper_confirm_%j.err
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"
B="$CVFEM_SRC/build/cvfem_hex8_ns_upwind_bench"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
N=100; REPEAT=20; REPS=11; DT=0.01
rate() { awk '/^  MDOF\/s:/ {print $2; exit}'; }
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
spread() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {printf "%.1f", (a[NR]-a[1])/a[1]*100}'; }
cvfem_uenv "$B" --n 64 --repeat 3 --warmup 1 --layout packed > /dev/null 2>&1
# The two rows that came out negative, at eleven repetitions, to separate a change from noise.
# Plus the LAGGED-RC action: the assembled matrix encodes the frozen Rhie-Chow form, so that is
# the matrix-free operator carrying the same physics as --bsr-apply, and it is what the paper's
# "slower by 1.25" sentence should be comparing against.
echo "### key                       median   spread(11)"
for row in "res_bare|" "bsr_apply|--bsr-apply" "jac_bare|--jac-action" \
           "jac_lagged_rc_bnd_dt|--jac-action --lagged-rc --rhie-chow --boundary --transient $DT"; do
    IFS='|' read -r key flags <<<"$row"
    vals=()
    for _ in $(seq 1 $REPS); do
        # shellcheck disable=SC2086
        vals+=("$(cvfem_uenv "$B" --n $N --repeat $REPEAT --warmup 3 --layout packed $flags 2>/dev/null | rate)")
    done
    printf '%-28s %8.1f   %6s%%\n' "$key" "$(median "${vals[@]}")" "$(spread "${vals[@]}")"
done
echo "### done"
