#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-stepbud
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=stepbud_%j.out
#SBATCH --error=stepbud_%j.err
#
# THE NUMERICAL DISSIPATION WHERE IT IS SUPPOSED TO BE LARGE.
#
# The Poiseuille control in the verification matrix shows eps_num vanishing at second order,
# which is what a resolved laminar flow must do and is the only reason the number is
# believable anywhere else. It says nothing about how big eps_num is when the flow has a
# separated shear layer in it -- and that is the quantity the whole first step of this work
# exists to produce, because the convection operator's |mdot| term IS this scheme's subgrid
# model and nothing had ever measured it.
#
# The backward-facing step is the one non-box domain here. It separates at the lip, so the
# shear layer sits across cells rather than along them and the cell Peclet number is large
# exactly where the gradients are.
#
# TWO AXES, AND THEY MUST NOT BE CONFOUNDED:
#
#   at fixed Reynolds number, REFINING must REDUCE eps_num -- it is a truncation term;
#   at fixed mesh, raising Re must RAISE eps_num/(eps_visc + eps_num) -- the cell Peclet
#   number is what makes the upwind term bite.
#
# Reporting one without the other cannot distinguish "the scheme is dissipative" from "this
# run was under-resolved", which is the distinction the measurement is for.
#
# The third axis is the SCHEME. Every cell runs first-order and deferred-correction, because
# the point of eps_num is to be the thing a better convection scheme reduces. If the
# correction does not reduce it here, then either the correction is not doing what the MMS
# ladder says it is, or eps_num is not measuring what it claims to.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "stepbud: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_stepbud/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# Nothing inherited. SFEM_FGMRES, SFEM_GMG and the scheme knobs are all named, because every
# one of them has silently decided a result at some point in this work.
# FGMRES + Vanka, no multigrid. Not a preference: measured, the step case on this geometry
# does NOT converge under multigrid -- the first run of this job read conv=0 after 6 Newton
# steps and 7,000 linear iterations at Re = 40 -- and a budget read off an unconverged state
# is not a measurement of anything. Vanka is the stack the port study established converges
# without multigrid, and taking the solver out of the question is the same reason the step
# group in the verification matrix uses a dense LU.
BASE="SFEM_CASE=step SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0"

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    stdbuf -oL -eL env $BASE "$@" SFEM_DIAG_CSV="$OUT/$tag.csv" \
        uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv ev en cl nt li
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    # From the CSV rather than the echoed line: the echo shares stdout with the driver's
    # other output and interleaves with stderr under load, which has already truncated one
    # number in this work.
    if [ -s "$OUT/$tag.csv" ]; then
        ev=$(tail -1 "$OUT/$tag.csv" | cut -d, -f8)
        en=$(tail -1 "$OUT/$tag.csv" | cut -d, -f9)
        cl=$(tail -1 "$OUT/$tag.csv" | cut -d, -f10)
    fi
    local frac="--"
    [ -n "${ev:-}" ] && [ -n "${en:-}" ] && frac=$(python3 -c "
ev=abs(float('${ev:-0}')); en=abs(float('${en:-0}'))
print('%.4f' % (en/(ev+en)) if (ev+en) > 0 else '--')" 2>/dev/null || echo "--")
    printf '%-22s %9s %5s %7s %8s %13s %13s %9s %9s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${ev:---}" "${en:---}" "$frac" "${cl:---}"
}
HDR() { printf '%-22s %9s %5s %7s %8s %13s %13s %9s %9s\n' \
        cell ndof conv newton lin_it eps_visc eps_num num_frac closure; }

# Re = rho U Ly / mu with Ly = 2: mu 0.05 is Re 40, 0.02 is Re 100, 0.01 is Re 200.
for MU in 0.05 0.02 0.01; do
    RE=$(python3 -c "print(int(1*1*2/$MU))")
    echo "== Re = $RE  (mu = $MU)"
    HDR
    # N must be EVEN. create_hex8_lshape refuses a step that does not fall on a grid line --
    # step_y/ymax*ny must be an integer, and with ymax = 2 and step_y = 1 that means ny even.
    # N=1 aborts in the mesh generator, which is how this job first failed.
    for NL in "2:4" "4:4" "4:8"; do
        NN=${NL%%:*}; LV=${NL##*:}
        # A dense LU is affordable at 7,060 dofs and refused above 20,000, so the smallest
        # mesh removes the linear solver entirely and the larger two use Vanka.
        if [ "$NN" = 2 ]; then PC="SFEM_PRECOND=direct"; else PC="SFEM_PRECOND=vanka"; fi
        for HO in 0 1; do
            run_one "re${RE}_n${NN}l${LV}_ho${HO}" SFEM_MU=$MU SFEM_N=$NN \
                SFEM_ELEMENT_REFINE_LEVEL=$LV $PC SFEM_CONV_HO=$HO SFEM_CONV_LIMITER=1
        done
    done
    echo
done
echo "logs in $OUT"
echo "done $(date)"
