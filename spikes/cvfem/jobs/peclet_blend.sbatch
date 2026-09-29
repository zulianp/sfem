#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-peclet
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=peclet_%j.out
#SBATCH --error=peclet_%j.err
#
# DOES SWITCHING THE UPWIND TERM OFF FOR MOMENTUM WORK HERE.
#
# Nalu and Nalu-Wind run this same discretisation and do not choose between upwind and central
# at all: they blend by cell Peclet, phi_ip = eta phi_upw + (1 - eta) phi_cds, and for VELOCITY
# their tanh blend is centred at Peclet 50,000 with width 200 against 2 and 1 for every other
# scalar. At any Peclet a real flow reaches, eta is zero -- the momentum equations run
# essentially unstabilised central and upwinding is kept for scalars.
#
# That is a far larger lever on eps_num than reconstructing the upwind term more accurately.
# Measured on this same case, the deferred correction moves eps_num by 2.6% to 17.6%; this
# removes the term.
#
# WHETHER IT IS STABLE HERE IS THE EXPERIMENT AND NOT A DEDUCTION. Colocated central
# differencing is viable because Rhie-Chow supplies the pressure-velocity coupling, which this
# code has. But Nalu is a projection scheme with an LES subgrid model carrying the dissipation,
# and this is a monolithic Newton formulation with neither. Three things can happen and all
# three are results:
#
#   it converges and eps_num collapses         -- the upwind term was the subgrid model, and
#                                                 the budget now has almost nothing in it;
#   it converges and eps_num barely moves      -- the dissipation was never in that term;
#   it does not converge                       -- central momentum needs what Nalu has and we
#                                                 do not, and the eta sweep says where it breaks.
#
# So the blend is SWEPT rather than switched: a ladder of transition Peclet numbers walks eta
# from 1 to 0 continuously, with the endpoints as controls. A single on/off pair could only
# distinguish the first outcome from the third.
#
# The Jacobian keeps the full first-order upwind dissipation in every arm. That is defect
# correction, the structure the deferred correction already uses, and the literature's reason
# for it is that a blend's contribution to an implicit Jacobian cannot readily be evaluated.
# An upwinded Jacobian for a less-upwinded residual is also the better-conditioned pairing.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "peclet: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_peclet/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

NN=${NN:-4}; LV=${LV:-4}
BASE="SFEM_CASE=step SFEM_U=1 SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0
      SFEM_N=$NN SFEM_ELEMENT_REFINE_LEVEL=$LV SFEM_CONV_HO=0"

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log" csv="$OUT/$tag.csv"
    local t0=$SECONDS
    stdbuf -oL -eL env $BASE "$@" SFEM_DIAG_CSV="$csv" \
        uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv nt li ev en sre
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | tail -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | tail -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | tail -1)
    sre=$(sed -n 's/.*highest Re SOLVED = \([0-9.e+-]*\) .*/\1/p' "$log" | head -1)
    if [ -s "$csv" ]; then
        ev=$(tail -1 "$csv" | cut -d, -f8); en=$(tail -1 "$csv" | cut -d, -f9)
    fi
    local frac="--"
    [ -n "${ev:-}" ] && [ -n "${en:-}" ] && frac=$(python3 -c "
ev=abs(float('${ev:-0}')); en=abs(float('${en:-0}'))
print('%.4f' % (en/(ev+en)) if (ev+en) > 0 else '--')" 2>/dev/null || echo "--")
    printf '%-18s %9s %5s %7s %9s %9s %13s %9s %7s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${sre:---}" \
        "${en:---}" "$frac" "$((SECONDS - t0))s"
}
printf '%-18s %9s %5s %7s %9s %9s %13s %9s %7s\n' \
    arm ndof conv newton lin_it stall_Re eps_num num_frac wall

# The control: no blend at all, the scheme as it stands.
run_one upwind
# The ladder. Pe_trans walks the transition point down through the Peclet numbers this flow
# actually contains, so eta goes from ~1 (upwind everywhere) to ~0 (central everywhere). The
# width is kept at a fifth of the transition so the blend stays smooth rather than a switch.
for PT in 1000 200 40 8 2; do
    PW=$(python3 -c "print(max($PT/5.0, 0.5))")
    run_one "tanh_pe${PT}" SFEM_PECLET_BLEND=tanh SFEM_PECLET_TRANS=$PT SFEM_PECLET_WIDTH=$PW
done
# Nalu's own velocity setting, which on this flow is pure central.
run_one nalu_velocity SFEM_PECLET_BLEND=tanh SFEM_PECLET_TRANS=50000 SFEM_PECLET_WIDTH=200
# And the classic form at its default, which is a much gentler blend.
run_one classic SFEM_PECLET_BLEND=classic
echo
echo "logs in $OUT"
echo "done $(date)"
