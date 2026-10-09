#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rcrem
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=rcrem_%j.out
#SBATCH --error=rcrem_%j.err
#
# Two candidate remedies for the stages that abandon under the corrected time scale.
#
# The coefficient fix makes Newton converge fast enough that a continuation stage routinely
# STARTS at or near its residual floor. Two of the sweep's configurations then abandoned a
# stage on "linear solve diverged after 0 iterations", and the two are not the same failure:
#
#   cav_Re100  rel 4.31e-07, rate 0.612  -- at the floor, should be accepted
#   mms_N64    rel 3.81e-03, rate 0.548  -- genuinely unconverged, should not be
#
# BiCGStab's divergence test already measures growth against its own starting residual, so
# unlike the FGMRES one it is not reporting a false positive -- it is a real BiCGStab blow-up
# on a right-hand side that is already solved. That leaves two places to act, and this job
# measures both rather than picking one:
#
#   A. SFEM_FGMRES=1 -- FGMRES instead of BiCGStab. Costs a Krylov basis; FGMRES minimises
#      over the whole space and does not have BiCGStab's erratic single steps. Changes no
#      tolerance, so nothing is accepted that was not converged.
#   B. SFEM_NL_LS_FLOOR=1e-6 -- raise the relative floor at which a failed linear solve is
#      read as "converged" rather than "diverged". Accepts cav_Re100 and still rejects
#      mms_N64, which is the discrimination wanted -- but it is a LOOSENING, and the default
#      of 1e-7 was chosen deliberately as ten times nl_rtol with the stated reason that
#      anything looser starts accepting unconverged states. That trade is a judgement call
#      rather than a fix, which is why it is measured beside A instead of merged.
#
# The control is the same two configurations with neither remedy, so the rows are comparable.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "rcrem: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_rcrem/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

run_one() {
    tag=$1; shift
    log="$OUT/$tag.log"
    stdbuf -oL -eL env "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/^newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nwt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    lit=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    hi=$(sed -n 's/.*highest Re SOLVED = \([0-9.]*\).*/\1/p' "$log" | head -1)
    ul=$(sed -n 's/.*u_linf: \([0-9.e+-]*\).*/\1/p' "$log" | head -1)
    ab=$(grep -c "abandoning stage" "$log")
    printf '%-30s %9s %4s %6s %9s %8s %5s %12s\n' \
        "$tag" "${ndof:--}" "${conv:--}" "${nwt:--}" "${lit:--}" "${hi:--}" "$ab" "${ul:--}"
}
hdr() { printf '%-30s %9s %4s %6s %9s %8s %5s %12s\n' \
        tag ndof conv newton lin_it hiRe aband u_linf; }

CAV="SFEM_RC_TAU=1 SFEM_CASE=cavity SFEM_N=48 SFEM_RHO=1 SFEM_U=1 SFEM_MU=0.01 SFEM_NL_MAX_IT=30 SFEM_ELEMENT_REFINE_LEVEL=1"
MMS="SFEM_RC_TAU=1 SFEM_CASE=mms SFEM_N=64 SFEM_MU=1 SFEM_RHO=1 SFEM_NL_MAX_IT=20 SFEM_ELEMENT_REFINE_LEVEL=1"

echo "== control: neither remedy (reproduces the sweep's two failures)"
hdr
run_one cav_control $CAV
run_one mms_control $MMS
echo
echo "== remedy A: FGMRES instead of BiCGStab, no tolerance changed"
hdr
run_one cav_fgmres $CAV SFEM_FGMRES=1
run_one mms_fgmres $MMS SFEM_FGMRES=1
echo
echo "== remedy B: the relative floor raised from 1e-7 to 1e-6"
echo "   mms_N64 must STILL fail here; if it passes, the floor is accepting an unconverged"
echo "   state and the remedy is worse than the problem."
hdr
run_one cav_floor $CAV SFEM_NL_LS_FLOOR=1e-6
run_one mms_floor $MMS SFEM_NL_LS_FLOOR=1e-6
echo
echo "== both, for completeness"
hdr
run_one cav_both $CAV SFEM_FGMRES=1 SFEM_NL_LS_FLOOR=1e-6
run_one mms_both $MMS SFEM_FGMRES=1 SFEM_NL_LS_FLOOR=1e-6
echo
echo "== and the old coefficient on the same two, so the comparison has a baseline"
hdr
run_one cav_oldtau SFEM_RC_TAU=0 SFEM_CASE=cavity SFEM_N=48 SFEM_RHO=1 SFEM_U=1 SFEM_MU=0.01 \
    SFEM_NL_MAX_IT=30 SFEM_ELEMENT_REFINE_LEVEL=1
run_one mms_oldtau SFEM_RC_TAU=0 SFEM_CASE=mms SFEM_N=64 SFEM_MU=1 SFEM_RHO=1 \
    SFEM_NL_MAX_IT=20 SFEM_ELEMENT_REFINE_LEVEL=1
echo
echo "logs in $OUT"
echo "done $(date)"
