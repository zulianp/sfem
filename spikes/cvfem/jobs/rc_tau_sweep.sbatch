#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rctau
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=rctau_%j.out
#SBATCH --error=rctau_%j.err
#
# The Rhie-Chow time scale, swept over the quantity it was wrong about.
#
# The coefficient is V/a_P -- the control volume over the momentum equation's own diagonal --
# and a_P carries inertia, convection and diffusion. It used to evaluate only the third, as
# Df = rc_scale h^2 / (2 mu), which is the Stokes limit. The error in that is the cell Peclet
# number rho|u|h/mu, so the severity of the approximation IS Pe_cell and the sweep has to be
# over Pe_cell rather than over anything else.
#
#   SFEM_RC_TAU=0   the previous diffusion-only coefficient, bit-for-bit
#   SFEM_RC_TAU=1   tau = [(2 a0/dt)^2 + (2|u|/h)^2 + (4 nu/h^2)^2]^(-1/2)  (default)
#
# There is NO zero-severity row in this sweep, and an earlier draft of this comment wrongly
# claimed Re=1 was one. The two forms differ by a factor of two even at Pe_cell = 0, because
# the standard time scale's diffusive branch is h^2/(4 nu) where the old coefficient was
# h^2/(2 nu). So Re=1 measures that factor of two and nothing else; the Peclet effect is what
# the ROWS ABOVE it add on top. A true zero-severity control needs the diffusive limits
# matched -- SFEM_RC_TAU=0 with SFEM_RHIE_CHOW_SCALE=0.5 -- which is the last block below.
#
# Two axes that must not be confounded. At fixed mesh, raising Re raises Pe_cell and should
# widen the gap. At fixed Re, refining the mesh LOWERS Pe_cell and should narrow it -- and
# the observed order of accuracy must be unchanged either way, because the stabilisation is
# a consistent O(h^2) term under both time scales and only its constant moved.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "rctau: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_rctau/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# One run, with the whole log kept and the per-iteration lines echoed as they appear, so a
# stalled configuration is distinguishable from a slow one while the job is still going.
run_one() {  # run_one <tag> <env assignments...>
    tag=$1; shift
    log="$OUT/$tag.log"
    stdbuf -oL -eL env "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/^newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nwt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    lit=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    ts=$(sed -n 's/^t_solve: \([0-9.e+-]*\).*/\1/p' "$log" | head -1)
    ul=$(sed -n 's/.*u_linf: \([0-9.e+-]*\).*/\1/p' "$log" | head -1)
    pl=$(sed -n 's/.*p_linf: \([0-9.e+-]*\).*/\1/p' "$log" | head -1)
    printf '%-26s %10s %4s %6s %8s %10s %12s %12s\n' \
        "$tag" "${ndof:--}" "${conv:--}" "${nwt:--}" "${lit:--}" "${ts:--}" "${ul:--}" "${pl:--}"
}

hdr() { printf '%-26s %10s %4s %6s %8s %10s %12s %12s\n' \
        tag ndof conv newton lin_it t_solve u_linf p_linf; }

echo "== axis 1: fixed mesh, rising cell Peclet (cavity, N=48, Pe_cell = Re/48)"
echo "   Re=1 is NOT a zero-severity control: the two forms differ by a factor of two even"
echo "   at Pe_cell = 0, because the standard diffusive branch is h^2/(4 nu) where the old"
echo "   coefficient was h^2/(2 nu). The Re=1 row measures that factor; the rows above it"
echo "   add the Peclet effect. The matched-diffusive-limit control is the last block."
hdr
for RE in 1 10 100 1000; do
    MU=$(awk -v r=$RE 'BEGIN{printf "%.10g", 1.0/r}')
    for T in 0 1; do
        run_one "cav_Re${RE}_tau${T}" SFEM_RC_TAU=$T SFEM_CASE=cavity SFEM_N=48 \
            SFEM_RHO=1 SFEM_U=1 SFEM_MU=$MU SFEM_NL_MAX_IT=30 SFEM_ELEMENT_REFINE_LEVEL=1
    done
done
echo

echo "== axis 2: fixed Re, refining mesh (mms, the only case with a manufactured solution)"
echo "   The observed order must be the same under both time scales. A changed order would"
echo "   mean the new coefficient is not a consistent O(h^2) term, which would be fatal."
hdr
for N in 8 16 32 64; do
    for T in 0 1; do
        run_one "mms_N${N}_tau${T}" SFEM_RC_TAU=$T SFEM_CASE=mms SFEM_N=$N \
            SFEM_MU=1 SFEM_RHO=1 SFEM_NL_MAX_IT=20 SFEM_ELEMENT_REFINE_LEVEL=1
    done
done
echo

echo "== the two exact-solution cases, for the solver cost at 75,140 dof"
echo "   These reproduced 4.9x and 3.8x fewer linear iterations on a laptop; this is the"
echo "   same comparison on the machine every other number in the reports came from."
hdr
for C in poiseuille couette; do
    for T in 0 1; do
        run_one "${C}_tau${T}" SFEM_RC_TAU=$T SFEM_CASE=$C SFEM_N=16 \
            SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_NL_MAX_IT=20
    done
done
echo

echo "== the true zero-severity control: the diffusive limits matched"
echo "   SFEM_RC_TAU=0 with half the scale has the same h^2/(4 nu) diffusive branch as the"
echo "   combined form, so at Re=1 -- where Pe_cell is 1/48 and the advective branch is"
echo "   nothing -- these two must agree. If they do not, the change does something other"
echo "   than what it claims and no row above this one means anything."
hdr
run_one "ctrl_Re1_tau0_half" SFEM_RC_TAU=0 SFEM_RHIE_CHOW_SCALE=0.5 SFEM_CASE=cavity SFEM_N=48 \
    SFEM_RHO=1 SFEM_U=1 SFEM_MU=1 SFEM_NL_MAX_IT=30 SFEM_ELEMENT_REFINE_LEVEL=1
run_one "ctrl_Re1_tau1"      SFEM_RC_TAU=1 SFEM_CASE=cavity SFEM_N=48 \
    SFEM_RHO=1 SFEM_U=1 SFEM_MU=1 SFEM_NL_MAX_IT=30 SFEM_ELEMENT_REFINE_LEVEL=1
echo

echo "== semi-structured, so the SS path is covered by the same sweep"
hdr
for T in 0 1; do
    run_one "ss_cav_tau${T}" SFEM_RC_TAU=$T SFEM_CASE=cavity SFEM_N=8 \
        SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_RHO=1 SFEM_U=1 SFEM_MU=0.01 \
        SFEM_GMG=1 SFEM_NL_MAX_IT=30
done
echo
echo "logs in $OUT"
echo "done $(date)"
