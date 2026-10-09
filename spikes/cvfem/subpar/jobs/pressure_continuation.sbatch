#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pcont
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pcont_%j.out
#SBATCH --error=pcont_%j.err
#
# The prescribed-pressure continuation, on the path the laptop cannot check.
#
# WHAT THE LOCAL SWEEP ALREADY ESTABLISHED, so this does not repeat it: on a flat
# 48x12x12 mesh (33,124 dof) the prescribed pressure is a PURE GAUGE SHIFT -- p_bar in
# {-0.16, 0, 0.16, 0.5} leaves u_linf at round-off and moves p_linf by exactly
# |p_bar + 0.16|. The level predictor (SFEM_P_PREDICT) exploits that and is exact there;
# the ramp (SFEM_P_STAGES) is the fallback. With both on, every p_bar from -3 to +10
# converges, where the previous code failed from p_bar = 1.0 upward.
#
# WHAT THIS JOB IS FOR, and it is not more of the same:
#
#  1. THE SEMI-STRUCTURED PATH IS UNTESTED. The predictor needs the seed's pressure level
#     at the port, and it gets it from create_nodeset_from_sideset on the mesh AFTER
#     to_semistructured has rebuilt it. That call is known to work there -- the step case
#     uses it for its skin -- but it has never been exercised on a PRESSURE sideset, and a
#     wrong or empty nodeset would silently give p_seed_port = 0 instead of -0.16. The
#     symptom would be a ramp that starts from the wrong place, not a crash. Section 1
#     prints the measured level next to the flat one, which is the whole check: they must
#     agree, because it is the same geometry.
#
#  2. SIZE, but only where the solver that runs at that size is the one being measured.
#     33,124 dof is below anything this operator saturates on, and the linear solver is the
#     component that actually failed in the original port runs. The first version of this
#     job simply raised N with the default solver and every row failed, p_bar = -3 included
#     -- at 1.85M dof with no multigrid there is no linear solve, so the table said nothing
#     about pressure. Section 3 runs semi-structured with multigrid instead, and sections 1
#     and 2 stay small with a dense LU so the linear solver is removed rather than strained.
#
#  3. THE CONTROL HAS TO RUN HERE TOO. SFEM_P_PREDICT=0 SFEM_P_STAGES=0 is the behaviour
#     before any of this existed. Without it in the same table on the same machine, the
#     "now it converges" column is an assertion rather than a comparison.
#
# Partition: normal rather than debug. The matrix is about 40 runs and several are minutes
# each at the larger sizes, which does not fit a debug allocation, and the debug QOS submit
# limit is not worth contending for when nothing here needs a fast turnaround.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "pcont: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_pcont/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo "driver $D"
echo

# One run. Echoes the per-stage lines to the job log as they are produced -- the matrix is
# long and a table that only appears at the end is indistinguishable from a hung job.
# SFEM_FGMRES=1 named explicitly rather than left to the driver default, so this script
# measures the intended Krylov method whatever a default does later.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_LZ=1 SFEM_MU=0.01 SFEM_U=1
      SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet SFEM_PRESSURE_SIDESET=outlet SFEM_FGMRES=1"

run_one() {  # run_one <tag> <env...>
    local tag=$1; shift
    local log="$OUT/$tag.log"
    stdbuf -oL -eL env $BASE "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$?
    stdbuf -oL grep -E "prescribed pressure:|pressure continuation:|^stage |SHORT OF TARGET" "$log" \
        | sed "s/^/    [$tag] /"
    local ndof conv nt li u p seed
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    u=$(sed -n 's/^u_linf: \([^ ]*\).*/\1/p' "$log" | head -1)
    p=$(sed -n 's/.*p_linf: \(.*\)$/\1/p' "$log" | head -1)
    seed=$(sed -n "s/.*seed level at '[^']*' = \([^,]*\),.*/\1/p" "$log" | head -1)
    printf '%-26s %9s %9s %5s %7s %9s %14s %14s\n' \
        "$tag" "${ndof:--}" "${seed:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${u:--}" "${p:--}"
}

HDR() { printf '%-26s %9s %9s %5s %7s %9s %14s %14s\n' \
        tag ndof seed_p conv newton lin_it u_linf p_linf; }

echo "== 1: semi-structured against flat, same geometry, same dof count, same answer"
echo "   SFEM_PRECOND=direct -- a dense LU of the fine Jacobian. That is affordable at this"
echo "   size and it is the point: it takes the linear solver out of the question entirely,"
echo "   so whatever these rows show is a statement about the continuation and the"
echo "   discretisation and about nothing else. It is also what the verification matrix runs"
echo "   the port group with, so these numbers are comparable with it."
echo
echo "   seed_p is the level the predictor measured at the port. It is a property of the"
echo "   GEOMETRY and the case, not of the discretisation, so the flat and semi-structured"
echo "   rows must print the same -0.16 -- and the semi-structured one goes through"
echo "   create_nodeset_from_sideset on a mesh to_semistructured has rebuilt, which no test"
echo "   has ever exercised on a PRESSURE sideset. A 0 there means the nodeset came back"
echo "   empty and the predictor is silently starting from the wrong place."
echo "   p_linf must be |p_bar + 0.16| in every converged row; that is the gauge identity."
# N=8, not 12. SFEM_PRECOND=direct builds a DENSE matrix of the whole system and refuses
# anything over SFEM_DIRECT_MAX_DOF=20000; 12 cells per unit length is 33,124 dofs and the
# first attempt at this section was refused outright, five rows of nothing. 8 gives 10,692,
# whose dense factor is under a gigabyte. Raising the cap instead would ask for an 8.8 GB
# matrix to prove a point about pressure.
DIRECT="SFEM_PRECOND=direct SFEM_GMG=0"
HDR
for PB in -3.0 0.5 1.0 3.0 10.0; do
    run_one "flat_p${PB}" SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=1 $DIRECT SFEM_PRESSURE=$PB
done
for PB in -3.0 0.5 1.0 3.0 10.0; do
    run_one "ss4_p${PB}"  SFEM_N=2  SFEM_ELEMENT_REFINE_LEVEL=4 $DIRECT SFEM_PRESSURE=$PB
done
echo

echo "== 2: the control, same size and same exact linear solve -- both mechanisms off"
echo "   This is the code as it stood before the continuation existed. Every row from"
echo "   p_bar = 1.0 upward is expected to FAIL. With a dense LU underneath, a failure here"
echo "   cannot be blamed on the Krylov method, which is what makes the comparison mean"
echo "   something; if one of these passes, the diagnosis needs revisiting."
HDR
for PB in 0.5 1.0 1.5 3.0 10.0; do
    run_one "ctrl_p${PB}" SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=1 $DIRECT \
                          SFEM_P_PREDICT=0 SFEM_P_STAGES=0 SFEM_PRESSURE=$PB
done
echo

echo "== 3: the two mechanisms separated, at a size the solver is actually used at"
echo "   Semi-structured with multigrid, which is the configuration the spike runs at scale."
echo "   A dense LU does not exist at this size and neither does plain BiCGStab: the first"
echo "   version of this job asked for 1.85M dof with NO multigrid and every row failed,"
echo "   including p_bar = -3 which solves comfortably at 33k. That measured the linear"
echo "   solver, not the continuation, and is exactly the confound this section avoids."
echo
echo "   The ramp-alone column is the interesting one: it is what a code with no gauge"
echo "   structure to exploit would have, so it says how much of the rescue is generic"
echo "   continuation and how much is this problem's algebra."
GMG="SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1"
HDR
for PB in 1.0 3.0 10.0; do
    run_one "pred_p${PB}"  SFEM_N=2 $GMG SFEM_P_PREDICT=1 SFEM_P_STAGES=0 SFEM_PRESSURE=$PB
    run_one "ramp_p${PB}"  SFEM_N=2 $GMG SFEM_P_PREDICT=0 SFEM_P_STAGES=8 SFEM_PRESSURE=$PB
    run_one "both_p${PB}"  SFEM_N=2 $GMG SFEM_P_PREDICT=1 SFEM_P_STAGES=4 SFEM_PRESSURE=$PB
done
echo

echo "logs in $OUT"
echo "done $(date)"
