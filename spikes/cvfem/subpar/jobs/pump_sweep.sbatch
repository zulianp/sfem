#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pump
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pump_%j.out
#SBATCH --error=pump_%j.err
#
# The diaphragm pump at higher resolution and higher Reynolds number.
#
#   sbatch --export=ALL,DRIVER=<path>/cvfem_hex8_ns_ssgmg jobs/pump_sweep.sbatch
#
# The verification matrix runs this case at N=8 (2,916 dof), mu=0.05 (Re=20) and with a
# dense LU, which is deliberate there -- the identity being checked is exact at any
# resolution, so the cheapest mesh that resolves the port is the right one for a gate. None
# of that survives here: a dense LU is cubic in the dof count and Re=20 is not a test of the
# upwinding at all.
#
# Re = rho U Ly / mu, and the pump is a unit cube with U = 1, so Re = 1/mu exactly.
#
# The identity stays the measure of correctness. The domain is fixed and the flow
# incompressible, so the flux through the port must equal the volume the diaphragm sweeps,
# V Lx Lz, with no closed form needed anywhere. Resolution and Reynolds number must not
# change it, and the run prints |port - swept| so a failure names its own size.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_pump/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"

hdr() { printf "%-22s %-9s %-6s %-6s %-9s %-9s %-11s %s\n" \
        run ndof Re conv Re_solved lin_it t_solve "|port-swept|"; }

one() {  # one <tag> <env...>
    local tag=$1; shift
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && timeout 600 uenv run $U -- env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        OMP_PROC_BIND=true OMP_PLACES=cores SFEM_CASE=pump "$@" \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd c li ts re rs err
    nd=$(grep -o 'ndof: [0-9]*'           "$r/log" | head -1 | awk '{print $2}')
    c=$( grep -o 'newton_converged: [01]' "$r/log" | head -1 | awk '{print $2}')
    li=$(grep -o 'lin_it_total: [0-9]*'   "$r/log" | head -1 | awk '{print $2}')
    ts=$(grep -o 't_solve: [0-9.]*'       "$r/log" | head -1 | awk '{print $2}')
    re=$(grep -o 'mms_Re.*' "$r/log" | head -1 | awk '{print $NF}')
    rs=$(grep -o 'highest Re SOLVED = [0-9.e+-]*' "$r/log" | head -1 | awk '{print $NF}')
    err=$(grep -o '|port - swept| [0-9.e+-]*' "$r/log" | head -1 | awk '{print $NF}')
    printf "%-22s %-9s %-6s %-6s %-9s %-9s %-11s %s\n" \
        "$tag" "${nd:-?}" "${RE_LABEL:-?}" "${c:-?}" "${rs:-?}" "${li:-?}" "${ts:-?}" "${err:-?}"
}

# ---- part 1: which linear solver survives the resolution ladder, at the gate's Re ----
#
# N=8 is the verification point and is here as the control: whatever the ladder shows has to
# agree with it where they overlap.
echo
echo "=== resolution ladder, Re = 20 (mu = 0.05) ==="
hdr
RE_LABEL=20
for n in 8 16 24 32; do
    one "lu_N$n"      SFEM_N=$n SFEM_MU=0.05 SFEM_GMG=0 SFEM_PRECOND=direct SFEM_NL_MAX_IT=40
    one "bjac_N$n"    SFEM_N=$n SFEM_MU=0.05 SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 \
                      SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=40
done

# ---- part 2: the Reynolds ladder, on the finest resolution that part 1 leaves standing ----
#
# mu alone sets Re here. The adaptive continuation ramps rho geometrically and inserts a
# stage at the geometric mean whenever one fails, so a run that cannot reach the target says
# how far it got rather than only that it failed -- which is what `Re_solved` reports.
echo
echo "=== Reynolds ladder, N = ${PUMP_N:-24} ==="
hdr
for mu in 0.05 0.01 0.002 0.001; do
    RE_LABEL=$(awk -v m=$mu 'BEGIN{printf "%g", 1/m}')
    one "re${RE_LABEL}_bjac" SFEM_N=${PUMP_N:-24} SFEM_MU=$mu SFEM_GMG=0 SFEM_FGMRES=1 \
        SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=40
done

echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
