#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gmres
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=gmres_%j.out
#SBATCH --error=gmres_%j.err
#
# What does a GMRES iteration cost, and how much of it is the operator?
#
# Kernel benchmarks measure an apply with nothing around it. A solve measures an iteration with
# everything around it. The gap between those two is usually quoted as though it were overhead
# of the Krylov method, and it is not: most of it is the operator's own companions -- the
# direction's gradient reconstruction, the boundary closure, the constraint copy -- which a
# bench that applies only the element kernel never pays for.
#
# So rather than compare two binaries, this decomposes ONE run. The compiled-in trace nests:
#
#   FGMRES/BiCGStab::apply      the whole Krylov iteration, in context
#     Function::apply             operator plus constraints
#       CVFEMNavierStokes::apply    the operator
#         element sweep             the kernel alone, in isolation
#
# and each level's difference from the one inside it is that layer's cost, measured on the
# same state, the same mesh and the same allocation. Five GMRES steps and one Newton step,
# because this is a measurement of an iteration rather than of a solve.
set -u
D=${DRIVER:?set DRIVER}
OUT=${OUT:-$SCRATCH/cvfem_gmres/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo "out  $OUT"

one() {  # one <tag> <n> <level> <sspack>
    local tag=$1 n=$2 lvl=$3 ps=$4
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && timeout 500 uenv run $U -- env OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_SS_PACK_SIZE=$ps \
        SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=30 SFEM_PRECOND=bjacobi \
        SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=5 SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0 \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd; nd=$(grep -o 'ndof: [0-9]*' "$r/log" | head -1 | awk '{print $2}')
    printf "\n=== %s: ndof=%s ===\n" "$tag" "${nd:-?}"
    # The NAMED layers, not the largest scopes.
    #
    # Sorting the trace by cost puts one-off setup on top -- initialize, the node-to-node
    # graph, the block diagonal -- and buries the per-iteration layers this exists to show.
    # And us_per_lin_it must not be quoted from a five-iteration run: it divides a wall clock
    # that contains a 227 ms setup by five, and reads ten times the true per-iteration cost.
    # Per-call trace figures do not have that problem, which is why they are what is printed.
    for scope in \
        "Function::apply" \
        "CVFEMNavierStokes::apply" \
        "cvfem_hex8_ns_steady::apply_jacobian_action_packed" \
        "sscvfem::apply_macro_local_hoisted" \
        "cvfem_hex8_ns_steady::nodal_grad_strided" \
        "sscvfem::nodal_grad_strided" \
        "cvfem_hex8_ns_steady::apply_boundary_scs_jacobian_action" \
        "DirichletConditions::copy_constrained_dofs"; do
        awk -F, -v s="$scope" 'NR>1 && $1==s && $2>0 {printf "  %-56s %6d calls %10.1f us/call\n", $1, $2, $4/$2*1e6}' \
            "$r/smesh.trace.csv" 2>/dev/null
    done
}

one flat      64 1 0
one ss_L8_pack 8 8 8
one ss_L8_scat 8 8 0
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
