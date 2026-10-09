#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-sslevel
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=sslevel_%j.out
#SBATCH --error=sslevel_%j.err
#
# How much of the semi-structured operator's gap to the flat one is the macro-element level?
#
# The scatter splits nodes geometrically: a lattice node strictly inside a macro-element is
# touched by that macro-element alone and is written straight out, while the faces, edges and
# corners are staged and reduced. That skin is (L+1)^3 - (L-1)^3 of (L+1)^3 nodes -- at L=2
# it is 26 of 27, so nearly everything is staged, and the fraction falls fast with L.
#
# So before engineering the scatter, measure whether the level already buys what the
# engineering would. N*L is held at 64 so every row is the same 4,343,300 dof and the same
# fine mesh; only the macro-element size changes. The flat run is the target to reach.
#
# One allocation, because a 5-11% node-to-node spread would otherwise swamp the comparison.
set -u
D=${DRIVER:?set DRIVER}
OUT=${OUT:-$SCRATCH/cvfem_sslevel/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo "out  $OUT"

one() {  # one <tag> <N> <level>
    local tag=$1 n=$2 lvl=$3
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r"
      export SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl
      export SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
      export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
      timeout 500 uenv run $U -- "$D" "$r/out" > "$r/log" 2>&1 )
    local ndof; ndof=$(grep -m1 -o 'ndof: [0-9]*' "$r/log" | awk '{print $2}')
    printf "  %-12s ndof=%-9s %s\n" "$tag" "${ndof:-?}" \
        "$(grep -m1 -o 't_solve: [0-9.]* s' "$r/log" | awk '{print "t_solve="$2}')"
}

# N*L = 64 throughout: same fine mesh, different macro-element size.
one flat_N64  64 1
one ss_L2_N32 32 2
one ss_L4_N16 16 4
one ss_L8_N8   8 8
one ss_L16_N4  4 16
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
