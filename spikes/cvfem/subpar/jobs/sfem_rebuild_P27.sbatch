#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=sfem-rebuild-P27
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=sfem_rebuild_P27_%j.out
#SBATCH --error=sfem_rebuild_P27_%j.err
# Rebuild the SFEM OpenMP install after the Alps scratch migration.
#
# /capstor is gone and $SCRATCH is now /ritom/scratch/cscs/zulianp, which carried none of the
# previous work: the sfem-omp and sfem-omp-zsg prefixes, the spike trees and every build with
# them. This rebuilds the one prefix everything else needs, from the source rsynced by
# scripts/synch_alps.sh, which is the repository at HEAD plus the uncommitted work in the tree.
#
# SFEM installs its dependencies (smesh, ryml, matrixio, SCCD, ssdf) into the same prefix, so
# this one install is what the spike configures against. Options follow the daint recipe in the
# repository README: OpenMP on, MPI on (mpicc/mpicxx, which also supply the include paths the
# installed headers need), Release.
set -u
SRC=$SCRATCH/sfem
BUILD=$SRC/build-omp
PREFIX=$SCRATCH/installations/sfem-omp
echo "host $(hostname) src $SRC prefix $PREFIX $(date)"

cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSFEM_ENABLE_OPENMP=ON \
    -DSFEM_ENABLE_MPI=ON \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" > "$SRC/configure_omp.log" 2>&1
echo "configure rc $?"
grep -E "CMake Error" -A6 "$SRC/configure_omp.log" | head -20

cmake --build "$BUILD" -j 72 > "$SRC/build_omp.log" 2>&1
echo "build rc $?"
grep -E "error:|Error [0-9]" "$SRC/build_omp.log" | head -20

cmake --install "$BUILD" > "$SRC/install_omp.log" 2>&1
echo "install rc $?"

echo "== prefix layout"
ls "$PREFIX" 2>&1 | head
for d in lib lib64; do echo "-- $d/cmake"; ls "$PREFIX/$d/cmake" 2>&1 | head -10; done
grep -c set_initial_guess_zero "$PREFIX"/include/sfem_Multigrid.hpp 2>&1
echo "=== DONE $(date)"
