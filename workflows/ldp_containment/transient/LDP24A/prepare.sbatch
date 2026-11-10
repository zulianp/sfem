#!/bin/bash
#SBATCH --job-name=ldp24a-prepare
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/ldp24a_prepare_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# LDP24A's case for the paper, a new folder CASE (the job stops if it exists): the valves' mesh and
# their obstacle from the FSI's case LDP24AT_v3 (db/solid: TET4, sfc order, with the side-sets PV_TOP,
# SV_TOP, PV_PIN, SV_PIN; db/sdf_valves), copied; db/solid_p2: db/solid promoted to TET10 (smesh
# mesh_promote) and put in sfc order (smesh sfc), as the FF200 cases' prepare.sh; db/solid_r1:
# db/solid refined once (smesh refine, every element into eight). Prints the nodes and elements.
#   [CASE=ldp_cases/LDP24A] [SRC=ldp_cases/LDP24AT_v3]
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
SMESH=$D/build-cuda/external/smesh
CASE=${CASE:-/ritom/scratch/cscs/zulianp/ldp_cases/LDP24A}
SRC=${SRC:-/ritom/scratch/cscs/zulianp/ldp_cases/LDP24AT_v3}
[ -e $CASE ] && { echo "prepare: $CASE exists"; exit 1; }
mkdir -p $CASE/db
cp -r $SRC/db/solid $SRC/db/sdf_valves $CASE/db/ || exit 1
cd $CASE/db
export OMP_NUM_THREADS=72
srun -n 1 -c 72 $SMESH/mesh_promote TET10 solid solid_p2_promoted > promote.log 2>&1 || { echo "prepare: mesh_promote failed"; exit 1; }
srun -n 1 -c 72 $SMESH/sfc solid_p2_promoted solid_p2 > sfc.log 2>&1 || { echo "prepare: sfc failed"; exit 1; }
SMESH_REFINEMENT_LEVELS=1 srun -n 1 -c 72 $D/build-x28/external/smesh/refine solid solid_r1 > refine.log 2>&1 || { echo "prepare: refine failed"; exit 1; }
for m in solid solid_p2 solid_r1; do
  echo "$m nodes $(( $(stat -c %s $m/x.float32) / 4 )) sidesets: $(ls $m/sidesets 2>/dev/null | tr '\n' ' ')"
  for b in $m/blocks/*/; do echo "  $(basename $b) elements $(( $(stat -c %s $b/i0.int32) / 4 ))"; done
done
