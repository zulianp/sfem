#!/bin/bash
#SBATCH --job-name=scale-mesh
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/scale_mesh_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# FF200's TET4 mesh refined uniformly once (smesh refine: every element into eight), into a new
# folder db/solid_r1 (the job stops if it exists), for the storage at about ten times the size
# (scale_storage: tangent9_valves.sbatch with NO_IC=1, mesh.path ../db/solid_r1, TET10 by
# mesh.promote_to_p2). Prints the elements per block and the nodes of both meshes.
set -u
D=/ritom/scratch/cscs/zulianp/sfem-prony-md
DB=/ritom/scratch/cscs/zulianp/ldp_cases/FF200_ExodusMesh/db
OUT=$DB/solid_r1
[ -e $OUT ] && { echo "scale_mesh: $OUT exists"; exit 1; }
export OMP_NUM_THREADS=72
SMESH_REFINEMENT_LEVELS=1 srun -n 1 -c 72 $D/build-x28/external/smesh/refine $DB/solid $OUT || exit 1
for m in $DB/solid $OUT; do
  echo "$m nodes $(( $(stat -c %s $m/x.float32) / 4 ))"
  for b in $m/blocks/*/; do echo "  $(basename $b) elements $(( $(stat -c %s $b/i0.int32) / 4 ))"; done
done
