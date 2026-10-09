#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-noz-setuptrace-P14
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_setuptrace_P14_%j.out
#SBATCH --error=nozzle_setuptrace_P14_%j.err
# Where the Vanka setup's time goes (build_p9: HEAD f201fd76d plus trace scopes on the fine
# matrix assembly's steps). The setup phase is 17.8 s of an 81 s run at 893,924 dof and the
# existing scopes account for ~7.7 s; perf found no serial hot symbol. A short run -- Re 1, one
# continuation stage, a handful of Newton steps -- with the trace is enough to split it.
# Warped FDA nozzle, macro core 2, FGMRES (SFEM_FGMRES=1, restart 480) + geometric multigrid,
# driver defaults otherwise. Complete logs via scripts/cvrun.sh.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
cd "$SPIKE_ROOT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=$SCRATCH/cvfem_nozzle_setuptrace/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname) out $OUT $(date)"
tag=L8_re1; mkdir -p "$OUT/$tag"
( cd "$OUT/$tag" && CVRUN_DIR=$OUT CVRUN_OUT=$OUT/$tag/out \
  CVRUN_FILTER="^newton |^stage |newton_converged|lin_it_total|t_solve|ndof:|gmg:" \
  uenv run $UENV_ARGS -- timeout 400 zsh "$SPIKE_ROOT/scripts/cvrun.sh" "$tag" "$SPIKE_ROOT/build_p9/cvfem_hex8_ns_ssgmg" \
    SFEM_CASE=nozzle SFEM_NOZZLE_RE=1 SFEM_GEOM=affine \
    SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
    SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 \
    "SFEM_NOZZLE_NAXIAL=4 3 8 12" )
echo "--- $tag exit $?"
grep -E "vanka_setup|galerkin_assembly|coarse_factor|precond_total|t_precond" "$OUT/$tag.log"
f="$OUT/$tag/smesh.trace.csv"
(head -1 "$f"; grep -iE "galerkin|vanka|assemble|alloc_acc|copy_out|pattern|scatter|inverse" "$f" | sort -t, -k3 -gr) | column -s, -t
echo "=== DONE $(date)"
