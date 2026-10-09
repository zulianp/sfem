#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-dt-campaign-P40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=dt_campaign_P40_%j.out
#SBATCH --error=dt_campaign_P40_%j.err
# Stage 6's one outstanding measurement. The campaign bound (39.6-52.1 socket-days) is a FLOOR
# because it uses t_step measured at dt = 1e-3, while the campaign runs at dt ~ 1.75e-4 -- below
# the measured range, where lin/newton was still climbing steeply: 14.2 -> 21.1 -> 34.8 as dt
# fell 4e-3 -> 1e-3 -> 2.5e-4. That climb is the unmeasured multiplier on the whole campaign.
#
# L8 extends the existing sweep so the two are directly comparable; L16 measures the actual
# campaign point, which is the number the bound is missing. No rebuild -- another session is
# using build_gate and this only reads it.
set -u
# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path
# says nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was
# run from; SPIKE_ROOT overrides it for a manual run.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "dt_campaign.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
B="${BUILD:-$SPIKE_ROOT/build_gate}"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_dt_campaign/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
one() {   # tag level dt timeout
    local tag=$1 lvl=$2 dt=$3 secs=$4 d=$OUT/$1; mkdir -p "$d"
    ( cd "$d" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_DT=$dt SFEM_NSTEPS=20 SFEM_BDF_ORDER=2 \
        SFEM_DIAG_CSV="$d/diag.csv" SFEM_ENABLE_OUTPUT=0 SFEM_WRITE_STEPS=0 \
        "$B/cvfem_hex8_ns_ssgmg" "$d/out" > "$d/log" 2>&1 )
    echo "--- $tag exit $?"
}
one L8_dt175  8  1.75e-4 1800
one L8_dt1e4  8  1e-4    1800
one L16_dt175 16 1.75e-4 3000
echo "=== DONE $(date)"
