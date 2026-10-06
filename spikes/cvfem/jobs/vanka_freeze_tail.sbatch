#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-freeze-tail-P35
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vanka_freeze_tail_P35_%j.out
#SBATCH --error=vanka_freeze_tail_P35_%j.err
# The one row P34 (job 4685821) did not reach before its two-hour limit: L=8 with the Peclet
# blend on and SFEM_VANKA_FREEZE=4. Everything else in that table completed.
#
# P34 measured, at 893,924 and 7,014,340 dof, that freezing costs NO iterations: newton/step and
# lin/step are unchanged to the digit at every stride tried, while wall time falls with the
# share vanka_setup held -- 7.5% at L=8 blend off, 24.0% at L=16, 9.8% at L=8 with the blend on
# at stride 2. This run closes the blend arm at stride 4, which is the configuration with the
# least margin: its defect correction contracts at only r ~ 0.49, so if a staler preconditioner
# is going to cost iterations anywhere, it is here.
#
# One run of 20 steps at ~16 s/step is about six minutes. Normal rather than debug: the debug
# QOS allows two queued jobs per USER and this account's other session holds both, so a debug
# submission is refused outright (QOSMaxSubmitJobPerUserLimit). Normal does not share that cap.
set -u
# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path
# says nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was
# run from; SPIKE_ROOT overrides it for a manual run.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "vanka_freeze_tail.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
B="${BUILD:-$SPIKE_ROOT/build_gate}"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_freeze/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
ls -la "$B/cvfem_hex8_ns_ssgmg"

NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"

printf "%-20s %-10s %-7s %-7s %-9s %-9s %-9s %-9s %s\n" \
       run ndof freeze steps t_step_s newton/st lin/st rebuilds note

one() {  # one <tag> <level> <steps> <freeze> <timeout> [extra env...]
    local tag=$1 lvl=$2 nst=$3 frz=$4 secs=$5; shift 5
    local d=$OUT/$tag; mkdir -p "$d"
    ( cd "$d" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_VANKA_FREEZE=$frz \
        SFEM_DT=1e-3 SFEM_NSTEPS=$nst SFEM_BDF_ORDER=2 \
        SFEM_DIAG_CSV="$d/diag.csv" SFEM_ENABLE_OUTPUT=0 SFEM_WRITE_STEPS=0 "$@" \
        "$B/cvfem_hex8_ns_ssgmg" "$d/out" > "$d/log" 2>&1 )
    local rc=$?
    local rb
    rb=$(awk '/^ *vanka_setup /{print $3; exit}' "$d/log" 2>/dev/null); rb=${rb:-"-"}
    awk -F, -v tag="$tag" -v frz="$frz" -v rc="$rc" -v rb="$rb" '
        NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
        { n++; t=$(c["t_step"]); ni=$(c["newton_it"]); li=$(c["lin_it"]); nd=$(c["ndof"]);
          if (n>1) { st+=t; sn+=ni; sl+=li-pl; k++ }
          pl=li }
        END { if (k>0) printf "%-20s %-10d %-7s %-7d %-9.3f %-9.2f %-9.2f %-9s %s\n",
                     tag, nd, frz, n, st/k, sn/k, sl/k, rb, (rc==0?"":"exit="rc);
              else    printf "%-20s %-10s %-7s %-7s %-9s %-9s %-9s %-9s %s\n",
                     tag, "?", frz, "0", "-", "-", "-", rb, "NO ROWS exit="rc }
    ' "$d/diag.csv" 2>/dev/null || printf "%-20s %s\n" "$tag" "no diag.csv, exit=$rc"
}

one L8_tanh_f4  8  20 4 1200 SFEM_PECLET_BLEND=tanh SFEM_NL_MAX_IT=120

echo
echo "=== phase table"
grep -E "^ *(vanka_setup|precond_total|outer_op) " "$OUT/L8_tanh_f4/log" 2>/dev/null | head -4
echo "=== floor acceptances / step failures"
grep -l "residual floor, accepting as converged" "$OUT"/*/log 2>/dev/null || echo "  none"
grep -h "FAILED to converge" "$OUT"/*/log 2>/dev/null | head -3 || true
echo "=== DONE $(date)"
