#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vanka-freeze-P34
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vanka_freeze_P34_%j.out
#SBATCH --error=vanka_freeze_P34_%j.err
# Does freezing the Vanka smoother across Newton iterations pay?
#
# The cost being attacked was measured in P33 (job 4682351): vanka_setup is 18.1 ms per rebuild
# at 116,212 dof, 126.8 ms at 893,924 and 3,604 ms at 7,014,340 -- roughly dof^1.65 between the
# top two rungs -- which is 11.2% of a time step at L=8 and 31.6% at L=16. It is rebuilt every
# Newton iteration, four of them per step, while a DNS step moves the state by O(dt).
#
# SFEM_VANKA_FREEZE=N rebuilds on the first Newton iteration of each time step and then every
# Nth within it, reusing the previous smoother in between. N=1 is the default and is the old
# behaviour bit for bit.
#
# THE MEASUREMENT IS NOT WALL TIME ALONE. A preconditioner built at an older linearisation can
# raise the Newton and linear counts faster than the skipped rebuilds save, and the blended arm
# is where that is most likely: its defect correction already contracts at only r ~ 0.49 at
# L = 8, so a staler preconditioner has the least margin there. Every row therefore reports
# Newton and linear iterations per step beside t_step, and the rebuild count scraped from the
# phase table, which says directly how many rebuilds the run actually paid for.
#
# THE CONTROL IS N=1. Those rows must reproduce P33 -- L8_off 4.864 s / 4.00 / 84.42,
# L16_off 45.365 s / 4.00 / 59.11, L8_tanh 16.379 s / 26.68 / 221.89. If they do not, the cache
# has changed behaviour at its default and nothing else in the table means anything.
#
# L16 runs 10 steps rather than 20: at ~45 s/step a 20-step run is 15 minutes, and six of them
# do not fit two hours beside the L8 sweep. Step 1 is excluded from every mean either way, so
# 10 steps still averages 9.
#
# The tanh arm gets SFEM_NL_MAX_IT=120. In P33 L16_tanh hit the default cap of 40 after 473
# Newton iterations with the residual still descending monotonically -- slow, not broken -- and
# produced no comparable row at all. The floor fix now accepts a capped stage that is at the
# residual floor, but a stage that is merely slow still needs the budget.
set -u
# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path
# says nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was
# run from; SPIKE_ROOT overrides it for a manual run.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "vanka_freeze.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
B="${BUILD:-$SPIKE_ROOT/build_gate}"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_freeze/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
echo "binary $B/cvfem_hex8_ns_ssgmg"
ls -la "$B/cvfem_hex8_ns_ssgmg"

NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"

printf "%-20s %-10s %-7s %-7s %-9s %-9s %-9s %-9s %s\n" \
       run ndof freeze steps t_step_s newton/st lin/st rebuilds note

one() {  # one <tag> <level> <steps> <freeze> <timeout> [extra env...]
    local tag=$1 lvl=$2 nst=$3 frz=$4 secs=$5; shift 5
    local d=$OUT/$tag; mkdir -p "$d"
    # A real output directory, never /dev/null: the driver writes a coarse mesh there and
    # MPI_Aborts after printing its summary if it cannot.
    ( cd "$d" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_VANKA_FREEZE=$frz \
        SFEM_DT=1e-3 SFEM_NSTEPS=$nst SFEM_BDF_ORDER=2 \
        SFEM_DIAG_CSV="$d/diag.csv" SFEM_ENABLE_OUTPUT=0 SFEM_WRITE_STEPS=0 "$@" \
        "$B/cvfem_hex8_ns_ssgmg" "$d/out" > "$d/log" 2>&1 )
    local rc=$?
    # Rebuilds actually paid for. The phase table prints "vanka_setup <sec> <count> <us> <pct>",
    # and phase_add counts one call per rebuild because the freeze guard wraps the timer too.
    local rb
    rb=$(awk '/^ *vanka_setup /{print $3; exit}' "$d/log" 2>/dev/null); rb=${rb:-"-"}
    # Means over steps 2..N: t_step and newton_it are per step, lin_it is cumulative over the
    # run and so is differenced.
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

echo "=== L=8, 893,924 dof, blend OFF (the shipping scheme) ==="
one L8_off_f1   8  20 1 1800
one L8_off_f2   8  20 2 1800
one L8_off_f4   8  20 4 1800

echo "=== L=16, 7,014,340 dof, blend OFF -- where vanka_setup is 31.6% of the step ==="
one L16_off_f1  16 10 1 2400
one L16_off_f2  16 10 2 2400
one L16_off_f4  16 10 4 2400

echo "=== L=8, blend TANH -- the arm with the least margin (r ~ 0.49) ==="
one L8_tanh_f1  8  20 1 2700 SFEM_PECLET_BLEND=tanh SFEM_NL_MAX_IT=120
one L8_tanh_f2  8  20 2 2700 SFEM_PECLET_BLEND=tanh SFEM_NL_MAX_IT=120
one L8_tanh_f4  8  20 4 2700 SFEM_PECLET_BLEND=tanh SFEM_NL_MAX_IT=120

echo
echo "=== phase table, vanka_setup and the step, for the L16 rows ==="
for tag in L16_off_f1 L16_off_f2 L16_off_f4; do
    printf -- "--- %s\n" "$tag"
    grep -E "^ *(vanka_setup|fine_assembly|newton_residual|precond_total) " "$OUT/$tag/log" 2>/dev/null | head -6
done

echo
echo "=== any floor acceptances or step failures? (the floor fix under a stale preconditioner) ==="
grep -l "residual floor, accepting as converged" "$OUT"/*/log 2>/dev/null || echo "  none"
grep -h "FAILED to converge" "$OUT"/*/log 2>/dev/null | head -5 || true

echo "=== DONE $(date)"
