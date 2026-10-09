#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-dns-cost-P33
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=dns_cost_P33_%j.out
#SBATCH --error=dns_cost_P33_%j.err
# Stage 1 of the transient-DNS plan: measure what a time step costs, so the production mesh is
# sized from data instead of from an extrapolation.
#
# Three things have to be measured rather than assumed, and each has a reason:
#
#   a       per-dof cost of a linear iteration. The two anchors in hand are STEADY solves
#           (2.0 s/Newton at 893,924 dof, 0.19 s at 116,212) and imply an exponent of 1.15,
#           mildly superlinear. A transient step is not that: its diagonal carries rho*a0/dt.
#   D(dt)   the transient discount. The rho*a0/dt term regularises the system as dt shrinks,
#           so linear counts fall -- possibly far. Swept here at CFL ~0.5 / 2 / 8.
#   r       the defect-correction rate with SFEM_PECLET_BLEND=tanh. The blend scales the
#           residual's |m| only; the Jacobian keeps full upwind, so Newton stops being Newton
#           and converges linearly at rate r. THIS IS THE GO/NO-GO: n_newton ~ log(rtol)/log(r),
#           so r > 0.5 makes central convection cost ~10x a plain step and the DNS with it.
#
# Sizes from the verified generator formula (core 2 / bore 1 / outer 1, axial "4 3 8 12"):
# L=4 -> 116,212 dof, L=8 -> 893,924, L=16 -> 7,014,340. Levels must be powers of two.
#
# Normal partition, not debug: 20 steps at 7M dof will not fit a 30-minute cap. One socket,
# which is the machine the campaign is planned for.
#
# t_step comes from the diagnostics CSV column added for exactly this. Step 1 is EXCLUDED from
# every mean: a fresh run has no previous step to guess from, so it walks the whole Reynolds
# ramp and costs several times what a settled step does.
set -u
# SLURM copies this script to /var/spool/slurmd/job<id>/slurm_script, so its own path
# says nothing about where the spike is. SLURM_SUBMIT_DIR is the directory sbatch was
# run from; SPIKE_ROOT overrides it for a manual run.
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "dns_cost.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
B="${BUILD:-$SPIKE_ROOT/build_gate}"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_dns_cost/$SLURM_JOB_ID; mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"

NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"

printf "%-22s %-10s %-9s %-8s %-9s %-9s %-9s %s\n" \
       run ndof dt steps t_step_s newton/st lin/st note

one() {  # one <tag> <level> <dt> <timeout> [extra env...]
    local tag=$1 lvl=$2 dt=$3 secs=$4; shift 4
    local d=$OUT/$tag; mkdir -p "$d"
    ( cd "$d" && stdbuf -oL -eL timeout "$secs" env $NOZ SFEM_ELEMENT_REFINE_LEVEL=$lvl \
        "SFEM_NOZZLE_NAXIAL=4 3 8 12" SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=1 \
        SFEM_DT=$dt SFEM_NSTEPS=20 SFEM_BDF_ORDER=2 \
        SFEM_DIAG_CSV="$d/diag.csv" SFEM_ENABLE_OUTPUT=0 SFEM_WRITE_STEPS=0 "$@" \
        "$B/cvfem_hex8_ns_ssgmg" "$d/out" > "$d/log" 2>&1 )
    local rc=$?
    # Means over steps 2..N, from the CSV: t_step directly, newton_it directly, and lin_it by
    # first difference because that column is cumulative over the run.
    awk -F, -v tag="$tag" -v dt="$dt" -v rc="$rc" '
        NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
        { n++; s=$(c["step"]); t=$(c["t_step"]); ni=$(c["newton_it"]); li=$(c["lin_it"]);
          nd=$(c["ndof"]);
          if (n>1) { st+=t; sn+=ni; sl+=li-pl; k++ }
          pl=li }
        END { if (k>0) printf "%-22s %-10d %-9s %-8d %-9.3f %-9.2f %-9.2f %s\n",
                     tag, nd, dt, n, st/k, sn/k, sl/k, (rc==0?"":"exit="rc);
              else    printf "%-22s %-10s %-9s %-8s %-9s %-9s %-9s %s\n",
                     tag, "?", dt, "0", "-", "-", "-", "NO ROWS exit="rc }
    ' "$d/diag.csv" 2>/dev/null || printf "%-22s %s\n" "$tag" "no diag.csv, exit=$rc"
}

echo "=== size ladder, blend OFF (the shipping scheme) ==="
one L4_off   4  1e-3  600
one L8_off   8  1e-3  1800
one L16_off  16 1e-3  5400

echo "=== size ladder, blend TANH (central momentum, defect correction) ==="
one L4_tanh  4  1e-3  900  SFEM_PECLET_BLEND=tanh
one L8_tanh  8  1e-3  2700 SFEM_PECLET_BLEND=tanh
one L16_tanh 16 1e-3  5400 SFEM_PECLET_BLEND=tanh

echo "=== dt sweep at L=8, for D(dt) ==="
one L8_dt025 8  2.5e-4 1800
one L8_dt4e3 8  4e-3   1800

echo
echo "=== defect-correction rate r: consecutive ||R|| within the last step of each tanh run ==="
for tag in L4_tanh L8_tanh; do
    echo "--- $tag"
    awk '/^=== step/ { split("",R); n=0 }
         /^newton [0-9]+ +\|\|R\|\|/ { R[++n]=$4 }
         END { for (i=2;i<=n;i++) if (R[i-1]+0>0) printf "  ratio %d/%d = %.3f\n", i, i-1, R[i]/R[i-1] }' \
        "$OUT/$tag/log" 2>/dev/null | head -8
done

echo "=== DONE $(date)"
