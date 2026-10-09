#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-f32
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=f32_%j.out
#SBATCH --error=f32_%j.err
#
# THE KERNELS FULLY IN SINGLE PRECISION, against double, on Grace.
#
# cvfem_hex8_ns_upwind_bench_f32 is the same driver compiled with the computation type set to
# float: every field, staged pack, cached adjugate table and kernel is f32, and the lane-blocked
# sweeps run 32 elements per lane group where double runs 16 (one 128-byte group either way).
# Its operators are held to the double build's by src/tests/verification/cvfem_bench_f32.sh.
#
# Both binaries in ONE allocation, order alternating every pass, medians of PASSES -- the only
# form that survives the 5-11% node-to-node variation. The best of the passes is printed beside
# the median because the paper reports bests (the campaign's convention: background load only
# ever slows a run). Every layout: packed, the element-coloured (`ecolor`, the paper's "col."),
# the pack-coloured (`colored`) and the standard atomic one, because a precision change that helps
# one and not another is the format talking, not the precision. Pack colouring runs the
# first-order operators only: it has no higher-order correction, and the driver refuses it there.
# The timing state is the unperturbed one every other job in jobs/ measures.
#
# The four residual rungs (res, res_rc, res_bnd, res_bnd_dt) are the completeness ladder of
# scripts/layout_campaign.sh with the same flags, so the paper's ladder table can carry an f32
# group measured here.
#
#   SIZES   problem sizes (cube edge n; dofs = 4 (n+1)^3). 128 is the paper's; the others say
#           where the float build saturates, since it moves half the bytes per dof.
#   ARMS    which operators run at every size; the full list runs at 128 only.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1
B64=$PWD/build/cvfem_hex8_ns_upwind_bench
B32=$PWD/build/cvfem_hex8_ns_upwind_bench_f32
for b in "$B64" "$B32"; do [ -x "$b" ] || { echo "not built: $b (build both on the login node)" >&2; exit 2; }; done
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
PASSES=${PASSES:-5}; REPEAT=${REPEAT:-10}
OUT=${OUT:-$SCRATCH/cvfem_f32/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"; CSV=$OUT/f32.csv
echo "### host $(hostname)  passes=$PASSES repeat=$REPEAT threads=72  $(date '+%F %T')"
echo "### csv: $CSV"

ALL="packed ecolor colored atomic"; HO="packed ecolor atomic"
FULL=( "res|residual, bare||$ALL"
       "res_rc|residual + Rhie-Chow|--rhie-chow|$ALL"
       "res_bnd|residual + Rhie-Chow + boundary|--rhie-chow --boundary|$ALL"
       "res_bnd_dt|residual + Rhie-Chow + boundary + transient|--rhie-chow --boundary --transient 1e-2|$ALL"
       "res_ho0|residual + correction, unlimited|--rhie-chow --conv-ho 0|$HO"
       "res_ho2|residual + correction, Venkatakrishnan|--rhie-chow --conv-ho 2|$HO"
       "res_ho3|residual + correction, Darwish-Moukalled|--rhie-chow --conv-ho 3|$HO"
       "jac|Jex bare|--jac-action|$ALL"
       "jac_rc|Jex + Rhie-Chow|--jac-action --rhie-chow|$ALL"
       "jac_ho0|Jhoex unlimited|--jac-action --rhie-chow --conv-ho 0|$HO"
       "jac_ho2|Jhoex Venkatakrishnan|--jac-action --rhie-chow --conv-ho 2|$HO"
       "jac_ho3|Jhoex Darwish-Moukalled|--jac-action --rhie-chow --conv-ho 3|$HO"
       "ngrad3|nodal velocity gradient, 9 components|--nodal-grad 3|$ALL"
       "spmv|assembled SpMV (first order, lagged)|--bsr-apply|packed" )
SWEEP=( "res_rc|residual + Rhie-Chow|--rhie-chow|packed"
        "res_ho3|residual + correction, Darwish-Moukalled|--rhie-chow --conv-ho 3|packed"
        "jac_rc|Jex + Rhie-Chow|--jac-action --rhie-chow|packed"
        "jac_ho3|Jhoex Darwish-Moukalled|--jac-action --rhie-chow --conv-ho 3|packed" )

declare -A V
run() {  # <n> <key> <layout> <extra>
    local n=$1 key=$2 lay=$3 extra=$4
    for p in $(seq 1 "$PASSES"); do
        if [ $((p % 2)) -eq 1 ]; then order="64 32"; else order="32 64"; fi
        for prec in $order; do
            local b=$B64; [ "$prec" = 32 ] && b=$B32
            local v
            v=$(cvfem_uenv "$b" --n "$n" --repeat "$REPEAT" --warmup 3 --layout "$lay" $extra \
                    --csv "$CSV" --tag "f${prec}_${key}_${lay}_n${n}_p${p}" 2>&1 \
                    | sed -n 's/^  MDOF\/s: //p' | head -1)
            [ -n "$v" ] || { echo "### MISSING f$prec $key $lay n=$n pass $p"; continue; }
            V["$n|$key|$lay|$prec"]+="$v "
        done
    done
}
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
bestof() { printf '%s\n' "$@" | sort -g | tail -1; }
report() {  # <n> <array name>
    local n=$1; local -n arms=$2
    echo
    echo "### n=$n  ($((4 * (n + 1) * (n + 1) * (n + 1))) dofs)"
    printf '%-10s %-46s %-7s %9s %9s %8s %9s %9s\n' key operator layout f64 f32 f32/f64 f64_best f32_best
    for a in "${arms[@]}"; do
        IFS='|' read -r key label extra lays <<<"$a"
        for lay in $lays; do
            [ -n "${V["$n|$key|$lay|64"]:-}" ] || continue
            local m64 m32 b64 b32
            m64=$(median ${V["$n|$key|$lay|64"]}); m32=$(median ${V["$n|$key|$lay|32"]:-0})
            b64=$(bestof ${V["$n|$key|$lay|64"]}); b32=$(bestof ${V["$n|$key|$lay|32"]:-0})
            printf '%-10s %-46s %-7s %9.1f %9.1f %8.3f %9.1f %9.1f\n' "$key" "$label" "$lay" "$m64" "$m32" \
                   "$(awk -v a="$m64" -v b="$m32" 'BEGIN{print (a>0)?b/a:0}')" "$b64" "$b32"
        done
    done
}

for a in "${FULL[@]}"; do
    IFS='|' read -r key label extra lays <<<"$a"
    for lay in $lays; do run 128 "$key" "$lay" "$extra"; done
done
report 128 FULL
for n in ${SIZES:-64 96 160 192}; do
    for a in "${SWEEP[@]}"; do
        IFS='|' read -r key label extra lays <<<"$a"
        for lay in $lays; do run "$n" "$key" "$lay" "$extra"; done
    done
    report "$n" SWEEP
done
echo
echo "### done $(date '+%F %T')"
