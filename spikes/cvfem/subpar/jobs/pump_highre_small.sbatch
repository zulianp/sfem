#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-highre
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=highre_%j.out
#SBATCH --error=highre_%j.err
#
# High Reynolds number on a small mesh with a fine timestep, to find out whether the case
# runs there at all before a day of wall clock is committed to it.
#
# Small in space and fine in time is the right shape for this question rather than a
# compromise. What fails at high Re is the nonlinear solve, and that is a property of the
# Reynolds number and the timestep, not of the mesh -- so the cheapest mesh that still
# resolves the port answers it. dt = 1/256 of the period is 8x finer than the cycles run so
# far, which is what a jet at Re >= 1000 needs to be followed rather than stepped over.
#
# Reported per configuration: whether it converged, how many continuation stages it needed
# (one per step means the ramp never had to reappear), and the swept-volume identity, which
# is the only correctness measure this case has and must hold at any Reynolds number.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_highre/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
DT=${HIGHRE_DT:-0.00390625}   # 1/256 of the period
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT    dt=$DT (1/256 period)"
echo
printf "%-18s %-9s %-6s %-7s %-8s %-9s %-9s %-10s %s\n" \
    run ndof Re steps stages conv lin_it s_per_step "|port-swept|"

one() {  # one <tag> <N> <mu> <nsteps>
    local tag=$1 n=$2 mu=$3 ns=$4
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && timeout 560 uenv run $U -- env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=pump SFEM_N=$n SFEM_MU=$mu SFEM_U=1 \
        SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi \
        SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=6000 \
        SFEM_DT=$DT SFEM_NSTEPS=$ns SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 \
        SFEM_WRITE_STEPS=1 \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd st sg c li ts sp err
    nd=$( grep -o 'ndof: [0-9]*'           "$r/log" | head -1 | awk '{print $2}')
    st=$( grep -cE "^pump: t ="            "$r/log")
    sg=$( grep -cE "^stage [0-9]+/"        "$r/log")
    c=$(  grep -o 'newton_converged: [01]' "$r/log" | head -1 | awk '{print $2}')
    li=$( grep -o 'lin_it_total: [0-9]*'   "$r/log" | head -1 | awk '{print $2}')
    ts=$( grep -o 't_solve: [0-9.]*'       "$r/log" | head -1 | awk '{print $2}')
    err=$(grep -o '|port - swept| [0-9.e+-]*' "$r/log" | head -1 | awk '{print $NF}')
    sp=$(awk -v t="${ts:-0}" -v s="${st:-1}" 'BEGIN{printf "%.1f", (s>0)?t/s:0}')
    printf "%-18s %-9s %-6s %-7s %-8s %-9s %-9s %-10s %s\n" \
        "$tag" "${nd:-?}" "$(awk -v m=$mu 'BEGIN{printf "%g", 1/m}')" \
        "${st:-0}/$ns" "${sg:-?}" "${c:-?}" "${li:-?}" "$sp" "${err:-not reached}"
}

one n16_re1000  16 0.001  48
one n16_re2000  16 0.0005 48
one n16_re5000  16 0.0002 48
one n24_re1000  24 0.001  24

echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
