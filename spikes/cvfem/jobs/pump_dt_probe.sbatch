#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pumpdt
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pumpdt_%j.out
#SBATCH --error=pumpdt_%j.err
#
# Does a smaller timestep pay for itself?
#
# The transient term puts a0 rho / dt times the control volume on each VELOCITY diagonal and
# nothing on pressure (cvfem_hex8_layout_common.hpp:725), so halving dt doubles exactly the
# block that block-Jacobi inverts. Each solve gets easier. It also doubles the number of
# solves, so the only comparison that means anything is cost for the same PHYSICAL time --
# which is what this measures, by holding t_end fixed and varying dt and the step count
# together.
#
# Reported per configuration: linear iterations and wall clock for the same 0.125 of a
# period, so a row that halves dt and more than halves the work is a win and one that does
# not is a loss.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_pumpdt/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
N=${PUMP_N:-32}
MU=${PUMP_MU:-0.005}
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "N=$N  mu=$MU (Re=$(awk -v m=$MU 'BEGIN{printf "%g", 1/m}'))  t_end=0.125 of one period"
echo
printf "%-10s %-7s %-9s %-8s %-10s %-11s %-12s %s\n" \
    dt steps ndof conv lin_it t_solve it_per_step "s per 0.125 period"
for spec in "0.0625 2" "0.03125 4" "0.015625 8" "0.0078125 16"; do
    set -- $spec; dt=$1; ns=$2
    r=$OUT/dt$dt; mkdir -p "$r"
    ( cd "$r" && timeout 700 uenv run $U -- env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=pump SFEM_N=$N SFEM_MU=$MU SFEM_U=1 \
        SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi \
        SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=4000 \
        SFEM_DT=$dt SFEM_NSTEPS=$ns SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 \
        "$D" "$r/out" > "$r/log" 2>&1 )
    nd=$(grep -o 'ndof: [0-9]*'           "$r/log" | head -1 | awk '{print $2}')
    c=$( grep -o 'newton_converged: [01]' "$r/log" | tail -1 | awk '{print $2}')
    li=$(grep -o 'lin_it_total: [0-9]*'   "$r/log" | awk '{s+=$2} END{print s}')
    ts=$(grep -o 't_solve: [0-9.]*'       "$r/log" | awk '{s+=$2} END{printf "%.2f", s}')
    ips=$(awk -v l="${li:-0}" -v n=$ns 'BEGIN{printf "%.0f", l/n}')
    printf "%-10s %-7s %-9s %-8s %-10s %-11s %-12s %s\n" \
        "$dt" "$ns" "${nd:-?}" "${c:-?}" "${li:-?}" "${ts:-?}" "$ips" "${ts:-?}"
done
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
