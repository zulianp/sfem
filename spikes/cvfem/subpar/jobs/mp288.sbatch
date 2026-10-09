#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-mp288
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=mp288_%j.out
#SBATCH --error=mp288_%j.err

# INVARIANCE AT A FULL NODE ON A MULTI-PACK CASE -- SIZED SO IT ACTUALLY FITS.
#
# Job 4720087 ran cavity N=16 refine 4 (4096 macro elements, two packs) across ranks and got
# vanka 7/7/7/7 and block-Jacobi 10/10/10/10 at 1/8/32/128 -- but the 288-rank arm was OOM
# KILLED and the job then hit its wall clock before block-Jacobi's 288 arm reported.
#
# The cause is not convergence. That case's coarsest level is 19,652 dofs and is solved by a
# DENSE LU that every rank holds in full: 19652^2 * 8 B = 3.09 GB per rank, so 288 ranks need
# ~890 GB against a node with ~870 GB. It fits at 128 (~395 GB) and does not at 288. The same
# fact shows up in the timings -- coarse_factor was 139.1 s of ~160 s, 46.6% -- which is why
# that case showed no speedup at all from 1 to 128 ranks and must NOT be quoted as a scaling
# result.
#
# WHY NOT JUST SWITCH TO THE ITERATIVE COARSE SOLVE. SFEM_GMG_DENSE_LU_BELOW (default 1<<30,
# in dofs, gating local AND global) would divert the coarsest level to BiCGStab and fit easily.
# But that changes the preconditioner, so the counts could not be compared with the 7 and 10
# above, and the driver's own comment gives the reason to avoid it here: an iterative coarse
# solve that stagnates "hands up a correction that is noise while reporting success" and
# "inherits every nondeterminism below it". A non-flat result would then be ambiguous between a
# decomposition effect and the coarse solver's tolerance behaviour.
#
# So instead the case is SIZED to fit while staying multi-pack:
#
#     cavity N=14 refine 4 -> 2744 macro elements  (> 2048, so TWO packs at the stock default)
#                             57^3 = 185,193 nodes, 740,772 dof
#                             coarsest 15^3 = 3,375 nodes = 13,500 dofs
#                             13500^2 * 8 B = 1.46 GB per rank -> 288 ranks ~ 420 GB, fits
#
# Same dense-LU coarse solve as every other measurement in this session, so the result extends
# the invariance claim without an asterisk.
#
# SELF-VALIDATING. Two preconditions are checked per arm rather than assumed, because an arm
# that silently took the single-pack or the iterative path would look like a clean result:
#   n_packs must be 2      -- the case really is multi-pack
#   "coarse LU:" present   -- the dense path really was taken
#
# READ, committed before the run:
#
#   vanka flat at 1..288        -> convergence is rank independent on a multi-pack case across a
#                                  full node, with the coarse solver unchanged
#   flat but != the 1-rank value -> parallel reaches a different operator than serial; a defect
#   moving with rank count      -> a decomposition defect that only a multi-pack case exposes,
#                                  and it outranks every serial result in this branch
#   any arm OOM or at the cap   -> not a result: OOM is a resource limit and a capped count is a
#                                  floor. Both must be reported as such, not as convergence.
#
# Block-Jacobi is omitted deliberately: it is already flat at 1/8/32/64 on the 143,748-dof case
# (11 everywhere) and the wall-clock budget here is tight, since the 1-rank arm factorises
# 13,500 dofs on a single core.

set -u

_cvfem_probe=scripts/alps_env.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || { echo "error: could not locate spikes/cvfem" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_mp288/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# The 1-rank arm factorises 13,500 dofs on ONE core, so it is the slow one; 280 s bounds a
# deadlock rather than a slow run, and an arm that is merely slow must not read as a timeout.
ARM_TIMEOUT="${ARM_TIMEOUT:-280}"

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=14 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=14 refine 4 -- 2744 macro elements (TWO packs at the stock 2048), 740,772 dof,"
echo "coarsest level 13,500 dofs -> 1.46 GB per rank, fits at 288.  MG + Vanka, pure MPI."
echo
printf "  %-7s %-11s %-11s %-8s %-10s %s\n" "ranks" "lin_it" "converged" "n_packs" "denseLU?" "seconds"
for n in 1 8 32 128 288; do
    tag="mp_n${n}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                      SFEM_SMOOTHER=vanka stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    np=$(grep -a "^n_packs:" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    lu=$(grep -ac "^coarse LU:" "$RAW/${tag}.log")
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
    [ "$rc" = "124" ] && it="TIMEOUT"
    [ "$it" = "4000" ] && it="4000(cap)"
    grep -aq "oom_kill" "$RAW/${tag}.log" && it="OOM-KILLED"
    printf "  %-7s %-11s %-11s %-8s %-10s %s\n" \
        "$n" "${it:-NONE}" "${cv:-?}" "${np:-none}" "$([ "${lu:-0}" -gt 0 ] && echo yes || echo NO)" "${sec:-n/a}"
done
echo
echo "preconditions: n_packs must read 2 and denseLU must read yes on every arm,"
echo "otherwise the arm did not run the configuration this job claims to test."
grep -ah "^nnodes:" "$RAW"/mp_n1.log | head -1
grep -ah "^coarse LU:" "$RAW"/mp_n1.log | head -1
echo "raw logs: $RAW"
