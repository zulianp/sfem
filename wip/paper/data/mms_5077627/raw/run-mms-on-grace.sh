#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
[[ ${SLURM_CPUS_PER_TASK:-0} -ge 8 ]] || { echo 'Request at least 8 CPU cores'; exit 1; }
mkdir -p "$root/results"
out=$(mktemp -d "$root/results/mms-${SLURM_JOB_ID:-local}-XXXXXX")
exec > >(tee "$out/run.log") 2>&1
cd "$root"
for name in ${!SFEM_@} ${!CVFEM_@}; do unset "$name"; done
date -u
hostname
printf 'RESULTS=%s\n' "$out"
sha256sum -c SOURCE-SHA256.txt > "$out/source-check.log"
cp SOURCE-SHA256.txt SOURCE-REVISIONS.txt run-mms-on-grace.sh "$out/"
mpicxx --version | tee "$out/compiler.txt"
lscpu > "$out/lscpu.txt"
cmake -S sfem/spikes/cvfem -B build/cvfem \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
    -DSFEM_DIR="$root/install/lib/cmake" -DCMAKE_PREFIX_PATH="$root/install" \
    -DCVFEM_ENABLE_CUDA=OFF
cmake --build build/cvfem --parallel 8 --target cvfem_hex8_ns_ssgmg
cp build/cvfem/CMakeCache.txt "$out/"
sha256sum build/cvfem/cvfem_hex8_ns_ssgmg > "$out/BUILD-SHA256.txt"
python_bin=$(command -v python3)
if [[ -x /user-environment/env/default/bin/python3 ]]; then
    python_bin=/user-environment/env/default/bin/python3
fi
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SFEM_NL_RTOL=1e-10 SFEM_LSOLVE_RTOL=1e-6 SFEM_RC_EXACT_JAC=1
export SFEM_RC_TAU=1 SFEM_RC_TAU_TRANSIENT=0 SFEM_VERIFY_TOL=0.01
export SFEM_GEOM=affine SFEM_LX=2 SFEM_LY=2 SFEM_LZ=2
export SFEM_PACK_SIZE=1024 SFEM_U=1
export VERIFY_GROUPS=conv CONV_LADDER='16 24 32'
export CONV_ARMS='lo ho_lim0 ho_lim1 ho_lim2_k0 ho_lim3'
export VERIFY_LEVEL=1 VERIFY_SOLVER=fgmres VERIFY_RESTART=480 NL_MAX_IT=40
export VERIFY_RUN_TIMEOUT=300 VERIFY_PYTHON="$python_bin"
export DRIVER="$root/build/cvfem/cvfem_hex8_ns_ssgmg" OUT="$out/study"
env | LC_ALL=C sort | grep -E '^(SFEM_|OMP_|OPENBLAS_|MKL_|VERIFY_|CONV_|NL_MAX_IT|DRIVER=|OUT=)' > "$out/settings.txt"
bash sfem/spikes/cvfem/scripts/verify_report.sh
"$python_bin" sfem/spikes/cvfem/scripts/mms_refinement_summary.py "$out/study"
echo "MMS REFINEMENT COLLECTED; inspect the separate accuracy gates: $out"
