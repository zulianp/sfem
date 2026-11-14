#!/usr/bin/env bash
#SBATCH --job-name=bench-mamal-mi300a
#SBATCH --account=def-teseo_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=mi300a:1
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-bench-mamal-mi300a-%j.out
# The MaMAL benchmark (bench_MaMAL.sbatch.sh, whose header is GH200's on Alps) on one AMD MI300A of
# Nibi: an APU, so the run takes its 24 Zen 4 cores of the node's 96. Submitted from this folder:
#   sbatch --export=ALL,SFEM_BIN=<HIP build>/sfem_MGSDFContactTest,HIP_ENV=<env script> bench_MaMAL_mi300a.sbatch.sh
# The environment script loads the HIP toolchain the build used (Nibi: hip-mpi-env.sh).
set -u
if [[ -n "${HIP_ENV:-}" ]]; then
    source "${HIP_ENV}"
fi
exec bash "${SLURM_SUBMIT_DIR}/bench_MaMAL.sbatch.sh"
