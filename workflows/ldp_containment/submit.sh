#!/bin/bash
# submit.sh: the benchmark's standard set as five debug jobs, each within 30 minutes. One node
# on Grace (one module, a rank per module, the node in 16 x 18) and with 1 and 4 GPUs; two full
# nodes, the node's shapes on all eight modules. The debug QOS takes two submitted jobs per user, so each
# submission waits for a free slot: run it under nohup.
#   LDP_CASES=... SFEM_BUILD=... [LDP_RESULTS=...] [LDP_REPEATS=3] workflows/ldp_containment/submit.sh
set -eu
: "${LDP_CASES:?}" "${SFEM_BUILD:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
LDP_RESULTS=${LDP_RESULTS:-$SCRATCH/ldp_bench}
mkdir -p "$LDP_RESULTS"
export LDP_CASES SFEM_BUILD LDP_RESULTS
submit() { # nodes configs
  while [ "$(squeue -u "$USER" -p debug -h | wc -l)" -ge 2 ]; do sleep 60; done
  LDP_CONFIGS="$2" sbatch --parsable --nodes="$1" --output="$LDP_RESULTS/slurm-%j.out" \
    --export=ALL,LDP_CONFIGS="$2" "$HERE/bench.sbatch.sh"
}
echo "grace, 1 module:  $(submit 1 'host_1x72')"
echo "grace, 1 node:    $(submit 1 'host_4x72 host_16x18')"
echo "gpu, 1 node:      $(submit 1 'gpu_1 gpu_4')"
echo "grace, 2 nodes:   $(submit 2 'host_8x72 host_32x18')"
echo "gpu, 2 nodes:     $(submit 2 'gpu_8')"
