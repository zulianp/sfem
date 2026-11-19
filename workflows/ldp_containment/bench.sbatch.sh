#!/bin/bash
#SBATCH --job-name=ldp-bench
#SBATCH --account=c40
#SBATCH --partition=debug
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The valve-containment benchmark (submit.sh sets the nodes and the configurations): every case of
# $LDP_CASE_NAMES under every configuration of $LDP_CONFIGS, $LDP_REPEATS times, the repeats
# outermost. Each run gets $LDP_RESULTS/<job>/<case>/<config>/r<k>/ with the playground's log,
# trace.csv, history.csv and env.txt; the mesh and fields only for the first repeat.
# python/collect.py turns the folders into CSVs.
#
# Configurations, over the job's nodes (4 GH200 modules of 72 cores and a GPU each):
#   host_RxT  R ranks of T threads on Grace, MPICH's GPU support off (its check of every buffer
#             is slow since driver 590, CSCS cray-mpich known issues)
#   gpu_R     R ranks, a GPU and its module's 72 cores each; GPU-aware MPI, with CSCS's
#             cuPointerGetAttribute override preloaded when it is there
set -u
: "${LDP_CASES:?the folder of the case folders (case.yaml, db/)}"
: "${SFEM_BUILD:?the build folder holding playground}"
LDP_CASE_NAMES=${LDP_CASE_NAMES:-FF200_ExodusMesh FP70_similar_coarse FP70_similar FF200_mesh_refined}
LDP_CONFIGS=${LDP_CONFIGS:-host_1x72 host_4x72 host_16x18 gpu_1 gpu_4}
LDP_REPEATS=${LDP_REPEATS:-3}
LDP_RESULTS=${LDP_RESULTS:-$SCRATCH/ldp_bench}
LDP_SET=${LDP_SET:-}  # more --set key=value for every run
OVERRIDE=/capstor/store/cscs/cscs/public/temp/cuptrgetattr_override.so

PG=$SFEM_BUILD/playground
NODES=${SLURM_JOB_NUM_NODES:-1}
OUT=$LDP_RESULTS/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores

# config -> kind ranks threads, or nothing for a config this job's nodes cannot hold. On several
# nodes a config must fill each one (288 Grace cores, 4 GPUs): a second node only after the first.
parse() {
  local c=$1 kind ranks threads
  case $c in
    host_*x*) kind=host; ranks=${c#host_}; threads=${ranks#*x}; ranks=${ranks%x*} ;;
    gpu_*) kind=gpu; ranks=${c#gpu_}; threads=72 ;;
    *) return 1 ;;
  esac
  local per=$((ranks / NODES))
  [ $((per * NODES)) -eq "$ranks" ] && [ "$per" -ge 1 ] || return 1
  [ $((per * threads)) -le 288 ] || return 1
  [ "$kind" = host ] || [ "$per" -le 4 ] || return 1
  [ "$NODES" -eq 1 ] || { [ "$kind" = host ] && [ $((per * threads)) -eq 288 ]; } || \
    { [ "$kind" = gpu ] && [ "$per" -eq 4 ]; } || return 1
  echo "$kind $ranks $threads $per"
}

run() { # case config repeat
  local case=$1 config=$2 repeat=$3 kind ranks threads per
  read -r kind ranks threads per <<< "$(parse "$config")" || return 0
  local dir=$OUT/$case/$config/r$repeat
  rm -rf "$dir"; mkdir -p "$dir"
  local extra=() pre="" gpu_support=0
  if [ "$kind" = gpu ]; then
    extra=(--set execution_space=device)
    gpu_support=1
    [ -f "$OVERRIDE" ] && pre=$OVERRIDE
  fi
  cd "$LDP_CASES/$case" || return 0
  local start end rc
  start=$(date +%s.%N)
  # shellcheck disable=SC2086
  LD_PRELOAD=$pre MPICH_GPU_SUPPORT_ENABLED=$gpu_support OMP_NUM_THREADS=$threads SMESH_TRACE_FILE=$dir/trace.csv \
    srun -N "$NODES" -n "$ranks" --ntasks-per-node="$per" --cpus-per-task="$threads" --cpu-bind=cores \
    --distribution=block:block "$PG" case.yaml --set output.path="$dir" --set output.history_csv="$dir/history.csv" \
    "${extra[@]}" $LDP_SET > "$dir/run.log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  cat > "$dir/env.txt" <<EOF
case=$case
config=$config
kind=$kind
nodes=$NODES
ranks=$ranks
ranks_per_node=$per
threads=$threads
gpus=$([ "$kind" = gpu ] && echo "$ranks" || echo 0)
repeat=$repeat
job=${SLURM_JOB_ID:-local}
rc=$rc
wall=$(echo "$end - $start" | bc)
start=$(date -d @"${start%.*}" -Iseconds)
hosts=${SLURM_JOB_NODELIST:-$(hostname)}
mpich_gpu_support=$gpu_support
preload=$pre
playground=$PG
playground_mtime=$(stat -c %Y "$PG")
case_yaml_sha=$(sha256sum case.yaml | cut -c1-16)
set=$LDP_SET
EOF
  [ "$repeat" -gt 1 ] && rm -rf "$dir/mesh" "$dir/out" "$dir/sdf"
  printf "%-20s %-12s r%d rc=%d %7.2f s\n" "$case" "$config" "$repeat" "$rc" "$(echo "$end - $start" | bc)"
}

echo "ldp bench: job ${SLURM_JOB_ID:-local}, $NODES node(s), configs: $LDP_CONFIGS, repeats: $LDP_REPEATS -> $OUT"
for config in $LDP_CONFIGS; do
  parse "$config" > /dev/null || echo "  $config does not fit $NODES node(s): skipped"
done
for repeat in $(seq 1 "$LDP_REPEATS"); do
  for config in $LDP_CONFIGS; do
    for case in $LDP_CASE_NAMES; do run "$case" "$config" "$repeat"; done
  done
done
