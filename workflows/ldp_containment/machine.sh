# The machine the LDP paper's job scripts run on (workflows/ldp_containment and
# workflows/prony_torsion/strip), sourced once SFEM_ROOT, the sfem tree the scripts come from, is set.
# Each variable keeps its value from the environment; the defaults are daint's, where the paper's runs
# were made. The paper's harness (xsesfsi_papers, ldp_containment/reproduce) sets them elsewhere.
#   SFEM_BUILD          the build whose playground, bench_prony_history, smesh tools (external/smesh)
#                       and sdf_round (bin) the runs use; unset, each script keeps its daint build
#   SFEM_DIAG_BUILD     a host build with SFEM_ENABLE_PRONY_DIAGNOSTICS=ON (the updates a format loses)
#   SFEM_PY             python with numpy, scipy and pyyaml
#   SFEM_RUNS           the collections, renders and logs
#   LDP_CASES           the pumps' case folders (LDP24A_r100, ...)
#   PAPER_PY            the paper's collectors (xsesfsi_papers/ldp_containment/python)
#   SFEM_THREADS        cores per run
#   SFEM_RUNS_PER_NODE  runs at once in one job
#   SFEM_LAUNCH_GPU     the launch of one run on one GPU and SFEM_THREADS cores ("": the binary itself)
#   SFEM_LAUNCH_HOST    the launch of one run on SFEM_THREADS cores ("": the binary itself)
#   SFEM_PVBATCH        ParaView's pvbatch, for the renders
SFEM_DIAG_BUILD=${SFEM_DIAG_BUILD:-$SFEM_ROOT/build-diag}
SFEM_PY=${SFEM_PY:-$SFEM_ROOT/venv/bin/python}
SFEM_RUNS=${SFEM_RUNS:-$SFEM_ROOT/runs}
LDP_CASES=${LDP_CASES:-/ritom/scratch/cscs/zulianp/ldp_cases}
PAPER_PY=${PAPER_PY:-/ritom/scratch/cscs/zulianp/xsesfsi_papers/ldp_containment/python}
SFEM_THREADS=${SFEM_THREADS:-72}
SFEM_RUNS_PER_NODE=${SFEM_RUNS_PER_NODE:-4}
SFEM_LAUNCH_GPU=${SFEM_LAUNCH_GPU-srun --exact -n 1 -c $SFEM_THREADS --gpus-per-task=1}
SFEM_LAUNCH_HOST=${SFEM_LAUNCH_HOST-srun --exact -n 1 -c $SFEM_THREADS}
SFEM_PVBATCH=${SFEM_PVBATCH:-pvbatch}
export OMP_NUM_THREADS=$SFEM_THREADS OMP_PROC_BIND=true OMP_PLACES=cores MPICH_GPU_SUPPORT_ENABLED=0

# Waits while SFEM_RUNS_PER_NODE runs are going.
sfem_throttle() { while [ "$(jobs -rp | wc -l)" -ge "$SFEM_RUNS_PER_NODE" ]; do wait -n; done; }
