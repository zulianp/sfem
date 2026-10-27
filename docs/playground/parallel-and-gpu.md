# Parallel and GPU runs

## Threads

On the host the operators and solvers are threaded with OpenMP (a build with
`SFEM_ENABLE_OPENMP=ON`). Set the threads and pin them:

```sh
OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores build/playground case.yaml
```

## MPI

The same case runs on several ranks:

```sh
mpiexec -n 4 build/playground case.yaml
```

The mesh read from `mesh.path` is partitioned across the ranks; output is gathered by rank 0 onto
the whole mesh, so `output.path` needs `mesh.path`. Under MPI the following are not available:
`mesh.refine`, `mesh.promote_to_p2`, the `ssmgc` obstacle and `output.quantities`; initial
conditions are given in the mesh file's node numbering. An SDF grid read from file under MPI is
split into z-slabs; each rank gathers the slab its nodes need.

## GPU

`execution_space: device` (or `--set execution_space=device`) runs the assembly, the solvers and
the contact QP on the GPU, in a build with `SFEM_ENABLE_CUDA=ON`. On the device, rotations
(`constraints`) and Dirichlet conditions with a profile are not available.

With MPI each rank uses one GPU. Within a node, `nccl: true` lets the exchange between ranks and
the contact QP's reductions go through NCCL instead of MPI. NCCL's set-up costs about 3 s, which
pays off from about 90,000 QP iterations on Alps; between nodes MPI is used. It needs a build in
which `SFEM_ENABLE_NCCL` found NCCL.

## On Alps (CSCS)

The GH200 nodes have four modules, each a 72-core Grace CPU and a Hopper GPU. Build on the login
node in the `prgenv-gnu/24.11:v2` uenv (see [building](../building.md)) and run on the compute
nodes. Fill whole nodes: 4 ranks of 72 threads, or 4 GPU ranks, per node.

```sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
export OMP_PROC_BIND=true OMP_PLACES=cores

# CPU: 4 ranks x 72 threads; MPICH's GPU support off
MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=72 \
  srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=cores \
  $SFEM_BUILD/playground case.yaml --set output.path=$SCRATCH/run_cpu

# GPU: 4 ranks, one GPU each; GPU-aware MPI
MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=72 \
  srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=72 --cpu-bind=cores \
  $SFEM_BUILD/playground case.yaml --set output.path=$SCRATCH/run_gpu --set execution_space=device
```

`MPICH_GPU_SUPPORT_ENABLED=1` makes MPICH check every buffer it sends, which slows host runs:
keep it at 0 unless the run is on the device. `workflows/ldp_containment/bench.sbatch.sh` is a
complete benchmark job, with a timing trace per run (`SMESH_TRACE_FILE`).
