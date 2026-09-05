# CVFEM semi-structured multigrid: kernels, per level

What runs at each level of the V-cycle, and what it costs.

**Measurement configuration.** 12x3x3 macro-elements (108) at lattice level 8, Poiseuille,
Re=100, on one Grace socket (72 cores, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`). Fixed work:
one Newton step, 60 preconditioner applications, 40 linear iterations. SFEM and smesh built
**with OpenMP** (`installations/sfem-omp`); the default build has it off, which makes every
smesh kernel serial -- see the note at the end.

## The hierarchy

| level | lattice | nodes | **dofs** | operator | smoother |
|-------|---------|-------|----------|----------|----------|
| L0 | 8 | 60,625 | **242,500** | matrix-free | damped block-Jacobi |
| L1 | 4 |  8,281 |  **33,124** | assembled BSR 4x4 | damped block-Jacobi |
| L2 | 2 |  1,225 |   **4,900** | assembled BSR 4x4 | damped block-Jacobi |
| L3 | 1 |    208 |     **832** | assembled BSR 4x4 | dense LU (exact) |

Coarse operators are **element-wise Galerkin**: level 1 is assembled from the fine
macro-elements as `sum_e P_e^T A_e P_e`, and each level below is one element-local coarsening
hop from the level above. Nothing is probed at any level.

## Kernels

| role | kernel | where |
|------|--------|-------|
| L0 operator / smoother apply | `sscvfem_apply_macro_local_hoisted` | `cvfem_sshex8_ns.hpp` |
| L0 scatter | `sscvfem_scatter_element_w<4>` + `sscvfem_reduce_shared_w<4>` | two-pass, no atomics, bitwise reproducible |
| L1-L3 operator | `sfem::bsr_spmv_static_4x4` | `sfem_BSR.hpp` |
| L0-L2 smoother | `BlockJacobi::apply` (4x4 blocks) | driver |
| L3 solve | `DenseLU::apply` | driver, densified from the BSR in O(nnz) |
| prolongation | `smesh::sshex8_prolongate` | smesh |
| restriction | `smesh::sshex8_restrict` | smesh |
| last-hop restriction | `smesh::sshex8_hierarchical_restriction` | smesh |
| micro-cell matrix | `cvfem_hex8_ns_upwind_jacobian_add_slots` with identity slots | yields the dense 8x8-block cell matrix |
| coarse assembly | `galerkin_assemble`, `galerkin_hop` | `cvfem_ss_galerkin.hpp` |

## Cost per call, and throughput

Throughput is the level's dofs divided by the call time; for transfers it is the *fine* side's
dofs, since that is what the kernel streams.

| phase | us/call | dofs | MDOF/s | share of cycle |
|-------|---------|------|--------|----------------|
| `smooth[L0]` (3 sweeps) | 1240.8 | 242,500 | 195 | **50.7%** |
| `op[L0]` | 375.3 | 242,500 | 646 | 7.7% |
| `outer_op` | 375.7 | 242,500 | 646 | 7.8% |
| `smooth[L1]` | 153.2 | 33,124 | 216 | 6.3% |
| `smooth[L2]` | 129.3 | 4,900 | **38** | 5.3% |
| `coarse_solve` (L3) | 208.3 | 832 | **4** | 4.3% |
| `restrict[L0->1]` | 134.3 | 242,500 | 1806 | 2.7% |
| `prolong[L1->0]` | 86.2 | 242,500 | 2813 | 1.8% |
| `restrict[L2->3]` | 179.1 | 4,900 | **27** | 3.7% |
| `galerkin_assembly` (per Newton step) | 3711 | -- | -- | 3.7% |

Total 0.401 s in phases, `t_solve` 0.406 s, 10160 us per linear iteration.

## What the numbers say

**The fine level dominates and is efficient.** `smooth[L0]` is half the cycle at 195 MDOF/s;
the bare operator apply runs at 646 MDOF/s, and the smoother is three sweeps plus a block-
diagonal solve, so those are consistent. Counters for the whole cycle: IPC 3.43, 0.70% cache
misses, 41.6% backend stalls, 2.73 GHz effective per core -- loaded, and approaching
memory-bound rather than holding idle headroom.

**Efficiency collapses on the coarse levels, by two orders of magnitude.** 195 MDOF/s at L0,
216 at L1, 38 at L2, 4 at the coarsest. The work per level falls 8x per hop while the parallel
overhead does not, so by L2 the barriers cost more than the arithmetic. `restrict[L2->3]` at
179 us is *dearer than* `restrict[L0->1]` at 134 us despite moving 50x fewer dofs -- the
clearest single symptom.

**Barrier waiting is now the largest item after the fine apply.** `libgomp` is about 46% of
profile samples, spread over a dozen entries. That is threads spinning, not runtime cost: 108
macro-elements over 72 threads is 1.5 each, so the critical path is 2 elements while half the
threads idle. 54 threads (108/54 = 2 exactly) measures 0.359 s against 72 threads' 0.406 s.

**Two environment settings matter more than they should.** `OMP_WAIT_POLICY=active` is worth
4%; `passive` is **4.2x worse** and `GOMP_SPINCOUNT=0` **4.8x worse**. A job script that sets a
conservative wait policy will lose most of the performance with no visible symptom.

## Build requirement

`SFEM_ENABLE_OPENMP` and `SMESH_ENABLE_OPENMP` default to **OFF**, and with them off every
`#pragma omp` in smesh compiles to nothing -- `libsmesh.a` contains zero `GOMP_parallel`
symbols. The transfers then run serially: `restrict[L0->1]` costs 1456 us instead of 134 us,
and the whole cycle 0.678 s instead of 0.397 s. Check with

    nm -C <prefix>/lib64/libsmesh.a | grep -c GOMP_parallel   # must be non-zero

before trusting any timing from this hierarchy.
