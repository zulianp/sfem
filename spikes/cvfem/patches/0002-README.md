# sshex8_restrict: persistent thread-private scratch

Applies to `external/smesh` (the submodule the SFEM build compiles via
`add_subdirectory`, see `cmake/SFEMDependencies.cmake:55`). Already applied in the working
tree; this file exists so the change can be carried to the standalone smesh repo.

`sshex8_restrict` allocated its scratch with `SMESH_ALLOC` (malloc) *inside* the OpenMP
parallel region and freed it at the end -- `2 + 2*vec_size` buffers, once per thread per
call. Those buffers carry no data between calls, and a multigrid cycle calls the routine
once per level per sweep, so a solve ran hundreds of thousands of allocations for nothing.
The patch replaces them with function-local `thread_local` vectors; `resize()` is a no-op
once they are large enough, so the steady state allocates nothing.

**This is a cleanliness fix, not a speed fix, and the measurement says so:** 1446 to 1521 us
per restriction before and after at 242,500 dofs on 72 Grace cores, inside run-to-run spread.
It was written while chasing why the transfers do not scale, and it is not the reason.

**The reason is that SFEM and smesh are compiled without OpenMP.** The build carries
`SFEM_ENABLE_OPENMP=OFF` and `SMESH_ENABLE_OPENMP=OFF`, `libsmesh.a` contains zero
`GOMP_parallel` symbols, and the smesh compile line is `-O3 -DNDEBUG -std=gnu++17 -fPIC`
with no `-fopenmp`. Every `#pragma omp` in smesh compiles to nothing, so the transfers are
serial regardless of what the source says. Enabling OpenMP in the build is the actual fix;
this patch is worth keeping for when that happens, since the allocations become per-thread
per-call at that point.

Correctness after the patch, all gates: element-wise Galerkin levels 1.8e-16 / 1.9e-17 /
1.1e-16, their block diagonals 4.1e-18 / 4.4e-18 / 0, coarse LU 0.0, transfer adjointness
1.000000 at every hop.
