#include "sfem_InertiaPotential_cuda.hpp"

#include <algorithm>

#include "sfem_cuda_base.hpp"

namespace sfem {

    namespace {

        /// One thread per row: walk the row and add to the entry on the
        /// diagonal.  A grid-stride loop, like the rest of this tree's
        /// vector-shaped kernels, so the launch does not have to size itself to
        /// the problem.
        __global__ void inertia_potential_crs_diagonal(const ptrdiff_t                    ndofs,
                                                       const count_t *const SFEM_RESTRICT rowptr,
                                                       const idx_t *const SFEM_RESTRICT   colidx,
                                                       const real_t *const SFEM_RESTRICT  mass,
                                                       const real_t                       alpha,
                                                       real_t *const SFEM_RESTRICT        values) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ndofs;
                 i += blockDim.x * gridDim.x) {
                const count_t begin = rowptr[i];
                const count_t end   = rowptr[i + 1];
                for (count_t k = begin; k < end; ++k) {
                    if (colidx[k] == i) {
                        values[k] += alpha * mass[i];
                        break;
                    }
                }
            }
        }

        /// One thread per node.  The diagonal block takes one mass entry per
        /// component, which is the same contraction the host loop makes.
        __global__ void inertia_potential_bsr_diagonal(const ptrdiff_t                    n_nodes,
                                                       const int                          block_size,
                                                       const count_t *const SFEM_RESTRICT rowptr,
                                                       const idx_t *const SFEM_RESTRICT   colidx,
                                                       const real_t *const SFEM_RESTRICT  mass,
                                                       const real_t                       alpha,
                                                       real_t *const SFEM_RESTRICT        values) {
            const ptrdiff_t bs2 = (ptrdiff_t)block_size * block_size;
            for (ptrdiff_t node = blockIdx.x * blockDim.x + threadIdx.x; node < n_nodes;
                 node += blockDim.x * gridDim.x) {
                const count_t begin = rowptr[node];
                const count_t end   = rowptr[node + 1];
                for (count_t k = begin; k < end; ++k) {
                    if (colidx[k] == node) {
                        real_t *const block = &values[k * bs2];
                        for (int d = 0; d < block_size; ++d) {
                            block[d * block_size + d] += alpha * mass[node * block_size + d];
                        }
                        break;
                    }
                }
            }
        }

        ptrdiff_t grid_for(const ptrdiff_t n, const int block) {
            return std::max(ptrdiff_t(1), (n + block - 1) / block);
        }

    }  // namespace

    int cu_inertia_potential_hessian_crs(const ptrdiff_t      ndofs,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         const real_t *const  mass,
                                         const real_t         alpha,
                                         real_t *const        values) {
        SFEM_DEBUG_SYNCHRONIZE();

        const int block = 128;
        inertia_potential_crs_diagonal<<<grid_for(ndofs, block), block>>>(
                ndofs, rowptr, colidx, mass, alpha, values);

        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

    int cu_inertia_potential_hessian_bsr(const ptrdiff_t      n_nodes,
                                         const int            block_size,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         const real_t *const  mass,
                                         const real_t         alpha,
                                         real_t *const        values) {
        SFEM_DEBUG_SYNCHRONIZE();

        const int block = 128;
        inertia_potential_bsr_diagonal<<<grid_for(n_nodes, block), block>>>(
                n_nodes, block_size, rowptr, colidx, mass, alpha, values);

        SFEM_DEBUG_SYNCHRONIZE();
        return SFEM_SUCCESS;
    }

}  // namespace sfem
