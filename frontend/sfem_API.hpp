#ifndef SFEM_API_HPP
#define SFEM_API_HPP

// An umbrella over the library. What it used to define lives with the module that owns it:
//   algebra:  sfem_blas_factory.hpp, sfem_operator_factory.hpp, sfem_solver_factory.hpp, sfem_device.hpp
//   frontend: sfem_Transfer.hpp, sfem_LinearOperatorFactory.hpp, and the condition factories in
//             sfem_DirichletConditions.hpp, sfem_NeumannConditions.hpp, sfem_ContactConditions.hpp
//   smesh:    derefine_elements (smesh_semistructured.hpp)
// It still includes everything it used to, so existing includers build; new code includes what it uses.
#include "sfem_LinearOperatorFactory.hpp"
#include "sfem_Transfer.hpp"
#include "sfem_blas_factory.hpp"
#include "sfem_device.hpp"
#include "sfem_operator_factory.hpp"
#include "sfem_solver_factory.hpp"

// C includes
#include "matrixio_array.h"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
// #include "sfem_mask.hpp"
#include "smesh_mesh.hpp"
#include "smesh_prolongation.hpp"
#include "smesh_restrict.hpp"
#include "smesh_restriction.hpp"
#include "smesh_sideset.hpp"
#include "smesh_sshex8.hpp"
#include "smesh_sshex8_prolongation.hpp"
#include "smesh_sshex8_restriction.hpp"
#include "smesh_ssquad4.hpp"
#include "smesh_ssquad4_prolongation.hpp"
#include "smesh_sstet4.hpp"
#include "smesh_sstet4_prolongation.hpp"

// C++ includes
#include "acrs.hpp"
#include "scrs.hpp"
#include "sdacrs.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_ContactConditions.hpp"

#include "sfem_CooSym.hpp"
#include "sfem_Function.hpp"
#include "sfem_MixedPrecisionShiftableBlockSymJacobi.hpp"
#include "sfem_Multigrid.hpp"
#include "sfem_ParallelMatrixFreeOperator.hpp"
#include "smesh_env.hpp"
#include "smesh_exchange.hpp"

#include "sell.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BSRTangent.hpp"
#include "sfem_CRS.hpp"
#include "sfem_DIA.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_bcrs_sym_SpMV.hpp"
#include "sfem_cg.hpp"
#include "sfem_parallel_cg.hpp"
#include "sfem_crs_sym_SpMV.hpp"
#include "sfem_mprgp.hpp"
#include "smesh_glob.hpp"
#include "smesh_semistructured.hpp"

// CUDA includes
#ifdef SFEM_ENABLE_CUDA
#include "sfem_ContactConditions_cuda.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_ShiftableJacobi.hpp"
#include "sfem_cuda_blas.hpp"
#include "sfem_cuda_crs_SpMV.hpp"
#include "sfem_cuda_mprgp_impl.hpp"
#include "sfem_cuda_solver.hpp"
#include "smesh_sshex8_prolongation.cuh"
#include "smesh_sshex8_restriction.cuh"
#include "smesh_tet4_prolongation_restriction.cuh"
#endif

// Externals
#include "matrixio_crs.h"

// System
#include <sys/stat.h>

#include <algorithm>
#include <vector>

namespace sfem {

    //     template <typename T>
    //     static SharedBuffer<T *> create_buffer(const std::ptrdiff_t n0, const std::ptrdiff_t n1, const MemorySpace es) {
    // #ifdef SFEM_ENABLE_CUDA
    //         if (es == MEMORY_SPACE_DEVICE) return sfem::create_device_buffer<T>(n0, n1);
    // #endif  // SFEM_ENABLE_CUDA
    //         return sfem::create_host_buffer<T>(n0, n1);
    //     }

    //     template <typename T>
    //     static SharedBuffer<T *> create_buffer(const std::ptrdiff_t n0, const std::ptrdiff_t n1, const ExecutionSpace es) {
    // #ifdef SFEM_ENABLE_CUDA
    //         if (es == EXECUTION_SPACE_DEVICE) return sfem::create_device_buffer<T>(n0, n1);
    // #endif  // SFEM_ENABLE_CUDA
    //         return sfem::create_host_buffer<T>(n0, n1);
    //     }

    //     template <typename T>
    //     static SharedBuffer<T> create_buffer(const std::ptrdiff_t n, const MemorySpace es) {
    // #ifdef SFEM_ENABLE_CUDA
    //         if (es == MEMORY_SPACE_DEVICE) return sfem::create_device_buffer<T>(n);
    // #endif  // SFEM_ENABLE_CUDA
    //         return sfem::create_host_buffer<T>(n);
    //     }

    //     template <typename T>
    //     static SharedBuffer<T> create_buffer(const std::ptrdiff_t n, const ExecutionSpace es) {
    // #ifdef SFEM_ENABLE_CUDA
    //         if (es == EXECUTION_SPACE_DEVICE) return sfem::create_device_buffer<T>(n);
    // #endif  // SFEM_ENABLE_CUDA
    //         return sfem::create_host_buffer<T>(n);
    //     }

    static inline SharedBuffer<idx_t *> sshex8_derefine_element_connectivity(const int                    from_level,
                                                                      const int                    to_level,
                                                                      const SharedBuffer<idx_t *> &elements) {
        const int       step_factor = from_level / to_level;
        const int       nxe         = (to_level + 1) * (to_level + 1) * (to_level + 1);
        const ptrdiff_t nelements   = elements->extent(1);

#ifdef SFEM_ENABLE_CUDA
        if (elements->mem_space() == MEMORY_SPACE_DEVICE) {
            std::vector<idx_t *> host_buff_from(elements->extent(0));
            buffer_device_to_host(elements->extent(0) * sizeof(idx_t *), elements->data(), host_buff_from.data());

            std::vector<idx_t *> host_dev_ptrs(nxe);
            for (int zi = 0; zi <= to_level; zi++) {
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        const int from_lidx =
                                smesh::sshex8_lidx(from_level, xi * step_factor, yi * step_factor, zi * step_factor);
                        const int to_lidx = smesh::sshex8_lidx(to_level, xi, yi, zi);

                        assert(from_lidx < elements->extent(0));
                        assert(to_lidx < host_dev_ptrs.size());

                        host_dev_ptrs[to_lidx] = host_buff_from[from_lidx];
                    }
                }
            }

            idx_t **dev_buff_to = (idx_t **)d_buffer_alloc(nxe * sizeof(idx_t *));
            buffer_host_to_device(nxe * sizeof(idx_t *), host_dev_ptrs.data(), dev_buff_to);
            return std::make_shared<Buffer<idx_t *>>(
                    nxe,
                    nelements,
                    dev_buff_to,
                    [nxe, host_dev_ptrs](int, void **ptr) { d_buffer_destroy(ptr); },
                    MEMORY_SPACE_DEVICE);
        }
#endif

        auto view = std::make_shared<Buffer<idx_t *>>(
                nxe,
                nelements,
                (idx_t **)malloc(nxe * sizeof(idx_t *)),
                [keep_alive = elements](int, void **v) {
                    (void)keep_alive;
                    free(v);
                },
                elements->mem_space());

        auto d     = view->data();
        auto elems = elements->data();

        for (int zi = 0; zi <= to_level; zi++) {
            for (int yi = 0; yi <= to_level; yi++) {
                for (int xi = 0; xi <= to_level; xi++) {
                    const int from_lidx = smesh::sshex8_lidx(from_level, xi * step_factor, yi * step_factor, zi * step_factor);
                    const int to_lidx   = smesh::sshex8_lidx(to_level, xi, yi, zi);
                    d[to_lidx]          = elems[from_lidx];
                }
            }
        }

        return view;
    }

    /// Singular skin: one block only. Multi-block: \c smesh::skin_sidesets.
    static inline std::shared_ptr<sfem::Sideset> create_skin_sideset(const std::shared_ptr<sfem::Mesh> &mesh) {
        return smesh::skin_sideset(mesh);
    }

}  // namespace sfem

#endif  // SFEM_API_HPP

