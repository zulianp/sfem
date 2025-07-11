#include "sfem_cuda_ShiftableJacobi.hpp"

#include "sfem_base.h"
#include "sfem_cuda_base.h"

#include "sfem_mask.h"

#include "cu_mask.cuh"

namespace sfem {
    namespace private_ {

        template <typename T>
        __global__ void inverse3_inplace_apply_AoS(const ptrdiff_t n_blocks, T* const SFEM_RESTRICT inout) {
            for (ptrdiff_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += blockDim.x * gridDim.x) {
                auto ddi = &inout[b * 9];

                const T mat_0 = ddi[0];
                const T mat_1 = ddi[1];
                const T mat_2 = ddi[2];
                const T mat_3 = ddi[3];
                const T mat_4 = ddi[4];
                const T mat_5 = ddi[5];
                const T mat_6 = ddi[6];
                const T mat_7 = ddi[7];
                const T mat_8 = ddi[8];

                const T x0 = mat_4 * mat_8;
                const T x1 = mat_5 * mat_7;
                const T x2 = mat_1 * mat_5;
                const T x3 = mat_1 * mat_8;
                const T x4 = mat_2 * mat_4;
                const T x5 = (T)1 / (mat_0 * x0 - mat_0 * x1 + mat_2 * mat_3 * mat_7 - mat_3 * x3 + mat_6 * x2 - mat_6 * x4);

                assert(x5 == x5);

                ddi[0] = x5 * (x0 - x1);
                ddi[1] = x5 * (mat_2 * mat_7 - x3);
                ddi[2] = x5 * (x2 - x4);
                ddi[3] = x5 * (-mat_3 * mat_8 + mat_5 * mat_6);
                ddi[4] = x5 * (mat_0 * mat_8 - mat_2 * mat_6);
                ddi[5] = x5 * (-mat_0 * mat_5 + mat_2 * mat_3);
                ddi[6] = x5 * (mat_3 * mat_7 - mat_4 * mat_6);
                ddi[7] = x5 * (-mat_0 * mat_7 + mat_1 * mat_6);
                ddi[8] = x5 * (mat_0 * mat_4 - mat_1 * mat_3);
            }
        }

        template <typename T>
        struct Invert3_CUDA {
            static void inplace_apply_AoS(const ptrdiff_t n_blocks, T* const inout) {
                SFEM_DEBUG_SYNCHRONIZE();

                int       kernel_block_size = 128;
                ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
                inverse3_inplace_apply_AoS<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, inout);

                SFEM_DEBUG_SYNCHRONIZE();
            }
        };

        template <typename T>
        __global__ void mask3_apply(const ptrdiff_t n_blocks, const mask_t* const SFEM_RESTRICT constraints_mask, T* const SFEM_RESTRICT inout) {
            for (ptrdiff_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += blockDim.x * gridDim.x) {
                auto ivi = &inout[b * 9];

                if (cu_mask_get(b * 3 + 0, constraints_mask)) {
                    ivi[0] = 1;
                    ivi[1] = 0;
                    ivi[2] = 0;
                }

                if (cu_mask_get(b * 3 + 1, constraints_mask)) {
                    ivi[3] = 0;
                    ivi[4] = 1;
                    ivi[5] = 0;
                }

                if (cu_mask_get(b * 3 + 2, constraints_mask)) {
                    ivi[6] = 0;
                    ivi[7] = 0;
                    ivi[8] = 1;
                }
            }
        }

        template <typename T>
        struct Mask3_CUDA {
            static void apply(const ptrdiff_t n_blocks, const mask_t* const constraints_mask, T* const inout) {
                SFEM_DEBUG_SYNCHRONIZE();

                int       kernel_block_size = 128;
                ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
                mask3_apply<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, constraints_mask, inout);

                SFEM_DEBUG_SYNCHRONIZE();
            }
        };

        template <typename In, typename Out>
        __global__ void sbv3_add_sym_diag_to_diag(const ptrdiff_t n_blocks, const In* const SFEM_RESTRICT in, Out* const SFEM_RESTRICT out) {
            for (ptrdiff_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += blockDim.x * gridDim.x) {
                auto di  = &in[b * 6];
                auto ivi = &out[b * 9];

                // row 0
                ivi[0] += di[0];
                ivi[1] += di[1];
                ivi[2] += di[2];

                // row 1
                ivi[3] += di[1];
                ivi[4] += di[3];
                ivi[5] += di[4];

                // row 2
                ivi[6] += di[2];
                ivi[7] += di[4];
                ivi[8] += di[5];
            }
        }

        template <typename DD, typename S, typename IVD>
        __global__ void sbv3_add_sparse_sym_diag_to_diag(const ptrdiff_t    n_blocks,
                                                         const idx_t* const SFEM_RESTRICT idx,
                                                         const DD* const    SFEM_RESTRICT dd,
                                                         const S* const     SFEM_RESTRICT s,
                                                         IVD* const         SFEM_RESTRICT ivd) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks; i += blockDim.x * gridDim.x) {
                auto di = &dd[i * 6];
                auto si = s[i];

                const ptrdiff_t b   = idx[i];
                auto            ivi = &ivd[b * 9];

                // row 0
                ivi[0] += si * di[0];
                ivi[1] += si * di[1];
                ivi[2] += si * di[2];

                // row 1
                ivi[3] += si * di[1];
                ivi[4] += si * di[3];
                ivi[5] += si * di[4];

                // row 2
                ivi[6] += si * di[2];
                ivi[7] += si * di[4];
                ivi[8] += si * di[5];
            }
        }

        template <typename In, typename Out>
        __global__ void sbv3_sym_diag_to_diag(const ptrdiff_t n_blocks, const In* const SFEM_RESTRICT in, Out* const SFEM_RESTRICT out) {
            for (ptrdiff_t b = blockIdx.x * blockDim.x + threadIdx.x; b < n_blocks; b += blockDim.x * gridDim.x) {
                auto di  = &in[b * 6];
                auto ivi = &out[b * 9];

                // row 0
                ivi[0] = di[0];
                ivi[1] = di[1];
                ivi[2] = di[2];

                // row 1
                ivi[3] = di[1];
                ivi[4] = di[3];
                ivi[5] = di[4];

                // row 2
                ivi[6] = di[2];
                ivi[7] = di[4];
                ivi[8] = di[5];
            }
        }

        template <typename B, typename In, typename Out>
        __global__ void sbv3_apply(const ptrdiff_t n_blocks, const B* const SFEM_RESTRICT blocks, const In* const SFEM_RESTRICT in, Out* const SFEM_RESTRICT out) {
            for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks; i += blockDim.x * gridDim.x) {
                const auto* const xi = &in[i * 3];
                auto* const       yi = &out[i * 3];
                const auto* const di = &blocks[i * 3 * 3];

                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        yi[d1] += (Out)di[d1 * 3 + d2] * xi[d2];
                    }
                }
            }
        }

        template <typename In, typename Out>
        static void SparseBlockSymOps_CUDA_add_sym_diag_to_diag(const ptrdiff_t n_blocks, const In* const in, Out* const out) {
            SFEM_DEBUG_SYNCHRONIZE();

            int       kernel_block_size = 128;
            ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
            sbv3_add_sym_diag_to_diag<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, in, out);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        template <typename DD, typename S, typename IVD>
        static void SparseBlockSymOps_CUDA_add_sparse_sym_diag_to_diag(const ptrdiff_t    n_blocks,
                                                                       const idx_t* const idx,
                                                                       const DD* const    dd,
                                                                       const S* const     s,
                                                                       IVD* const         ivd) {
            SFEM_DEBUG_SYNCHRONIZE();

            int       kernel_block_size = 128;
            ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
            sbv3_add_sparse_sym_diag_to_diag<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, idx, dd, s, ivd);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        template <typename In, typename Out>
        static void SparseBlockSymOps_CUDA_sym_diag_to_diag(const ptrdiff_t n_blocks, const In* const in, Out* const out) {
            SFEM_DEBUG_SYNCHRONIZE();

            int       kernel_block_size = 128;
            ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
            sbv3_sym_diag_to_diag<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, in, out);

            SFEM_DEBUG_SYNCHRONIZE();
        }

        template <typename InvDiag, typename In, typename Out>
        static void SparseBlockSymOps_CUDA_apply(const ptrdiff_t      n_blocks,
                                                 const InvDiag* const inv_diag,
                                                 const In* const      in,
                                                 Out* const           out) {
            SFEM_DEBUG_SYNCHRONIZE();

            int       kernel_block_size = 128;
            ptrdiff_t kernel_n_blocks   = std::max(ptrdiff_t(1), (n_blocks + kernel_block_size - 1) / kernel_block_size);
            sbv3_apply<<<kernel_n_blocks, kernel_block_size>>>(n_blocks, inv_diag, in, out);

            SFEM_DEBUG_SYNCHRONIZE();
        }

    }  // namespace private_

    template <typename HP, typename LP>
    int ShiftableBlockSymJacobi_CUDA<HP, LP>::build(const int dim, struct ShiftableBlockSymJacobi_Tpl<HP, LP>& tpl) {
        if (dim != 3) {
            SFEM_ERROR("ShiftableBlockSymJacobi_CUDA::build(dim=%d) not supported!\n", dim);
            return SFEM_FAILURE;
        }

        tpl.add_sparse_sym_diag_to_diag = &private_::SparseBlockSymOps_CUDA_add_sparse_sym_diag_to_diag<HP, HP, LP>;
        tpl.add_sym_diag_to_diag        = &private_::SparseBlockSymOps_CUDA_add_sym_diag_to_diag<HP, LP>;
        tpl.sym_diag_to_diag            = &private_::SparseBlockSymOps_CUDA_sym_diag_to_diag<HP, LP>;
        tpl.apply                       = &private_::SparseBlockSymOps_CUDA_apply<LP, HP, HP>;

        tpl.apply_mask     = &private_::Mask3_CUDA<LP>::apply;
        tpl.inplace_invert = &private_::Invert3_CUDA<LP>::inplace_apply_AoS;
        return SFEM_SUCCESS;
    }

    template class ShiftableBlockSymJacobi_CUDA<double>;
    template class ShiftableBlockSymJacobi_CUDA<float>;
    template class ShiftableBlockSymJacobi_CUDA<double, float>;
    // template class ShiftableBlockSymJacobi_CUDA<float, half>;

}  // namespace sfem
