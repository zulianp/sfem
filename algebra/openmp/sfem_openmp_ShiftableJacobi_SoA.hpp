#ifndef SFEM_OPENMP_SHIFTABLEJACOBI_HPP
#define SFEM_OPENMP_SHIFTABLEJACOBI_HPP

#include "sfem_base.h"
#include "sfem_mask.h"

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace sfem {

    template <typename HP, typename LP = HP>
    struct ShiftableBlockSymJacobi_SoA_Tpl {
        std::function<void(const ptrdiff_t, const idx_t* const, const HP* const, const HP* const, LP* const)>
                                                                                          add_sparse_sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const HP* const, LP* const)>                  add_sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const HP* const, LP* const)>                  sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const LP* const, const HP* const, HP* const)> apply;
        std::function<void(const ptrdiff_t, const mask_t* const, LP* const)>              apply_mask;
        std::function<void(const ptrdiff_t, LP* const)>                                   inplace_invert;
    };

    namespace private_ {

        template <typename T>
        struct Invert3_OpenMP_SoA {
            static SFEM_INLINE void inverse3(
                    // Input
                    const T mat_0,
                    const T mat_1,
                    const T mat_2,
                    const T mat_3,
                    const T mat_4,
                    const T mat_5,
                    const T mat_6,
                    const T mat_7,
                    const T mat_8,
                    // Output
                    T* const mat_inv_0,
                    T* const mat_inv_1,
                    T* const mat_inv_2,
                    T* const mat_inv_3,
                    T* const mat_inv_4,
                    T* const mat_inv_5,
                    T* const mat_inv_6,
                    T* const mat_inv_7,
                    T* const mat_inv_8) {
                const T x0 = mat_4 * mat_8;
                const T x1 = mat_5 * mat_7;
                const T x2 = mat_1 * mat_5;
                const T x3 = mat_1 * mat_8;
                const T x4 = mat_2 * mat_4;
                const T x5 = 1 / (mat_0 * x0 - mat_0 * x1 + mat_2 * mat_3 * mat_7 - mat_3 * x3 + mat_6 * x2 - mat_6 * x4);
                assert(x5 == x5);
                *mat_inv_0 = x5 * (x0 - x1);
                *mat_inv_1 = x5 * (mat_2 * mat_7 - x3);
                *mat_inv_2 = x5 * (x2 - x4);
                *mat_inv_3 = x5 * (-mat_3 * mat_8 + mat_5 * mat_6);
                *mat_inv_4 = x5 * (mat_0 * mat_8 - mat_2 * mat_6);
                *mat_inv_5 = x5 * (-mat_0 * mat_5 + mat_2 * mat_3);
                *mat_inv_6 = x5 * (mat_3 * mat_7 - mat_4 * mat_6);
                *mat_inv_7 = x5 * (-mat_0 * mat_7 + mat_1 * mat_6);
                *mat_inv_8 = x5 * (mat_0 * mat_4 - mat_1 * mat_3);
            }

            static void SFEM_INLINE inplace_apply(const ptrdiff_t n_blocks, T** const inout) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    inverse3(inout[0][b],
                             inout[1][b],
                             inout[2][b],
                             inout[3][b],
                             inout[4][b],
                             inout[5][b],
                             inout[6][b],
                             inout[7][b],
                             inout[8][b],
                             &inout[0][b],
                             &inout[1][b],
                             &inout[2][b],
                             &inout[3][b],
                             &inout[4][b],
                             &inout[5][b],
                             &inout[6][b],
                             &inout[7][b],
                             &inout[8][b]);
                }
            }
        };

        template <typename T>
        struct Mask3_SoA_OpenMP {
            static void apply(const ptrdiff_t n_blocks, const mask_t* const constraints_mask, T** const inout) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    if (mask_get(b * 3 + 0, constraints_mask)) {
                        inout[0][b] = 1;
                        inout[1][b] = 0;
                        inout[2][b] = 0;
                    }

                    if (mask_get(b * 3 + 1, constraints_mask)) {
                        inout[3][b] = 0;
                        inout[4][b] = 1;
                        inout[5][b] = 0;
                    }

                    if (mask_get(b * 3 + 2, constraints_mask)) {
                        inout[6][b] = 0;
                        inout[7][b] = 0;
                        inout[8][b] = 1;
                    }
                }
            }
        };

        template <typename In, typename Out>
        static void SparseBlockSymOps_SoA_OpenMP_add_sym_diag_to_diag(const ptrdiff_t  n_blocks,
                                                                      const In** const in,
                                                                      Out** const      out) {
#pragma omp parallel for
            for (ptrdiff_t b = 0; b < n_blocks; b++) {
                // row 0
                out[0][b] += di[0][b];
                out[1][b] += di[1][b];
                out[2][b] += di[2][b];

                // row 1
                out[3][b] += di[1][b];
                out[4][b] += di[3][b];
                out[5][b] += di[4][b];

                // row 2
                out[6][b] += di[2][b];
                out[7][b] += di[4][b];
                out[8][b] += di[5][b];
            }
        }

        template <typename DD, typename S, typename IVD>
        static void SparseBlockSymOps_SoA_OpenMP_add_sparse_sym_diag_to_diag(const ptrdiff_t    n_blocks,
                                                                             const idx_t* const idx,
                                                                             const DD** const   dd,
                                                                             const S* const     s,
                                                                             IVD** const        ivd) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_blocks; i++) {
                auto        si = s[i];
                const idx_t b  = idx[i];

                // row 0
                ivd[0][b] += si * dd[0][b];
                ivd[1][b] += si * dd[1][b];
                ivd[2][b] += si * dd[2][b];

                // row 1
                ivd[3][b] += si * dd[1][b];
                ivd[4][b] += si * dd[3][b];
                ivd[5][b] += si * dd[4][b];

                // row 2
                ivd[6][b] += si * dd[2][b];
                ivd[7][b] += si * dd[4][b];
                ivd[8][b] += si * dd[5][b];
            }
        }

        template <typename In, typename Out>
        static void SparseBlockSymOps_SoA_OpenMP_sym_diag_to_diag(const ptrdiff_t n_blocks, const In* const in, Out* const out) {
#pragma omp parallel for
            for (ptrdiff_t b = 0; b < n_blocks; b++) {
                // row 0
                out[0][b] = in[0][b];
                out[1][b] = in[1][b];
                out[2][b] = in[2][b];

                // row 1
                out[3][b] = in[1][b];
                out[4][b] = in[3][b];
                out[5][b] = in[4][b];

                // row 2
                out[6][b] = in[2][b];
                out[7][b] = in[4][b];
                out[8][b] = in[5][b];
            }
        }

        template <typename InvDiag, typename In, typename Out>
        static void SparseBlockSymOps_SoA_OpenMP_apply(const ptrdiff_t      n_blocks,
                                                       const InvDiag* const inv_diag,
                                                       const In* const      in,
                                                       Out* const           out) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_blocks; i++) {
                const auto* const xi = &in[i * 3];
                auto* const       yi = &out[i * 3];

                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        yi[d1] += inv_diag[d1 * 3 + d2][i] * xi[d2];
                    }
                }
            }
        }

    }  // namespace private_

    template <typename HP, typename LP = HP>
    struct ShiftableBlockSymJacobi_SoA_OpenMP {
        static int build(const int dim, ShiftableBlockSymJacobi_Tpl<HP, LP>& tpl) {
            if (dim != 3) {
                SFEM_ERROR("ShiftableBlockSymJacobi_SoA_OpenMP::build(dim=%d) not supported!\n", dim);
                return SFEM_FAILURE;
            }

            tpl.add_sparse_sym_diag_to_diag = &private_::SparseBlockSymOps_SoA_OpenMP_add_sparse_sym_diag_to_diag<HP, HP, LP>;
            tpl.add_sym_diag_to_diag        = &private_::SparseBlockSymOps_SoA_OpenMP_add_sym_diag_to_diag<HP, LP>;
            tpl.sym_diag_to_diag            = &private_::SparseBlockSymOps_SoA_OpenMP_sym_diag_to_diag<HP, LP>;
            tpl.apply                       = &private_::SparseBlockSymOps_SoA_OpenMP_apply<LP, HP, HP>;

            tpl.apply_mask     = &private_::Mask3_SoA_OpenMP<LP>::apply;
            tpl.inplace_invert = &private_::Invert3_SoA_OpenMP<LP>::inplace_apply;
            return SFEM_SUCCESS;
        }
    };

}  // namespace sfem

#endif  // SFEM_OPENMP_SHIFTABLEJACOBI_HPP
