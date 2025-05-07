#ifndef SFEM_OPENMP_SHIFTABLEJACOBI_HPP
#define SFEM_OPENMP_SHIFTABLEJACOBI_HPP

#include "sfem_base.h"
#include "sfem_mask.h"

#include <cstdio>
#include <cstdlib>
#include <functional>

namespace sfem {

    template <typename T>
    struct ShiftableBlockSymJacobi_Tpl {
        std::function<void(const ptrdiff_t, const idx_t* const, const T* const, const T* const, T* const)>
                                                                                       add_sparse_sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const T* const, T* const)>                 add_sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const T* const, T* const)>                 sym_diag_to_diag;
        std::function<void(const ptrdiff_t, const T* const, const T* const, T* const)> apply;
        std::function<void(const ptrdiff_t, const mask_t* const, T* const)>            apply_mask;
        std::function<void(const ptrdiff_t, T* const)>                                 inplace_invert;
    };

    namespace private_ {

        template <typename T>
        struct Invert3_OpenMP {
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

            static void inplace_apply_AoS(const ptrdiff_t n_blocks, T* const inout) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    auto ddi = &inout[b * 9];

                    inverse3(ddi[0],
                             ddi[1],
                             ddi[2],
                             ddi[3],
                             ddi[4],
                             ddi[5],
                             ddi[6],
                             ddi[7],
                             ddi[8],
                             &ddi[0],
                             &ddi[1],
                             &ddi[2],
                             &ddi[3],
                             &ddi[4],
                             &ddi[5],
                             &ddi[6],
                             &ddi[7],
                             &ddi[8]);
                }
            }
        };

        template <typename T>
        struct Mask3_OpenMP {
            static void apply(const ptrdiff_t n_blocks, const mask_t* const constraints_mask, T* const inout) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
                    auto ivi = &inout[b * 9];

                    if (mask_get(b * 3 + 0, constraints_mask)) {
                        ivi[0] = 1;
                        ivi[1] = 0;
                        ivi[2] = 0;
                    }

                    if (mask_get(b * 3 + 1, constraints_mask)) {
                        ivi[3] = 0;
                        ivi[4] = 1;
                        ivi[5] = 0;
                    }

                    if (mask_get(b * 3 + 2, constraints_mask)) {
                        ivi[6] = 0;
                        ivi[7] = 0;
                        ivi[8] = 1;
                    }
                }
            }
        };

        template <typename T>
        struct SparseBlockSymOps_OpenMP {
            static void add_sym_diag_to_diag(const ptrdiff_t n_blocks, const T* const in, T* const out) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
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

            static void add_sparse_sym_diag_to_diag(const ptrdiff_t    n_blocks,
                                                    const idx_t* const idx,
                                                    const T* const     dd,
                                                    const T* const     s,
                                                    T* const           ivd) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    auto di = &dd[i * 6];
                    auto si = s[i];

                    const idx_t b   = idx[i];
                    auto        ivi = &ivd[b * 9];

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

            static void sym_diag_to_diag(const ptrdiff_t n_blocks, const T* const in, T* const out) {
#pragma omp parallel for
                for (ptrdiff_t b = 0; b < n_blocks; b++) {
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
            static void apply(const ptrdiff_t n_blocks, const T* const inv_diag, const T* const in, T* const out) {
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_blocks; i++) {
                    const T* const xi = &in[i * 3];
                    T* const       yi = &out[i * 3];
                    const T* const di = &inv_diag[i * 3 * 3];

                    for (int d1 = 0; d1 < 3; d1++) {
                        for (int d2 = 0; d2 < 3; d2++) {
                            yi[d1] += di[d1 * 3 + d2] * xi[d2];
                        }
                    }
                }
            }
        };
    }  // namespace private_

    template <typename T>
    struct ShiftableBlockSymJacobi_OpenMP {
        static int build(const int dim, ShiftableBlockSymJacobi_Tpl<T>& tpl) {
            if (dim != 3) {
                SFEM_ERROR("ShiftableBlockSymJacobi_OpenMP::build(dim=%d) not supported!\n", dim);
                return SFEM_FAILURE;
            }

            tpl.add_sym_diag_to_diag        = &private_::SparseBlockSymOps_OpenMP<T>::add_sym_diag_to_diag;
            tpl.add_sparse_sym_diag_to_diag = &private_::SparseBlockSymOps_OpenMP<T>::add_sparse_sym_diag_to_diag;
            tpl.sym_diag_to_diag            = &private_::SparseBlockSymOps_OpenMP<T>::sym_diag_to_diag;
            tpl.apply                       = &private_::SparseBlockSymOps_OpenMP<T>::apply;

            tpl.apply_mask     = &private_::Mask3_OpenMP<T>::apply;
            tpl.inplace_invert = &private_::Invert3_OpenMP<T>::inplace_apply_AoS;
            return SFEM_SUCCESS;
        }
    };
}  // namespace sfem

#endif  // SFEM_OPENMP_SHIFTABLEJACOBI_HPP
