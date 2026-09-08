#include "sfem_ContactSolveKernels.hpp"
#include "sfem_aliases.hpp"
#include "sfem_macros.hpp"
// Gather / Scatter done outside

#include <stddef.h>
#include <stdlib.h>
#include "sfem_base.hpp"

#include "sfem_Function.hpp"
#include "sfem_aliases.hpp"
#include "sfem_macros.hpp"

#include "sfem_API.hpp"

namespace sfem {

    static SFEM_INLINE count_t contact_find_col(const idx_t target, const idx_t* const SFEM_RESTRICT row, const count_t lenrow) {
        if (lenrow <= 32) {
            count_t k = 0;
            for (; k + 3 < lenrow; k += 4) {
                if (row[k] == target) return k;
                if (row[k + 1] == target) return k + 1;
                if (row[k + 2] == target) return k + 2;
                if (row[k + 3] == target) return k + 3;
            }

            for (; k < lenrow; ++k) {
                if (row[k] == target) return k;
            }
        } else {
            count_t left  = 0;
            count_t right = lenrow;
            while (left < right) {
                const count_t mid = left + ((right - left) >> 1);
                if (row[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            return left;
        }

        return 0;
    }

    static SFEM_INLINE void contact_add_block3(real_t* const SFEM_RESTRICT block,
                                               const real_t                c,
                                               const real_t                b00,
                                               const real_t                b01,
                                               const real_t                b02,
                                               const real_t                b11,
                                               const real_t                b12,
                                               const real_t                b22) {
        const real_t v0 = c * b00;
        const real_t v1 = c * b01;
        const real_t v2 = c * b02;
        const real_t v4 = c * b11;
        const real_t v5 = c * b12;
        const real_t v8 = c * b22;

#pragma omp atomic update
        block[0] += v0;
#pragma omp atomic update
        block[1] += v1;
#pragma omp atomic update
        block[2] += v2;
#pragma omp atomic update
        block[3] += v1;
#pragma omp atomic update
        block[4] += v4;
#pragma omp atomic update
        block[5] += v5;
#pragma omp atomic update
        block[6] += v2;
#pragma omp atomic update
        block[7] += v5;
#pragma omp atomic update
        block[8] += v8;
    }

    static SFEM_INLINE void contact_add_block(real_t* const SFEM_RESTRICT       block,
                                              const int                         dim,
                                              const real_t                      c,
                                              const real_t* const SFEM_RESTRICT b) {
        for (int d1 = 0; d1 < dim; ++d1) {
            const int row_offset = d1 * dim;
            for (int d2 = 0; d2 < dim; ++d2) {
                const int    offset = row_offset + d2;
                const real_t v      = c * b[offset];
#pragma omp atomic update
                block[offset] += v;
            }
        }
    }

    void compute_macaulay_term(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const SFEM_RESTRICT                      distances,
                               const real_t* const SFEM_RESTRICT                      agumentation,
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t* const SFEM_RESTRICT penalty,
                               const ptrdiff_t                                        in_stride,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                               real_t* const                                          macaulay) {
        SFEM_TRACE_SCOPE("compute_macaulay_term");

        if (dim == 3) {
            const real_t* const in0 = in[0];
            const real_t* const in1 = in[1];
            const real_t* const in2 = in[2];
            const real_t* const n0  = normals[0];
            const real_t* const n1  = normals[1];
            const real_t* const n2  = normals[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                const count_t begin = cm_rowptr[i];
                const count_t end   = cm_rowptr[i + 1];
                if (begin == end) {
                    macaulay[i] = 0;
                    continue;
                }

                const ptrdiff_t dof1 = i * in_stride;
                real_t          u20  = 0;
                real_t          u21  = 0;
                real_t          u22  = 0;

                for (count_t j = begin; j < end; j++) {
                    const ptrdiff_t dof2 = cm_colidx[j] * in_stride;
                    const real_t    w    = cm_vals[j];
                    u20 += w * in0[dof2];
                    u21 += w * in1[dof2];
                    u22 += w * in2[dof2];
                }

                const real_t normal_diff = n0[i] * (in0[dof1] - u20) + n1[i] * (in1[dof1] - u21) + n2[i] * (in2[dof1] - u22);
                macaulay[i]              = std::max(normal_diff - distances[i] + agumentation[i] / penalty[i], real_t(0));
            }
        } else {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                auto lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
                if (lenrow == 0) {
                    macaulay[i] = 0;
                    continue;
                }

                auto row     = &cm_colidx[cm_rowptr[i]];
                auto weights = &cm_vals[cm_rowptr[i]];

                real_t normal_diff = 0;
                for (int d = 0; d < dim; d++) {
                    const ptrdiff_t dof1 = i * in_stride;
                    real_t          u2   = 0;
                    for (count_t j = 0; j < lenrow; j++) {
                        const ptrdiff_t dof2 = row[j] * in_stride;
                        u2 += weights[j] * in[d][dof2];
                    }

                    normal_diff += normals[d][i] * (in[d][dof1] - u2);
                }

                macaulay[i] = std::max(normal_diff - distances[i] + agumentation[i] / penalty[i], real_t(0));
            }
        }
    }

    void assemble_contact_gradient(const int                                dim,
                                   const ptrdiff_t                          nnodes,
                                   const real_t* const SFEM_RESTRICT penalty,
                                   const count_t* const SFEM_RESTRICT       cm_rowptr,
                                   const idx_t* const SFEM_RESTRICT         cm_colidx,
                                   const real_t* const SFEM_RESTRICT        cm_vals,
                                   const real_t* const SFEM_RESTRICT        distances,
                                   const real_t* const SFEM_RESTRICT        agumentation,
                                   const real_t* const* const SFEM_RESTRICT normals,
                                   const real_t* const SFEM_RESTRICT        mass,
                                   const real_t* const SFEM_RESTRICT        macaulay,
                                   //    Output
                                   real_t* const SFEM_RESTRICT grad) {
        SFEM_TRACE_SCOPE("assemble_contact_gradient");

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            if (macaulay[i] == 0) continue;

            auto lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
            if (lenrow == 0) continue;

            auto row     = &cm_colidx[cm_rowptr[i]];
            auto weights = &cm_vals[cm_rowptr[i]];

            real_t normal[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                normal[d] = normals[d][i];
            }

            real_t force[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                // Point-force we scale it by the mass-density at the contact point
                force[d] = mass[i] * penalty[i] * macaulay[i] * normal[d];
            }

            for (int d = 0; d < dim; d++) {
#pragma omp atomic update
                grad[i * dim + d] += force[d];
            }

            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                    grad[row[j] * dim + d] -= force[d] * weights[j];
                }
            }
        }
    }

    void assemble_contact_hessian_diag_block(const int                                dim,
                                             const ptrdiff_t                          nnodes,
                                             const count_t* const SFEM_RESTRICT       cm_rowptr,
                                             const idx_t* const SFEM_RESTRICT         cm_colidx,
                                             const real_t* const SFEM_RESTRICT        cm_vals,
                                             const real_t* const SFEM_RESTRICT        distances,
                                             const real_t* const SFEM_RESTRICT        agumentation,
                                             const real_t* const* const SFEM_RESTRICT normals,
                                             const real_t* const SFEM_RESTRICT        mass,
                                             const real_t* const SFEM_RESTRICT penalty,
                                             const real_t* const SFEM_RESTRICT        macaulay,
                                             //  Output
                                             const ptrdiff_t                                  diag_stride,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("assemble_contact_hessian_diag_block");

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            if (macaulay[i] == 0) continue;
            auto lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
            if (lenrow == 0) continue;

            auto row     = &cm_colidx[cm_rowptr[i]];
            auto weights = &cm_vals[cm_rowptr[i]];

            real_t normal[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                normal[d] = normals[d][i];
            }

            real_t nnT[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (int d1 = 0; d1 < dim; d1++) {
                for (int d2 = 0; d2 < dim; d2++) {
                    nnT[d1 * dim + d2] = mass[i] * penalty[i] * normal[d1] * normal[d2];
                }
            }

            // Assemble H11
            for (int d = 0; d < dim * dim; d++) {
#pragma omp atomic update
                diag_values[d][i * diag_stride] += nnT[d];
            }

            // Assemble H22
            for (int d = 0; d < dim * dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                    diag_values[d][row[j] * diag_stride] += weights[j] * weights[j] * nnT[d];
                }
            }
        }
    }

    void assemble_contact_hessian_block_diag(const int                                        dim,
                                             const ptrdiff_t                                  nnodes,
                                             const count_t* const SFEM_RESTRICT               cm_rowptr,
                                             const idx_t* const SFEM_RESTRICT                 cm_colidx,
                                             const real_t* const SFEM_RESTRICT                cm_vals,
                                             const real_t* const* const SFEM_RESTRICT         normals,
                                             const real_t* const SFEM_RESTRICT                mass,
                                             const real_t* const SFEM_RESTRICT penalty,
                                             const real_t* const SFEM_RESTRICT                active,
                                             const ptrdiff_t                                  diag_stride,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("assemble_contact_hessian_block_diag");
        assert(dim == 3);

        const real_t* const nx = normals[0];
        const real_t* const ny = normals[1];
        const real_t* const nz = normals[2];

        real_t* const d0 = diag_values[0];
        real_t* const d1 = diag_values[1];
        real_t* const d2 = diag_values[2];
        real_t* const d3 = diag_values[3];
        real_t* const d4 = diag_values[4];
        real_t* const d5 = diag_values[5];

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            if (active[i] == 0) continue;

            const count_t row_begin = cm_rowptr[i];
            const count_t row_end   = cm_rowptr[i + 1];
            if (row_begin == row_end) continue;

            const real_t s  = mass[i] * penalty[i];
            const real_t n0 = nx[i];
            const real_t n1 = ny[i];
            const real_t n2 = nz[i];

            const real_t b0 = s * n0 * n0;
            const real_t b1 = s * n0 * n1;
            const real_t b2 = s * n0 * n2;
            const real_t b3 = s * n1 * n1;
            const real_t b4 = s * n1 * n2;
            const real_t b5 = s * n2 * n2;

            const ptrdiff_t ii = i * diag_stride;

#pragma omp atomic update
            d0[ii] += b0;
#pragma omp atomic update
            d1[ii] += b1;
#pragma omp atomic update
            d2[ii] += b2;
#pragma omp atomic update
            d3[ii] += b3;
#pragma omp atomic update
            d4[ii] += b4;
#pragma omp atomic update
            d5[ii] += b5;

            for (count_t k = row_begin; k < row_end; ++k) {
                const ptrdiff_t r  = cm_colidx[k] * diag_stride;
                const real_t    w  = cm_vals[k];
                const real_t    w2 = w * w;

#pragma omp atomic update
                d0[r] += w2 * b0;
#pragma omp atomic update
                d1[r] += w2 * b1;
#pragma omp atomic update
                d2[r] += w2 * b2;
#pragma omp atomic update
                d3[r] += w2 * b3;
#pragma omp atomic update
                d4[r] += w2 * b4;
#pragma omp atomic update
                d5[r] += w2 * b5;
            }
        }
    }

    void contact_hessian_apply(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t* const SFEM_RESTRICT penalty,
                               const real_t* const SFEM_RESTRICT                      macaulay,
                               const ptrdiff_t                                        in_stride,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                               //  Output
                               const ptrdiff_t                                  out_stride,
                               real_t* const SFEM_RESTRICT* const SFEM_RESTRICT out_values) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            if (macaulay[i] == 0) continue;

            auto lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
            if (lenrow == 0) continue;

            auto row     = &cm_colidx[cm_rowptr[i]];
            auto weights = &cm_vals[cm_rowptr[i]];

            real_t u1[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                u1[d] = in[d][i * in_stride];
            }

            real_t u2[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
                    const ptrdiff_t dof = row[j] * in_stride;
                    u2[d] += weights[j] * in[d][dof];
                }
            }

            real_t normal[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                normal[d] = normals[d][i];
            }

            real_t diff[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                diff[d] = u1[d] - u2[d];
            }

            real_t normal_diff = 0;
            for (int d = 0; d < dim; d++) {
                normal_diff += normal[d] * diff[d];
            }

            real_t applied[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                // Point-applied we scale it by the density at the contact point
                applied[d] = penalty[i] * mass[i] * normal_diff * normal[d];
            }

            for (int d = 0; d < dim; d++) {
#pragma omp atomic update
                out_values[d][i * out_stride] += applied[d];
            }

            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                    out_values[d][row[j] * out_stride] -= applied[d] * weights[j];
                }
            }
        }
    }

    void contact_hessian_bsr(const int                                dim,
                             const ptrdiff_t                          nnodes,
                             const count_t* const SFEM_RESTRICT       cm_rowptr,
                             const idx_t* const SFEM_RESTRICT         cm_colidx,
                             const real_t* const SFEM_RESTRICT        cm_vals,
                             const real_t* const* const SFEM_RESTRICT normals,
                             const real_t* const SFEM_RESTRICT        mass,
                             const real_t* const SFEM_RESTRICT penalty,
                             const real_t* const SFEM_RESTRICT        macaulay,
                             const count_t* const SFEM_RESTRICT       rowptr,
                             const idx_t* const SFEM_RESTRICT         colidx,
                             real_t* const SFEM_RESTRICT              values) {
        if (dim == 3) {
            const real_t* const nx = normals[0];
            const real_t* const ny = normals[1];
            const real_t* const nz = normals[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                if (macaulay[i] == 0) continue;

                const count_t contact_begin = cm_rowptr[i];
                const count_t contact_end   = cm_rowptr[i + 1];
                if (contact_begin == contact_end) continue;

                const real_t s  = penalty[i] * mass[i];
                const real_t n0 = nx[i];
                const real_t n1 = ny[i];
                const real_t n2 = nz[i];

                const real_t b00 = s * n0 * n0;
                const real_t b01 = s * n0 * n1;
                const real_t b02 = s * n0 * n2;
                const real_t b11 = s * n1 * n1;
                const real_t b12 = s * n1 * n2;
                const real_t b22 = s * n2 * n2;

                const count_t      bsr_i_begin = rowptr[i];
                const count_t      bsr_i_len   = rowptr[i + 1] - bsr_i_begin;
                const idx_t* const bsr_i_cols  = &colidx[bsr_i_begin];
                real_t* const      bsr_i_vals  = &values[bsr_i_begin * 9];

                {
                    const count_t block = contact_find_col(i, bsr_i_cols, bsr_i_len);
                    contact_add_block3(&bsr_i_vals[block * 9], 1, b00, b01, b02, b11, b12, b22);
                }

                for (count_t k = contact_begin; k < contact_end; ++k) {
                    const idx_t  j = cm_colidx[k];
                    const real_t w = cm_vals[k];

                    {
                        const count_t block = contact_find_col(j, bsr_i_cols, bsr_i_len);
                        contact_add_block3(&bsr_i_vals[block * 9], -w, b00, b01, b02, b11, b12, b22);
                    }

                    const count_t      bsr_j_begin = rowptr[j];
                    const count_t      bsr_j_len   = rowptr[j + 1] - bsr_j_begin;
                    const idx_t* const bsr_j_cols  = &colidx[bsr_j_begin];
                    real_t* const      bsr_j_vals  = &values[bsr_j_begin * 9];

                    {
                        const count_t block = contact_find_col(i, bsr_j_cols, bsr_j_len);
                        contact_add_block3(&bsr_j_vals[block * 9], -w, b00, b01, b02, b11, b12, b22);
                    }

                    for (count_t l = contact_begin; l < contact_end; ++l) {
                        const count_t block = contact_find_col(cm_colidx[l], bsr_j_cols, bsr_j_len);
                        contact_add_block3(&bsr_j_vals[block * 9], w * cm_vals[l], b00, b01, b02, b11, b12, b22);
                    }
                }
            }
        } else {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                if (macaulay[i] == 0) continue;

                const count_t contact_begin = cm_rowptr[i];
                const count_t contact_end   = cm_rowptr[i + 1];
                if (contact_begin == contact_end) continue;

                const real_t s = penalty[i] * mass[i];
                real_t       b[9];
                for (int d1 = 0; d1 < dim; ++d1) {
                    const real_t sn = s * normals[d1][i];
                    for (int d2 = 0; d2 < dim; ++d2) {
                        b[d1 * dim + d2] = sn * normals[d2][i];
                    }
                }

                const int          block_size  = dim * dim;
                const count_t      bsr_i_begin = rowptr[i];
                const count_t      bsr_i_len   = rowptr[i + 1] - bsr_i_begin;
                const idx_t* const bsr_i_cols  = &colidx[bsr_i_begin];
                real_t* const      bsr_i_vals  = &values[bsr_i_begin * block_size];

                {
                    const count_t block = contact_find_col(i, bsr_i_cols, bsr_i_len);
                    contact_add_block(&bsr_i_vals[block * block_size], dim, 1, b);
                }

                for (count_t k = contact_begin; k < contact_end; ++k) {
                    const idx_t  j = cm_colidx[k];
                    const real_t w = cm_vals[k];

                    {
                        const count_t block = contact_find_col(j, bsr_i_cols, bsr_i_len);
                        contact_add_block(&bsr_i_vals[block * block_size], dim, -w, b);
                    }

                    const count_t      bsr_j_begin = rowptr[j];
                    const count_t      bsr_j_len   = rowptr[j + 1] - bsr_j_begin;
                    const idx_t* const bsr_j_cols  = &colidx[bsr_j_begin];
                    real_t* const      bsr_j_vals  = &values[bsr_j_begin * block_size];

                    {
                        const count_t block = contact_find_col(i, bsr_j_cols, bsr_j_len);
                        contact_add_block(&bsr_j_vals[block * block_size], dim, -w, b);
                    }

                    for (count_t l = contact_begin; l < contact_end; ++l) {
                        const count_t block = contact_find_col(cm_colidx[l], bsr_j_cols, bsr_j_len);
                        contact_add_block(&bsr_j_vals[block * block_size], dim, w * cm_vals[l], b);
                    }
                }
            }
        }
    }

    void apply_contact_hessian(const int                                dim,
                               const ptrdiff_t                          nnodes,
                               const count_t* const SFEM_RESTRICT       cm_rowptr,
                               const idx_t* const SFEM_RESTRICT         cm_colidx,
                               const real_t* const SFEM_RESTRICT        cm_vals,
                               const idx_t* const SFEM_RESTRICT         node_mapping,
                               const real_t* const* const SFEM_RESTRICT normals,
                               const real_t* const SFEM_RESTRICT        mass,
                               const real_t* const SFEM_RESTRICT        active,
                               const real_t* const SFEM_RESTRICT penalty,
                               const real_t* const SFEM_RESTRICT        x,
                               real_t* const SFEM_RESTRICT              y) {
        assert(dim == 3);

        const real_t* const nx = normals[0];
        const real_t* const ny = normals[1];
        const real_t* const nz = normals[2];

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const real_t a = active[i];
            if (a == 0) continue;

            const count_t row_begin = cm_rowptr[i];
            const count_t row_end   = cm_rowptr[i + 1];
            if (row_begin == row_end) continue;

            const ptrdiff_t dof1 = node_mapping[i] * dim;

            const real_t x10 = x[dof1];
            const real_t x11 = x[dof1 + 1];
            const real_t x12 = x[dof1 + 2];

            real_t x20 = 0;
            real_t x21 = 0;
            real_t x22 = 0;

            for (count_t k = row_begin; k < row_end; ++k) {
                const real_t    w    = cm_vals[k];
                const ptrdiff_t dof2 = node_mapping[cm_colidx[k]] * dim;

                x20 += w * x[dof2];
                x21 += w * x[dof2 + 1];
                x22 += w * x[dof2 + 2];
            }

            const real_t n0 = nx[i];
            const real_t n1 = ny[i];
            const real_t n2 = nz[i];
            const real_t s  = a * penalty[i] * mass[i] * (n0 * (x10 - x20) + n1 * (x11 - x21) + n2 * (x12 - x22));
            const real_t f0 = s * n0;
            const real_t f1 = s * n1;
            const real_t f2 = s * n2;

#pragma omp atomic update
            y[dof1] += f0;

#pragma omp atomic update
            y[dof1 + 1] += f1;

#pragma omp atomic update
            y[dof1 + 2] += f2;

            for (count_t k = row_begin; k < row_end; ++k) {
                const real_t    w    = cm_vals[k];
                const ptrdiff_t dof2 = node_mapping[cm_colidx[k]] * dim;

#pragma omp atomic update
                y[dof2] -= w * f0;

#pragma omp atomic update
                y[dof2 + 1] -= w * f1;

#pragma omp atomic update
                y[dof2 + 2] -= w * f2;
            }
        }
    }

    void gather_combine_hessian_diag(const int                                              dim,
                                     const ptrdiff_t                                        n_contact_nodes,
                                     const idx_t* const                                     node_mapping,
                                     const ptrdiff_t                                        elasticity_diag_stride,
                                     const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT elasticity_diag_values,
                                     const ptrdiff_t                                        contact_diag_stride,
                                     real_t* const SFEM_RESTRICT* const SFEM_RESTRICT       contact_diag_values) {
        SFEM_TRACE_SCOPE("gather_combine_hessian_diag");

        if (dim == 3) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact_nodes; ++i) {
                const ptrdiff_t global_node = node_mapping[i] * elasticity_diag_stride;
                const ptrdiff_t local_node  = i * contact_diag_stride;

                contact_diag_values[0][local_node] += elasticity_diag_values[0][global_node];
                contact_diag_values[1][local_node] += elasticity_diag_values[1][global_node];
                contact_diag_values[2][local_node] += elasticity_diag_values[2][global_node];
                contact_diag_values[3][local_node] += elasticity_diag_values[1][global_node];
                contact_diag_values[4][local_node] += elasticity_diag_values[3][global_node];
                contact_diag_values[5][local_node] += elasticity_diag_values[4][global_node];
                contact_diag_values[6][local_node] += elasticity_diag_values[2][global_node];
                contact_diag_values[7][local_node] += elasticity_diag_values[4][global_node];
                contact_diag_values[8][local_node] += elasticity_diag_values[5][global_node];
            }
        } else {
            SMESH_ASSERT(dim == 2);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact_nodes; ++i) {
                const ptrdiff_t global_node = node_mapping[i] * elasticity_diag_stride;
                const ptrdiff_t local_node  = i * contact_diag_stride;
                contact_diag_values[0][local_node] += elasticity_diag_values[0][global_node];
                contact_diag_values[1][local_node] += elasticity_diag_values[1][global_node];
                contact_diag_values[2][local_node] += elasticity_diag_values[1][global_node];
                contact_diag_values[3][local_node] += elasticity_diag_values[2][global_node];
            }
        }
    }

    void compute_penetration(const int                                              dim,
                             const ptrdiff_t                                        nnodes,
                             const count_t* const SFEM_RESTRICT                     cm_rowptr,
                             const idx_t* const SFEM_RESTRICT                       cm_colidx,
                             const real_t* const SFEM_RESTRICT                      cm_vals,
                             const real_t* const* const SFEM_RESTRICT               normals,
                             const real_t* const SFEM_RESTRICT                      gap,
                             const ptrdiff_t                                        in_stride,
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                             real_t* const SFEM_RESTRICT                            penetration) {
        SFEM_TRACE_SCOPE("compute_penetration");

        if (dim == 3) {
            const real_t* const in0 = in[0];
            const real_t* const in1 = in[1];
            const real_t* const in2 = in[2];
            const real_t* const n0  = normals[0];
            const real_t* const n1  = normals[1];
            const real_t* const n2  = normals[2];

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                const count_t begin = cm_rowptr[i];
                const count_t end   = cm_rowptr[i + 1];
                if (begin == end) {
                    penetration[i] = 0;
                    continue;
                }

                const ptrdiff_t dof1 = i * in_stride;
                real_t          u20  = 0;
                real_t          u21  = 0;
                real_t          u22  = 0;

                for (count_t j = begin; j < end; j++) {
                    const ptrdiff_t dof2 = cm_colidx[j] * in_stride;
                    const real_t    w    = cm_vals[j];
                    u20 += w * in0[dof2];
                    u21 += w * in1[dof2];
                    u22 += w * in2[dof2];
                }

                const real_t normal_diff = n0[i] * (in0[dof1] - u20) + n1[i] * (in1[dof1] - u21) + n2[i] * (in2[dof1] - u22);
                penetration[i]           = std::max(real_t(0), normal_diff - gap[i]);
            }
        } else {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nnodes; i++) {
                const count_t lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
                if (lenrow == 0) {
                    penetration[i] = 0;
                    continue;
                }

                const idx_t* const  row     = &cm_colidx[cm_rowptr[i]];
                const real_t* const weights = &cm_vals[cm_rowptr[i]];
                const ptrdiff_t     dof1    = i * in_stride;

                real_t normal_diff = 0;
                for (int d = 0; d < dim; d++) {
                    real_t u2 = 0;
                    for (count_t j = 0; j < lenrow; j++) {
                        const ptrdiff_t dof2 = row[j] * in_stride;
                        u2 += weights[j] * in[d][dof2];
                    }

                    normal_diff += normals[d][i] * (in[d][dof1] - u2);
                }

                penetration[i] = std::max(real_t(0), normal_diff - gap[i]);
            }
        }
    }

    // `reference` may be null, in which case the undeformed configuration is used.
    void contact_gather_displacement(const int                                        dim,
                                     const ptrdiff_t                                  n_contact,
                                     const idx_t* const SFEM_RESTRICT                 node_mapping,
                                     const real_t* const SFEM_RESTRICT                in,
                                     const real_t* const SFEM_RESTRICT                reference,
                                     real_t* const SFEM_RESTRICT* const SFEM_RESTRICT out) {
        if (!reference) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                const ptrdiff_t dof = node_mapping[i] * dim;
                for (int d = 0; d < dim; ++d) {
                    out[d][i] = in[dof + d];
                }
            }
            return;
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            const ptrdiff_t dof = node_mapping[i] * dim;
            for (int d = 0; d < dim; ++d) {
                out[d][i] = in[dof + d] - reference[dof + d];
            }
        }
    }

    void contact_scatter_displacement(const int                                              dim,
                                      const ptrdiff_t                                        n_contact,
                                      const idx_t* const SFEM_RESTRICT                       node_mapping,
                                      const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                                      real_t* const SFEM_RESTRICT                            out) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            const ptrdiff_t dof = node_mapping[i] * dim;
            for (int d = 0; d < dim; ++d) {
                out[dof + d] = in[d][i];
            }
        }
    }

    void contact_objective_steps(const int                                dim,
                                 const ptrdiff_t                          nnodes,
                                 const count_t* const SFEM_RESTRICT       cm_rowptr,
                                 const idx_t* const SFEM_RESTRICT         cm_colidx,
                                 const real_t* const SFEM_RESTRICT        cm_vals,
                                 const real_t* const SFEM_RESTRICT        distances,
                                 const real_t* const SFEM_RESTRICT        agumentation,
                                 const real_t* const* const SFEM_RESTRICT normals,
                                 const real_t* const SFEM_RESTRICT        mass,
                                 const real_t* const SFEM_RESTRICT penalty,
                                 const real_t* const SFEM_RESTRICT        disp,
                                 const real_t* const SFEM_RESTRICT        inc,
                                 const int                                nsteps,
                                 const real_t* const SFEM_RESTRICT        steps,
                                 real_t* const SFEM_RESTRICT              values) {
        SFEM_TRACE_SCOPE("contact_objective_steps");

        if (nsteps <= 0) return;

        if (dim == 3) {
            const real_t* const n0 = normals[0];
            const real_t* const n1 = normals[1];
            const real_t* const n2 = normals[2];

#pragma omp parallel
            {
                real_t* const local_values = (real_t*)calloc(nsteps, sizeof(real_t));

#pragma omp for
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    const count_t row_begin = cm_rowptr[i];
                    const count_t row_end   = cm_rowptr[i + 1];
                    if (row_begin == row_end) continue;

                    const ptrdiff_t dof1 = i * 3;

                    real_t d20 = 0;
                    real_t d21 = 0;
                    real_t d22 = 0;
                    real_t c20 = 0;
                    real_t c21 = 0;
                    real_t c22 = 0;

                    for (count_t k = row_begin; k < row_end; ++k) {
                        const real_t    w    = cm_vals[k];
                        const ptrdiff_t dof2 = cm_colidx[k] * 3;

                        d20 += w * disp[dof2 + 0];
                        d21 += w * disp[dof2 + 1];
                        d22 += w * disp[dof2 + 2];

                        c20 += w * inc[dof2 + 0];
                        c21 += w * inc[dof2 + 1];
                        c22 += w * inc[dof2 + 2];
                    }

                    const real_t nx = n0[i];
                    const real_t ny = n1[i];
                    const real_t nz = n2[i];

                    const real_t disp_normal =
                            nx * (disp[dof1 + 0] - d20) + ny * (disp[dof1 + 1] - d21) + nz * (disp[dof1 + 2] - d22);
                    const real_t inc_normal =
                            nx * (inc[dof1 + 0] - c20) + ny * (inc[dof1 + 1] - c21) + nz * (inc[dof1 + 2] - c22);
                    const real_t m           = mass[i];
                    const real_t aug         = agumentation[i];
                    const real_t inv_penalty = real_t(1) / penalty[i];
                    const real_t shift       = -distances[i] + aug * inv_penalty;
                    const real_t c           = -real_t(0.5) * m * aug * aug * inv_penalty;
                    const real_t scale       = m * real_t(0.5) * penalty[i];

                    for (int s = 0; s < nsteps; ++s) {
                        const real_t v = disp_normal + steps[s] * inc_normal + shift;
                        const real_t p = v > 0 ? v : 0;
                        local_values[s] += scale * p * p + c;
                    }
                }

                for (int s = 0; s < nsteps; ++s) {
#pragma omp atomic update
                    values[s] += local_values[s];
                }

                free(local_values);
            }
        } else {
#pragma omp parallel
            {
                real_t* const local_values = (real_t*)calloc(nsteps, sizeof(real_t));

#pragma omp for
                for (ptrdiff_t i = 0; i < nnodes; ++i) {
                    const count_t row_begin = cm_rowptr[i];
                    const count_t row_end   = cm_rowptr[i + 1];
                    if (row_begin == row_end) continue;

                    const ptrdiff_t dof1        = i * dim;
                    real_t          disp_normal = 0;
                    real_t          inc_normal  = 0;

                    for (int d = 0; d < dim; ++d) {
                        real_t disp_secondary = 0;
                        real_t inc_secondary  = 0;

                        for (count_t k = row_begin; k < row_end; ++k) {
                            const ptrdiff_t dof2 = cm_colidx[k] * dim;
                            const real_t    w    = cm_vals[k];

                            disp_secondary += w * disp[dof2 + d];
                            inc_secondary += w * inc[dof2 + d];
                        }

                        const real_t normal = normals[d][i];
                        disp_normal += normal * (disp[dof1 + d] - disp_secondary);
                        inc_normal += normal * (inc[dof1 + d] - inc_secondary);
                    }

                    const real_t m           = mass[i];
                    const real_t aug         = agumentation[i];
                    const real_t inv_penalty = real_t(1) / penalty[i];
                    const real_t shift       = -distances[i] + aug * inv_penalty;
                    const real_t c           = -real_t(0.5) * m * aug * aug * inv_penalty;
                    const real_t scale       = m * real_t(0.5) * penalty[i];

                    for (int s = 0; s < nsteps; ++s) {
                        const real_t v = disp_normal + steps[s] * inc_normal + shift;
                        const real_t p = v > 0 ? v : 0;
                        local_values[s] += scale * p * p + c;
                    }
                }

                for (int s = 0; s < nsteps; ++s) {
#pragma omp atomic update
                    values[s] += local_values[s];
                }

                free(local_values);
            }
        }
    }

    class ContactJacobi::Impl {
    public:
        std::shared_ptr<ContactData> cd;

        SharedBuffer<real_t> penalty;
        int    n_loops{10};
        bool   enable_augmentation{false};
        real_t relaxation_parameter{1. / 3};

        struct Workspace {
            smesh::SharedBuffer<real_t>  material_grad;
            smesh::SharedBuffer<real_t>  elast_diag_values;
            smesh::SharedBuffer<mask_t>  contact_node_mask;
            smesh::SharedBuffer<real_t>  contact_grad;
            smesh::SharedBuffer<real_t>  penetration;
            smesh::SharedBuffer<real_t>  macaulay;
            smesh::SharedBuffer<real_t*> displacement;
            smesh::SharedBuffer<real_t*> diag_values;
        };

        Workspace ws;

        void init() {
            auto f               = cd->f;
            auto contact_surface = cd->surface;
            auto es              = f->execution_space();

            const int       dim            = f->space()->mesh_ptr()->spatial_dimension();
            const ptrdiff_t ndofs          = f->space()->n_dofs();
            const ptrdiff_t n_nodes        = ndofs / dim;
            const ptrdiff_t n_contact      = contact_surface->n_nodes();
            const int       sym_block_size = (dim * (dim + 1)) / 2;

            // Material-related buffers
            ws.material_grad     = sfem::create_buffer<real_t>(ndofs, es);
            ws.elast_diag_values = sfem::create_buffer<real_t>(n_nodes * sym_block_size, es);
            ws.contact_node_mask = sfem::create_buffer<mask_t>(mask_count(n_nodes), es);

            // Contact-related buffers
            ws.contact_grad = sfem::create_buffer<real_t>(n_contact * dim, es);
            ws.penetration  = sfem::create_buffer<real_t>(n_contact, es);
            ws.macaulay     = sfem::create_buffer<real_t>(n_contact, es);
            ws.displacement = sfem::create_buffer<real_t>(dim, n_contact, es);
            ws.diag_values  = sfem::create_buffer<real_t>(dim * dim, n_contact, es);
        }

        Impl(const std::shared_ptr<ContactData>& cd) : cd(cd) { init(); }

        void smooth(const SharedBuffer<real_t>& x) {
            SFEM_TRACE_SCOPE("ContactJacobi::smooth");

            auto      space = cd->f->space();
            auto      mesh  = space->mesh_ptr();
            const int dim   = mesh->spatial_dimension();
            assert(dim == 3);

            const ptrdiff_t ndofs          = space->n_dofs();
            const ptrdiff_t n_nodes        = ndofs / dim;
            const ptrdiff_t n_contact      = cd->surface->node_mapping()->size();
            const int       sym_block_size = (dim * (dim + 1)) / 2;
            const real_t    omega          = 1. / dim;
            auto            es             = cd->f->execution_space();
            auto            blas           = sfem::blas<real_t>(es);

            auto material_grad     = ws.material_grad;
            auto elast_diag_values = ws.elast_diag_values;
            auto contact_node_mask = ws.contact_node_mask;
            auto contact_grad      = ws.contact_grad;
            auto macaulay          = ws.macaulay;
            auto displacement      = ws.displacement;
            auto diag_values       = ws.diag_values;
            auto constraints_mask  = cd->constraints_mask;
            assert(constraints_mask);

            const idx_t* const nm        = cd->surface->node_mapping()->data();
            const real_t* const reference =
                    cd->reference_displacement ? cd->reference_displacement->data() : nullptr;
            auto               cm        = cd->coupling_matrix;
            const count_t*     cm_rowptr = cm->row_ptr->data();
            const idx_t*       cm_colidx = cm->col_idx->data();
            const real_t*      cm_vals   = cm->values->data();
            const real_t*      distances = cd->distances->data();
            const real_t*      aug       = cd->agumentation->data();
            const real_t*      mass      = cd->mass_vector->data();
            real_t* const*     normals   = cd->normals->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < contact_node_mask->size(); ++i) {
                contact_node_mask->data()[i] = 0;
            }

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                mask_set(nm[i], contact_node_mask->data());
            }

            const mask_t* const mask         = constraints_mask->data();
            const mask_t* const contact_mask = contact_node_mask->data();
            real_t* const       xd           = x->data();

            // If the material is nonlinear should be inside the loop
            blas->values(elast_diag_values->size(), 0, elast_diag_values->data());
            cd->f->hessian_block_diag_sym(x->data(), elast_diag_values->data());

            ptrdiff_t each = std::min(n_loops, 1);
            for (int loop = 0; loop < n_loops; ++loop) {
                blas->values(ndofs, 0, material_grad->data());

                cd->f->gradient(x->data(), material_grad->data());

                const real_t* const eg = material_grad->data();
                const real_t* const ed = elast_diag_values->data();

                contact_gather_displacement(dim, n_contact, nm, x->data(), reference, displacement->data());

                blas->values(contact_grad->size(), 0, contact_grad->data());
                for (int d = 0; d < dim * dim; ++d) {
                    blas->values(n_contact, 0, diag_values->data()[d]);
                }

                compute_macaulay_term(dim,
                                      n_contact,
                                      cm_rowptr,
                                      cm_colidx,
                                      cm_vals,
                                      distances,
                                      aug,
                                      normals,
                                      mass,
                                      penalty->data(),
                                      1,
                                      displacement->data(),
                                      macaulay->data());

                assemble_contact_gradient(dim,
                                          n_contact,
                                          penalty->data(),
                                          cm_rowptr,
                                          cm_colidx,
                                          cm_vals,
                                          distances,
                                          aug,
                                          normals,
                                          mass,
                                          macaulay->data(),
                                          contact_grad->data());

                assemble_contact_hessian_diag_block(dim,
                                                    n_contact,
                                                    cm_rowptr,
                                                    cm_colidx,
                                                    cm_vals,
                                                    distances,
                                                    aug,
                                                    normals,
                                                    mass,
                                                    penalty->data(),
                                                    macaulay->data(),
                                                    1,
                                                    diag_values->data());

                const real_t* const ed_soa[6] = {ed, ed + 1, ed + 2, ed + 3, ed + 4, ed + 5};
                gather_combine_hessian_diag(dim, n_contact, nm, sym_block_size, ed_soa, 1, diag_values->data());

                real_t* const* const dv_mask = diag_values->data();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_contact; ++i) {
                    const ptrdiff_t dof = nm[i] * 3;
                    if (mask_get(dof, mask)) {
                        dv_mask[0][i] = 1;
                        dv_mask[1][i] = 0;
                        dv_mask[2][i] = 0;
                    }

                    if (mask_get(dof + 1, mask)) {
                        dv_mask[3][i] = 0;
                        dv_mask[4][i] = 1;
                        dv_mask[5][i] = 0;
                    }

                    if (mask_get(dof + 2, mask)) {
                        dv_mask[6][i] = 0;
                        dv_mask[7][i] = 0;
                        dv_mask[8][i] = 1;
                    }
                }

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_nodes; ++i) {
                    if (mask_get(i, contact_mask)) continue;

                    real_t a0 = ed[i * 6 + 0], a1 = ed[i * 6 + 1], a2 = ed[i * 6 + 2];
                    real_t a3 = ed[i * 6 + 1], a4 = ed[i * 6 + 3], a5 = ed[i * 6 + 4];
                    real_t a6 = ed[i * 6 + 2], a7 = ed[i * 6 + 4], a8 = ed[i * 6 + 5];

                    const ptrdiff_t dof = i * 3;
                    if (mask_get(dof, mask)) {
                        a0 = 1;
                        a1 = 0;
                        a2 = 0;
                    }

                    if (mask_get(dof + 1, mask)) {
                        a3 = 0;
                        a4 = 1;
                        a5 = 0;
                    }

                    if (mask_get(dof + 2, mask)) {
                        a6 = 0;
                        a7 = 0;
                        a8 = 1;
                    }

                    const real_t g0 = eg[dof + 0];
                    const real_t g1 = eg[dof + 1];
                    const real_t g2 = eg[dof + 2];

                    const real_t x0  = a4 * a8;
                    const real_t x1  = a5 * a7;
                    const real_t x2  = a1 * a5;
                    const real_t x3  = a1 * a8;
                    const real_t x4  = a2 * a4;
                    const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

                    if (!std::isfinite(det) || det == 0) {
                        if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
                        if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
                        if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
                        continue;
                    }

                    const real_t inv_det = 1 / det;

                    const real_t i0 = inv_det * (x0 - x1);
                    const real_t i1 = inv_det * (a2 * a7 - x3);
                    const real_t i2 = inv_det * (x2 - x4);
                    const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
                    const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
                    const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
                    const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
                    const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
                    const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

                    xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
                    xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
                    xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
                }

                const real_t* const* const dv = diag_values->data();
                const real_t* const        cg = contact_grad->data();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n_contact; ++i) {
                    const ptrdiff_t local_node  = i;
                    const ptrdiff_t global_node = nm[i];
                    const ptrdiff_t dof         = global_node * 3;
                    const ptrdiff_t local_dof   = local_node * 3;

                    const real_t g0 = eg[dof + 0] + (mask_get(dof + 0, mask) ? 0 : cg[local_dof + 0]);
                    const real_t g1 = eg[dof + 1] + (mask_get(dof + 1, mask) ? 0 : cg[local_dof + 1]);
                    const real_t g2 = eg[dof + 2] + (mask_get(dof + 2, mask) ? 0 : cg[local_dof + 2]);

                    if (g0 == 0 && g1 == 0 && g2 == 0) continue;

                    const real_t a0 = dv[0][local_node], a1 = dv[1][local_node], a2 = dv[2][local_node];
                    const real_t a3 = dv[3][local_node], a4 = dv[4][local_node], a5 = dv[5][local_node];
                    const real_t a6 = dv[6][local_node], a7 = dv[7][local_node], a8 = dv[8][local_node];

                    const real_t x0  = a4 * a8;
                    const real_t x1  = a5 * a7;
                    const real_t x2  = a1 * a5;
                    const real_t x3  = a1 * a8;
                    const real_t x4  = a2 * a4;
                    const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

                    if (!std::isfinite(det) || det == 0) {
                        if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
                        if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
                        if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
                        continue;
                    }

                    const real_t inv_det = 1 / det;

                    const real_t i0 = inv_det * (x0 - x1);
                    const real_t i1 = inv_det * (a2 * a7 - x3);
                    const real_t i2 = inv_det * (x2 - x4);
                    const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
                    const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
                    const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
                    const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
                    const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
                    const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

                    xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
                    xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
                    xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
                }

                real_t* const aug = cd->agumentation->data();
                if ((loop + 1) % each == 0 && enable_augmentation) {
                    contact_gather_displacement(dim, n_contact, nm, x->data(), reference, displacement->data());
                    compute_macaulay_term(dim,
                                          n_contact,
                                          cm_rowptr,
                                          cm_colidx,
                                          cm_vals,
                                          distances,
                                          aug,
                                          normals,
                                          mass,
                                          penalty->data(),
                                          1,
                                          displacement->data(),
                                          macaulay->data());

                    const real_t* const m = macaulay->data();
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n_contact; ++i) {
                        aug[i] = penalty->data()[i] * m[i];
                    }
                }
            }
        }
    };

    ContactJacobi::ContactJacobi(const std::shared_ptr<ContactData>& cd) : impl_(std::make_unique<Impl>(cd)) {}
    ContactJacobi::~ContactJacobi() {}

    void ContactJacobi::smooth(const SharedBuffer<real_t>& x) { impl_->smooth(x); }

    void ContactJacobi::set_penalty(const SharedBuffer<real_t>& penalty) { impl_->penalty = penalty; }

    void ContactJacobi::set_penalty(const real_t penalty) {
        const ptrdiff_t n    = impl_->cd->surface->node_mapping()->size();
        auto            buff = sfem::create_buffer<real_t>(n, impl_->cd->f->execution_space());
        real_t* const   d    = buff->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            d[i] = penalty;
        }

        impl_->penalty = buff;
    }

    void ContactJacobi::set_n_loops(int n_loops) { impl_->n_loops = n_loops; }

    void ContactJacobi::set_enable_augmentation(bool enable_augmentation) { impl_->enable_augmentation = enable_augmentation; }

    void ContactJacobi::set_relaxation_parameter(real_t relaxation_parameter) {
        impl_->relaxation_parameter = relaxation_parameter;
    }

    void ContactJacobi::contact_data_changed() { impl_->init(); }

}  // namespace sfem
