#include "sfem_ContactSolveKernels.hpp"
#include "sfem_aliases.hpp"
#include "sfem_macros.hpp"
// Gather / Scatter done outside

#include <stddef.h>
#include "sfem_base.hpp"

namespace sfem {

    void compute_macaulay_term(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const SFEM_RESTRICT                      distances,
                               const real_t* const SFEM_RESTRICT                      agumentation,
                               const real_t* const* SFEM_RESTRICT const SFEM_RESTRICT normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t                                           penalty,
                               const ptrdiff_t                                        in_stride,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in_old,
                               const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                               real_t* const                                          macaulay) {
        SFEM_TRACE_SCOPE("compute_macaulay_term");

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            auto lenrow = cm_rowptr[i + 1] - cm_rowptr[i];
            if (lenrow == 0) {
                macaulay[i] = 0;
                continue;
            }

            auto row     = &cm_colidx[cm_rowptr[i]];
            auto weights = &cm_vals[cm_rowptr[i]];

            real_t u1[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                u1[d] = in[d][i * in_stride] - in_old[d][i * in_stride];
            }

            real_t u2[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
                    const ptrdiff_t dof = row[j] * dim + d;
                    u2[d] += weights[j] * (in[d][j * in_stride] - in_old[d][j * in_stride]);
                }
            }

            const real_t g         = distances[i];
            real_t       normal[3] = {0, 0, 0};
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

            real_t pen       = normal_diff - g;
            real_t lagr_mult = agumentation[i];
            macaulay[i]      = std::max(pen + lagr_mult / penalty, real_t(0));
        }
    }

    void assemble_contact_gradient(const int                                dim,
                                   const ptrdiff_t                          nnodes,
                                   const real_t                             penalty,
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
                force[d] = mass[i] * penalty * macaulay[i] * normal[d];
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
                                             const real_t                             penalty,
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
                    nnT[d1 * dim + d2] = mass[i] * penalty * normal[d1] * normal[d2];
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

    void contact_hessian_apply(const int                                              dim,
                               const ptrdiff_t                                        nnodes,
                               const count_t* const SFEM_RESTRICT                     cm_rowptr,
                               const idx_t* const SFEM_RESTRICT                       cm_colidx,
                               const real_t* const SFEM_RESTRICT                      cm_vals,
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t                                           penalty,
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
                    const ptrdiff_t dof = row[j] * dim + d;
                    u2[d] += weights[j] * in[d][j * in_stride];
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
                applied[d] = mass[i] * normal_diff * normal[d];
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
                             const real_t* const* SFEM_RESTRICT const SFEM_RESTRICT normals,
                             const real_t* const SFEM_RESTRICT                      gap,
                             const ptrdiff_t                                        in_stride,
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in_old,
                             const real_t* const SFEM_RESTRICT* const SFEM_RESTRICT in,
                             real_t* const SFEM_RESTRICT                            penetration) {
        SFEM_TRACE_SCOPE("compute_penetration");

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
                    const ptrdiff_t dof2 = row[j] * in_stride + d;
                    u2 += weights[j] * (in[d][dof2] - in_old[d][dof2]);
                }

                normal_diff += normals[d][i] * (in[d][dof1] - in_old[d][dof1] - u2);
            }

            penetration[i] = std::max(real_t(0), normal_diff - gap[i]);
        }
    }

}  // namespace sfem
