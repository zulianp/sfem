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
                               const real_t* const* const SFEM_RESTRICT               normals,
                               const real_t* const SFEM_RESTRICT                      mass,
                               const real_t                                           penalty,
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

                const real_t normal_diff =
                        n0[i] * (in0[dof1] - u20) + n1[i] * (in1[dof1] - u21) + n2[i] * (in2[dof1] - u22);
                macaulay[i] = std::max(normal_diff - distances[i] + agumentation[i] / penalty, real_t(0));
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

                macaulay[i] = std::max(normal_diff - distances[i] + agumentation[i] / penalty, real_t(0));
            }
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

                const real_t normal_diff =
                        n0[i] * (in0[dof1] - u20) + n1[i] * (in1[dof1] - u21) + n2[i] * (in2[dof1] - u22);
                penetration[i] = std::max(real_t(0), normal_diff - gap[i]);
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

    void contact_gather_displacement(const int                                        dim,
                                     const ptrdiff_t                                  n_contact,
                                     const idx_t* const SFEM_RESTRICT                 node_mapping,
                                     const real_t* const SFEM_RESTRICT                in,
                                     real_t* const SFEM_RESTRICT* const SFEM_RESTRICT out) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_contact; ++i) {
            const ptrdiff_t dof = node_mapping[i] * dim;
            for (int d = 0; d < dim; ++d) {
                out[d][i] = in[dof + d];
            }
        }
    }

    //     struct ContactJacobiParameters {
    //         real_t penalty{100};
    //         int    n_loops{10};
    //         real_t solver_tol{1e-6};
    //         bool   enable_augmentation{true};
    //         real_t relaxation_parameter{1. / 3};
    //     };

    //     struct ContactJacobiWorkspace {
    //         smesh::SharedBuffer<real_t> material_grad;
    //         smesh::SharedBuffer<real_t> elast_diag_values;
    //         smesh::SharedBuffer<mask_t> contact_node_mask;
    //         smesh::SharedBuffer<real_t> contact_grad;
    //         smesh::SharedBuffer<real_t> penetration;
    //         smesh::SharedBuffer<real_t> macaulay;
    //         smesh::SharedBuffer<real_t> diag_values;
    //     };

    //     void contact_jacobi_workspace_init(const std::shared_ptr<sfem::Function>& f,
    //                                        const std::shared_ptr<sfem::Mesh>&     contact_surface,
    //                                        ContactJacobiWorkspace&                ws) {
    //         SFEM_TRACE_SCOPE("contact_jacobi_workspace_init");

    //         int             dim     = f->space()->mesh_ptr()->spatial_dimension();
    //         const ptrdiff_t ndofs   = f->space()->n_dofs();
    //         const ptrdiff_t n_nodes = ndofs / dim;

    //         const ptrdiff_t n_contact = contact_surface->n_nodes();

    //         const int sym_block_size = (dim * (dim + 1)) / 2;
    //         auto      es             = f->execution_space();

    //         // Material-related buffers
    //         ws.material_grad     = sfem::create_buffer<real_t>(ndofs, es);
    //         ws.elast_diag_values = sfem::create_buffer<real_t>(n_nodes * sym_block_size, es);
    //         ws.contact_node_mask = sfem::create_buffer<mask_t>(mask_count(n_nodes), es);

    //         // Contact-related buffers
    //         ws.contact_grad = sfem::create_buffer<real_t>(ndofs, es);
    //         ws.penetration  = sfem::create_buffer<real_t>(n_contact, es);
    //         ws.macaulay     = sfem::create_buffer<real_t>(n_contact, es);
    //         ws.diag_values  = sfem::create_buffer<real_t>(dim * dim, n_contact, es);
    //     }

    //     void contact_jacobi_apply(const ContactJacobiParameters&               params,
    //                               const ContactJacobiContactData&              cd,
    //                               const std::shared_ptr<sfem::Function>&       f,
    //                               const std::shared_ptr<sfem::Buffer<real_t>>& x,
    //                               ContactJacobiWorkspace&                      ws) {
    //         SFEM_TRACE_SCOPE("contact_jacobi_step");

    //         auto      space = f->space();
    //         auto      mesh  = space->mesh_ptr();
    //         const int dim   = mesh->spatial_dimension();
    //         assert(dim == 3);

    //         const ptrdiff_t ndofs          = space->n_dofs();
    //         const ptrdiff_t n_nodes        = ndofs / dim;
    //         const ptrdiff_t n_contact      = cd.surface->node_mapping()->size();
    //         const int       sym_block_size = (dim * (dim + 1)) / 2;
    //         const real_t    omega          = 1. / 3;
    //         auto            es             = f->execution_space();
    //         auto            blas           = sfem::blas<real_t>(es);

    //         auto material_grad     = ws.material_grad;
    //         auto elast_diag_values = ws.elast_diag_values;
    //         auto contact_node_mask = ws.contact_node_mask;
    //         auto contact_grad      = ws.contact_grad;
    //         auto penetration       = ws.penetration;
    //         auto macaulay          = ws.macaulay;
    //         auto diag_values       = ws.diag_values;
    //         auto constraints_mask  = cd.constraints_mask;

    //         assert(constraints_mask);
    //         ContactKernelWorkspace contact_ws(dim, n_contact, es);

    //         const idx_t* const nm = cd.surface->node_mapping()->data();
    // #pragma omp parallel for
    //         for (ptrdiff_t i = 0; i < contact_node_mask->size(); ++i) {
    //             contact_node_mask->data()[i] = 0;
    //         }

    // #pragma omp parallel for
    //         for (ptrdiff_t i = 0; i < n_contact; ++i) {
    //             mask_set(nm[i], contact_node_mask->data());
    //         }

    //         const mask_t* const mask         = constraints_mask->data();
    //         const mask_t* const contact_mask = contact_node_mask->data();
    //         real_t* const       xd           = x->data();

    //         // If the material is nonlinear should be inside the loop
    //         blas->values(elast_diag_values->size(), 0, elast_diag_values->data());
    //         f->hessian_block_diag_sym(x->data(), elast_diag_values->data());

    //         ptrdiff_t each = std::min(n_loops, 1);
    //         for (int loop = 0; loop < n_loops; ++loop) {
    //             blas->values(ndofs, 0, material_grad->data());

    //             f->gradient(x->data(), material_grad->data());

    //             const real_t* const eg = material_grad->data();
    //             const real_t* const ed = elast_diag_values->data();

    //             blas->values(ndofs, 0, contact_grad->data());
    //             for (int d = 0; d < dim * dim; ++d) {
    //                 blas->values(n_contact, 0, diag_values->data()[d]);
    //             }

    //             compute_macaulay_term(cd, contact_ws, penalty, x->data(), macaulay->data());
    //             assemble_contact_gradient(cd, contact_ws, penalty, macaulay->data(), contact_grad->data());
    //             assemble_contact_hessian_diag(cd, penalty, macaulay->data(), 1, diag_values->data());
    //             gather_combine_hessian_diag(cd, constraints_mask->data(), elast_diag_values->data(), 1, diag_values->data());

    // #pragma omp parallel for
    //             for (ptrdiff_t i = 0; i < n_nodes; ++i) {
    //                 if (mask_get(i, contact_mask)) continue;

    //                 real_t a0 = ed[i * 6 + 0], a1 = ed[i * 6 + 1], a2 = ed[i * 6 + 2];
    //                 real_t a3 = ed[i * 6 + 1], a4 = ed[i * 6 + 3], a5 = ed[i * 6 + 4];
    //                 real_t a6 = ed[i * 6 + 2], a7 = ed[i * 6 + 4], a8 = ed[i * 6 + 5];

    //                 const ptrdiff_t dof = i * 3;
    //                 if (mask_get(dof, mask)) {
    //                     a0 = 1;
    //                     a1 = 0;
    //                     a2 = 0;
    //                 }

    //                 if (mask_get(dof + 1, mask)) {
    //                     a3 = 0;
    //                     a4 = 1;
    //                     a5 = 0;
    //                 }

    //                 if (mask_get(dof + 2, mask)) {
    //                     a6 = 0;
    //                     a7 = 0;
    //                     a8 = 1;
    //                 }

    //                 const real_t g0 = eg[dof + 0];
    //                 const real_t g1 = eg[dof + 1];
    //                 const real_t g2 = eg[dof + 2];

    //                 const real_t x0  = a4 * a8;
    //                 const real_t x1  = a5 * a7;
    //                 const real_t x2  = a1 * a5;
    //                 const real_t x3  = a1 * a8;
    //                 const real_t x4  = a2 * a4;
    //                 const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

    //                 if (!std::isfinite(det) || det == 0) {
    //                     if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
    //                     if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
    //                     if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
    //                     continue;
    //                 }

    //                 const real_t inv_det = 1 / det;

    //                 const real_t i0 = inv_det * (x0 - x1);
    //                 const real_t i1 = inv_det * (a2 * a7 - x3);
    //                 const real_t i2 = inv_det * (x2 - x4);
    //                 const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
    //                 const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
    //                 const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
    //                 const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
    //                 const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
    //                 const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

    //                 xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
    //                 xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
    //                 xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
    //             }

    //             const real_t* const* const dv = diag_values->data();
    //             const real_t* const        cg = contact_grad->data();

    // #pragma omp parallel for
    //             for (ptrdiff_t i = 0; i < n_contact; ++i) {
    //                 const ptrdiff_t local_node  = i;
    //                 const ptrdiff_t global_node = nm[i];
    //                 const ptrdiff_t dof         = global_node * 3;

    //                 const real_t g0 = eg[dof + 0] + (mask_get(dof + 0, mask) ? 0 : cg[dof + 0]);
    //                 const real_t g1 = eg[dof + 1] + (mask_get(dof + 1, mask) ? 0 : cg[dof + 1]);
    //                 const real_t g2 = eg[dof + 2] + (mask_get(dof + 2, mask) ? 0 : cg[dof + 2]);

    //                 if (g0 == 0 && g1 == 0 && g2 == 0) continue;

    //                 const real_t a0 = dv[0][local_node], a1 = dv[1][local_node], a2 = dv[2][local_node];
    //                 const real_t a3 = dv[3][local_node], a4 = dv[4][local_node], a5 = dv[5][local_node];
    //                 const real_t a6 = dv[6][local_node], a7 = dv[7][local_node], a8 = dv[8][local_node];

    //                 const real_t x0  = a4 * a8;
    //                 const real_t x1  = a5 * a7;
    //                 const real_t x2  = a1 * a5;
    //                 const real_t x3  = a1 * a8;
    //                 const real_t x4  = a2 * a4;
    //                 const real_t det = a0 * x0 - a0 * x1 + a2 * a3 * a7 - a3 * x3 + a6 * x2 - a6 * x4;

    //                 if (!std::isfinite(det) || det == 0) {
    //                     if (std::isfinite(a0) && a0 != 0) xd[dof + 0] -= omega * g0 / a0;
    //                     if (std::isfinite(a4) && a4 != 0) xd[dof + 1] -= omega * g1 / a4;
    //                     if (std::isfinite(a8) && a8 != 0) xd[dof + 2] -= omega * g2 / a8;
    //                     continue;
    //                 }

    //                 const real_t inv_det = 1 / det;

    //                 const real_t i0 = inv_det * (x0 - x1);
    //                 const real_t i1 = inv_det * (a2 * a7 - x3);
    //                 const real_t i2 = inv_det * (x2 - x4);
    //                 const real_t i3 = inv_det * (-a3 * a8 + a5 * a6);
    //                 const real_t i4 = inv_det * (a0 * a8 - a2 * a6);
    //                 const real_t i5 = inv_det * (-a0 * a5 + a2 * a3);
    //                 const real_t i6 = inv_det * (a3 * a7 - a4 * a6);
    //                 const real_t i7 = inv_det * (-a0 * a7 + a1 * a6);
    //                 const real_t i8 = inv_det * (a0 * a4 - a1 * a3);

    //                 xd[dof + 0] -= omega * (i0 * g0 + i1 * g1 + i2 * g2);
    //                 xd[dof + 1] -= omega * (i3 * g0 + i4 * g1 + i5 * g2);
    //                 xd[dof + 2] -= omega * (i6 * g0 + i7 * g1 + i8 * g2);
    //             }

    //             real_t* const aug = cd.agumentation->data();
    //             if ((loop + 1) % each == 0 && enable_augmentation) {
    //                 compute_macaulay_term(cd, contact_ws, penalty, x->data(), macaulay->data());

    //                 const real_t* const m = macaulay->data();
    // #pragma omp parallel for
    //                 for (ptrdiff_t i = 0; i < n_contact; ++i) {
    //                     aug[i] = penalty * m[i];
    //                 }
    //             }

    //             compute_penetration(cd, contact_ws, x->data(), penetration->data());

    //             real_t              penetration_norm2 = 0;
    //             real_t              lagr_mult_norm2   = 0;
    //             const real_t* const p                 = penetration->data();
    // #pragma omp parallel for reduction(+ : penetration_norm2, lagr_mult_norm2)
    //             for (ptrdiff_t i = 0; i < n_contact; ++i) {
    //                 penetration_norm2 += p[i] * p[i];
    //                 lagr_mult_norm2 += aug[i] * aug[i];
    //             }

    //             real_t              full_grad_norm2 = 0;
    //             const real_t* const mg              = material_grad->data();
    // #pragma omp parallel for reduction(+ : full_grad_norm2)
    //             for (ptrdiff_t i = 0; i < ndofs; ++i) {
    //                 const real_t g = mg[i] + cg[i];
    //                 full_grad_norm2 += g * g;
    //             }

    //             full_grad_norm2   = std::sqrt(full_grad_norm2);
    //             penetration_norm2 = std::sqrt(penetration_norm2);
    //             lagr_mult_norm2   = std::sqrt(lagr_mult_norm2);

    //             if (full_grad_norm2 < solver_tol && penetration_norm2 < solver_tol && lagr_mult_norm2 < solver_tol) {
    //                 break;
    //             }

    //             if (loop % 100 == 0) {
    //                 printf("%d) full_grad_norm = %g, penetration_norm = %g, lagr_mult_norm = %g\n",
    //                        loop,
    //                        full_grad_norm2,
    //                        penetration_norm2,
    //                        lagr_mult_norm2);
    //             }
    //         }
    //     }

}  // namespace sfem
