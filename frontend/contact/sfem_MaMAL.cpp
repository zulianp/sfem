#include "sfem_MaMAL.hpp"

#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"

#include "sfem_CRS.hpp"
#include "sfem_SelfContact.hpp"
#include "smesh_crs_graph.hpp"

#include "sfem_API.hpp"
#include "smesh_ssquad4_prolongation.hpp"

namespace sfem {
    class Memory {
    public:
        SharedBuffer<real_t> rhs;
        SharedBuffer<real_t> solution;
        SharedBuffer<real_t> work;
        SharedBuffer<real_t> diag;
        inline ptrdiff_t     size() const { return solution->size(); }
        ~Memory() {}
    };

    struct MaMALParams {
        int    max_iterations{100};
        real_t tolerance{1e-6};
        real_t margin{0.01};
        real_t search_radius{1e-4};

#ifdef SFEM_ENABLE_YAML
        void from_yaml(const ryml::ConstNodeRef& node) {
            // TODO: implement
            max_iterations = node["max_iterations"].val<int>();
            tolerance      = node["tolerance"].val<real_t>();
            margin         = node["margin"].val<real_t>();
            search_radius  = node["search_radius"].val<real_t>();
        }
#endif
    };

    using CRSGraph_t = smesh::CRSGraph<count_t, idx_t>;
    using CRS_t      = sfem::CRS<count_t, idx_t, real_t>;

    struct GalerkinRAP {
        std::shared_ptr<CRS_t> R, P;

        std::shared_ptr<CRS_t> apply(const std::shared_ptr<CRS_t>& A) const { return sfem::rap(R, A, P); }
    };

    struct ContactData {
        std::shared_ptr<sfem::Function> f;
        std::shared_ptr<smesh::Mesh>    surface;
        std::shared_ptr<CRS_t>          coupling_matrix;
        smesh::SharedBuffer<real_t>     values;
        smesh::SharedBuffer<real_t>     mass_vector;
        smesh::SharedBuffer<real_t*>    normals;
        smesh::SharedBuffer<real_t>     distances;
        smesh::SharedBuffer<real_t>     frozen_displacement;
        SharedBuffer<mask_t>            constraints_mask;
        smesh::SharedBuffer<real_t>     agumentation;
    };

    void compute_penetration(ContactData& cd, const real_t* const disp, real_t* const penetration) {
        SFEM_TRACE_SCOPE("compute_penetration");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = coupling_matrix->values->data();
        ptrdiff_t n               = coupling_matrix->rows();

        auto d       = cd.distances->data();
        auto normals = cd.normals->data();
        auto disp0   = cd.frozen_displacement->data();
        auto nm      = cd.surface->node_mapping()->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            const count_t lenrow = rowptr[i + 1] - rowptr[i];
            if (lenrow == 0) {
                penetration[i] = 0;
                continue;
            }

            const idx_t* const  row     = &colidx[rowptr[i]];
            const real_t* const weights = &vals[rowptr[i]];
            const ptrdiff_t     dof1    = nm[i] * dim;

            real_t normal_diff = 0;
            for (int d = 0; d < dim; d++) {
                real_t u2 = 0;
                for (count_t j = 0; j < lenrow; j++) {
                    const ptrdiff_t dof2 = nm[row[j]] * dim + d;
                    u2 += weights[j] * (disp[dof2] - disp0[dof2]);
                }

                normal_diff += normals[d][i] * (disp[dof1 + d] - disp0[dof1 + d] - u2);
            }

            penetration[i] = std::max(real_t(0), normal_diff - d[i]);
        }
    }

    void compute_macaulay_term_from_penetration(ContactData&        cd,
                                                const real_t        penalty,
                                                const real_t* const penetration,
                                                real_t* const       macaulay) {
        SFEM_TRACE_SCOPE("compute_macaulay_term_from_penetration");
        auto            coupling_matrix = cd.coupling_matrix;
        auto            rowptr          = coupling_matrix->row_ptr->data();
        auto            aug             = cd.agumentation->data();
        const ptrdiff_t n               = coupling_matrix->rows();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            if (rowptr[i + 1] == rowptr[i]) {
                macaulay[i] = 0;
                continue;
            }

            macaulay[i] = std::max(penetration[i] + aug[i] / penalty, real_t(0));
        }
    }

    void displace_points(const std::shared_ptr<smesh::Mesh>&     surface,
                         const std::shared_ptr<Buffer<real_t>>&  displacement,
                         const std::shared_ptr<Buffer<real_t*>>& inout) {
        auto p = inout->data();
        auto u = displacement->data();
        auto m = surface->node_mapping()->data();

        const ptrdiff_t n   = surface->node_mapping()->size();
        const int       dim = surface->spatial_dimension();

        for (int d = 0; d < dim; d++) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; i++) {
                p[d][i] += u[m[i] * dim + d];
            }
        }
    }

    void compute_macaulay_term(ContactData& cd, const real_t penalty, const real_t* const disp, real_t* const macaulay) {
        SFEM_TRACE_SCOPE("compute_macaulay_term");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      values          = cd.values;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = values->data();
        ptrdiff_t n               = coupling_matrix->row_ptr->size() - 1;

        auto d       = cd.distances->data();
        auto aug     = cd.agumentation->data();
        auto normals = cd.normals->data();
        auto mass    = cd.mass_vector->data();
        auto disp0   = cd.frozen_displacement->data();

        auto nm = cd.surface->node_mapping()->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            auto lenrow = rowptr[i + 1] - rowptr[i];
            if (lenrow == 0) {
                macaulay[i] = 0;
                continue;
            }

            auto row = &colidx[rowptr[i]];

            auto weights = &vals[rowptr[i]];

            real_t u1[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                const ptrdiff_t dof = nm[i] * dim + d;
                u1[d]               = disp[dof] - disp0[dof];
            }

            real_t u2[3] = {0, 0, 0};
            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
                    const ptrdiff_t dof = nm[row[j]] * dim + d;
                    u2[d] += weights[j] * (disp[dof] - disp0[dof]);
                }
            }

            const real_t g         = d[i];
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
            real_t lagr_mult = aug[i];
            macaulay[i]      = std::max(pen + lagr_mult / penalty, real_t(0));
        }
    }

    void assemble_contact_gradient(ContactData& cd, const real_t penalty, const real_t* const macaulay, real_t* const grad) {
        SFEM_TRACE_SCOPE("assemble_contact_gradient");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      values          = coupling_matrix->values;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = values->data();
        ptrdiff_t n               = coupling_matrix->row_ptr->size() - 1;

        auto d       = cd.distances->data();
        auto aug     = cd.agumentation->data();
        auto normals = cd.normals->data();
        auto mass    = cd.mass_vector->data();

        auto nm = cd.surface->node_mapping()->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            if (macaulay[i] == 0) continue;

            auto lenrow = rowptr[i + 1] - rowptr[i];
            if (lenrow == 0) continue;

            auto row     = &colidx[rowptr[i]];
            auto weights = &vals[rowptr[i]];

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
                grad[nm[i] * dim + d] += force[d];
            }

            for (int d = 0; d < dim; d++) {
                for (count_t j = 0; j < lenrow; j++) {
#pragma omp atomic update
                    grad[nm[row[j]] * dim + d] -= force[d] * weights[j];
                }
            }
        }
    }

    void assemble_contact_hessian_diag(ContactData&                                     cd,
                                       const real_t                                     penalty,
                                       const real_t* const                              macaulay,
                                       const ptrdiff_t                                  diag_stride,
                                       real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("assemble_contact_hessian_diag");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = coupling_matrix->values->data();
        ptrdiff_t n               = coupling_matrix->rows();

        auto d       = cd.distances->data();
        auto aug     = cd.agumentation->data();
        auto normals = cd.normals->data();
        auto mass    = cd.mass_vector->data();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            if (macaulay[i] == 0) continue;

            auto lenrow = rowptr[i + 1] - rowptr[i];
            if (lenrow == 0) continue;

            auto row     = &colidx[rowptr[i]];
            auto weights = &vals[rowptr[i]];

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

    // Gather the diagonal values from the symmetric representation elast_diag_values (uses node mapping to read), add them to the
    // diag_values, mask (uses node mapping to read) them for the constraint rows with an identiity row
    void gather_combine_hessian_diag(ContactData&                                     cd,
                                     const mask_t* const                              is_constrained,
                                     const real_t* const                              elast_diag_values,
                                     const ptrdiff_t                                  diag_stride,
                                     real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("gather_combine_hessian_diag");
        const int       dim = cd.surface->spatial_dimension();
        const ptrdiff_t n   = cd.surface->node_mapping()->size();
        const idx_t*    nm  = cd.surface->node_mapping()->data();

        if (dim == 3) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t global_node = nm[i];
                const real_t*   ed          = &elast_diag_values[global_node * 6];
                const ptrdiff_t local_node  = i * diag_stride;

                diag_values[0][local_node] += ed[0];
                diag_values[1][local_node] += ed[1];
                diag_values[2][local_node] += ed[2];
                diag_values[3][local_node] += ed[1];
                diag_values[4][local_node] += ed[3];
                diag_values[5][local_node] += ed[4];
                diag_values[6][local_node] += ed[2];
                diag_values[7][local_node] += ed[4];
                diag_values[8][local_node] += ed[5];

                const ptrdiff_t dof = global_node * 3;
                if (mask_get(dof, is_constrained)) {
                    diag_values[0][local_node] = 1;
                    diag_values[1][local_node] = 0;
                    diag_values[2][local_node] = 0;
                }

                if (mask_get(dof + 1, is_constrained)) {
                    diag_values[3][local_node] = 0;
                    diag_values[4][local_node] = 1;
                    diag_values[5][local_node] = 0;
                }

                if (mask_get(dof + 2, is_constrained)) {
                    diag_values[6][local_node] = 0;
                    diag_values[7][local_node] = 0;
                    diag_values[8][local_node] = 1;
                }
            }
        } else {
            const int sym_block_size = (dim * (dim + 1)) / 2;

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t global_node = nm[i];
                const real_t*   ed          = &elast_diag_values[global_node * sym_block_size];
                const ptrdiff_t local_node  = i * diag_stride;

                int s = 0;
                for (int d1 = 0; d1 < dim; ++d1) {
                    diag_values[d1 * dim + d1][local_node] += ed[s++];
                    for (int d2 = d1 + 1; d2 < dim; ++d2) {
                        const real_t e = ed[s++];
                        diag_values[d1 * dim + d2][local_node] += e;
                        diag_values[d2 * dim + d1][local_node] += e;
                    }
                }

                const ptrdiff_t dof = global_node * dim;
                for (int d1 = 0; d1 < dim; ++d1) {
                    if (!mask_get(dof + d1, is_constrained)) continue;
                    for (int d2 = 0; d2 < dim; ++d2) {
                        diag_values[d1 * dim + d2][local_node] = (d1 == d2) ? 1 : 0;
                    }
                }
            }
        }
    }

    class ContactJacobi {
    public:
        ContactData cd;

        real_t penalty{100};
        int    n_loops{3};
        bool   enable_augmentation{false};

        void smooth(const SharedBuffer<real_t>& x) {
            SFEM_TRACE_SCOPE("ContactJacobi::smooth");

            auto      space = cd.f->space();
            auto      mesh  = space->mesh_ptr();
            const int dim   = mesh->spatial_dimension();
            assert(dim == 3);

            const ptrdiff_t ndofs          = space->n_dofs();
            const ptrdiff_t n_nodes        = ndofs / dim;
            const ptrdiff_t n_contact      = cd.surface->node_mapping()->size();
            const int       sym_block_size = (dim * (dim + 1)) / 2;
            const real_t    omega          = 1. / dim;
            auto            es             = cd.f->execution_space();
            auto            blas           = sfem::blas<real_t>(es);

            auto material_grad     = sfem::create_buffer<real_t>(ndofs, es);
            auto elast_diag_values = sfem::create_buffer<real_t>(n_nodes * sym_block_size, es);
            auto contact_node_mask = sfem::create_buffer<mask_t>(mask_count(n_nodes), es);
            auto contact_grad      = sfem::create_buffer<real_t>(ndofs, es);
            auto penetration       = sfem::create_buffer<real_t>(n_contact, es);
            auto macaulay          = sfem::create_buffer<real_t>(n_contact, es);
            auto diag_values       = sfem::create_buffer<real_t>(dim * dim, n_contact, es);
            auto constraints_mask  = cd.constraints_mask;
            assert(constraints_mask);

            const idx_t* const nm = cd.surface->node_mapping()->data();
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
            cd.f->hessian_block_diag_sym(x->data(), elast_diag_values->data());

            ptrdiff_t each = std::min(n_loops, 1);
            for (int loop = 0; loop < n_loops; ++loop) {
                blas->values(ndofs, 0, material_grad->data());

                cd.f->gradient(x->data(), material_grad->data());

                const real_t* const eg = material_grad->data();
                const real_t* const ed = elast_diag_values->data();

                blas->values(ndofs, 0, contact_grad->data());
                for (int d = 0; d < dim * dim; ++d) {
                    blas->values(n_contact, 0, diag_values->data()[d]);
                }

                compute_macaulay_term(cd, penalty, x->data(), macaulay->data());
                assemble_contact_gradient(cd, penalty, macaulay->data(), contact_grad->data());
                assemble_contact_hessian_diag(cd, penalty, macaulay->data(), 1, diag_values->data());
                gather_combine_hessian_diag(cd, constraints_mask->data(), elast_diag_values->data(), 1, diag_values->data());

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

                    const real_t g0 = eg[dof + 0] + (mask_get(dof + 0, mask) ? 0 : cg[dof + 0]);
                    const real_t g1 = eg[dof + 1] + (mask_get(dof + 1, mask) ? 0 : cg[dof + 1]);
                    const real_t g2 = eg[dof + 2] + (mask_get(dof + 2, mask) ? 0 : cg[dof + 2]);

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

                real_t* const aug = cd.agumentation->data();
                if ((loop + 1) % each == 0 && enable_augmentation) {
                    compute_macaulay_term(cd, penalty, x->data(), macaulay->data());

                    const real_t* const m = macaulay->data();
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n_contact; ++i) {
                        aug[i] = penalty * m[i];
                    }
                }
            }
        }
    };

    std::vector<GalerkinRAP> create_galerkin_rap(const SharedBuffer<idx_t*>& elements, const std::vector<int>& levels) {
        const ptrdiff_t nelements = elements->extent(1);
        int             nlevels   = levels.size();

        ptrdiff_t fine_n_nodes = sfem::ss_elements_max_node_id(elements) + 1;

        std::vector<GalerkinRAP> ret;
        auto                     elems = elements;
        for (int i = 0; i < nlevels - 1; i++) {
            auto rowptr = create_host_buffer<count_t>(fine_n_nodes);
            smesh::ssquad4_prolongation_crs_nnz(levels[i], nelements, elems->data(), fine_n_nodes, rowptr->data());

            auto colidx = create_host_buffer<idx_t>(rowptr->data()[fine_n_nodes]);
            auto values = create_host_buffer<real_t>(rowptr->data()[fine_n_nodes]);

            smesh::ssquad4_prolongation_crs_fill(
                    levels[i], nelements, elems->data(), fine_n_nodes, rowptr->data(), colidx->data(), values->data());

            // FIXME This method should be moved to SMESH
            elems                    = sfem::ssquad4_derefine_element_connectivity(levels[i], levels[i + 1], elements);
            ptrdiff_t coarse_n_nodes = sfem::ss_elements_max_node_id(elems) + 1;

            auto P = h_crs_spmv(fine_n_nodes, coarse_n_nodes, rowptr, colidx, values, real_t(0));
            auto R = P->transpose();
            ret.push_back({.R = R, .P = P});

            fine_n_nodes = coarse_n_nodes;
        }

        return ret;
    }

    class MaMAL::Impl {
    public:
        std::shared_ptr<Function>                      f;
        MaMALParams                                    params;
        std::shared_ptr<Contact>                       contact;
        std::shared_ptr<MultigridData>                 data;
        std::vector<std::shared_ptr<Operator<real_t>>> operators;
        // std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers;

        std::vector<std::shared_ptr<CRS_t>> coupling_matrices;
        std::vector<SharedBuffer<real_t*>>  normals;
        std::vector<SharedBuffer<real_t>>   mass_vectors;
        std::vector<GalerkinRAP>            galerkin_restrictions;

        std::vector<std::shared_ptr<Memory>> memory;

        std::shared_ptr<smesh::Mesh>   contact_surface;
        std::shared_ptr<ContactJacobi> contact_jacobi;
        SharedBuffer<mask_t>           constraints_mask;
        SharedBuffer<real_t>           agumentation;

        Impl(const std::shared_ptr<Function>& f) : f(f) {}

        int n_levels() const { return data->semistructured_levels.size(); }

#ifdef SFEM_ENABLE_YAML
        void init(const ryml::ConstNodeRef& node) {
            params.from_yaml(node);
            init();
        }
#endif

        void init() {
            auto space             = f->space();
            bool is_semistructured = space->has_semi_structured_mesh();

            if (!is_semistructured) {
                SFEM_ERROR("MaMAL is not supported for non-semistructured meshes!\n");
                return;
            }

            auto mesh        = space->mesh_ptr();
            auto block_size  = space->block_size();
            auto spatial_dim = mesh->spatial_dimension();
            auto es          = f->execution_space();

            data = sfem::create_gmg_data(f);
            if (!data) {
                SFEM_ERROR("[Error] MaMAL could not build gmg data!\n");
                return;
            }

            operators = sfem::create_gmg_operators(data, op_type::MATRIX_FREE);

            contact_surface = smesh::skin(mesh);
            contact = create_contact(space, contact_surface, params.margin, params.search_radius * params.search_radius, es);

            // FIXME multiblock should still work!
            galerkin_restrictions = create_galerkin_rap(contact_surface->block(0)->elements(), data->semistructured_levels);

            contact_jacobi = std::make_shared<ContactJacobi>();

            constraints_mask = sfem::create_buffer<mask_t>(mask_count(space->n_dofs()), es);
            agumentation     = sfem::create_buffer<real_t>(contact->mass_vector()->size(), es);

            memory.resize(n_levels());
            for (int i = 0; i < n_levels(); i++) {
                memory[i]           = std::make_shared<Memory>();
                const ptrdiff_t n   = data->functions[i]->space()->n_dofs();
                memory[i]->solution = create_buffer<real_t>(n, es);
                memory[i]->rhs      = create_buffer<real_t>(n, es);
                memory[i]->work     = create_buffer<real_t>(n, es);
                memory[i]->diag     = create_buffer<real_t>(n, es);
            }
        }

        void resample_contact_conditions(const smesh::SharedBuffer<real_t>& displacement) {
            contact->recompute(displacement);
            coupling_matrices.clear();
            coupling_matrices.push_back(h_crs_spmv(contact->graph()->n_nodes(),
                                                   contact->graph()->n_nodes(),
                                                   contact->graph()->rowptr(),
                                                   contact->graph()->colidx(),
                                                   contact->values(),
                                                   real_t(0)));

            const int spatial_dim = contact_surface->spatial_dimension();
            const int nlevels     = data->semistructured_levels.size();

            if (normals.empty()) {
                normals.resize(nlevels);
                normals[0] = contact->normals();

                for (int i = 1; i < nlevels; i++) {
                    normals[i] = create_host_buffer<real_t>(spatial_dim, galerkin_restrictions[i - 1].R->rows());
                }

                mass_vectors.resize(nlevels);
                mass_vectors[0] = contact->mass_vector();
                for (int i = 1; i < nlevels; i++) {
                    mass_vectors[i] = create_host_buffer<real_t>(galerkin_restrictions[i - 1].R->rows());
                }
            } else {
                normals[0]      = contact->normals();
                mass_vectors[0] = contact->mass_vector();
            }

            for (int i = 0; i < nlevels - 1; i++) {
                auto A_fine   = coupling_matrices[i];
                auto A_coarse = galerkin_restrictions[i].apply(A_fine);
                coupling_matrices.push_back(A_coarse);

                galerkin_restrictions[i].R->multi_apply(spatial_dim, normals[i]->data(), normals[i + 1]->data());
                galerkin_restrictions[i].R->apply(mass_vectors[i]->data(), mass_vectors[i + 1]->data());
            }

            contact_jacobi->cd = {.f                   = f,
                                  .surface             = contact_surface,
                                  .coupling_matrix     = coupling_matrices[0],
                                  .values              = contact->values(),
                                  .mass_vector         = contact->mass_vector(),
                                  .normals             = contact->normals(),
                                  .distances           = contact->distances(),
                                  .frozen_displacement = contact->frozen_displacement(),
                                  .constraints_mask    = constraints_mask,
                                  .agumentation        = agumentation};
        }

        void nonlinear_smooth(const SharedBuffer<real_t>& x) { contact_jacobi->smooth(x); }

        // TODO: complete this (similar to ShiftedPenaltyMultigrid but we have the Galerkin.R for the restriction and Galerkin.P
        // for the prolongation)
        void nonlinear_cycle() {
            nonlinear_smooth(memory[0]->solution);

            // Restrict residual and contact hessian diagonal component (see ShiftedPenaltyMultigrid)
            linear_cycle(1);

            nonlinear_smooth(memory[0]->solution);

            // update agumentation
        }

        void linear_cycle(int level) {
            if (level == n_levels() - 1) {
                // Solve coarse grid system
                return;
            }
        }

        ~Impl() = default;
    };

    MaMAL::MaMAL(const std::shared_ptr<Function>& f) : impl_(std::make_unique<Impl>(f)) {}

    MaMAL::~MaMAL() = default;

    std::shared_ptr<MaMAL> MaMAL::create(const std::shared_ptr<Function>& f) {
        auto ret = std::make_shared<MaMAL>(f);
        ret->impl_->init();
        return ret;
    }

#ifdef SFEM_ENABLE_YAML
    std::shared_ptr<MaMAL> MaMAL::create(const std::shared_ptr<Function>& f, const ryml::ConstNodeRef& node) {
        auto ret = std::make_shared<MaMAL>(f);
        ret->impl_->init(node);
        return ret;
    }
#endif
}  // namespace sfem
