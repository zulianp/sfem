#include "sfem_MaMAL.hpp"

#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"

#include "sfem_CRS.hpp"
#include "sfem_SelfContact.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_env.hpp"

#include "sfem_API.hpp"
#include "smesh_ssquad4_prolongation.hpp"

#include <cmath>

namespace sfem {
    class Memory {
    public:
        SharedBuffer<real_t> rhs;
        SharedBuffer<real_t> solution;
        SharedBuffer<real_t> work;
        SharedBuffer<real_t> correction;
        SharedBuffer<real_t> diag;
        inline ptrdiff_t     size() const { return solution->size(); }
        ~Memory() {}
    };

    struct MaMALParams {
        int    max_iterations{100};
        real_t tolerance{1e-6};
        real_t margin{1e-8};
        real_t search_radius{1e-2};
        real_t correction_damping{1};
        real_t min_correction_damping{1e-2};
        real_t augmentation_relaxation{1};
        int    line_search_steps{0};
        int    contact_update_frequency{0};
        int    contact_jacobi_loops{20};
        bool   line_search_recompute_contact{false};
        bool   enable_augmentation{false};

        void from_env() {
            max_iterations           = smesh::Env::read("SFEM_MAMAL_MAX_ITERATIONS", max_iterations);
            tolerance                = smesh::Env::read("SFEM_MAMAL_TOLERANCE", tolerance);
            margin                   = smesh::Env::read("SFEM_MAMAL_MARGIN", margin);
            search_radius            = smesh::Env::read("SFEM_MAMAL_SEARCH_RADIUS", search_radius);
            correction_damping       = smesh::Env::read("SFEM_MAMAL_CORRECTION_DAMPING", correction_damping);
            min_correction_damping   = smesh::Env::read("SFEM_MAMAL_MIN_CORRECTION_DAMPING", min_correction_damping);
            augmentation_relaxation  = smesh::Env::read("SFEM_MAMAL_AUGMENTATION_RELAXATION", augmentation_relaxation);
            line_search_steps        = smesh::Env::read("SFEM_MAMAL_LINE_SEARCH_STEPS", line_search_steps);
            contact_update_frequency = smesh::Env::read("SFEM_MAMAL_CONTACT_UPDATE_FREQUENCY", contact_update_frequency);
            contact_jacobi_loops     = smesh::Env::read("SFEM_MAMAL_CONTACT_JACOBI_LOOPS", contact_jacobi_loops);
            line_search_recompute_contact =
                    smesh::Env::read("SFEM_MAMAL_LINE_SEARCH_RECOMPUTE_CONTACT", line_search_recompute_contact);
            enable_augmentation = smesh::Env::read("SFEM_MAMAL_ENABLE_AUGMENTATION", enable_augmentation);
        }

#ifdef SFEM_ENABLE_YAML
        void from_yaml(const ryml::ConstNodeRef& node) {
            // TODO: implement
            max_iterations = node["max_iterations"].val<int>();
            tolerance      = node["tolerance"].val<real_t>();
            margin         = node["margin"].val<real_t>();
            search_radius  = node["search_radius"].val<real_t>();
            from_env();
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

    void assemble_contact_hessian_block_diag(ContactData&                                     cd,
                                             const real_t                                     penalty,
                                             const real_t* const                              macaulay,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("assemble_contact_hessian_block_diag");
        const int dim = cd.surface->spatial_dimension();
        assert(dim == 3);

        auto            coupling_matrix = cd.coupling_matrix;
        auto            rowptr          = coupling_matrix->row_ptr->data();
        auto            colidx          = coupling_matrix->col_idx->data();
        auto            vals            = coupling_matrix->values->data();
        const ptrdiff_t n               = coupling_matrix->rows();

        auto normals = cd.normals->data();
        auto mass    = cd.mass_vector->data();
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            if (macaulay[i] == 0) continue;

            const count_t lenrow = rowptr[i + 1] - rowptr[i];
            if (lenrow == 0) continue;

            const idx_t* const  row     = &colidx[rowptr[i]];
            const real_t* const weights = &vals[rowptr[i]];

            const real_t s  = mass[i] * penalty;
            const real_t n0 = normals[0][i];
            const real_t n1 = normals[1][i];
            const real_t n2 = normals[2][i];

            const real_t block[6] = {s * n0 * n0, s * n0 * n1, s * n0 * n2, s * n1 * n1, s * n1 * n2, s * n2 * n2};

            for (int d = 0; d < 6; ++d) {
#pragma omp atomic update
                diag_values[d][i] += block[d];
            }

            for (count_t j = 0; j < lenrow; j++) {
                const idx_t  r  = row[j];
                const real_t w2 = weights[j] * weights[j];
                for (int d = 0; d < 6; ++d) {
#pragma omp atomic update
                    diag_values[d][r] += w2 * block[d];
                }
            }
        }
    }

    void apply_contact_hessian(const std::shared_ptr<smesh::Mesh>& surface,
                               const std::shared_ptr<CRS_t>&       coupling_matrix,
                               const SharedBuffer<real_t*>&        normals,
                               const SharedBuffer<real_t>&         mass_vector,
                               const SharedBuffer<real_t>&         active,
                               const SharedBuffer<mask_t>&         constraints_mask,
                               const real_t                        penalty,
                               const real_t* const                 x,
                               real_t* const                       y) {
        SFEM_TRACE_SCOPE("apply_contact_hessian");

        const int dim = surface->spatial_dimension();
        assert(dim == 3);

        const count_t* const rowptr = coupling_matrix->row_ptr->data();
        const idx_t* const   colidx = coupling_matrix->col_idx->data();
        const real_t* const  vals   = coupling_matrix->values->data();
        const idx_t* const   nm     = surface->node_mapping()->data();
        const real_t* const  mass   = mass_vector->data();
        const real_t* const  alpha  = active->data();
        const real_t* const  nx     = normals->data()[0];
        const real_t* const  ny     = normals->data()[1];
        const real_t* const  nz     = normals->data()[2];
        const mask_t* const  mask   = constraints_mask ? constraints_mask->data() : nullptr;
        const ptrdiff_t      n      = coupling_matrix->rows();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t a = alpha[i];
            if (a == 0) continue;

            const count_t row_begin = rowptr[i];
            const count_t row_end   = rowptr[i + 1];
            if (row_begin == row_end) continue;

            const ptrdiff_t dof1 = nm[i] * dim;

            const real_t x10 = (mask && mask_get(dof1, mask)) ? 0 : x[dof1];
            const real_t x11 = (mask && mask_get(dof1 + 1, mask)) ? 0 : x[dof1 + 1];
            const real_t x12 = (mask && mask_get(dof1 + 2, mask)) ? 0 : x[dof1 + 2];

            real_t x20 = 0;
            real_t x21 = 0;
            real_t x22 = 0;

            for (count_t k = row_begin; k < row_end; ++k) {
                const real_t    w    = vals[k];
                const ptrdiff_t dof2 = nm[colidx[k]] * dim;

                x20 += w * ((mask && mask_get(dof2, mask)) ? 0 : x[dof2]);
                x21 += w * ((mask && mask_get(dof2 + 1, mask)) ? 0 : x[dof2 + 1]);
                x22 += w * ((mask && mask_get(dof2 + 2, mask)) ? 0 : x[dof2 + 2]);
            }

            const real_t n0 = nx[i];
            const real_t n1 = ny[i];
            const real_t n2 = nz[i];
            const real_t s  = a * penalty * mass[i] * (n0 * (x10 - x20) + n1 * (x11 - x21) + n2 * (x12 - x22));
            const real_t f0 = s * n0;
            const real_t f1 = s * n1;
            const real_t f2 = s * n2;

            if (!mask || !mask_get(dof1, mask)) {
#pragma omp atomic update
                y[dof1] += f0;
            }

            if (!mask || !mask_get(dof1 + 1, mask)) {
#pragma omp atomic update
                y[dof1 + 1] += f1;
            }

            if (!mask || !mask_get(dof1 + 2, mask)) {
#pragma omp atomic update
                y[dof1 + 2] += f2;
            }

            for (count_t k = row_begin; k < row_end; ++k) {
                const real_t    w    = vals[k];
                const ptrdiff_t dof2 = nm[colidx[k]] * dim;

                if (!mask || !mask_get(dof2, mask)) {
#pragma omp atomic update
                    y[dof2] -= w * f0;
                }

                if (!mask || !mask_get(dof2 + 1, mask)) {
#pragma omp atomic update
                    y[dof2 + 1] -= w * f1;
                }

                if (!mask || !mask_get(dof2 + 2, mask)) {
#pragma omp atomic update
                    y[dof2 + 2] -= w * f2;
                }
            }
        }
    }

    void apply_scaled_block_diag(const std::shared_ptr<SparseBlockVector<real_t>>& block_diag,
                                 const SharedBuffer<real_t>&                       scaling,
                                 const real_t                                      sign,
                                 const real_t* const                               x,
                                 real_t* const                                     y) {
        SFEM_TRACE_SCOPE("apply_scaled_block_diag");

        const ptrdiff_t      n_blocks = block_diag->n_blocks();
        const idx_t* const   idx      = block_diag->idx()->data();
        const real_t* const  blocks   = block_diag->data()->data();
        const real_t* const  scale    = scaling->data();
        static constexpr int dim      = 3;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_blocks; ++i) {
            const real_t* const b  = &blocks[i * 6];
            const real_t        s  = sign * scale[i];
            const ptrdiff_t     d  = idx[i] * dim;
            const real_t* const xi = &x[d];
            real_t* const       yi = &y[d];

            const real_t y0 = s * (b[0] * xi[0] + b[1] * xi[1] + b[2] * xi[2]);
            const real_t y1 = s * (b[1] * xi[0] + b[3] * xi[1] + b[4] * xi[2]);
            const real_t y2 = s * (b[2] * xi[0] + b[4] * xi[1] + b[5] * xi[2]);

            yi[0] += y0;
            yi[1] += y1;
            yi[2] += y2;
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

    std::vector<GalerkinRAP> create_galerkin_rap(const std::vector<std::shared_ptr<smesh::Mesh>>& surfaces,
                                                 const std::vector<int>&                          levels) {
        int nlevels = levels.size();

        std::vector<GalerkinRAP> ret;
        for (int i = 0; i < nlevels - 1; i++) {
            auto            elems        = surfaces[i]->block(0)->elements();
            const ptrdiff_t nelements    = elems->extent(1);
            const ptrdiff_t fine_n_nodes = surfaces[i]->n_nodes();

            auto rowptr = create_host_buffer<count_t>(fine_n_nodes + 1);
            smesh::ssquad4_prolongation_crs_nnz(levels[i], nelements, elems->data(), fine_n_nodes, rowptr->data());

            auto colidx = create_host_buffer<idx_t>(rowptr->data()[fine_n_nodes]);
            auto values = create_host_buffer<real_t>(rowptr->data()[fine_n_nodes]);

            smesh::ssquad4_prolongation_crs_fill(
                    levels[i], nelements, elems->data(), fine_n_nodes, rowptr->data(), colidx->data(), values->data());

            const ptrdiff_t    coarse_n_nodes = surfaces[i + 1]->n_nodes();
            std::vector<idx_t> fine_to_coarse(fine_n_nodes, SFEM_IDX_INVALID);
            const idx_t* const fine_to_volume   = surfaces[i]->node_mapping()->data();
            const idx_t* const coarse_to_volume = surfaces[i + 1]->node_mapping()->data();

            idx_t max_coarse_volume_node = 0;
            for (ptrdiff_t c = 0; c < coarse_n_nodes; ++c) {
                max_coarse_volume_node = std::max(max_coarse_volume_node, coarse_to_volume[c]);
            }

            std::vector<idx_t> volume_to_coarse(max_coarse_volume_node + 1, SFEM_IDX_INVALID);
            for (ptrdiff_t c = 0; c < coarse_n_nodes; ++c) {
                volume_to_coarse[coarse_to_volume[c]] = c;
            }

            for (ptrdiff_t f = 0; f < fine_n_nodes; ++f) {
                const idx_t volume_node = fine_to_volume[f];
                if (volume_node <= max_coarse_volume_node) {
                    fine_to_coarse[f] = volume_to_coarse[volume_node];
                }
            }

            for (ptrdiff_t k = 0; k < colidx->size(); ++k) {
                assert(colidx->data()[k] < fine_n_nodes);
                assert(fine_to_coarse[colidx->data()[k]] != SFEM_IDX_INVALID);
                colidx->data()[k] = fine_to_coarse[colidx->data()[k]];
            }

            auto P = h_crs_spmv(fine_n_nodes, coarse_n_nodes, rowptr, colidx, values, real_t(0));
            auto R = P->transpose();
            ret.push_back({.R = R, .P = P});
        }

        return ret;
    }

    class MaMAL::Impl {
    public:
        std::shared_ptr<Function>                                    f;
        MaMALParams                                                  params;
        std::shared_ptr<Contact>                                     contact;
        std::shared_ptr<MultigridData>                               data;
        std::vector<std::shared_ptr<Operator<real_t>>>               operators;
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers;

        std::vector<std::shared_ptr<CRS_t>> coupling_matrices;
        std::vector<SharedBuffer<real_t*>>  normals;
        std::vector<SharedBuffer<real_t*>>  weighted_normals;
        std::vector<SharedBuffer<real_t>>   mass_vectors;
        std::vector<SharedBuffer<real_t>>   contact_active;
        std::vector<GalerkinRAP>            galerkin_restrictions;

        std::vector<std::shared_ptr<Memory>>                    memory;
        std::vector<std::shared_ptr<smesh::Mesh>>               contact_surfaces;
        std::vector<SharedBuffer<real_t*>>                      contact_block_diag_soa;
        std::vector<SharedBuffer<real_t>>                       contact_block_diag_aos;
        std::vector<SharedBuffer<idx_t>>                        contact_block_idx;
        std::vector<std::shared_ptr<SparseBlockVector<real_t>>> contact_block_diag;
        std::vector<SharedBuffer<mask_t>>                       level_constraints_mask;
        SharedBuffer<real_t>                                    contact_grad;
        SharedBuffer<real_t>                                    macaulay;

        std::shared_ptr<smesh::Mesh>   contact_surface;
        std::shared_ptr<smesh::Mesh>   contact_eval_surface;
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
            params.from_env();

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

            std::vector<std::shared_ptr<smesh::Mesh>> galerkin_contact_surfaces(n_levels());
            contact_surface              = smesh::skin(mesh);
            galerkin_contact_surfaces[0] = contact_surface;
            for (int i = 1; i < n_levels(); ++i) {
                galerkin_contact_surfaces[i] = smesh::skin(data->functions[i]->space()->mesh_ptr());
            }

            contact_eval_surface = contact_surface;
            if (smesh::is_semistructured_type(contact_surface->element_type(0))) {
                contact_eval_surface = smesh::ssquad_to_quad4(contact_surface);
                contact_eval_surface->block(0)->set_element_type(smesh::QUADSHELL4);
            }

            contact = create_contact(space, contact_eval_surface, params.margin, params.search_radius * params.search_radius, es);

            // FIXME multiblock should still work!
            galerkin_restrictions = create_galerkin_rap(galerkin_contact_surfaces, data->semistructured_levels);

            contact_jacobi          = std::make_shared<ContactJacobi>();
            contact_jacobi->n_loops = params.contact_jacobi_loops;

            constraints_mask = sfem::create_buffer<mask_t>(mask_count(space->n_dofs()), es);
            agumentation     = sfem::create_buffer<real_t>(contact->mass_vector()->size(), es);
            contact_grad     = sfem::create_buffer<real_t>(space->n_dofs(), es);
            macaulay         = sfem::create_buffer<real_t>(contact->mass_vector()->size(), es);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < constraints_mask->size(); ++i) {
                constraints_mask->data()[i] = 0;
            }
            f->constraints_mask(constraints_mask->data());

            smoothers = sfem::create_gmg_default_smoothers_and_solver(data, operators, 3, false);

            contact_surfaces.resize(n_levels());
            contact_surfaces[0] = contact_eval_surface;
            for (int i = 1; i < n_levels(); ++i) {
                auto surface = galerkin_contact_surfaces[i];
                if (smesh::is_semistructured_type(surface->element_type(0))) {
                    surface = smesh::ssquad_to_quad4(surface);
                    surface->block(0)->set_element_type(smesh::QUADSHELL4);
                }

                contact_surfaces[i] = surface;
            }

            memory.resize(n_levels());
            contact_block_diag_soa.resize(n_levels());
            contact_block_diag_aos.resize(n_levels());
            contact_block_idx.resize(n_levels());
            contact_block_diag.resize(n_levels());
            level_constraints_mask.resize(n_levels());
            contact_active.resize(n_levels());
            for (int i = 0; i < n_levels(); i++) {
                memory[i]             = std::make_shared<Memory>();
                const ptrdiff_t n     = data->functions[i]->space()->n_dofs();
                memory[i]->solution   = create_buffer<real_t>(n, es);
                memory[i]->rhs        = create_buffer<real_t>(n, es);
                memory[i]->work       = create_buffer<real_t>(n, es);
                memory[i]->correction = create_buffer<real_t>(n, es);

                const ptrdiff_t n_contact = contact_surfaces[i]->node_mapping()->size();
                memory[i]->diag           = create_buffer<real_t>(n_contact, es);
                contact_block_diag_soa[i] = create_buffer<real_t>(6, n_contact, es);
                contact_block_diag_aos[i] = create_buffer<real_t>(n_contact * 6, es);
                contact_block_idx[i]      = contact_surfaces[i]->node_mapping();
                contact_block_diag[i]     = create_sparse_block_vector(contact_block_idx[i], contact_block_diag_aos[i]);

                level_constraints_mask[i] = sfem::create_buffer<mask_t>(mask_count(n), es);
                contact_active[i]         = sfem::create_buffer<real_t>(n_contact, es);
            }

            configure_coarse_solver_preconditioner();
        }

        void configure_coarse_solver_preconditioner() {
            const int level = n_levels() - 1;
            auto      cf    = data->functions[level];
            auto      fs    = cf->space();
            auto      es    = f->execution_space();

            if (fs->block_size() != 3) return;

            auto diag = sfem::create_buffer<real_t>(fs->n_dofs() / fs->block_size() * 6, es);
            auto mask = sfem::create_buffer<mask_t>(mask_count(fs->n_dofs()), es);

            cf->constraints_mask(mask->data());
            cf->hessian_block_diag_sym(nullptr, diag->data());

            auto jacobi                  = sfem::create_shiftable_block_sym_jacobi<real_t>(fs->block_size(), diag, mask, es);
            jacobi->relaxation_parameter = real_t(1) / fs->block_size();
            smoothers[level]->set_preconditioner_op(jacobi);
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

            const int spatial_dim = contact_eval_surface->spatial_dimension();
            const int nlevels     = data->semistructured_levels.size();

            if (normals.empty()) {
                normals.resize(nlevels);
                normals[0] = contact->normals();

                for (int i = 1; i < nlevels; i++) {
                    normals[i] = create_host_buffer<real_t>(spatial_dim, galerkin_restrictions[i - 1].R->rows());
                }

                weighted_normals.resize(nlevels);
                weighted_normals[0] = create_host_buffer<real_t>(spatial_dim, contact_eval_surface->node_mapping()->size());
                for (int i = 1; i < nlevels; i++) {
                    weighted_normals[i] = create_host_buffer<real_t>(spatial_dim, galerkin_restrictions[i - 1].R->rows());
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

            {
                const ptrdiff_t n  = mass_vectors[0]->size();
                const real_t*   m  = mass_vectors[0]->data();
                real_t** const  wn = weighted_normals[0]->data();
                real_t** const  nr = normals[0]->data();

                for (int d = 0; d < spatial_dim; ++d) {
#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; ++i) {
                        wn[d][i] = m[i] * nr[d][i];
                    }
                }
            }

            for (int i = 0; i < nlevels - 1; i++) {
                auto A_fine   = coupling_matrices[i];
                auto A_coarse = galerkin_restrictions[i].apply(A_fine);
                coupling_matrices.push_back(A_coarse);

                galerkin_restrictions[i].R->apply(mass_vectors[i]->data(), mass_vectors[i + 1]->data());
                galerkin_restrictions[i].R->multi_apply(
                        spatial_dim, weighted_normals[i]->data(), weighted_normals[i + 1]->data());

                {
                    const ptrdiff_t n  = mass_vectors[i + 1]->size();
                    real_t** const  nr = normals[i + 1]->data();
                    real_t** const  wn = weighted_normals[i + 1]->data();

#pragma omp parallel for
                    for (ptrdiff_t k = 0; k < n; ++k) {
                        real_t norm = 0;
                        for (int d = 0; d < spatial_dim; ++d) {
                            norm += wn[d][k] * wn[d][k];
                        }

                        norm                  = std::sqrt(norm);
                        const real_t inv_norm = norm > 0 ? real_t(1) / norm : real_t(0);
                        for (int d = 0; d < spatial_dim; ++d) {
                            nr[d][k] = wn[d][k] * inv_norm;
                        }
                    }
                }
            }

            contact_jacobi->cd = {.f                   = f,
                                  .surface             = contact_eval_surface,
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

        void pack_contact_block_diag(const int level) {
            auto            src  = contact_block_diag_soa[level]->data();
            real_t* const   dst  = contact_block_diag_aos[level]->data();
            const ptrdiff_t n    = memory[level]->diag->size();
            const int       dim  = data->functions[level]->space()->block_size();
            const mask_t*   mask = level_constraints_mask[level]->data();
            const idx_t*    idx  = contact_block_idx[level]->data();

            assert(dim == 3);

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                const ptrdiff_t dof = idx[i] * dim;
                real_t* const   b   = &dst[i * 6];

                b[0] = src[0][i];
                b[1] = src[1][i];
                b[2] = src[2][i];
                b[3] = src[3][i];
                b[4] = src[4][i];
                b[5] = src[5][i];

                const bool c0 = mask_get(dof, mask);
                const bool c1 = mask_get(dof + 1, mask);
                const bool c2 = mask_get(dof + 2, mask);

                if (c0) {
                    b[0] = 0;
                    b[1] = 0;
                    b[2] = 0;
                }

                if (c1) {
                    b[1] = 0;
                    b[3] = 0;
                    b[4] = 0;
                }

                if (c2) {
                    b[2] = 0;
                    b[4] = 0;
                    b[5] = 0;
                }
            }
        }

        void restrict_contact_active_set() {
            const real_t* const m = macaulay->data();
            real_t* const       a = contact_active[0]->data();
            const ptrdiff_t     n = contact_active[0]->size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                a[i] = m[i] > 0 ? real_t(1) : real_t(0);
            }

            for (int l = 0; l < n_levels() - 1; ++l) {
                galerkin_restrictions[l].R->apply(contact_active[l]->data(), contact_active[l + 1]->data());

                real_t* const   coarse = contact_active[l + 1]->data();
                const ptrdiff_t nc     = contact_active[l + 1]->size();

#pragma omp parallel for
                for (ptrdiff_t i = 0; i < nc; ++i) {
                    coarse[i] = coarse[i] > 0 ? real_t(1) : real_t(0);
                }
            }
        }

        ContactData linearized_contact_data(const int level) {
            return {.f                   = data->functions[level],
                    .surface             = contact_surfaces[level],
                    .coupling_matrix     = coupling_matrices[level],
                    .values              = coupling_matrices[level]->values,
                    .mass_vector         = mass_vectors[level],
                    .normals             = normals[level],
                    .distances           = nullptr,
                    .frozen_displacement = nullptr,
                    .constraints_mask    = level_constraints_mask[level],
                    .agumentation        = nullptr};
        }

        void assemble_level_contact_hessian_block_diag() {
            auto blas = sfem::blas<real_t>(f->execution_space());

            for (int l = 0; l < n_levels(); ++l) {
                blas->values(memory[l]->diag->size(), 1, memory[l]->diag->data());
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < level_constraints_mask[l]->size(); ++i) {
                    level_constraints_mask[l]->data()[i] = 0;
                }
                data->functions[l]->constraints_mask(level_constraints_mask[l]->data());

                auto diag = contact_block_diag_soa[l]->data();
                for (int d = 0; d < 6; ++d) {
                    blas->values(contact_block_diag_soa[l]->extent(1), 0, diag[d]);
                }

                ContactData cd = linearized_contact_data(l);
                assemble_contact_hessian_block_diag(cd, contact_jacobi->penalty, contact_active[l]->data(), diag);
                pack_contact_block_diag(l);
            }
        }

        std::shared_ptr<Operator<real_t>> contact_hessian_op(const int level, const bool offdiag_only) {
            auto            surface          = contact_surfaces[level];
            auto            coupling_matrix  = coupling_matrices[level];
            auto            level_normals    = normals[level];
            auto            level_mass       = mass_vectors[level];
            auto            active           = contact_active[level];
            auto            constraints      = level_constraints_mask[level];
            auto            block_diag       = contact_block_diag[level];
            auto            block_diag_scale = memory[level]->diag;
            const real_t    penalty          = contact_jacobi->penalty;
            const ptrdiff_t n                = operators[level]->rows();
            auto            es               = f->execution_space();

            return sfem::make_op<real_t>(
                    n,
                    n,
                    [=](const real_t* const x, real_t* const y) {
                        apply_contact_hessian(
                                surface, coupling_matrix, level_normals, level_mass, active, constraints, penalty, x, y);

                        if (offdiag_only) {
                            apply_scaled_block_diag(block_diag, block_diag_scale, real_t(-1), x, y);
                        }
                    },
                    es);
        }

        std::shared_ptr<Operator<real_t>> shifted_op(const int level) {
            return operators[level] + contact_hessian_op(level, false);
        }

        real_t eval_fine_residual(const SharedBuffer<real_t>& x, const SharedBuffer<real_t>& residual) {
            SFEM_TRACE_SCOPE("MaMAL::eval_fine_residual");
            auto            blas  = sfem::blas<real_t>(f->execution_space());
            auto            mem   = memory[0];
            const ptrdiff_t ndofs = x->size();

            blas->values(ndofs, 0, residual->data());
            f->gradient(x->data(), residual->data());
            blas->scal(ndofs, -1, residual->data());

            blas->values(ndofs, 0, contact_grad->data());
            compute_macaulay_term(contact_jacobi->cd, contact_jacobi->penalty, x->data(), macaulay->data());
            assemble_contact_gradient(contact_jacobi->cd, contact_jacobi->penalty, macaulay->data(), contact_grad->data());
            blas->axpy(ndofs, -1, contact_grad->data(), residual->data());

            return blas->norm2(residual->size(), residual->data());
        }

        real_t eval_fine_residual_and_jacobian() {
            SFEM_TRACE_SCOPE("MaMAL::eval_fine_residual_and_jacobian");

            const real_t grad_norm = eval_fine_residual(memory[0]->solution, memory[0]->work);
            restrict_contact_active_set();

            assemble_level_contact_hessian_block_diag();
            return grad_norm;
        }

        void update_augmentation() {
            compute_macaulay_term(contact_jacobi->cd, contact_jacobi->penalty, memory[0]->solution->data(), macaulay->data());

            real_t* const       aug = agumentation->data();
            const real_t* const m   = macaulay->data();
            const real_t        p   = contact_jacobi->penalty;
            const real_t        r   = params.augmentation_relaxation;
            const real_t        c   = real_t(1) - r;
            const ptrdiff_t     n   = agumentation->size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                aug[i] = c * aug[i] + r * p * m[i];
            }
        }

        void nonlinear_cycle() { nonlinear_iteration(); }

        real_t nonlinear_iteration() {
            nonlinear_smooth(memory[0]->solution);

            const real_t grad_norm = eval_fine_residual_and_jacobian();

            auto                    blas = sfem::blas<real_t>(f->execution_space());
            std::shared_ptr<Memory> mem  = memory[0];

            if (n_levels() > 1) {
                auto mem_coarse = memory[1];

                blas->values(mem_coarse->rhs->size(), 0, mem_coarse->rhs->data());
                data->restrictions[0]->apply(mem->work->data(), mem_coarse->rhs->data());
                blas->values(mem_coarse->solution->size(), 0, mem_coarse->solution->data());

                linear_cycle(1);

                blas->values(mem->correction->size(), 0, mem->correction->data());
                data->prolongations[1]->apply(mem_coarse->solution->data(), mem->correction->data());

                blas->copy(mem->solution->size(), mem->solution->data(), mem->rhs->data());

                real_t alpha = params.correction_damping;
                for (int ls = 0; ls < params.line_search_steps; ++ls) {
                    const real_t* const x0 = mem->rhs->data();
                    const real_t* const dx = mem->correction->data();
                    real_t* const       x  = mem->solution->data();
                    const ptrdiff_t     n  = mem->solution->size();

#pragma omp parallel for
                    for (ptrdiff_t i = 0; i < n; ++i) {
                        x[i] = x0[i] + alpha * dx[i];
                    }

                    if (params.line_search_recompute_contact) {
                        resample_contact_conditions(mem->solution);
                    }

                    const real_t trial_norm = eval_fine_residual(mem->solution, mem->work);
                    if (trial_norm <= grad_norm || alpha <= params.min_correction_damping) {
                        break;
                    }

                    alpha *= real_t(0.5);
                }
            }

            nonlinear_smooth(memory[0]->solution);

            if (params.enable_augmentation) update_augmentation();

            return grad_norm;
        }

        void linear_cycle(int level) {
            SFEM_TRACE_SCOPE("MaMAL::linear_cycle");

            auto blas     = sfem::blas<real_t>(f->execution_space());
            auto mem      = memory[level];
            auto smoother = smoothers[level];
            auto op       = operators[level];

            if (level == n_levels() - 1) {
                smoother->set_op_and_diag_shift(
                        op + contact_hessian_op(level, true), contact_block_diag[level], memory[level]->diag);
                blas->values(mem->solution->size(), 0, mem->solution->data());
                smoother->apply(mem->rhs->data(), mem->solution->data());
                return;
            }

            auto sop        = shifted_op(level);
            auto mem_coarse = memory[level + 1];

            smoother->set_op_and_diag_shift(op + contact_hessian_op(level, true), contact_block_diag[level], memory[level]->diag);
            smoother->apply(mem->rhs->data(), mem->solution->data());

            blas->values(mem->work->size(), 0, mem->work->data());
            sop->apply(mem->solution->data(), mem->work->data());
            blas->axpby(mem->work->size(), 1, mem->rhs->data(), -1, mem->work->data());

            blas->values(mem_coarse->rhs->size(), 0, mem_coarse->rhs->data());
            data->restrictions[level]->apply(mem->work->data(), mem_coarse->rhs->data());
            blas->values(mem_coarse->solution->size(), 0, mem_coarse->solution->data());

            linear_cycle(level + 1);

            blas->values(mem->work->size(), 0, mem->work->data());
            data->prolongations[level + 1]->apply(mem_coarse->solution->data(), mem->work->data());
            blas->axpy(mem->solution->size(), 1, mem->work->data(), mem->solution->data());

            smoother->apply(mem->rhs->data(), mem->solution->data());
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

    int MaMAL::solve(const smesh::SharedBuffer<real_t>& x) {
        auto blas = sfem::blas<real_t>(impl_->f->execution_space());
        auto mem  = impl_->memory[0];

        blas->copy(x->size(), x->data(), mem->solution->data());
        impl_->f->apply_constraints(mem->solution->data());

        int iter = 0;
        for (; iter < impl_->params.max_iterations; ++iter) {
            if (iter == 0 || (impl_->params.contact_update_frequency > 0 && iter % impl_->params.contact_update_frequency == 0)) {
                impl_->resample_contact_conditions(mem->solution);
            }

            const real_t grad_norm = impl_->nonlinear_iteration();
            printf("MaMAL::solve %d gradient_norm %e\n", iter, (double)grad_norm);
            fflush(stdout);

            if (grad_norm < impl_->params.tolerance) {
                ++iter;
                break;
            }
        }

        blas->copy(mem->solution->size(), mem->solution->data(), x->data());
        return SFEM_SUCCESS;
    }
}  // namespace sfem
