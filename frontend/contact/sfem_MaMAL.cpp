#include "sfem_MaMAL.hpp"

#include "sfem_ContactSkin.hpp"
#include "sfem_ContactSolveKernels.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"

#include "sfem_CRS.hpp"
#include "sfem_CRS_X_BSR.hpp"
#include "sfem_SelfContact.hpp"
#include "smesh_crs_graph.hpp"
#include "smesh_env.hpp"

#include "sfem_API.hpp"
#include "smesh_ssquad4_prolongation.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

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
        int    max_iterations{20};
        real_t tolerance{1e-6};
        real_t margin{1e-8};
        real_t search_radius{5e-2};
        real_t correction_damping{1};
        real_t min_correction_damping{1e-2};
        real_t augmentation_relaxation{1};
        int    contact_update_frequency{0};
        int    contact_jacobi_loops{10};
        bool   line_search_recompute_contact{false};
        bool   enable_augmentation{false};
        bool   enable_self_contact{false};
        bool   contact_hessian_galerkin_assembly{true};

        void from_env() {
            max_iterations           = smesh::Env::read("SFEM_MAMAL_MAX_ITERATIONS", max_iterations);
            tolerance                = smesh::Env::read("SFEM_MAMAL_TOLERANCE", tolerance);
            margin                   = smesh::Env::read("SFEM_MAMAL_MARGIN", margin);
            search_radius            = smesh::Env::read("SFEM_MAMAL_SEARCH_RADIUS", search_radius);
            correction_damping       = smesh::Env::read("SFEM_MAMAL_CORRECTION_DAMPING", correction_damping);
            min_correction_damping   = smesh::Env::read("SFEM_MAMAL_MIN_CORRECTION_DAMPING", min_correction_damping);
            augmentation_relaxation  = smesh::Env::read("SFEM_MAMAL_AUGMENTATION_RELAXATION", augmentation_relaxation);
            contact_update_frequency = smesh::Env::read("SFEM_MAMAL_CONTACT_UPDATE_FREQUENCY", contact_update_frequency);
            contact_jacobi_loops     = smesh::Env::read("SFEM_MAMAL_CONTACT_JACOBI_LOOPS", contact_jacobi_loops);
            line_search_recompute_contact =
                    smesh::Env::read("SFEM_MAMAL_LINE_SEARCH_RECOMPUTE_CONTACT", line_search_recompute_contact);
            enable_augmentation = smesh::Env::read("SFEM_MAMAL_ENABLE_AUGMENTATION", enable_augmentation);
            enable_self_contact = smesh::Env::read("SFEM_MAMAL_ENABLE_SELF_CONTACT", enable_self_contact);
            contact_hessian_galerkin_assembly =
                    smesh::Env::read("SFEM_MAMAL_CONTACT_HESSIAN_GALERKIN", contact_hessian_galerkin_assembly);
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
    using CRS_t      = sfem::ContactData::CRS_t;
    using BSR_t      = sfem::BSR<count_t, idx_t, real_t, real_t>;

    struct GalerkinRAP {
        std::shared_ptr<CRS_t> R, P;

        std::shared_ptr<CRS_t> apply(const std::shared_ptr<CRS_t>& A) const { return sfem::rap(R, A, P); }
    };

    std::shared_ptr<CRSGraph_t> create_contact_hessian_graph(const std::shared_ptr<CRS_t>& coupling_matrix) {
        SFEM_TRACE_SCOPE("create_contact_hessian_graph");

        const ptrdiff_t n      = coupling_matrix->rows();
        const count_t*  rowptr = coupling_matrix->row_ptr->data();
        const idx_t*    colidx = coupling_matrix->col_idx->data();

        std::vector<std::vector<idx_t>> graph_rows(n);
        std::vector<idx_t>              targets;

        for (ptrdiff_t i = 0; i < n; ++i) {
            targets.clear();
            targets.reserve(rowptr[i + 1] - rowptr[i] + 1);
            targets.push_back(i);
            for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
                targets.push_back(colidx[k]);
            }

            std::sort(targets.begin(), targets.end());
            targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

            for (const idx_t row : targets) {
                auto& dst = graph_rows[row];
                dst.insert(dst.end(), targets.begin(), targets.end());
            }
        }

        auto out_rowptr       = create_host_buffer<count_t>(n + 1);
        out_rowptr->data()[0] = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            auto& row = graph_rows[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            out_rowptr->data()[i + 1] = out_rowptr->data()[i] + row.size();
        }

        auto out_colidx = create_host_buffer<idx_t>(out_rowptr->data()[n]);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const auto& row = graph_rows[i];
            idx_t*      dst = &out_colidx->data()[out_rowptr->data()[i]];
            for (ptrdiff_t k = 0, len = row.size(); k < len; ++k) {
                dst[k] = row[k];
            }
        }

        return std::make_shared<CRSGraph_t>(out_rowptr, out_colidx);
    }

    void apply_contact_hessian_bsr(const std::shared_ptr<smesh::Mesh>& surface,
                                   const std::shared_ptr<BSR_t>&       hessian,
                                   const SharedBuffer<real_t>&         local_x,
                                   const SharedBuffer<real_t>&         local_y,
                                   const real_t* const                 x,
                                   real_t* const                       y) {
        SFEM_TRACE_SCOPE("apply_contact_hessian_bsr");

        const int          dim = surface->spatial_dimension();
        const ptrdiff_t    n   = surface->node_mapping()->size();
        const idx_t* const nm  = surface->node_mapping()->data();
        real_t* const      lx  = local_x->data();
        real_t* const      ly  = local_y->data();

        assert(dim == 3);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t gdof = nm[i] * dim;
            const ptrdiff_t ldof = i * dim;
            lx[ldof + 0]         = x[gdof + 0];
            lx[ldof + 1]         = x[gdof + 1];
            lx[ldof + 2]         = x[gdof + 2];
        }

        hessian->apply(lx, ly);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            const ptrdiff_t gdof = nm[i] * dim;
            const ptrdiff_t ldof = i * dim;

#pragma omp atomic update
            y[gdof + 0] += ly[ldof + 0];
#pragma omp atomic update
            y[gdof + 1] += ly[ldof + 1];
#pragma omp atomic update
            y[gdof + 2] += ly[ldof + 2];
        }
    }

    void extract_contact_hessian_bsr_diag(const std::shared_ptr<BSR_t>& hessian, real_t* const SFEM_RESTRICT block_diag) {
        SFEM_TRACE_SCOPE("extract_contact_hessian_bsr_diag");

        const count_t* const rowptr = hessian->row_ptr->data();
        const idx_t* const   colidx = hessian->col_idx->data();
        const real_t* const  values = hessian->values->data();
        const ptrdiff_t      n      = hessian->row_ptr->size() - 1;

        assert(hessian->row_block_size() == 3);
        assert(hessian->col_block_size() == 3);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            real_t* const dst = &block_diag[i * 6];
            dst[0]            = 0;
            dst[1]            = 0;
            dst[2]            = 0;
            dst[3]            = 0;
            dst[4]            = 0;
            dst[5]            = 0;

            for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
                if (colidx[k] == i) {
                    const real_t* const b = &values[k * 9];
                    dst[0]                = b[0];
                    dst[1]                = b[1];
                    dst[2]                = b[2];
                    dst[3]                = b[4];
                    dst[4]                = b[5];
                    dst[5]                = b[8];
                    break;
                }
            }
        }
    }

    void compute_macaulay_term_from_global_displacement(ContactData&        cd,
                                                        const real_t        penalty,
                                                        const real_t* const disp,
                                                        real_t* const       work,
                                                        real_t* const       macaulay) {
        SFEM_TRACE_SCOPE("compute_macaulay_term_from_global_displacement");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = coupling_matrix->values->data();
        auto      nm              = cd.surface->node_mapping()->data();
        ptrdiff_t n               = coupling_matrix->rows();
        assert(dim == 3);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            const ptrdiff_t global_dof = nm[i] * dim;
            const ptrdiff_t local_dof  = i * dim;
            work[local_dof + 0]        = disp[global_dof + 0];
            work[local_dof + 1]        = disp[global_dof + 1];
            work[local_dof + 2]        = disp[global_dof + 2];
        }

        const real_t* const local_disp[3] = {work + 0, work + 1, work + 2};
        compute_macaulay_term(dim,
                              n,
                              rowptr,
                              colidx,
                              vals,
                              cd.distances->data(),
                              cd.agumentation->data(),
                              cd.normals->data(),
                              cd.mass_vector->data(),
                              penalty,
                              dim,
                              local_disp,
                              macaulay);
    }

    void compute_penetration_from_global_displacement(ContactData&        cd,
                                                      const real_t* const disp,
                                                      real_t* const       work,
                                                      real_t* const       penetration) {
        SFEM_TRACE_SCOPE("compute_penetration_from_global_displacement");
        const int dim             = cd.surface->spatial_dimension();
        auto      coupling_matrix = cd.coupling_matrix;
        auto      rowptr          = coupling_matrix->row_ptr->data();
        auto      colidx          = coupling_matrix->col_idx->data();
        auto      vals            = coupling_matrix->values->data();
        auto      nm              = cd.surface->node_mapping()->data();
        ptrdiff_t n               = coupling_matrix->rows();
        assert(dim == 3);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; i++) {
            const ptrdiff_t global_dof = nm[i] * dim;
            const ptrdiff_t local_dof  = i * dim;
            work[local_dof + 0]        = disp[global_dof + 0];
            work[local_dof + 1]        = disp[global_dof + 1];
            work[local_dof + 2]        = disp[global_dof + 2];
        }

        const real_t* const local_disp[3] = {work + 0, work + 1, work + 2};
        compute_penetration(dim, n, rowptr, colidx, vals, cd.normals->data(), cd.distances->data(), dim, local_disp, penetration);
    }

    void assemble_contact_hessian_block_diag(ContactData&                                     cd,
                                             const real_t                                     penalty,
                                             const real_t* const                              macaulay,
                                             real_t* const SFEM_RESTRICT* const SFEM_RESTRICT diag_values) {
        SFEM_TRACE_SCOPE("assemble_contact_hessian_block_diag");
        const int dim = cd.surface->spatial_dimension();
        assert(dim == 3);

        auto coupling_matrix = cd.coupling_matrix;
        sfem::assemble_contact_hessian_block_diag(dim,
                                                  coupling_matrix->rows(),
                                                  coupling_matrix->row_ptr->data(),
                                                  coupling_matrix->col_idx->data(),
                                                  coupling_matrix->values->data(),
                                                  cd.normals->data(),
                                                  cd.mass_vector->data(),
                                                  penalty,
                                                  macaulay,
                                                  1,
                                                  diag_values);
    }

    void apply_contact_hessian(const std::shared_ptr<smesh::Mesh>& surface,
                               const std::shared_ptr<CRS_t>&       coupling_matrix,
                               const SharedBuffer<real_t*>&        normals,
                               const SharedBuffer<real_t>&         mass_vector,
                               const SharedBuffer<real_t>&         active,
                               const real_t                        penalty,
                               const real_t* const                 x,
                               real_t* const                       y) {
        SFEM_TRACE_SCOPE("apply_contact_hessian");

        const int dim = surface->spatial_dimension();
        assert(dim == 3);

        sfem::apply_contact_hessian(dim,
                                    coupling_matrix->rows(),
                                    coupling_matrix->row_ptr->data(),
                                    coupling_matrix->col_idx->data(),
                                    coupling_matrix->values->data(),
                                    surface->node_mapping()->data(),
                                    normals->data(),
                                    mass_vector->data(),
                                    active->data(),
                                    penalty,
                                    x,
                                    y);
    }

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

    std::shared_ptr<smesh::Mesh> derefine_contact_trace(const std::shared_ptr<smesh::Mesh>& fine_surface,
                                                        const int                           from_level,
                                                        const int                           to_level) {
        if (from_level == to_level) {
            return fine_surface;
        }

        assert(fine_surface->n_blocks() == 1);
        assert(from_level % to_level == 0);

        auto      coarse_view = ssquad4_derefine_element_connectivity(from_level, to_level, fine_surface->block(0)->elements());
        auto      coarse_view_data  = coarse_view->data();
        const int nxe               = coarse_view->extent(0);
        const ptrdiff_t nelements   = coarse_view->extent(1);
        idx_t           max_node_id = 0;

        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int v = 0; v < nxe; ++v) {
                const idx_t old = coarse_view_data[v][e];
                assert(old >= 0);
                max_node_id = std::max(max_node_id, old);
            }
        }

        std::vector<idx_t> old_to_new(max_node_id + 1, SFEM_IDX_INVALID);
        ptrdiff_t          coarse_n_nodes = 0;

        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int v = 0; v < nxe; ++v) {
                const idx_t old = coarse_view_data[v][e];
                if (old_to_new[old] == SFEM_IDX_INVALID) {
                    old_to_new[old] = coarse_n_nodes++;
                }
            }
        }

        auto elements = create_host_buffer<idx_t>(nxe, nelements);
        auto elems    = elements->data();
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int v = 0; v < nxe; ++v) {
                elems[v][e] = old_to_new[coarse_view_data[v][e]];
            }
        }

        auto            points       = create_host_buffer<geom_t>(fine_surface->spatial_dimension(), coarse_n_nodes);
        auto            node_mapping = create_host_buffer<idx_t>(coarse_n_nodes);
        auto            fine_points  = fine_surface->points()->data();
        auto            coarse_pts   = points->data();
        auto            fine_mapping = fine_surface->node_mapping()->data();
        auto            coarse_map   = node_mapping->data();
        const ptrdiff_t fine_n_nodes = fine_surface->n_nodes();

        std::vector<idx_t> volume_to_surface;
        if (max_node_id >= fine_n_nodes) {
            idx_t max_volume_node = max_node_id;
            for (ptrdiff_t i = 0; i < fine_n_nodes; ++i) {
                max_volume_node = std::max(max_volume_node, fine_mapping[i]);
            }

            volume_to_surface.resize(max_volume_node + 1, SFEM_IDX_INVALID);
            for (ptrdiff_t i = 0; i < fine_n_nodes; ++i) {
                volume_to_surface[fine_mapping[i]] = i;
            }
        }

        for (ptrdiff_t old = 0; old <= max_node_id; ++old) {
            const idx_t n = old_to_new[old];
            if (n == SFEM_IDX_INVALID) continue;

            const idx_t fine_node = old < fine_n_nodes ? old : volume_to_surface[old];
            assert(fine_node >= 0 && fine_node < fine_n_nodes);
            coarse_map[n] = fine_mapping[fine_node];
            for (int d = 0; d < fine_surface->spatial_dimension(); ++d) {
                coarse_pts[d][n] = fine_points[d][fine_node];
            }
        }

        auto block = std::make_shared<smesh::Mesh::Block>();
        block->set_name(fine_surface->block(0)->name());
        block->set_element_type(smesh::shell_type(smesh::proteus_hex_type(to_level)));
        block->set_elements(elements);

        std::vector<std::shared_ptr<smesh::Mesh::Block>> blocks;
        blocks.push_back(block);

        auto ret = std::make_shared<smesh::Mesh>(fine_surface->comm(), blocks, points);
        ret->set_node_mapping(node_mapping);
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
        std::vector<std::shared_ptr<BSR_t>> galerkin_contact_hessian_bsr;

        std::vector<std::shared_ptr<Memory>>                    memory;
        std::vector<std::shared_ptr<smesh::Mesh>>               contact_surfaces;
        std::vector<SharedBuffer<real_t*>>                      contact_block_diag_soa;
        std::vector<SharedBuffer<real_t>>                       contact_block_diag_aos;
        std::vector<SharedBuffer<idx_t>>                        contact_block_idx;
        std::vector<std::shared_ptr<SparseBlockVector<real_t>>> contact_block_diag;
        std::vector<SharedBuffer<mask_t>>                       level_constraints_mask;
        SharedBuffer<real_t>                                    contact_grad;
        SharedBuffer<real_t>                                    contact_local_grad;
        SharedBuffer<real_t>                                    macaulay;
        std::vector<SharedBuffer<real_t>>                       contact_hessian_local_x;
        std::vector<SharedBuffer<real_t>>                       contact_hessian_local_y;

        std::shared_ptr<smesh::Mesh>       contact_surface;
        std::shared_ptr<smesh::Mesh>       contact_eval_surface;
        std::shared_ptr<sfem::ContactData> contact_jacobi_data;
        std::shared_ptr<ContactJacobi>     contact_jacobi;
        SharedBuffer<mask_t>               constraints_mask;
        SharedBuffer<real_t>               agumentation;
        real_t                             contact_penalty{1000};

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

            constraints_mask = sfem::create_buffer<mask_t>(mask_count(space->n_dofs()), es);
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < constraints_mask->size(); ++i) {
                constraints_mask->data()[i] = 0;
            }
            f->constraints_mask(constraints_mask->data());

            std::vector<std::shared_ptr<smesh::Mesh>> galerkin_contact_surfaces(n_levels());
            contact_surface = smesh::skin(mesh);

            galerkin_contact_surfaces[0] = contact_surface;
            for (int i = 1; i < n_levels(); ++i) {
                galerkin_contact_surfaces[i] =
                        derefine_contact_trace(contact_surface, data->semistructured_levels[0], data->semistructured_levels[i]);
            }

            contact_eval_surface = contact_surface;
            if (smesh::is_semistructured_type(contact_surface->element_type(0))) {
                contact_eval_surface = smesh::ssquad_to_quad4(contact_surface);
                contact_eval_surface->block(0)->set_element_type(smesh::QUADSHELL4);
                remove_contraints_connected_elements(contact_eval_surface, constraints_mask, spatial_dim);
            }

            if (params.enable_self_contact) {
                contact = create_contact(
                        space, contact_eval_surface, params.margin, params.search_radius * params.search_radius, es);
            } else {
                contact = create_mulitbody_contact(
                        space, contact_eval_surface, params.margin, params.search_radius * params.search_radius, es);
            }

            // FIXME multiblock should still work!
            galerkin_restrictions = create_galerkin_rap(galerkin_contact_surfaces, data->semistructured_levels);

            agumentation       = sfem::create_buffer<real_t>(contact->mass_vector()->size(), es);
            contact_grad       = sfem::create_buffer<real_t>(space->n_dofs(), es);
            contact_local_grad = sfem::create_buffer<real_t>(contact->mass_vector()->size() * spatial_dim, es);
            macaulay           = sfem::create_buffer<real_t>(contact->mass_vector()->size(), es);
            sfem::blas<real_t>(es)->zeros(agumentation->size(), agumentation->data());

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
            contact_hessian_local_x.resize(n_levels());
            contact_hessian_local_y.resize(n_levels());
            for (int i = 0; i < n_levels(); i++) {
                memory[i]             = std::make_shared<Memory>();
                const ptrdiff_t n     = data->functions[i]->space()->n_dofs();
                memory[i]->solution   = create_buffer<real_t>(n, es);
                memory[i]->rhs        = create_buffer<real_t>(n, es);
                memory[i]->work       = create_buffer<real_t>(n, es);
                memory[i]->correction = create_buffer<real_t>(n, es);

                const ptrdiff_t n_contact  = contact_surfaces[i]->node_mapping()->size();
                memory[i]->diag            = create_buffer<real_t>(n_contact, es);
                contact_block_diag_soa[i]  = create_buffer<real_t>(6, n_contact, es);
                contact_block_diag_aos[i]  = create_buffer<real_t>(n_contact * 6, es);
                contact_block_idx[i]       = contact_surfaces[i]->node_mapping();
                contact_block_diag[i]      = create_sparse_block_vector(contact_block_idx[i], contact_block_diag_aos[i]);
                contact_hessian_local_x[i] = create_buffer<real_t>(n_contact * spatial_dim, es);
                contact_hessian_local_y[i] = create_buffer<real_t>(n_contact * spatial_dim, es);

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

            auto jacobi = sfem::create_shiftable_block_sym_jacobi<real_t>(fs->block_size(), diag, mask, es);
            jacobi->set_relaxation_parameter(real_t(1) / fs->block_size());
            smoothers[level]->set_preconditioner_op(jacobi);
            smoothers[level]->set_max_it(100000);
        }

        void resample_contact_conditions(const smesh::SharedBuffer<real_t>& displacement) {
            contact->recompute(displacement);
            coupling_matrices.clear();
            galerkin_contact_hessian_bsr.clear();
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

            if (!contact_jacobi_data) {
                contact_jacobi_data = std::make_shared<ContactData>();
            }

            *contact_jacobi_data = {.f                = f,
                                    .surface          = contact_eval_surface,
                                    .coupling_matrix  = coupling_matrices[0],
                                    .values           = contact->values(),
                                    .mass_vector      = contact->mass_vector(),
                                    .normals          = contact->normals(),
                                    .distances        = contact->distances(),
                                    .constraints_mask = constraints_mask,
                                    .agumentation     = agumentation};

            if (!contact_jacobi) {
                contact_jacobi = std::make_shared<ContactJacobi>(contact_jacobi_data);
            }

            contact_jacobi->set_penalty(contact_penalty);
            contact_jacobi->set_n_loops(params.contact_jacobi_loops);
            contact_jacobi->set_enable_augmentation(false);
        }

        void nonlinear_smooth(const SharedBuffer<real_t>& x) { contact_jacobi->smooth(x); }

        void pack_contact_block_diag(const int level) {
            auto            src = contact_block_diag_soa[level]->data();
            real_t* const   dst = contact_block_diag_aos[level]->data();
            const ptrdiff_t n   = memory[level]->diag->size();
            const int       dim = data->functions[level]->space()->block_size();
            const idx_t*    idx = contact_block_idx[level]->data();

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
            return {.f                = data->functions[level],
                    .surface          = contact_surfaces[level],
                    .coupling_matrix  = coupling_matrices[level],
                    .values           = coupling_matrices[level]->values,
                    .mass_vector      = mass_vectors[level],
                    .normals          = normals[level],
                    .distances        = nullptr,
                    .constraints_mask = level_constraints_mask[level],
                    .agumentation     = nullptr};
        }

        void assemble_level_contact_hessian_block_diag() {
            auto       blas = sfem::blas<real_t>(f->execution_space());
            const bool use_galerkin_contact_hessian =
                    params.contact_hessian_galerkin_assembly && f->execution_space() == EXECUTION_SPACE_HOST;

            if (use_galerkin_contact_hessian) {
                assemble_galerkin_contact_hessian_bsr();
            }

            for (int l = 0; l < n_levels(); ++l) {
                blas->values(memory[l]->diag->size(), 1, memory[l]->diag->data());
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < level_constraints_mask[l]->size(); ++i) {
                    level_constraints_mask[l]->data()[i] = 0;
                }
                data->functions[l]->constraints_mask(level_constraints_mask[l]->data());

                if (use_galerkin_contact_hessian) {
                    extract_contact_hessian_bsr_diag(galerkin_contact_hessian_bsr[l], contact_block_diag_aos[l]->data());
                } else {
                    auto diag = contact_block_diag_soa[l]->data();
                    for (int d = 0; d < 6; ++d) {
                        blas->values(contact_block_diag_soa[l]->extent(1), 0, diag[d]);
                    }

                    ContactData cd = linearized_contact_data(l);
                    assemble_contact_hessian_block_diag(cd, contact_penalty, contact_active[l]->data(), diag);
                    pack_contact_block_diag(l);
                }
            }
        }

        void ensure_fine_contact_hessian_bsr() {
            if (galerkin_contact_hessian_bsr.size() != static_cast<size_t>(n_levels())) {
                galerkin_contact_hessian_bsr.resize(n_levels());
            }

            if (galerkin_contact_hessian_bsr[0]) return;

            auto      graph  = create_contact_hessian_graph(coupling_matrices[0]);
            const int dim    = contact_surfaces[0]->spatial_dimension();
            auto      values = create_host_buffer<real_t>(graph->nnz() * dim * dim);

            galerkin_contact_hessian_bsr[0] = sfem::h_bsr_spmv<count_t, idx_t, real_t, real_t>(
                    graph->n_nodes(), graph->n_nodes(), dim, graph->rowptr(), graph->colidx(), values, real_t(0));
        }

        void assemble_galerkin_contact_hessian_bsr() {
            if (!params.contact_hessian_galerkin_assembly) return;
            if (f->execution_space() != EXECUTION_SPACE_HOST) return;

            SFEM_TRACE_SCOPE("assemble_galerkin_contact_hessian_bsr");
            ensure_fine_contact_hessian_bsr();

            auto fine = galerkin_contact_hessian_bsr[0];
            {
                real_t* const   values = fine->values->data();
                const ptrdiff_t n      = fine->values->size();
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < n; ++i) {
                    values[i] = 0;
                }
            }

            const int dim = contact_surfaces[0]->spatial_dimension();
            contact_hessian_bsr(dim,
                                coupling_matrices[0]->rows(),
                                coupling_matrices[0]->row_ptr->data(),
                                coupling_matrices[0]->col_idx->data(),
                                coupling_matrices[0]->values->data(),
                                normals[0]->data(),
                                mass_vectors[0]->data(),
                                contact_penalty,
                                contact_active[0]->data(),
                                fine->row_ptr->data(),
                                fine->col_idx->data(),
                                fine->values->data());

            for (int l = 0; l < n_levels() - 1; ++l) {
                galerkin_contact_hessian_bsr[l + 1] =
                        sfem::rap(galerkin_restrictions[l].R, galerkin_contact_hessian_bsr[l], galerkin_restrictions[l].P);
            }
        }

        std::shared_ptr<Operator<real_t>> contact_hessian_op(const int level) {
            if (params.contact_hessian_galerkin_assembly && f->execution_space() == EXECUTION_SPACE_HOST &&
                level < static_cast<int>(galerkin_contact_hessian_bsr.size()) && galerkin_contact_hessian_bsr[level]) {
                auto            surface = contact_surfaces[level];
                auto            hessian = galerkin_contact_hessian_bsr[level];
                auto            local_x = contact_hessian_local_x[level];
                auto            local_y = contact_hessian_local_y[level];
                const auto      es      = f->execution_space();
                const ptrdiff_t n       = operators[level]->rows();

                return sfem::make_op<real_t>(
                        n,
                        n,
                        [=](const real_t* const x, real_t* const y) {
                            apply_contact_hessian_bsr(surface, hessian, local_x, local_y, x, y);
                        },
                        es);
            }

            auto            surface         = contact_surfaces[level];
            auto            coupling_matrix = coupling_matrices[level];
            auto            level_normals   = normals[level];
            auto            level_mass      = mass_vectors[level];
            auto            active          = contact_active[level];
            const real_t    penalty         = contact_penalty;
            const ptrdiff_t n               = operators[level]->rows();
            auto            es              = f->execution_space();

            return sfem::make_op<real_t>(
                    n,
                    n,
                    [=](const real_t* const x, real_t* const y) {
                        apply_contact_hessian(surface, coupling_matrix, level_normals, level_mass, active, penalty, x, y);
                    },
                    es);
        }

        std::shared_ptr<Operator<real_t>> combined_op(const int level) { return operators[level] + contact_hessian_op(level); }

        real_t eval_fine_residual(const SharedBuffer<real_t>& x, const SharedBuffer<real_t>& residual) {
            SFEM_TRACE_SCOPE("MaMAL::eval_fine_residual");
            auto            blas  = sfem::blas<real_t>(f->execution_space());
            auto            mem   = memory[0];
            const ptrdiff_t ndofs = x->size();

            blas->values(ndofs, 0, residual->data());
            f->gradient(x->data(), residual->data());
            blas->scal(ndofs, -1, residual->data());

            blas->values(ndofs, 0, contact_grad->data());

            const int          dim       = contact_jacobi_data->surface->spatial_dimension();
            const ptrdiff_t    n_contact = contact_jacobi_data->coupling_matrix->rows();
            const auto         cm        = contact_jacobi_data->coupling_matrix;
            const auto         surface   = contact_jacobi_data->surface;
            real_t* const      cg        = contact_grad->data();
            const idx_t* const nm        = surface->node_mapping()->data();
            assert(dim == 3);

            compute_macaulay_term_from_global_displacement(
                    *contact_jacobi_data, contact_penalty, x->data(), contact_local_grad->data(), macaulay->data());

            blas->values(contact_local_grad->size(), 0, contact_local_grad->data());
            assemble_contact_gradient(dim,
                                      n_contact,
                                      contact_penalty,
                                      cm->row_ptr->data(),
                                      cm->col_idx->data(),
                                      cm->values->data(),
                                      contact_jacobi_data->distances->data(),
                                      contact_jacobi_data->agumentation->data(),
                                      contact_jacobi_data->normals->data(),
                                      contact_jacobi_data->mass_vector->data(),
                                      macaulay->data(),
                                      contact_local_grad->data());

            const real_t* const lcg = contact_local_grad->data();
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                const ptrdiff_t global_dof = nm[i] * dim;
                const ptrdiff_t local_dof  = i * dim;
                cg[global_dof + 0]         = lcg[local_dof + 0];
                cg[global_dof + 1]         = lcg[local_dof + 1];
                cg[global_dof + 2]         = lcg[local_dof + 2];
            }
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
            compute_macaulay_term_from_global_displacement(*contact_jacobi_data,
                                                           contact_penalty,
                                                           memory[0]->solution->data(),
                                                           contact_local_grad->data(),
                                                           macaulay->data());

            real_t* const       aug = agumentation->data();
            const real_t* const m   = macaulay->data();
            const real_t        p   = contact_penalty;
            const real_t        r   = params.augmentation_relaxation;
            const real_t        c   = real_t(1) - r;
            const ptrdiff_t     n   = agumentation->size();

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n; ++i) {
                aug[i] = c * aug[i] + r * p * m[i];
            }
        }

        real_t contact_penetration_norm(const SharedBuffer<real_t>& x) {
            auto blas = sfem::blas<real_t>(f->execution_space());
            compute_penetration_from_global_displacement(
                    *contact_jacobi_data, x->data(), contact_local_grad->data(), macaulay->data());
            return blas->norm2(macaulay->size(), macaulay->data());
        }

        real_t augmented_lagrangian_value(const SharedBuffer<real_t>& x) {
            static const real_t zero_step[1] = {0};
            real_t              value        = 0;
            f->value_steps(x->data(), x->data(), 1, zero_step, &value);

            const int           dim        = contact_jacobi_data->surface->spatial_dimension();
            const ptrdiff_t     n_contact  = contact_jacobi_data->coupling_matrix->rows();
            const auto          cm         = contact_jacobi_data->coupling_matrix;
            const idx_t* const  nm         = contact_jacobi_data->surface->node_mapping()->data();
            const real_t* const disp       = x->data();
            real_t* const       local_disp = contact_hessian_local_x[0]->data();

            assert(dim == 3);
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                const ptrdiff_t global_dof = nm[i] * dim;
                const ptrdiff_t local_dof  = i * dim;
                local_disp[local_dof + 0]  = disp[global_dof + 0];
                local_disp[local_dof + 1]  = disp[global_dof + 1];
                local_disp[local_dof + 2]  = disp[global_dof + 2];
            }

            contact_objective_steps(dim,
                                    n_contact,
                                    cm->row_ptr->data(),
                                    cm->col_idx->data(),
                                    cm->values->data(),
                                    contact_jacobi_data->distances->data(),
                                    contact_jacobi_data->agumentation->data(),
                                    contact_jacobi_data->normals->data(),
                                    contact_jacobi_data->mass_vector->data(),
                                    contact_penalty,
                                    local_disp,
                                    local_disp,
                                    1,
                                    zero_step,
                                    &value);

            return value;
        }

        real_t augmented_lagrangian_line_search_step(const std::shared_ptr<Memory>& mem) {
            static const int    n_line_search_steps         = 13;
            static const real_t steps[n_line_search_steps]  = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.};
            real_t              values[n_line_search_steps] = {0};

            f->value_steps(mem->solution->data(), mem->correction->data(), n_line_search_steps, steps, values);

            printf("MaMAL::nonlinear_cycle step fun: [\n");
            for (int i = 0; i < n_line_search_steps; ++i) {
                printf("%e -> %e\n", (double)steps[i], (double)values[i]);
            }
            printf("]\n");

            const int       dim        = contact_jacobi_data->surface->spatial_dimension();
            const ptrdiff_t n_contact  = contact_jacobi_data->coupling_matrix->rows();
            const auto      cm         = contact_jacobi_data->coupling_matrix;
            const idx_t*    nm         = contact_jacobi_data->surface->node_mapping()->data();
            const real_t*   disp       = mem->solution->data();
            const real_t*   inc        = mem->correction->data();
            real_t*         local_disp = contact_hessian_local_x[0]->data();
            real_t*         local_inc  = contact_hessian_local_y[0]->data();

            assert(dim == 3);
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_contact; ++i) {
                const ptrdiff_t global_dof = nm[i] * dim;
                const ptrdiff_t local_dof  = i * dim;
                local_disp[local_dof + 0]  = disp[global_dof + 0];
                local_disp[local_dof + 1]  = disp[global_dof + 1];
                local_disp[local_dof + 2]  = disp[global_dof + 2];
                local_inc[local_dof + 0]   = inc[global_dof + 0];
                local_inc[local_dof + 1]   = inc[global_dof + 1];
                local_inc[local_dof + 2]   = inc[global_dof + 2];
            }

            contact_objective_steps(dim,
                                    n_contact,
                                    cm->row_ptr->data(),
                                    cm->col_idx->data(),
                                    cm->values->data(),
                                    contact_jacobi_data->distances->data(),
                                    contact_jacobi_data->agumentation->data(),
                                    contact_jacobi_data->normals->data(),
                                    contact_jacobi_data->mass_vector->data(),
                                    contact_penalty,
                                    local_disp,
                                    local_inc,
                                    n_line_search_steps,
                                    steps,
                                    values);

            int    best_step_idx = 0;
            real_t best_value    = values[best_step_idx];
            for (int i = 1; i < n_line_search_steps; ++i) {
                if (values[i] < best_value) {
                    best_value    = values[i];
                    best_step_idx = i;
                }
            }

            printf("MaMAL::nonlinear_cycle step fun + contact: [\n");
            for (int i = 0; i < n_line_search_steps; ++i) {
                printf("%e -> %e\n", (double)steps[i], (double)values[i]);
            }
            printf("]\n");

            printf("MaMAL::nonlinear_cycle step fun + contact best: %e -> %e\n",
                   (double)steps[best_step_idx],
                   (double)best_value);

            return steps[best_step_idx];
        }

        void smooth_candidate_correction(const std::shared_ptr<Memory>& mem) {
            auto            blas = sfem::blas<real_t>(f->execution_space());
            const ptrdiff_t n    = mem->solution->size();
            const real_t*   sol  = mem->solution->data();
            real_t* const   cand = mem->work->data();
            real_t* const   corr = mem->correction->data();

            blas->copy(n, sol, cand);
            blas->axpy(n, 1, corr, cand);
            nonlinear_smooth(mem->work);
            blas->zaxpby(n, 1, cand, -1, sol, corr);
        }

        void nonlinear_cycle() {
            nonlinear_smooth(memory[0]->solution);

            const real_t grad_norm = eval_fine_residual_and_jacobian();

            auto                    blas = sfem::blas<real_t>(f->execution_space());
            std::shared_ptr<Memory> mem  = memory[0];

            if (n_levels() > 1) {
                auto mem_coarse = memory[1];

                {  // Restrict residual
                    blas->zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                    data->restrictions[0]->apply(mem->work->data(), mem_coarse->rhs->data());
                }
                linear_cycle(1);

                {  // Prolongate and correct
                    blas->zeros(mem->correction->size(), mem->correction->data());
                    data->prolongations[1]->apply(mem_coarse->solution->data(), mem->correction->data());

                    // smooth_candidate_correction(mem);
                    // const real_t current_energy        = augmented_lagrangian_value(mem->solution);
                    // const real_t candidate_energy      = augmented_lagrangian_value(mem->work);
                    // const real_t current_penetration   = contact_penetration_norm(mem->solution);
                    // const real_t candidate_penetration = contact_penetration_norm(mem->work);
                    // printf("MaMAL::nonlinear_cycle candidate energy %e -> %e penetration %e -> %e\n",
                    //        (double)current_energy,
                    //        (double)candidate_energy,
                    //        (double)current_penetration,
                    //        (double)candidate_penetration);

                    // const real_t step = augmented_lagrangian_line_search_step(mem);

                    const real_t step = smesh::Env::read("SFEM_MAMAL_LINE_SEARCH_STEP", real_t(1.0));
                    blas->axpy(mem->solution->size(), step, mem->correction->data(), mem->solution->data());
                }
            }

            nonlinear_smooth(memory[0]->solution);

            if (params.enable_augmentation) update_augmentation();
        }

        void linear_cycle(int level) {
            SFEM_TRACE_SCOPE("MaMAL::linear_cycle");

            auto blas     = sfem::blas<real_t>(f->execution_space());
            auto mem      = memory[level];
            auto smoother = smoothers[level];
            auto cop      = combined_op(level);

            if (level == n_levels() - 1) {
                // Coarse grid solve
                smoother->set_op_and_diag_shift(cop, contact_block_diag[level], memory[level]->diag);
                blas->zeros(mem->solution->size(), mem->solution->data());
                smoother->apply(mem->rhs->data(), mem->solution->data());
                return;
            }

            auto mem_coarse = memory[level + 1];

            {  // Pre-smooth
                smoother->set_op_and_diag_shift(cop, contact_block_diag[level], memory[level]->diag);
                blas->zeros(mem->solution->size(), mem->solution->data());
                smoother->apply(mem->rhs->data(), mem->solution->data());
            }

            {  // Restrict residual
                blas->zeros(mem->work->size(), mem->work->data());
                cop->apply(mem->solution->data(), mem->work->data());
                blas->axpby(mem->work->size(), 1, mem->rhs->data(), -1, mem->work->data());

                blas->zeros(mem_coarse->rhs->size(), mem_coarse->rhs->data());
                data->restrictions[level]->apply(mem->work->data(), mem_coarse->rhs->data());
            }

            linear_cycle(level + 1);

            {  // Prolongate and correct
                blas->zeros(mem->work->size(), mem->work->data());
                data->prolongations[level + 1]->apply(mem_coarse->solution->data(), mem->work->data());
                blas->axpy(mem->solution->size(), 1, mem->work->data(), mem->solution->data());
            }

            {  // Post-smooth
                smoother->apply(mem->rhs->data(), mem->solution->data());
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

    int MaMAL::solve(const smesh::SharedBuffer<real_t>& x) {
        SFEM_TRACE_SCOPE("MaMAL::solve");

        auto blas = sfem::blas<real_t>(impl_->f->execution_space());
        auto mem  = impl_->memory[0];

        blas->copy(x->size(), x->data(), mem->solution->data());
        impl_->f->apply_constraints(mem->solution->data());

        int iter = 0;
        for (; iter < impl_->params.max_iterations; ++iter) {
            if (iter == 0 || (impl_->params.contact_update_frequency > 0 && iter % impl_->params.contact_update_frequency == 0)) {
                impl_->resample_contact_conditions(mem->solution);
            }

            impl_->nonlinear_cycle();

            const real_t grad_norm = impl_->eval_fine_residual(mem->solution, mem->work);
            compute_penetration_from_global_displacement(*impl_->contact_jacobi_data,
                                                         mem->solution->data(),
                                                         impl_->contact_local_grad->data(),
                                                         impl_->macaulay->data());
            const real_t penetration_norm = blas->norm2(impl_->macaulay->size(), impl_->macaulay->data());

            printf("MaMAL::solve %d gradient_norm %e penetration_norm %e\n", iter, (double)grad_norm, (double)penetration_norm);
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
