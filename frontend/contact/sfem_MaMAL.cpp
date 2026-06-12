#include "sfem_MaMAL.hpp"

#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"

#include "sfem_CRS.hpp"
#include "sfem_SelfContact.hpp"

#include "sfem_API.hpp"
#include "smesh_ssquad4_prolongation.hpp"

namespace sfem {
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

    using CRS_t = sfem::CRS<count_t, idx_t, real_t>;

    struct GalerkinRAP {
        std::shared_ptr<CRS_t> R, P;
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
        std::shared_ptr<Function>                                    f;
        MaMALParams                                                  params;
        std::shared_ptr<Contact>                                     contact;
        std::shared_ptr<MultigridData>                               data;
        std::vector<std::shared_ptr<Operator<real_t>>>               operators;
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers;
        std::vector<std::shared_ptr<CRS_t>>                          coupling_matrices;

        std::vector<GalerkinRAP> galerkin_restrictions;

        std::shared_ptr<smesh::Mesh> contact_surface;

        Impl(const std::shared_ptr<Function>& f) : f(f) {}

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

            // TODO
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
