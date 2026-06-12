#include "sfem_MaMAL.hpp"

#include "sfem_Function.hpp"
#include "sfem_GeometricMultigrid.hpp"

#include "sfem_CRS.hpp"
#include "sfem_SelfContact.hpp"

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

    class MaMAL::Impl {
    public:
        using CRS_t = sfem::CRS<count_t, idx_t, real_t>;
        std::shared_ptr<Function>                                    f;
        MaMALParams                                                  params;
        std::shared_ptr<Contact>                                     contact;
        std::shared_ptr<MultigridData>                               data;
        std::vector<std::shared_ptr<Operator<real_t>>>               operators;
        std::vector<std::shared_ptr<MatrixFreeLinearSolver<real_t>>> smoothers;
        std::vector<std::shared_ptr<CRS_t>>                          coupling_matrices;

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
