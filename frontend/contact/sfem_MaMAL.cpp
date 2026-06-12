#include "sfem_MaMAL.hpp"

#include "sfem_Function.hpp"

namespace sfem {
    struct MaMALParams {
        int    max_iterations{100};
        real_t tolerance{1e-6};

#ifdef SFEM_ENABLE_YAML
        void from_yaml(const ryml::ConstNodeRef& node) {
            // TODO: implement
            max_iterations = node["max_iterations"].val<int>();
            tolerance      = node["tolerance"].val<real_t>();
        }
#endif
    };

    class MaMAL::Impl {
    public:
        std::shared_ptr<Function> f;
        ExecutionSpace            es;
        MaMALParams               params;

        Impl(const std::shared_ptr<Function>& f, const ExecutionSpace es) : f(f), es(es) {}

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
        }

        ~Impl() = default;
    };

    MaMAL::MaMAL(const std::shared_ptr<Function>& f, const ExecutionSpace es) : impl_(std::make_unique<Impl>(f, es)) {}

    MaMAL::~MaMAL() = default;

    std::shared_ptr<MaMAL> MaMAL::create(const std::shared_ptr<Function>& f, const ExecutionSpace es) {
        auto ret = std::make_shared<MaMAL>(f, es);
        ret->impl_->init();
        return ret;
    }

#ifdef SFEM_ENABLE_YAML
    std::shared_ptr<MaMAL> MaMAL::create(const std::shared_ptr<Function>& f,
                                         const ryml::ConstNodeRef&        node,
                                         const ExecutionSpace             es) {
        auto ret = std::make_shared<MaMAL>(f, es);
        ret->impl_->init(node);
        return ret;
    }
#endif
}  // namespace sfem
