#include "sfem_ConstraintsFactory.hpp"

#include <functional>
#include <map>

#include "sfem_DirichletConditions.hpp"
#include "sfem_Rotate.hpp"

namespace sfem {

    class ConstraintsFactory::Impl {
    public:
#ifdef SFEM_ENABLE_RYAML
        using ConstraintFactoryT = std::function<std::shared_ptr<Constraint>(const std::shared_ptr<FunctionSpace> &space,
                                                                             const ryml::NodeRef                  &node,
                                                                             const ExecutionSpace execution_space)>;
        std::map<std::string, ConstraintFactoryT> constraints;
#endif
    };

    ConstraintsFactory::ConstraintsFactory() : impl_(std::make_unique<Impl>()) {}
    ConstraintsFactory::~ConstraintsFactory() = default;

    ConstraintsFactory &ConstraintsFactory::instance() {
        static ConstraintsFactory instance_;
        return instance_;
    }

    std::shared_ptr<Constraint> ConstraintsFactory::create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                                    const ExecutionSpace                  execution_space) {
        return nullptr;
    }

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Constraint> ConstraintsFactory::create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                                     const ryml::NodeRef                  &node,
                                                                     const ExecutionSpace                  execution_space) {
        if (!node["type"].readable()) {
            SFEM_ERROR("Invalid constraint type (missing 'type' key)\n");
            return nullptr;
        }

        std::string name;
        node["type"] >> name;
        auto iterator = instance().impl_->constraints.find(name);
        if (iterator != instance().impl_->constraints.end()) {
            return iterator->second(space, node, execution_space);
        } else {
            SFEM_ERROR("Invalid constraint type %s\n", name.c_str());
        }
        return nullptr;
    }

    void ConstraintsFactory::register_constraint_type(
            const std::string                                                                      &name,
            const std::function<std::shared_ptr<Constraint>(const std::shared_ptr<FunctionSpace> &space,
                                                            const ryml::NodeRef                  &node,
                                                            const ExecutionSpace                  execution_space)> &factory) {
        instance().impl_->constraints[name] = factory;
    }

    void register_constraints() {
        ConstraintsFactory::register_constraint_type("dirichlet",
                                                     [](const std::shared_ptr<FunctionSpace> &space,
                                                        const ryml::NodeRef                  &node,
                                                        const ExecutionSpace execution_space) -> std::shared_ptr<Constraint> {
                                                         auto conds = DirichletConditions::create_from_yaml(space, node);

                                                         if (execution_space == EXECUTION_SPACE_DEVICE) {
                                                             return to_device(conds);
                                                         }
                                                         return conds;
                                                     });

        ConstraintsFactory::register_constraint_type(
                "rotate_xy",
                [](const std::shared_ptr<FunctionSpace> &space, const ryml::NodeRef &node, const ExecutionSpace execution_space)
                        -> std::shared_ptr<Constraint> { return RotateXY::create_from_yaml(space, node, execution_space); });

        ConstraintsFactory::register_constraint_type(
                "rotate_xz",
                [](const std::shared_ptr<FunctionSpace> &space, const ryml::NodeRef &node, const ExecutionSpace execution_space)
                        -> std::shared_ptr<Constraint> { return RotateXZ::create_from_yaml(space, node, execution_space); });

        ConstraintsFactory::register_constraint_type(
                "rotate_yz",
                [](const std::shared_ptr<FunctionSpace> &space, const ryml::NodeRef &node, const ExecutionSpace execution_space)
                        -> std::shared_ptr<Constraint> { return RotateYZ::create_from_yaml(space, node, execution_space); });
    }

#endif

}  // namespace sfem