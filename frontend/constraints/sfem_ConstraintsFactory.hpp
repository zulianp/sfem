#ifndef SFEM_CONSTRAINTS_FACTORY_HPP
#define SFEM_CONSTRAINTS_FACTORY_HPP

#include "sfem_Constraint.hpp"

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#endif

namespace sfem {

    class ConstraintsFactory final {
    public:
        static ConstraintsFactory &instance();

        ConstraintsFactory();
        ~ConstraintsFactory();

#ifdef SFEM_ENABLE_RYAML
        static std::shared_ptr<Constraint> create_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                            const ryml::NodeRef                  &node,
                                                            const ExecutionSpace                  execution_space);

        static void register_constraint_type(
                const std::string                                                                      &name,
                const std::function<std::shared_ptr<Constraint>(const std::shared_ptr<FunctionSpace> &space,
                                                                const ryml::NodeRef                  &node,
                                                                const ExecutionSpace                  execution_space)> &factory);
#endif
        static std::shared_ptr<Constraint> create_from_env(const std::shared_ptr<FunctionSpace> &space,
                                                           const ExecutionSpace                  execution_space);

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    void register_constraints();

#ifdef SFEM_ENABLE_RYAML
    std::shared_ptr<Constraint> constraints_from_yaml(const std::shared_ptr<FunctionSpace> &space,
                                                      const ryml::NodeRef                  &node,
                                                      const ExecutionSpace                  execution_space) {
        return ConstraintsFactory::create_from_yaml(space, node, execution_space);
    }
#endif

}  // namespace sfem

#endif