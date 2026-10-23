#pragma once

// A unit cube of HEX8 elements, clamped at x = 0 and pulled along x at x = 1, for the tests of the
// nonlinear solver components.

#include <cmath>
#include <memory>
#include <string>

#include "sfem_API.hpp"

namespace sfem {
    namespace test {

        struct CubeProblem {
            std::shared_ptr<Mesh>          mesh;
            std::shared_ptr<FunctionSpace> fs;
            std::shared_ptr<Function>      f;
            std::shared_ptr<Op>            op;
            SharedBuffer<real_t>           x;  // zero, with the boundary values applied
        };

        /// `op_name` a generated elasticity operator whose parameters are mu and lmbda.
        inline CubeProblem make_cube_problem(const std::string &op_name, const int n, const real_t stretch) {
            CubeProblem p;
            auto        comm = Communicator::world();
            p.mesh           = Mesh::create_hex8_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
            p.fs             = FunctionSpace::create(p.mesh, 3);
            p.f              = Function::create(p.fs);

            p.op = create_op(p.fs, op_name.c_str(), EXECUTION_SPACE_HOST);
            p.op->initialize();
            for (const auto &block : p.mesh->blocks()) {
                p.op->set_value_in_block(block->name(), "mu", 1);
                p.op->set_value_in_block(block->name(), "lmbda", 2);
            }
            p.f->add_operator(p.op);

            auto left  = Sideset::create_from_selector(p.mesh, [](const geom_t x, const geom_t, const geom_t) -> bool {
                return std::abs(x) < 1e-8;
            });
            auto right = Sideset::create_from_selector(p.mesh, [](const geom_t x, const geom_t, const geom_t) -> bool {
                return std::abs(x - 1) < 1e-8;
            });

            std::vector<DirichletConditions::Condition> conditions = {{.sidesets = {left}, .value = 0, .component = 0},
                                                                      {.sidesets = {left}, .value = 0, .component = 1},
                                                                      {.sidesets = {left}, .value = 0, .component = 2},
                                                                      {.sidesets = {right}, .value = stretch, .component = 0}};
            p.f->add_constraint(DirichletConditions::create(p.fs, conditions));

            p.x = create_host_buffer<real_t>(p.fs->n_dofs());
            p.f->apply_constraints(p.x->data());
            return p;
        }

    }  // namespace test
}  // namespace sfem
