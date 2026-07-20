#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"
#include "smesh_mesh.hpp"
#include "smesh_output.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

enum BoundaryMarker {
    MARKER_INTERIOR = 0,
    MARKER_INLET = 1,
    MARKER_OUTLET = 2,
    MARKER_WALL = 3,
    MARKER_SPAN = 4,
};

static real_t hump_bottom(const real_t x, const real_t hump_start, const real_t hump_length, const real_t hump_height) {
    if (x < hump_start || x > hump_start + hump_length) {
        return 0;
    }
    const real_t pi = static_cast<real_t>(std::acos(-1.0));
    const real_t s = (x - hump_start) / hump_length;
    const real_t wave = std::sin(pi * s);
    return hump_height * wave * wave;
}

static void zero_buffer(const std::shared_ptr<sfem::Buffer<real_t>> &buffer) {
    std::memset(buffer->data(), 0, buffer->size() * sizeof(real_t));
}

static real_t dot(const std::shared_ptr<sfem::Buffer<real_t>> &left, const std::shared_ptr<sfem::Buffer<real_t>> &right) {
    real_t value = 0;
    for (ptrdiff_t i = 0; i < left->size(); ++i) {
        value += left->data()[i] * right->data()[i];
    }
    return value;
}

static real_t norm2(const std::shared_ptr<sfem::Buffer<real_t>> &buffer) {
    return std::sqrt(dot(buffer, buffer));
}

static bool supports_generated_navier_stokes_solver(const smesh::ElemType element_type) {
    return element_type == smesh::HEX27 || element_type == smesh::PROTEUS_HEX27;
}

static int prepare_mesh_for_generated_navier_stokes(const std::shared_ptr<smesh::Mesh> &mesh,
                                                    const smesh::ElemType               requested_element_type) {
    if (requested_element_type == smesh::HEX27) {
        return SFEM_SUCCESS;
    }

    if (requested_element_type != smesh::PROTEUS_HEX27) {
        return SFEM_FAILURE;
    }

    static constexpr int hex27_to_cartesian[27] = {
            0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 10, 14, 16, 12, 4, 22, 13,
    };
    auto elements = mesh->elements(0);
    auto streams = elements->data();
    idx_t *cartesian[27];
    for (int node = 0; node < 27; ++node) {
        cartesian[node] = streams[node];
    }
    for (int node = 0; node < 27; ++node) {
        streams[node] = cartesian[hex27_to_cartesian[node]];
    }
    mesh->block(0)->set_element_type(smesh::HEX27);
    return SFEM_SUCCESS;
}

static int write_stage_log(const smesh::Path &output_dir,
                           const int          n_steps,
                           const real_t       dt,
                           const real_t *const residual_norm,
                           const real_t *const correction_norm,
                           const int *const    nonlinear_iterations,
                           const int *const    linear_iterations) {
    auto path = output_dir / "solve_stages.csv";
    FILE *file = std::fopen(path.c_str(), "w");
    if (!file) {
        SFEM_ERROR("wall_mounted_hump: failed to open %s\n", path.c_str());
        return SFEM_FAILURE;
    }
    std::fprintf(file, "step,time,stage,operator,residual_norm,correction_norm,nonlinear_iterations,linear_iterations\n");
    for (int step = 0; step < n_steps; ++step) {
        const double t0 = step * dt;
        std::fprintf(file, "%d,%.17g,1,previous_state_update,,,,\n", step, t0);
        std::fprintf(file,
                     "%d,%.17g,2,generated_navier_stokes_newton_bicgstab,%.17g,%.17g,%d,%d\n",
                     step,
                     t0 + 0.5 * dt,
                     residual_norm[step],
                     correction_norm[step],
                     nonlinear_iterations[step],
                     linear_iterations[step]);
        std::fprintf(file, "%d,%.17g,3,write_time_step,,,,\n", step, t0 + dt);
    }
    std::fclose(file);
    return SFEM_SUCCESS;
}

static void split_state(const std::shared_ptr<sfem::Buffer<real_t>> &state,
                        const std::shared_ptr<sfem::Buffer<real_t>> &u0,
                        const std::shared_ptr<sfem::Buffer<real_t>> &u1,
                        const std::shared_ptr<sfem::Buffer<real_t>> &u2,
                        const std::shared_ptr<sfem::Buffer<real_t>> &p) {
    for (ptrdiff_t node = 0; node < u0->size(); ++node) {
        u0->data()[node] = state->data()[4 * node + 0];
        u1->data()[node] = state->data()[4 * node + 1];
        u2->data()[node] = state->data()[4 * node + 2];
        p->data()[node]  = state->data()[4 * node + 3];
    }
}

static std::shared_ptr<sfem::DirichletConditions> create_hump_dirichlet_conditions(
        const std::shared_ptr<sfem::FunctionSpace> &space,
        const std::shared_ptr<smesh::Mesh>          &mesh,
        const std::shared_ptr<sfem::Buffer<real_t>> &state,
        const real_t                                 length,
        const real_t                                 height,
        const real_t                                 width,
        const real_t                                 hump_start,
        const real_t                                 hump_length,
        const real_t                                 hump_height) {
    auto points = mesh->points()->data();
    const real_t eps = 1.0e-6;
    ptrdiff_t n_velocity_nodes = 0;
    ptrdiff_t n_pressure_nodes = 0;
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        const real_t x = points[0][node];
        const real_t y = points[1][node];
        const real_t z = points[2][node];
        const real_t bottom = hump_bottom(x, hump_start, hump_length, hump_height);
        n_velocity_nodes += (x <= eps || y <= bottom + eps || y >= height - eps || z <= eps || z >= width - eps) ? 1 : 0;
        n_pressure_nodes += (x >= length - eps) ? 1 : 0;
    }

    auto dirichlet = std::make_shared<sfem::DirichletConditions>(space);

    if (n_velocity_nodes > 0) {
        idx_t  *u0_nodes  = static_cast<idx_t *>(std::malloc(sizeof(idx_t) * n_velocity_nodes));
        real_t *u0_values = static_cast<real_t *>(std::malloc(sizeof(real_t) * n_velocity_nodes));
        idx_t  *u1_nodes  = static_cast<idx_t *>(std::malloc(sizeof(idx_t) * n_velocity_nodes));
        idx_t  *u2_nodes  = static_cast<idx_t *>(std::malloc(sizeof(idx_t) * n_velocity_nodes));

        ptrdiff_t offset = 0;
        for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
            const real_t x = points[0][node];
            const real_t y = points[1][node];
            const real_t z = points[2][node];
            const real_t bottom = hump_bottom(x, hump_start, hump_length, hump_height);
            if (x <= eps || y <= bottom + eps || y >= height - eps || z <= eps || z >= width - eps) {
                u0_nodes[offset]  = static_cast<idx_t>(node);
                u0_values[offset] = state->data()[4 * node + 0];
                u1_nodes[offset]  = static_cast<idx_t>(node);
                u2_nodes[offset]  = static_cast<idx_t>(node);
                ++offset;
            }
        }

        dirichlet->add_condition(n_velocity_nodes, n_velocity_nodes, u0_nodes, 0, u0_values);
        dirichlet->add_condition(n_velocity_nodes, n_velocity_nodes, u1_nodes, 1, real_t(0));
        dirichlet->add_condition(n_velocity_nodes, n_velocity_nodes, u2_nodes, 2, real_t(0));
    }

    if (n_pressure_nodes > 0) {
        idx_t *p_nodes = static_cast<idx_t *>(std::malloc(sizeof(idx_t) * n_pressure_nodes));
        ptrdiff_t offset = 0;
        for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
            const real_t x = points[0][node];
            if (x >= length - eps) {
                p_nodes[offset++] = static_cast<idx_t>(node);
            }
        }

        dirichlet->add_condition(n_pressure_nodes, n_pressure_nodes, p_nodes, 3, real_t(0));
    }

    return dirichlet;
}

static int run_wall_mounted_hump(const std::shared_ptr<sfem::Communicator> &comm, const int argc, char **argv) {
    const smesh::Path output_dir = argc > 1 ? smesh::Path(argv[1]) : smesh::Path("wall_mounted_hump_output");
    const auto element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX27").c_str());
    const ptrdiff_t nx = static_cast<ptrdiff_t>(smesh::Env::read<int64_t>("SFEM_HUMP_NX", 16));
    const ptrdiff_t ny = static_cast<ptrdiff_t>(smesh::Env::read<int64_t>("SFEM_HUMP_NY", 6));
    const ptrdiff_t nz = static_cast<ptrdiff_t>(smesh::Env::read<int64_t>("SFEM_HUMP_NZ", 2));
    const real_t length = smesh::Env::read<real_t>("SFEM_HUMP_LENGTH", 9.0);
    const real_t height = smesh::Env::read<real_t>("SFEM_HUMP_HEIGHT", 3.0);
    const real_t width = smesh::Env::read<real_t>("SFEM_HUMP_WIDTH", 1.0);
    const real_t hump_start = smesh::Env::read<real_t>("SFEM_HUMP_START", 0.65);
    const real_t hump_length = smesh::Env::read<real_t>("SFEM_HUMP_BODY_LENGTH", 1.0);
    const real_t hump_height = smesh::Env::read<real_t>("SFEM_HUMP_BODY_HEIGHT", 0.128);
    const real_t inlet_velocity = smesh::Env::read<real_t>("SFEM_HUMP_INLET_U", 1.0);
    const real_t dt = smesh::Env::read<real_t>("SFEM_DT", 0.01);
    const real_t rho = smesh::Env::read<real_t>("SFEM_RHO", 1.0);
    const real_t nu = smesh::Env::read<real_t>("SFEM_NU", 1.0e-3);
    const real_t nonlinear_relaxation = smesh::Env::read<real_t>("SFEM_NONLINEAR_RELAXATION", 1.0);
    const int n_steps = smesh::Env::read<int>("SFEM_MAX_STEPS", 1);
    const int nonlinear_max_it = smesh::Env::read<int>("SFEM_NL_MAX_IT", 8);
    const int linear_max_it = smesh::Env::read<int>("SFEM_LSOLVE_MAX_IT", 1000);
    const real_t nonlinear_atol = smesh::Env::read<real_t>("SFEM_NL_ATOL", 1.0e-8);
    const real_t linear_atol = smesh::Env::read<real_t>("SFEM_LSOLVE_ATOL", 1.0e-10);
    const real_t nonlinear_alpha = smesh::Env::read<real_t>("SFEM_NL_ALPHA", nonlinear_relaxation);
    const bool solver_verbose = smesh::Env::read<bool>("SFEM_SOLVER_VERBOSE", false);

    if (!supports_generated_navier_stokes_solver(element_type)) {
        std::fprintf(stderr,
                     "wall_mounted_hump: SFEM_ELEM_TYPE=%s is unsupported by this M10 solver. "
                     "Use HEX27 or PROTEUS_HEX27 for the HEX27_HEX8 Taylor-Hood GeneratedNavierStokes path.\n",
                     smesh::type_to_string(element_type));
        return EXIT_FAILURE;
    }

    if (n_steps < 0 || dt <= 0 || rho <= 0 || nu < 0 || nonlinear_max_it <= 0 || linear_max_it <= 0 ||
        nonlinear_atol < 0 || linear_atol < 0 || nonlinear_alpha <= 0) {
        std::fprintf(stderr,
                     "wall_mounted_hump: invalid inputs: SFEM_MAX_STEPS=%d, SFEM_DT=%.17g, "
                     "SFEM_RHO=%.17g, SFEM_NU=%.17g, SFEM_NL_MAX_IT=%d, SFEM_LSOLVE_MAX_IT=%d, "
                     "SFEM_NL_ATOL=%.17g, SFEM_LSOLVE_ATOL=%.17g, SFEM_NL_ALPHA=%.17g\n",
                     n_steps,
                     static_cast<double>(dt),
                     static_cast<double>(rho),
                     static_cast<double>(nu),
                     nonlinear_max_it,
                     linear_max_it,
                     static_cast<double>(nonlinear_atol),
                     static_cast<double>(linear_atol),
                     static_cast<double>(nonlinear_alpha));
        return EXIT_FAILURE;
    }

    auto mesh = smesh::Mesh::create_wall_mounted_hump(
            comm, element_type, nx, ny, nz, length, height, width, hump_start, hump_length, hump_height);
    if (!mesh) {
        return SFEM_FAILURE;
    }
    if (prepare_mesh_for_generated_navier_stokes(mesh, element_type) != SFEM_SUCCESS) {
        return EXIT_FAILURE;
    }

    smesh::create_directory(output_dir);
    mesh->write(output_dir / "mesh");

    auto space = sfem::FunctionSpace::create(mesh, 4, smesh::HEX27);
    auto op = sfem::create_op(space, "GeneratedNavierStokes", sfem::EXECUTION_SPACE_HOST);
    if (!op) {
        SFEM_ERROR("wall_mounted_hump: failed to create GeneratedNavierStokes for element type %d\n", element_type);
        return SFEM_FAILURE;
    }
    op->set_value_in_block("fluid", "rho", rho);
    op->set_value_in_block("fluid", "nu", nu);
    op->set_value_in_block("fluid", "dt", dt);
    op->set_value_in_block("fluid", "convection_scale", 1);

    auto function = sfem::Function::create(space);
    function->add_operator(op);
    function->set_output_dir(output_dir / "time_steps");

    auto state = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto previous = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto residual = sfem::create_host_buffer<real_t>(space->n_dofs());
    auto correction = sfem::create_host_buffer<real_t>(space->n_dofs());
    zero_buffer(state);
    zero_buffer(previous);
    zero_buffer(residual);
    zero_buffer(correction);

    auto u0 = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    auto u1 = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    auto u2 = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    auto p = sfem::create_host_buffer<real_t>(mesh->n_nodes());
    auto marker = sfem::create_host_buffer<real_t>(mesh->n_nodes());

    auto points = mesh->points()->data();
    const real_t eps = 1.0e-6;
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        const real_t x = points[0][node];
        const real_t y = points[1][node];
        const real_t z = points[2][node];
        const real_t bottom = hump_bottom(x, hump_start, hump_length, hump_height);
        const real_t eta = height > bottom ? (y - bottom) / (height - bottom) : 0;
        state->data()[4 * node + 0] = inlet_velocity * 4 * eta * (1 - eta);
        state->data()[4 * node + 1] = 0;
        state->data()[4 * node + 2] = 0;
        state->data()[4 * node + 3] = 0;
        marker->data()[node] = MARKER_INTERIOR;
        if (x <= eps) {
            marker->data()[node] = MARKER_INLET;
        } else if (x >= length - eps) {
            marker->data()[node] = MARKER_OUTLET;
        } else if (y <= bottom + eps || y >= height - eps) {
            marker->data()[node] = MARKER_WALL;
        } else if (z <= eps || z >= width - eps) {
            marker->data()[node] = MARKER_SPAN;
        }
    }
    auto dirichlet = create_hump_dirichlet_conditions(
            space, mesh, state, length, height, width, hump_start, hump_length, hump_height);
    function->add_dirichlet_conditions(dirichlet);
    function->apply_constraints(state->data());

    auto residual_norm = sfem::create_host_buffer<real_t>(n_steps);
    auto correction_norm = sfem::create_host_buffer<real_t>(n_steps);
    auto nonlinear_iterations = sfem::create_host_buffer<int>(n_steps);
    auto linear_iterations = sfem::create_host_buffer<int>(n_steps);
    int status = SFEM_SUCCESS;
    for (int step = 0; step < n_steps; ++step) {
        std::copy(state->data(), state->data() + state->size(), previous->data());
        op->set_field("previous", previous, 0);

        residual_norm->data()[step] = 0;
        correction_norm->data()[step] = 0;
        nonlinear_iterations->data()[step] = 0;
        linear_iterations->data()[step] = 0;

        for (int nonlinear_it = 0; nonlinear_it < nonlinear_max_it; ++nonlinear_it) {
            if (function->update(state->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("wall_mounted_hump: GeneratedNavierStokes update failed at step %d iteration %d\n", step, nonlinear_it);
                return SFEM_FAILURE;
            }

            zero_buffer(residual);
            if (function->gradient(state->data(), residual->data()) != SFEM_SUCCESS) {
                SFEM_ERROR("wall_mounted_hump: GeneratedNavierStokes residual failed at step %d iteration %d\n", step, nonlinear_it);
                return SFEM_FAILURE;
            }

            const real_t rnorm = norm2(residual);
            residual_norm->data()[step] = rnorm;
            nonlinear_iterations->data()[step] = nonlinear_it + 1;
            if (rnorm <= nonlinear_atol) {
                break;
            }

            auto linear_op = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, function, state, sfem::EXECUTION_SPACE_HOST);
            auto solver = sfem::create_bcgs<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
            solver->set_max_it(linear_max_it);
            solver->set_atol(linear_atol);
            solver->verbose = solver_verbose;

            zero_buffer(correction);
            function->copy_constrained_dofs(residual->data(), correction->data());
            const int linear_status = solver->apply(residual->data(), correction->data());
            linear_iterations->data()[step] += solver->iterations();

            const real_t cnorm = norm2(correction);
            correction_norm->data()[step] = cnorm;
            if (linear_status != SFEM_SUCCESS) {
                std::fprintf(stderr,
                             "wall_mounted_hump: BiCGStab failed at step %d nonlinear iteration %d "
                             "(residual %.17g, correction %.17g)\n",
                             step,
                             nonlinear_it,
                             static_cast<double>(rnorm),
                             static_cast<double>(cnorm));
                return EXIT_FAILURE;
            }

            for (ptrdiff_t dof = 0; dof < state->size(); ++dof) {
                state->data()[dof] -= nonlinear_alpha * correction->data()[dof];
            }
            function->apply_constraints(state->data());

            if (cnorm <= nonlinear_atol) {
                break;
            }
        }

        status = function->output()->write_time_step("state", (step + 1) * dt, state->data());
        if (status != SFEM_SUCCESS) {
            std::fprintf(stderr, "wall_mounted_hump: failed to write state time step %d\n", step + 1);
            return EXIT_FAILURE;
        }
    }

    split_state(state, u0, u1, u2, p);

    auto output = smesh::Output::create(mesh, output_dir / "restart");
    status = SFEM_SUCCESS;
    status |= output->write_nodal("u0", smesh::TypeToEnum<real_t>::value(), u0->data());
    status |= output->write_nodal("u1", smesh::TypeToEnum<real_t>::value(), u1->data());
    status |= output->write_nodal("u2", smesh::TypeToEnum<real_t>::value(), u2->data());
    status |= output->write_nodal("p", smesh::TypeToEnum<real_t>::value(), p->data());
    status |= output->write_nodal("boundary_marker", smesh::TypeToEnum<real_t>::value(), marker->data());
    status |= write_stage_log(output_dir,
                              n_steps,
                              dt,
                              residual_norm->data(),
                              correction_norm->data(),
                              nonlinear_iterations->data(),
                              linear_iterations->data());
    return status == SFEM_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return run_wall_mounted_hump(ctx->communicator(), argc, argv);
}
