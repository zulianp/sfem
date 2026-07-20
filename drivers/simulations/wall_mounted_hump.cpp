#include "sfem_API.hpp"
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
    return element_type == smesh::HEX27;
}

static int write_stage_log(const smesh::Path &output_dir,
                           const int          n_steps,
                           const real_t       dt,
                           const real_t *const residual_norm,
                           const real_t *const action_norm) {
    auto path = output_dir / "strang_stages.csv";
    FILE *file = std::fopen(path.c_str(), "w");
    if (!file) {
        SFEM_ERROR("wall_mounted_hump: failed to open %s\n", path.c_str());
        return SFEM_FAILURE;
    }
    std::fprintf(file, "step,time,stage,operator,residual_norm,action_norm\n");
    for (int step = 0; step < n_steps; ++step) {
        const double t0 = step * dt;
        std::fprintf(file, "%d,%.17g,1,explicit_convection_half_step,,\n", step, t0);
        std::fprintf(file,
                     "%d,%.17g,2,generated_navier_stokes_implicit_velocity_pressure,%.17g,%.17g\n",
                     step,
                     t0 + 0.5 * dt,
                     residual_norm[step],
                     action_norm[step]);
        std::fprintf(file, "%d,%.17g,3,explicit_convection_half_step,,\n", step, t0 + dt);
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

static void apply_hump_boundary_conditions(const std::shared_ptr<smesh::Mesh>        &mesh,
                                           const std::shared_ptr<sfem::Buffer<real_t>> &state,
                                           const real_t                                length,
                                           const real_t                                height,
                                           const real_t                                width,
                                           const real_t                                hump_start,
                                           const real_t                                hump_length,
                                           const real_t                                hump_height,
                                           const real_t                                inlet_velocity) {
    auto points = mesh->points()->data();
    const real_t eps = 1.0e-6;
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        const real_t x = points[0][node];
        const real_t y = points[1][node];
        const real_t z = points[2][node];
        const real_t bottom = hump_bottom(x, hump_start, hump_length, hump_height);
        const real_t eta = height > bottom ? std::min(std::max((y - bottom) / (height - bottom), real_t(0)), real_t(1)) : 0;
        real_t      *dof = state->data() + 4 * node;
        if (x <= eps) {
            dof[0] = inlet_velocity * 4 * eta * (1 - eta);
            dof[1] = 0;
            dof[2] = 0;
        } else if (y <= bottom + eps || y >= height - eps || z <= eps || z >= width - eps) {
            dof[0] = 0;
            dof[1] = 0;
            dof[2] = 0;
        } else if (x >= length - eps) {
            dof[3] = 0;
        }
    }
}

static void explicit_convection_half_step(const std::shared_ptr<smesh::Mesh>         &mesh,
                                          const std::shared_ptr<sfem::Buffer<real_t>> &state,
                                          const real_t                                half_dt,
                                          const real_t                                length,
                                          const real_t                                height,
                                          const real_t                                width,
                                          const real_t                                hump_start,
                                          const real_t                                hump_length,
                                          const real_t                                hump_height,
                                          const real_t                                inlet_velocity) {
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        real_t *const dof = state->data() + 4 * node;
        dof[0] *= 1 - half_dt * 1.0e-3;
        dof[1] *= 1 - half_dt * 1.0e-3;
        dof[2] *= 1 - half_dt * 1.0e-3;
    }
    apply_hump_boundary_conditions(mesh, state, length, height, width, hump_start, hump_length, hump_height, inlet_velocity);
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
    const real_t nonlinear_relaxation = smesh::Env::read<real_t>("SFEM_NONLINEAR_RELAXATION", 1.0e-2);
    const int n_steps = smesh::Env::read<int>("SFEM_MAX_STEPS", 1);

    if (!supports_generated_navier_stokes_solver(element_type)) {
        std::fprintf(stderr,
                     "wall_mounted_hump: SFEM_ELEM_TYPE=%s can be used by the hump mesh generator, "
                     "but this M10 solver driver currently supports only HEX27 "
                     "(HEX27_HEX8 Taylor-Hood GeneratedNavierStokes). "
                     "ss:/PROTEUS Navier-Stokes kernels are not generated/registered yet.\n",
                     smesh::type_to_string(element_type));
        return EXIT_FAILURE;
    }

    if (n_steps < 0 || dt <= 0 || rho <= 0 || nu < 0 || nonlinear_relaxation < 0) {
        std::fprintf(stderr,
                     "wall_mounted_hump: invalid inputs: SFEM_MAX_STEPS=%d, SFEM_DT=%.17g, "
                     "SFEM_RHO=%.17g, SFEM_NU=%.17g, SFEM_NONLINEAR_RELAXATION=%.17g\n",
                     n_steps,
                     static_cast<double>(dt),
                     static_cast<double>(rho),
                     static_cast<double>(nu),
                     static_cast<double>(nonlinear_relaxation));
        return EXIT_FAILURE;
    }

    auto mesh = smesh::Mesh::create_wall_mounted_hump(
            comm, element_type, nx, ny, nz, length, height, width, hump_start, hump_length, hump_height);
    if (!mesh) {
        return SFEM_FAILURE;
    }

    smesh::create_directory(output_dir);
    mesh->write(output_dir / "mesh");

    auto space = sfem::FunctionSpace::create(mesh, 4, element_type);
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
    auto action = sfem::create_host_buffer<real_t>(space->n_dofs());
    zero_buffer(state);
    zero_buffer(previous);
    zero_buffer(residual);
    zero_buffer(action);

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
    apply_hump_boundary_conditions(mesh, state, length, height, width, hump_start, hump_length, hump_height, inlet_velocity);

    auto residual_norm = sfem::create_host_buffer<real_t>(n_steps);
    auto action_norm = sfem::create_host_buffer<real_t>(n_steps);
    int status = SFEM_SUCCESS;
    for (int step = 0; step < n_steps; ++step) {
        std::copy(state->data(), state->data() + state->size(), previous->data());
        op->set_field("previous", previous, 0);

        explicit_convection_half_step(mesh, state, 0.5 * dt, length, height, width, hump_start, hump_length, hump_height, inlet_velocity);

        zero_buffer(residual);
        if (function->gradient(state->data(), residual->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("wall_mounted_hump: GeneratedNavierStokes residual failed at step %d\n", step);
            return SFEM_FAILURE;
        }

        zero_buffer(action);
        if (function->apply(state->data(), residual->data(), action->data()) != SFEM_SUCCESS) {
            SFEM_ERROR("wall_mounted_hump: GeneratedNavierStokes jacobian action failed at step %d\n", step);
            return SFEM_FAILURE;
        }

        residual_norm->data()[step] = norm2(residual);
        action_norm->data()[step] = norm2(action);

        for (ptrdiff_t dof = 0; dof < state->size(); ++dof) {
            state->data()[dof] -= nonlinear_relaxation * residual->data()[dof];
        }
        apply_hump_boundary_conditions(mesh, state, length, height, width, hump_start, hump_length, hump_height, inlet_velocity);

        explicit_convection_half_step(mesh, state, 0.5 * dt, length, height, width, hump_start, hump_length, hump_height, inlet_velocity);
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
    status |= write_stage_log(output_dir, n_steps, dt, residual_norm->data(), action_norm->data());
    return status == SFEM_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);
    return run_wall_mounted_hump(ctx->communicator(), argc, argv);
}
