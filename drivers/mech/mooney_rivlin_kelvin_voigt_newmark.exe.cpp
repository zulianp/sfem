#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_NewmarkInertiaPotential.hpp"
#include "sfem_StateField.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"

namespace {

    struct EnvOptions {
        real_t mu;
        real_t lambda;
        real_t eta_s;
        real_t eta_b;
        real_t rho;
        real_t dt;
        real_t t_end;
        int    n_steps;
        real_t beta;
        real_t gamma;
        int    nl_max_it;
        real_t nl_tol;
        real_t nl_alpha;
        real_t lsolve_rtol;
        real_t lsolve_atol;
        int    lsolve_max_it;
        int    export_freq;
        real_t load_scale;
        real_t load_ramp_time;
        real_t load_pulse_time;
        real_t initial_disp_y;
        real_t initial_disp_z;
        std::string initial_displacement;
        std::string initial_displacement_components;
        std::string initial_velocity;
        std::string initial_velocity_components;
        bool   verbose;

        static EnvOptions read() {
            EnvOptions ret{};
            ret.mu              = smesh::Env::read("SFEM_MU", 1.0);
            ret.lambda          = smesh::Env::read("SFEM_LAMBDA", 1.0);
            ret.eta_s           = smesh::Env::read("SFEM_ETA_S", 0.1);
            ret.eta_b           = smesh::Env::read("SFEM_ETA_B", 0.0);
            ret.rho             = smesh::Env::read("SFEM_RHO", 1.0);
            ret.dt              = smesh::Env::read("SFEM_DT", 0.01);
            ret.t_end           = smesh::Env::read("SFEM_T_END", 1.0);
            ret.n_steps         = std::max<int>(1, smesh::Env::read("SFEM_STEPS", (int)std::ceil(ret.t_end / ret.dt)));
            ret.beta            = smesh::Env::read("SFEM_NEWMARK_BETA", 0.25);
            ret.gamma           = smesh::Env::read("SFEM_NEWMARK_GAMMA", 0.5);
            ret.nl_max_it       = smesh::Env::read("SFEM_NL_MAX_IT", 30);
            ret.nl_tol          = smesh::Env::read("SFEM_NL_TOL", 1e-9);
            ret.nl_alpha        = smesh::Env::read("SFEM_NL_ALPHA", 1.0);
            ret.lsolve_rtol     = smesh::Env::read("SFEM_LSOLVE_RTOL", 1e-3);
            ret.lsolve_atol     = smesh::Env::read("SFEM_LSOLVE_ATOL", 1e-12);
            ret.lsolve_max_it   = smesh::Env::read("SFEM_LSOLVE_MAX_IT", 20000);
            ret.export_freq     = std::max<int>(1, smesh::Env::read("SFEM_EXPORT_FREQ", 1));
            ret.load_scale      = smesh::Env::read("SFEM_LOAD_SCALE", 1.0);
            ret.load_ramp_time  = smesh::Env::read("SFEM_LOAD_RAMP_TIME", 0.0);
            ret.load_pulse_time = smesh::Env::read("SFEM_LOAD_PULSE_TIME", 0.0);
            ret.initial_disp_y  = smesh::Env::read("SFEM_INITIAL_DISP_Y", 0.0);
            ret.initial_disp_z  = smesh::Env::read("SFEM_INITIAL_DISP_Z", 0.0);
            ret.initial_displacement = smesh::Env::read_string("SFEM_INITIAL_DISPLACEMENT", "");
            ret.initial_displacement_components = smesh::Env::read_string("SFEM_INITIAL_DISPLACEMENT_COMPONENTS", "");
            ret.initial_velocity            = smesh::Env::read_string("SFEM_INITIAL_VELOCITY", "");
            ret.initial_velocity_components = smesh::Env::read_string("SFEM_INITIAL_VELOCITY_COMPONENTS", "");
            ret.verbose         = smesh::Env::read("SFEM_VERBOSE", false);
            return ret;
        }
    };

    void set_material_parameter(const std::shared_ptr<sfem::Op>   &op,
                                const std::shared_ptr<sfem::Mesh> &mesh,
                                const char *const                  parameter_name,
                                const real_t                       value) {
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), parameter_name, value);
        }
    }

    real_t load_factor(const EnvOptions &env, const real_t t) {
        if (env.load_pulse_time > 0) {
            if (t >= env.load_pulse_time) {
                return 0;
            }

            const real_t pi = std::acos(real_t(-1));
            return env.load_scale * std::sin(pi * t / env.load_pulse_time);
        }

        const real_t ramp = env.load_ramp_time > 0 ? std::min<real_t>(t / env.load_ramp_time, 1) : 1;
        return env.load_scale * ramp;
    }

    void initialize_cantilever_bend(const std::shared_ptr<sfem::Mesh> &mesh,
                                    const real_t                       tip_y,
                                    const real_t                       tip_z,
                                    real_t *const SFEM_RESTRICT        u) {
        if (tip_y == 0 && tip_z == 0) {
            return;
        }

        const ptrdiff_t n_nodes = mesh->n_nodes();
        const int       dim     = mesh->spatial_dimension();
        auto            points  = mesh->points()->data();
        const geom_t   *x       = points[0];

        geom_t xmin = x[0];
        geom_t xmax = x[0];
        for (ptrdiff_t i = 1; i < n_nodes; ++i) {
            xmin = std::min(xmin, x[i]);
            xmax = std::max(xmax, x[i]);
        }

        const real_t inv_length = xmax > xmin ? real_t(1) / (xmax - xmin) : real_t(0);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            real_t s = (x[i] - xmin) * inv_length;
            s        = std::max<real_t>(0, std::min<real_t>(1, s));

            const real_t shape = s * s * (3 - 2 * s);
            if (dim > 1) {
                u[dim * i + 1] = tip_y * shape;
            }
            if (dim > 2) {
                u[dim * i + 2] = tip_z * shape;
            }
        }
    }

    int read_initial_state(const std::shared_ptr<sfem::Mesh> &mesh,
                           const std::string                 &full_path,
                           const std::string                 &component_paths,
                           real_t *const                      state) {
        if (!full_path.empty() && !component_paths.empty()) {
            SFEM_ERROR("Specify either a full initial-state file or component files, not both\n");
            return SFEM_FAILURE;
        }
        if (!full_path.empty()) {
            return sfem::read_state_field(full_path, mesh->n_nodes() * mesh->spatial_dimension(), state);
        }
        if (!component_paths.empty()) {
            return sfem::read_state_field_components(component_paths, mesh->n_nodes(), mesh->spatial_dimension(), state);
        }
        return SFEM_SUCCESS;
    }

    void newmark_predictor(const ptrdiff_t n,
                           const real_t    dt,
                           const real_t    beta,
                           const real_t   *const SFEM_RESTRICT u,
                           const real_t   *const SFEM_RESTRICT v,
                           const real_t   *const SFEM_RESTRICT a,
                           real_t         *const SFEM_RESTRICT u_hat) {
        const real_t dt2_scale = dt * dt * (real_t(0.5) - beta);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            u_hat[i] = u[i] + dt * v[i] + dt2_scale * a[i];
        }
    }

    void newmark_velocity_shift(const ptrdiff_t n,
                                const real_t    dt,
                                const real_t    gamma,
                                const real_t    alpha_v,
                                const real_t   *const SFEM_RESTRICT v,
                                const real_t   *const SFEM_RESTRICT a,
                                const real_t   *const SFEM_RESTRICT u_hat,
                                real_t         *const SFEM_RESTRICT z) {
        const real_t a_scale = dt * (1 - gamma);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            z[i] = v[i] + a_scale * a[i] - alpha_v * u_hat[i];
        }
    }

    void newmark_update(const ptrdiff_t n,
                        const real_t    alpha_a,
                        const real_t    alpha_v,
                        const real_t   *const SFEM_RESTRICT u,
                        const real_t   *const SFEM_RESTRICT u_hat,
                        const real_t   *const SFEM_RESTRICT z,
                        real_t         *const SFEM_RESTRICT v,
                        real_t         *const SFEM_RESTRICT a) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            a[i] = alpha_a * (u[i] - u_hat[i]);
            v[i] = alpha_v * u[i] + z[i];
        }
    }

}  // namespace

int solve_mooney_rivlin_kelvin_voigt_newmark(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    if (argc != 5) {
        if (!comm->rank()) {
            std::fprintf(stderr, "usage: %s <mesh> <dirichlet.yaml|NONE> <neumann.yaml|NONE> <output>\n", argv[0]);
        }
        return SFEM_FAILURE;
    }

    const EnvOptions env = EnvOptions::read();
    if (env.dt <= 0 || env.beta <= 0) {
        SFEM_ERROR("SFEM_DT and SFEM_NEWMARK_BETA must be positive\n");
        return SFEM_FAILURE;
    }

    const smesh::Path mesh_path(argv[1]);
    const smesh::Path dirichlet_path(argv[2]);
    const smesh::Path neumann_path(argv[3]);
    const smesh::Path output_path(argv[4]);

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);
    auto fs   = sfem::FunctionSpace::create(mesh, mesh->spatial_dimension());
    auto f    = sfem::Function::create(fs);

    std::shared_ptr<sfem::DirichletConditions> dirichlet_conditions;
    if (dirichlet_path.to_string() != "NONE") {
        dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
        if (!dirichlet_conditions) return SFEM_FAILURE;
        f->add_constraint(dirichlet_conditions);
    }

    auto material_op = sfem::create_op(fs, "GeneratedMooneyRivlinKelvinVoigtNewmark", sfem::EXECUTION_SPACE_HOST);
    if (!material_op) {
        SFEM_ERROR("Unable to create GeneratedMooneyRivlinKelvinVoigtNewmark\n");
        return SFEM_FAILURE;
    }

    set_material_parameter(material_op, mesh, "mu", env.mu);
    set_material_parameter(material_op, mesh, "lmbda", env.lambda);
    set_material_parameter(material_op, mesh, "eta_s", env.eta_s);
    set_material_parameter(material_op, mesh, "eta_b", env.eta_b);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    auto u     = sfem::create_host_buffer<real_t>(ndofs);
    auto u_n   = sfem::create_host_buffer<real_t>(ndofs);
    auto v_n   = sfem::create_host_buffer<real_t>(ndofs);
    auto a_n   = sfem::create_host_buffer<real_t>(ndofs);
    auto v     = sfem::create_host_buffer<real_t>(ndofs);
    auto a     = sfem::create_host_buffer<real_t>(ndofs);
    auto z     = sfem::create_host_buffer<real_t>(ndofs);
    auto rhs   = sfem::create_host_buffer<real_t>(ndofs);
    auto incr  = sfem::create_host_buffer<real_t>(ndofs);

    blas->zeros(ndofs, u->data());
    blas->zeros(ndofs, u_n->data());
    blas->zeros(ndofs, v_n->data());
    blas->zeros(ndofs, a_n->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());
    blas->zeros(ndofs, z->data());
    blas->zeros(ndofs, rhs->data());
    blas->zeros(ndofs, incr->data());
    initialize_cantilever_bend(mesh, env.initial_disp_y, env.initial_disp_z, u_n->data());
    if (read_initial_state(mesh, env.initial_displacement, env.initial_displacement_components, u_n->data()) != SFEM_SUCCESS)
        return SFEM_FAILURE;
    if (read_initial_state(mesh, env.initial_velocity, env.initial_velocity_components, v_n->data()) != SFEM_SUCCESS)
        return SFEM_FAILURE;
    if (dirichlet_conditions && dirichlet_conditions->set_time(0) != SFEM_SUCCESS) return SFEM_FAILURE;
    f->apply_constraints(u_n->data());

    auto inertia_op = std::make_shared<sfem::NewmarkInertiaPotential>(fs);
    inertia_op->set_density(env.rho);
    if (inertia_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    auto u_hat = inertia_op->u_hat();

    material_op->set_field("previous", z, 0);
    if (material_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    f->add_operator(material_op);
    f->add_operator(inertia_op);

    std::shared_ptr<sfem::NeumannConditions> neumann_conditions;
    if (neumann_path.to_string() != "NONE") {
        neumann_conditions = sfem::NeumannConditions::create_from_file(fs, neumann_path);
        if (!neumann_conditions) return SFEM_FAILURE;
        f->add_operator(neumann_conditions);
    }

    auto linear_op = sfem::create_linear_operator("MF", f, u, sfem::EXECUTION_SPACE_HOST);
    auto bcgs      = sfem::create_bcgs<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
    bcgs->verbose  = env.verbose;
    bcgs->set_max_it(env.lsolve_max_it);
    bcgs->set_rtol(env.lsolve_rtol);
    bcgs->set_atol(env.lsolve_atol);

    smesh::create_directory(output_path);
    smesh::create_directory(output_path / "out");
    mesh->write(output_path / "mesh");

    auto out = f->output();
    out->set_output_dir(output_path / "out");
    out->enable_AoS_to_SoA(true);
    out->write_time_step("disp", 0, u_n->data());
    out->write_time_step("velocity", 0, v_n->data());
    out->write_time_step("acceleration", 0, a_n->data());
    out->log_time(0);

    const real_t alpha_a = 1 / (env.beta * env.dt * env.dt);
    const real_t alpha_v = env.gamma / (env.beta * env.dt);
    inertia_op->set_alpha(alpha_a);
    set_material_parameter(material_op, mesh, "newmark_velocity_alpha", alpha_v);

    if (!comm->rank()) {
        std::printf("Solving Mooney-Rivlin Kelvin-Voigt Newmark: ndofs=%td, dt=%g, steps=%d\n",
                    ndofs,
                    (double)env.dt,
                    env.n_steps);
        if (env.verbose) {
            std::printf("%-8s %-10s %-8s %-14s\n", "step", "newton", "bcgs", "residual");
        }
    }

    ptrdiff_t total_linear_iterations = 0;

    for (int step = 1; step <= env.n_steps; ++step) {
        const real_t t = step * env.dt;
        if (dirichlet_conditions && dirichlet_conditions->set_time(t) != SFEM_SUCCESS) return SFEM_FAILURE;
        if (neumann_conditions && neumann_conditions->set_time(t, load_factor(env, t)) != SFEM_SUCCESS) return SFEM_FAILURE;

        newmark_predictor(ndofs, env.dt, env.beta, u_n->data(), v_n->data(), a_n->data(), u_hat->data());
        newmark_velocity_shift(ndofs, env.dt, env.gamma, alpha_v, v_n->data(), a_n->data(), u_hat->data(), z->data());

        blas->copy(ndofs, u_n->data(), u->data());
        f->apply_constraints(u->data());

        bool converged = false;
        for (int it = 0; it < env.nl_max_it; ++it) {
            f->update(u->data());

            blas->zeros(ndofs, rhs->data());
            if (f->gradient(u->data(), rhs->data()) != SFEM_SUCCESS) {
                std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_newmark: gradient failed at step %d Newton iteration %d\n", step, it);
                return SFEM_FAILURE;
            }
            f->set_value_to_constrained_dofs(0, rhs->data());

            const real_t residual_norm = blas->norm2(ndofs, rhs->data());
            if (residual_norm < env.nl_tol) {
                converged = true;
                if (!comm->rank() && env.verbose) {
                    std::printf("%-8d %-10d %-8d %-14.4e\n", step, it, 0, (double)residual_norm);
                }
                break;
            }

            blas->zeros(ndofs, incr->data());
            f->copy_constrained_dofs(rhs->data(), incr->data());
            bcgs->set_op(linear_op);
            if (bcgs->apply(rhs->data(), incr->data()) != SFEM_SUCCESS) {
                std::fprintf(stderr, "mooney_rivlin_kelvin_voigt_newmark: BiCGStab failed at step %d Newton iteration %d\n", step, it);
                return SFEM_FAILURE;
            }
            total_linear_iterations += bcgs->iterations();

            blas->axpy(ndofs, -env.nl_alpha, incr->data(), u->data());
            f->apply_constraints(u->data());

            if (!comm->rank() && env.verbose) {
                std::printf("%-8d %-10d %-8d %-14.4e\n", step, it, bcgs->iterations(), (double)residual_norm);
            }
        }

        if (!converged) {
            SFEM_ERROR("Newton did not converge at step %d\n", step);
            return SFEM_FAILURE;
        }

        newmark_update(ndofs, alpha_a, alpha_v, u->data(), u_hat->data(), z->data(), v->data(), a->data());

        if (step % env.export_freq == 0 || step == env.n_steps) {
            out->write_time_step("disp", t, u->data());
            out->write_time_step("velocity", t, v->data());
            out->write_time_step("acceleration", t, a->data());
            out->log_time(t);
        }

        blas->copy(ndofs, u->data(), u_n->data());
        blas->copy(ndofs, v->data(), v_n->data());
        blas->copy(ndofs, a->data(), a_n->data());
    }

    if (!comm->rank()) {
        std::printf("Total BiCGStab iterations: %td\n", total_linear_iterations);
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize_serial(argc, argv);
    return solve_mooney_rivlin_kelvin_voigt_newmark(ctx->communicator(), argc, argv);
}
