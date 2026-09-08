// Prony-series viscoelastic torsion with release.
//
// A Mooney-Rivlin solid whose relaxation function is a Prony series
//
//     G(t) = g_inf + sum_i g_i exp(-t / tau_i),    g_inf = 1 - sum_i g_i
//
// is twisted about an axis, held, and then let go. While the twist is held the reaction
// torque decays through the spectrum of relaxation times; the instant the grip is released
// the torque drops to zero and the body recovers over the same spectrum. The two halves
// together characterise the series far better than either alone, which is why the release is
// part of the case rather than a separate run.
//
// The structure follows drivers/mech/hyperelasticity_bdf2.exe.cpp -- predictor, Newton with an
// assembled tangent, per-step export -- with three differences forced by the material:
//
//   * sfem::MooneyRivlinVisco provides no energy and no matrix-free apply, so the tangent is
//     assembled in BSR and the line search backtracks on the residual norm rather than on the
//     incremental potential.
//   * the material carries history, so update_history() closes each converged step and dt must
//     stay fixed: the Prony coefficients alpha_i = exp(-dt / tau_i) are computed once from it.
//   * inertia is optional. A relaxation experiment lives in the quasi-static regime, which is
//     the default here; `dynamics: {type: bdf2}` restores the second-order term.
//
// The material kernels are HEX8 only (see operators/mooney_rivlin_visco.cpp), so the mesh must
// be a hexahedral one.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BDF2InertiaPotential.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_NewmarkInertiaPotential.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"
#include "smesh_glob.hpp"

#include "prony_case.hpp"
#include "prony_torsion_bc.hpp"

namespace {

    idx_t nearest_node(const std::shared_ptr<sfem::Mesh> &mesh, const real_t target[3]) {
        const int       dim    = mesh->spatial_dimension();
        const ptrdiff_t nnodes = mesh->n_nodes();
        auto            points = mesh->points()->data();

        idx_t  best_node = 0;
        double best_d2   = std::numeric_limits<double>::max();
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            double d2 = 0;
            for (int d = 0; d < dim; ++d) {
                const double diff = (double)points[d][i] - (double)target[d];
                d2 += diff * diff;
            }

            if (d2 < best_d2) {
                best_d2   = d2;
                best_node = (idx_t)i;
            }
        }

        return best_node;
    }

    std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> create_linear_solver(
            const prony::Solver                             &opts,
            const std::shared_ptr<sfem::Operator<real_t>>   &op,
            const std::shared_ptr<sfem::Operator<real_t>>   &preconditioner,
            const ptrdiff_t                                  ndofs) {
        std::shared_ptr<sfem::MatrixFreeLinearSolver<real_t>> ret;

        if (opts.linear_type == "bcgs") {
            auto solver = sfem::create_bcgs<real_t>(op, sfem::EXECUTION_SPACE_HOST);
            solver->verbose = opts.lin_verbose;
            solver->set_rtol(opts.lin_rtol);
            solver->set_atol(opts.lin_atol);
            ret = solver;
        } else {
            auto solver = sfem::create_cg<real_t>(op, sfem::EXECUTION_SPACE_HOST);
            solver->verbose = opts.lin_verbose;
            solver->set_rtol(opts.lin_rtol);
            solver->set_atol(opts.lin_atol);
            ret = solver;
        }

        ret->set_max_it(opts.lin_max_it);
        ret->set_n_dofs(ndofs);
        if (preconditioner) {
            ret->set_preconditioner_op(preconditioner);
        }
        return ret;
    }

    void bdf2_predictor_be(const ptrdiff_t     n,
                           const real_t        dt,
                           const real_t *const u_n,
                           const real_t *const v_n,
                           real_t *const       u_hat) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            u_hat[i] = u_n[i] + dt * v_n[i];
        }
    }

    void bdf2_predictor(const ptrdiff_t     n,
                        const real_t        dt,
                        const real_t *const u_n,
                        const real_t *const u_nm1,
                        const real_t *const v_n,
                        const real_t *const v_nm1,
                        real_t *const       u_hat) {
        const real_t a0 = real_t(4.0 / 3.0);
        const real_t a1 = real_t(-1.0 / 3.0);
        const real_t b0 = real_t(8.0 / 9.0) * dt;
        const real_t b1 = real_t(-2.0 / 9.0) * dt;

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            u_hat[i] = a0 * u_n[i] + a1 * u_nm1[i] + b0 * v_n[i] + b1 * v_nm1[i];
        }
    }

    void update_velocity_be(const ptrdiff_t     n,
                            const real_t        inv_dt,
                            const real_t *const u_np1,
                            const real_t *const u_n,
                            real_t *const       v_np1) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            v_np1[i] = inv_dt * (u_np1[i] - u_n[i]);
        }
    }

    void update_velocity_bdf2(const ptrdiff_t     n,
                              const real_t        inv_2dt,
                              const real_t *const u_np1,
                              const real_t *const u_n,
                              const real_t *const u_nm1,
                              real_t *const       v_np1) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            v_np1[i] = inv_2dt * (3 * u_np1[i] - 4 * u_n[i] + u_nm1[i]);
        }
    }

    // The three below are lifted verbatim from drivers/mech/mooney_rivlin_kelvin_voigt_newmark,
    // the reference Newmark driver in this tree, which agrees line for line with the hand-rolled
    // scheme in frontend/tests/sfem_MooneyRivlinGravityTest.cpp.
    //
    // `z` is the constant part of the affine relation v = alpha_v * u + z. It exists because a
    // rate-dependent material needs v as a function of u *inside* the element kernel during the
    // Newton solve. MooneyRivlinVisco is displacement-based -- it takes prev_u, not a velocity --
    // so nothing here needs that, but the form is kept as-is so this stays diffable against the
    // reference.
    void newmark_predictor(const ptrdiff_t     n,
                           const real_t        dt,
                           const real_t        beta,
                           const real_t *const u,
                           const real_t *const v,
                           const real_t *const a,
                           real_t *const       u_hat) {
        const real_t dt2_scale = dt * dt * (real_t(0.5) - beta);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            u_hat[i] = u[i] + dt * v[i] + dt2_scale * a[i];
        }
    }

    void newmark_velocity_shift(const ptrdiff_t     n,
                                const real_t        dt,
                                const real_t        gamma,
                                const real_t        alpha_v,
                                const real_t *const v,
                                const real_t *const a,
                                const real_t *const u_hat,
                                real_t *const       z) {
        const real_t a_scale = dt * (1 - gamma);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            z[i] = v[i] + a_scale * a[i] - alpha_v * u_hat[i];
        }
    }

    void newmark_update(const ptrdiff_t     n,
                        const real_t        alpha_a,
                        const real_t        alpha_v,
                        const real_t *const u,
                        const real_t *const u_hat,
                        const real_t *const z,
                        real_t *const       v,
                        real_t *const       a) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            a[i] = alpha_a * (u[i] - u_hat[i]);
            v[i] = alpha_v * u[i] + z[i];
        }
    }

}  // namespace

int solve_prony_visco_torsion(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_prony_visco_torsion");

    if (argc != 2) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <case.yaml>\n", argv[0]);
        }
        return SFEM_FAILURE;
    }

    if (comm->size() > 1) {
        SFEM_ERROR("MPI runtimes are not supported by prony_visco_torsion!\n");
    }

    prony::Case c;
    if (prony::read_case(argv[1], c) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    if (!comm->rank()) {
        c.print(std::cout);
    }

    auto mesh = sfem::Mesh::create_from_file(comm, smesh::Path(c.mesh));
    if (!mesh) {
        SFEM_ERROR("[prony] unable to read mesh %s\n", c.mesh.c_str());
        return SFEM_FAILURE;
    }

    if (mesh->spatial_dimension() != 3) {
        SFEM_ERROR("[prony] the viscoelastic Mooney-Rivlin kernels are three dimensional (mesh has dim %d)\n",
                   mesh->spatial_dimension());
        return SFEM_FAILURE;
    }

    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);
    auto      f          = sfem::Function::create(fs);

    // ------------------------------------------------------------------ material
    auto visco = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    visco->set_C10(c.material.C10);
    visco->set_C01(c.material.C01);
    visco->set_K(c.material.K);
    visco->set_dt(c.time.dt);

    if (c.material.wlf_enabled) {
        // Order matters: the shift factor enters the coefficients that set_prony_terms and
        // initialize_history go on to compute.
        visco->set_wlf_params(c.material.wlf_C1, c.material.wlf_C2, c.material.wlf_T_ref);
        visco->set_temperature(c.material.temperature);
        visco->enable_wlf(true);
    }

    if (!c.material.prony.empty()) {
        std::vector<real_t> g, tau;
        g.reserve(c.material.prony.size());
        tau.reserve(c.material.prony.size());
        for (const auto &t : c.material.prony) {
            g.push_back(t.g);
            tau.push_back(t.tau);
        }
        visco->set_prony_terms((int)g.size(), g.data(), tau.data());
    }

    if (visco->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    visco->initialize_history();
    f->add_operator(visco);

    // ------------------------------------------------------------------ inertia
    // BDF2InertiaPotential and NewmarkInertiaPotential are the same operator -- a lumped-mass
    // penalty 1/2 alpha (u - u_hat)^T M (u - u_hat) -- and carry no scheme of their own. Which
    // integrator this is comes entirely from the alpha and u_hat the loop below feeds them, so
    // the two are held behind a setter rather than a common base class, which they lack.
    const bool                 dynamic = c.dynamics.integrator != prony::Integrator::QuasiStatic;
    const bool                 newmark = c.dynamics.integrator == prony::Integrator::Newmark;
    std::shared_ptr<sfem::Op>  inertia;
    std::function<void(real_t)> set_inertia_alpha;
    sfem::SharedBuffer<real_t>  u_hat;

    if (newmark) {
        auto op = std::make_shared<sfem::NewmarkInertiaPotential>(fs);
        op->set_density(c.dynamics.density);
        if (op->initialize() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        inertia           = op;
        u_hat             = op->u_hat();
        set_inertia_alpha = [op](const real_t alpha) { op->set_alpha(alpha); };
    } else if (dynamic) {
        auto op = std::make_shared<sfem::BDF2InertiaPotential>(fs);
        op->set_density(c.dynamics.density);
        if (op->initialize() != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        inertia           = op;
        u_hat             = op->u_hat();
        set_inertia_alpha = [op](const real_t alpha) { op->set_alpha(alpha); };
    }

    if (inertia) {
        f->add_operator(inertia);
    }

    // Newmark's coefficients do not change from step to step, unlike BDF2's, which switches
    // from the backward-Euler start to 9/4 after the first step.
    const real_t newmark_alpha_a = newmark ? 1 / (c.dynamics.beta * c.time.dt * c.time.dt) : 0;
    const real_t newmark_alpha_v = newmark ? c.dynamics.gamma / (c.dynamics.beta * c.time.dt) : 0;
    if (newmark) {
        set_inertia_alpha(newmark_alpha_a);
    }

    // ------------------------------------------------------------------ constraints
    std::shared_ptr<sfem::DirichletConditions> dirichlet;
    if (!c.dirichlet_file.empty()) {
        dirichlet = sfem::DirichletConditions::create_from_file(fs, c.dirichlet_file);
    } else if (!c.dirichlet_yaml.empty()) {
        // SFEM's parser looks up `dirichlet_conditions` at the document root, so the case file
        // is handed over whole and the dirichlet schema stays SFEM's.
        dirichlet = sfem::DirichletConditions::create_from_yaml(fs, c.dirichlet_yaml);
    }

    if (dirichlet) {
        f->add_constraint(dirichlet);
    }

    auto torsion = prony::create_torsion_bc(fs, c.torsion);
    if (torsion) {
        torsion->set_angle(prony::torsion_angle_at(c.torsion, 0));
        f->add_constraint(torsion->constraint);
    }

    // ------------------------------------------------------------------ state
    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    auto u      = sfem::create_host_buffer<real_t>(ndofs);
    auto u_n    = sfem::create_host_buffer<real_t>(ndofs);
    auto u_nm1  = sfem::create_host_buffer<real_t>(ndofs);
    auto v      = sfem::create_host_buffer<real_t>(ndofs);
    auto v_n    = sfem::create_host_buffer<real_t>(ndofs);
    auto v_nm1  = sfem::create_host_buffer<real_t>(ndofs);
    auto a      = newmark ? sfem::create_host_buffer<real_t>(ndofs) : nullptr;
    auto a_n    = newmark ? sfem::create_host_buffer<real_t>(ndofs) : nullptr;
    auto z      = newmark ? sfem::create_host_buffer<real_t>(ndofs) : nullptr;
    auto rhs    = sfem::create_host_buffer<real_t>(ndofs);
    auto trial  = sfem::create_host_buffer<real_t>(ndofs);
    auto incr   = sfem::create_host_buffer<real_t>(ndofs);
    auto diag   = sfem::create_host_buffer<real_t>(ndofs);
    auto react  = sfem::create_host_buffer<real_t>(ndofs);

    blas->zeros(ndofs, u->data());
    blas->zeros(ndofs, u_n->data());
    blas->zeros(ndofs, u_nm1->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, v_n->data());
    blas->zeros(ndofs, v_nm1->data());
    if (newmark) {
        blas->zeros(ndofs, a->data());
        blas->zeros(ndofs, a_n->data());
        blas->zeros(ndofs, z->data());
    }

    if (dirichlet && dirichlet->set_time(0) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    f->apply_constraints(u_n->data());
    f->apply_constraints(u_nm1->data());
    f->apply_constraints(u->data());

    // ------------------------------------------------------------------ solvers
    auto linear_op = sfem::create_linear_operator(sfem::op_type::BSR, f, u, sfem::EXECUTION_SPACE_HOST);
    if (!linear_op) {
        SFEM_ERROR("[prony] unable to assemble the BSR tangent\n");
        return SFEM_FAILURE;
    }

    auto jacobi = c.solver.preconditioner ? sfem::create_shiftable_jacobi(diag, sfem::EXECUTION_SPACE_HOST) : nullptr;
    auto lsolve = create_linear_solver(c.solver, linear_op, jacobi, ndofs);

    // Sum of the operator gradients, with no constraint handling on top. Function::gradient
    // finishes by calling constraints_gradient, which overwrites every constrained entry with
    // the constraint violation -- zero once the constraint is satisfied. That is what the
    // Newton solve wants and exactly what the reaction must not have: the constraint force
    // lives in those entries and would be erased. So the reaction is read from the operators
    // directly.
    auto internal_force = [&](const real_t *const x, real_t *const out) -> int {
        blas->zeros(ndofs, out);
        if (visco->gradient(x, out) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        if (inertia && inertia->gradient(x, out) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        return SFEM_SUCCESS;
    };

    // ------------------------------------------------------------------ output
    const smesh::Path output_path{c.output.path};
    smesh::create_directory(output_path);
    smesh::create_directory(output_path / "out");
    fs->mesh_ptr()->write(output_path / "mesh");

    // Exported field files are named by an export counter, so writing into a directory that
    // already holds a longer run's output leaves stale higher-index files behind and any
    // transient export downstream then reads more fields than there are time steps. The driver
    // does not delete anything it did not create -- the path comes from the case file and may be
    // anything -- so it says so instead and leaves the choice to the caller.
    {
        size_t existing = 0;
        for (const char *ext : {"*.raw", "*.float64", "*.float32"}) {
            existing += smesh::find_files(((output_path / "out") / ext).to_string()).size();
        }

        if (existing && !comm->rank()) {
            fprintf(stderr,
                    "[prony] warning: %s already holds %zu exported field files. They are named by\n"
                    "        export index, so any left over from a longer run get mixed into this one\n"
                    "        and the transient export then fails on the mismatch. Clear it first.\n",
                    (output_path / "out").to_string().c_str(),
                    existing);
        }
    }

    auto out = f->output();
    out->set_output_dir(output_path / "out");
    out->enable_AoS_to_SoA(true);
    // One place that decides what a time step consists of. Writing the initial state and the
    // per-step state from two separate lists is how they drift: the initial export used to write
    // only the displacement, so velocity and acceleration ended up one file short of time.txt and
    // the transient export failed downstream on the mismatch.
    auto export_state = [&](const real_t time, const real_t *const u_s, const real_t *const v_s, const real_t *const a_s) {
        out->write_time_step("disp", time, u_s);
        if (inertia) {
            out->write_time_step("velocity", time, v_s);
        }
        if (newmark) {
            out->write_time_step("acceleration", time, a_s);
        }
        out->log_time(time);
    };

    export_state(0, u_n->data(), v_n->data(), newmark ? a_n->data() : nullptr);

    FILE *history = fopen(c.output.history_csv.c_str(), "w");
    if (!history) {
        SFEM_ERROR("[prony] unable to open history CSV %s\n", c.output.history_csv.c_str());
        return SFEM_FAILURE;
    }
    fprintf(history, "time,angle,released,torque,ux,uy,uz,newton_it,lin_it,gnorm\n");

    idx_t control_node = 0;
    if (c.output.has_control_point) {
        control_node = nearest_node(mesh, c.output.control_point);
        if (!comm->rank()) {
            printf("control point: node %d at (%g, %g, %g)\n",
                   (int)control_node,
                   (double)mesh->points()->data()[0][control_node],
                   (double)mesh->points()->data()[1][control_node],
                   (double)mesh->points()->data()[2][control_node]);
        }
    }

    const int n_steps = std::max<int>(1, (int)std::ceil(c.time.t_end / c.time.dt));

    if (!comm->rank()) {
        printf("nnodes: %td  nelements: %td  ndof: %td\n",
               mesh->n_nodes(),
               mesh->n_elements(),
               ndofs);
        printf("Solving Prony-series viscoelastic torsion: dt=%g, steps=%d, t_end=%g, %s\n",
               (double)c.time.dt,
               n_steps,
               (double)c.time.t_end,
               newmark ? "Newmark" : (dynamic ? "BDF2" : "quasi-static"));
        printf("%-8s %-10s %-8s %-6s %-14s %-14s %-14s\n", "step", "time", "newton", "lin", "gnorm", "angle", "torque");
        fflush(stdout);
    }

    // ------------------------------------------------------------------ time loop
    bool      released                = false;
    ptrdiff_t total_linear_iterations = 0;

    // A reaction that flips sign on every single step is the 2*dt saw-tooth of a non-dissipative
    // time integrator ringing on an impulse, not a physical response. Each individual step still
    // converges -- the Newton solve is fine, it is the scheme that is wrong -- so nothing else in
    // the loop notices, and the run completes and hands out garbage. Count the flips instead.
    real_t previous_torque = 0;
    int    sign_flips      = 0;
    bool   sawtooth_warned = false;

    // Torque control solves torque(angle) = target for the angle, wrapping the whole nonlinear
    // solve in a scalar secant iteration. The reaction torque is already computed every step, so
    // the only new machinery is the scalar root find -- and torque(angle) is smooth and
    // monotone, so it converges in a couple of iterations once warm-started from the previous
    // step.
    const bool torque_control = torsion && c.torsion.control == prony::TorsionControl::Torque;
    // A small probe twist to start from: the response is very nearly linear near zero, so one
    // evaluation there plus the exact point torque(0) = 0 puts the secant on the answer at once.
    real_t     theta          = torque_control ? real_t(1e-2) : real_t(0);
    // The two most recent (angle, torque) evaluations, carried across steps to warm-start. It is
    // seeded with the exact pair (0, 0) rather than left empty -- that is a free data point.
    real_t     secant_theta   = 0;
    real_t     secant_torque  = 0;
    bool       secant_primed  = true;

    for (int step = 1; step <= n_steps; ++step) {
        const real_t t = step * c.time.dt;

        if (torsion) {
            if (!released && c.torsion.has_release && t >= c.torsion.release_time) {
                // Releasing means the twisted face stops being constrained at all: it becomes
                // traction free and the stored elastic energy drives the recovery. The
                // displacement reached at release stays in u as the initial state of that
                // recovery, and the fixed end alone still removes the rigid body modes.
                f->clear_constraints();
                if (dirichlet) {
                    f->add_constraint(dirichlet);
                }
                released = true;
                if (!comm->rank()) {
                    printf("--- torsion released at t = %g ---\n", (double)t);
                    fflush(stdout);
                }
            }

            if (!released) {
                torsion->set_angle(prony::torsion_angle_at(c.torsion, t));
            }
        }

        if (dirichlet && dirichlet->set_time(t) != SFEM_SUCCESS) {
            fclose(history);
            return SFEM_FAILURE;
        }

        if (newmark) {
            newmark_predictor(ndofs, c.time.dt, c.dynamics.beta, u_n->data(), v_n->data(), a_n->data(), u_hat->data());
            newmark_velocity_shift(ndofs,
                                   c.time.dt,
                                   c.dynamics.gamma,
                                   newmark_alpha_v,
                                   v_n->data(),
                                   a_n->data(),
                                   u_hat->data(),
                                   z->data());
        } else if (inertia) {
            if (step == 1) {
                set_inertia_alpha(1 / (c.time.dt * c.time.dt));
                bdf2_predictor_be(ndofs, c.time.dt, u_n->data(), v_n->data(), u_hat->data());
            } else {
                set_inertia_alpha(real_t(9.0 / 4.0) / (c.time.dt * c.time.dt));
                bdf2_predictor(ndofs, c.time.dt, u_n->data(), u_nm1->data(), v_n->data(), v_nm1->data(), u_hat->data());
            }
        }

        blas->copy(ndofs, u_n->data(), u->data());

        int    newton_it        = 0;
        int    step_linear_it   = 0;
        real_t gnorm            = 0;
        real_t gnorm_0          = 0;
        bool   converged        = false;
        bool   diverged         = false;
        real_t torque           = 0;
        real_t angle            = 0;
        int    outer_it         = 0;

        const real_t torque_target = torque_control && !released ? prony::load_ramp(c.torsion, t) * c.torsion.torque : real_t(0);
        const int    outer_max     = (torque_control && !released) ? c.torsion.torque_max_it : 1;

        for (; outer_it < outer_max; ++outer_it) {
        if (torsion && !released) {
            torsion->set_angle(torque_control ? theta : prony::torsion_angle_at(c.torsion, t));
        }
        f->apply_constraints(u->data());

        newton_it      = 0;
        step_linear_it = 0;
        converged      = false;

        for (; newton_it < c.solver.nl_max_it; ++newton_it) {
            f->update(u->data());

            blas->zeros(ndofs, rhs->data());
            if (f->gradient(u->data(), rhs->data()) != SFEM_SUCCESS) {
                fclose(history);
                return SFEM_FAILURE;
            }

            f->set_value_to_constrained_dofs(0, rhs->data());

            gnorm = blas->norm2(ndofs, rhs->data());
            if (newton_it == 0) {
                gnorm_0 = gnorm;
            }

            if (!std::isfinite((double)gnorm) || (gnorm_0 > 0 && gnorm > c.solver.divergence_factor * gnorm_0)) {
                // Abandon rather than run the iteration cap out: a residual this far above its
                // starting value is not going to come back, and the remaining iterations only
                // delay the report.
                fprintf(stderr,
                        "[prony] Newton diverged at step %d: |g| = %g, initial |g| = %g\n",
                        step,
                        (double)gnorm,
                        (double)gnorm_0);
                diverged = true;
                break;
            }

            if (gnorm < c.solver.nl_tol) {
                converged = true;
                break;
            }

            // MooneyRivlinVisco assembles but does not apply, so the tangent is rebuilt in BSR
            // at every iteration.
            linear_op = sfem::create_linear_operator(sfem::op_type::BSR, f, u, sfem::EXECUTION_SPACE_HOST);
            if (!linear_op) {
                SFEM_ERROR("[prony] unable to assemble the BSR tangent\n");
                fclose(history);
                return SFEM_FAILURE;
            }

            if (jacobi) {
                blas->zeros(ndofs, diag->data());
                if (f->hessian_diag(u->data(), diag->data()) != SFEM_SUCCESS) {
                    fclose(history);
                    return SFEM_FAILURE;
                }
                f->set_value_to_constrained_dofs(1, diag->data());
                jacobi->set_diag(diag);
            }

            blas->zeros(ndofs, incr->data());
            f->copy_constrained_dofs(rhs->data(), incr->data());
            lsolve->set_op(linear_op);
            lsolve->apply(rhs->data(), incr->data());
            step_linear_it += lsolve->iterations();
            total_linear_iterations += lsolve->iterations();

            // The increment solves J incr = g, so the Newton step is u <- u - alpha incr.
            real_t alpha = -c.solver.nl_alpha;

            if (c.solver.line_search) {
                // Backtracking on ||residual||. There is no energy to minimise here: the
                // viscoelastic operator implements neither value() nor value_steps().
                const real_t factors[] = {real_t(1), real_t(0.5), real_t(0.25), real_t(0.125), real_t(1.0 / 16)};

                real_t best_norm  = std::numeric_limits<real_t>::max();
                real_t best_alpha = 0;

                for (const real_t factor : factors) {
                    const real_t trial_alpha = -c.solver.nl_alpha * factor;

                    blas->copy(ndofs, u->data(), trial->data());
                    blas->axpy(ndofs, trial_alpha, incr->data(), trial->data());
                    f->apply_constraints(trial->data());

                    blas->zeros(ndofs, react->data());
                    if (f->gradient(trial->data(), react->data()) != SFEM_SUCCESS) {
                        fclose(history);
                        return SFEM_FAILURE;
                    }
                    f->set_value_to_constrained_dofs(0, react->data());

                    const real_t trial_norm = blas->norm2(ndofs, react->data());
                    if (std::isfinite((double)trial_norm) && trial_norm < best_norm) {
                        best_norm  = trial_norm;
                        best_alpha = trial_alpha;
                    }

                    if (std::isfinite((double)trial_norm) && trial_norm < gnorm) {
                        break;
                    }
                }

                if (best_alpha == 0) {
                    fprintf(stderr, "[prony] no step reduced the residual at step %d, abandoning\n", step);
                    diverged = true;
                    break;
                }

                alpha = best_alpha;
            }

            blas->axpy(ndofs, alpha, incr->data(), u->data());
            f->apply_constraints(u->data());

            if (c.verbose && !comm->rank()) {
                // The inner loop has to be watchable while it runs: a step that is grinding
                // down its residual and a step that has stalled look identical from the
                // per-step line alone.
                printf("    newton %3d: |g| = %-14.6e lin_it = %-6d alpha = %-10.4g\n",
                       newton_it,
                       (double)gnorm,
                       lsolve->iterations(),
                       (double)alpha);
                fflush(stdout);
            }
        }

        if (diverged) {
            fclose(history);
            return SFEM_FAILURE;
        }

        if (!converged && !comm->rank()) {
            fprintf(stderr,
                    "[prony] Newton did not reach %g in %d iterations at step %d (|g| = %g)\n",
                    (double)c.solver.nl_tol,
                    c.solver.nl_max_it,
                    step,
                    (double)gnorm);
        }

        if (internal_force(u->data(), react->data()) != SFEM_SUCCESS) {
            fclose(history);
            return SFEM_FAILURE;
        }

        angle  = (torsion && !released) ? (torque_control ? theta : prony::torsion_angle_at(c.torsion, t)) : real_t(0);
        torque = torsion ? prony::reaction_torque(*torsion, mesh, block_size, u->data(), react->data()) : real_t(0);

        if (outer_max == 1) {
            break;
        }

        const real_t residual = torque - torque_target;
        if (std::abs(residual) <= c.torsion.torque_tol * std::abs(torque_target)) {
            break;
        }

        // Secant on theta. The first evaluation of the run has no second point, so the initial
        // guess comes from the secant through the origin -- torque(0) = 0 exactly, and
        // torque(theta) is very nearly linear near it.
        real_t next_theta;
        if (!secant_primed || std::abs(torque - secant_torque) < std::abs(torque) * real_t(1e-14)) {
            next_theta = (std::abs(torque) > 0) ? theta * (torque_target / torque) : theta + real_t(1e-3);
        } else {
            next_theta = theta + (torque_target - torque) * (theta - secant_theta) / (torque - secant_torque);
        }

        // A secant step on a stiffening response can overshoot badly, so the change is clamped.
        // The bound has to be a span rather than a ratio: a multiplicative clamp collapses to
        // nothing at theta = 0 and the iteration then crawls away from the origin one clamp at a
        // time. The floor lets it leave zero in one step and the magnitude term lets it double
        // once it is away.
        const real_t span = std::max(std::abs(theta), real_t(1e-2));
        next_theta        = std::min(std::max(next_theta, theta - span), theta + span);

        secant_theta  = theta;
        secant_torque = torque;
        secant_primed = true;
        theta         = next_theta;
        }

        if (torque_control && !released && outer_it >= outer_max && !comm->rank()) {
            fprintf(stderr,
                    "[prony] torque control did not reach %g of the target %g in %d iterations at step %d "
                    "(torque = %g)\n",
                    (double)c.torsion.torque_tol,
                    (double)torque_target,
                    outer_max,
                    step,
                    (double)torque);
        }

        if (newmark) {
            newmark_update(ndofs,
                           newmark_alpha_a,
                           newmark_alpha_v,
                           u->data(),
                           u_hat->data(),
                           z->data(),
                           v->data(),
                           a->data());
        } else if (inertia) {
            if (step == 1) {
                update_velocity_be(ndofs, 1 / c.time.dt, u->data(), u_n->data(), v->data());
            } else {
                update_velocity_bdf2(ndofs, 1 / (2 * c.time.dt), u->data(), u_n->data(), u_nm1->data(), v->data());
            }
        }

        if (step > 1 && torque * previous_torque < 0) {
            ++sign_flips;
        } else {
            sign_flips = 0;
        }
        previous_torque = torque;

        if (sign_flips >= 8 && !sawtooth_warned && !comm->rank()) {
            sawtooth_warned = true;
            fprintf(stderr,
                    "\n[prony] the reaction has changed sign on %d consecutive steps. That is the\n"
                    "        2*dt saw-tooth of a non-dissipative time integrator, not a response of the\n"
                    "        material -- every step converges, the scheme is what is wrong.\n",
                    sign_flips);
            if (newmark) {
                fprintf(stderr,
                        "        This run uses Newmark with beta = %g, gamma = %g. At gamma = 0.5 the\n"
                        "        scheme is exactly non-dissipative, so anything impulsive -- the start of\n"
                        "        a ramp, the start of a sine, a release -- rings forever. Use gamma > 0.5\n"
                        "        with beta = (1+gamma)^2/4, e.g. gamma = 0.6 with beta = 0.64.\n",
                        (double)c.dynamics.beta,
                        (double)c.dynamics.gamma);
            }
            fflush(stderr);
        }

        if (sign_flips >= 32) {
            fprintf(stderr,
                    "[prony] abandoning at step %d: the reaction has alternated for %d consecutive\n"
                    "        steps, so the remaining %d steps would only produce more of the same.\n",
                    step,
                    sign_flips,
                    n_steps - step);
            fclose(history);
            return SFEM_FAILURE;
        }

        // The history advances once per step, on the converged displacement. Doing it inside
        // the Newton loop would make the material depend on the iterate rather than the state.
        if (visco->update_history(u->data()) != SFEM_SUCCESS) {
            fclose(history);
            return SFEM_FAILURE;
        }

        if (!comm->rank()) {
            printf("%-8d %-10.4g %-8d %-6d %-14.4e %-14.6g %-14.6e\n",
                   step,
                   (double)t,
                   newton_it,
                   step_linear_it,
                   (double)gnorm,
                   (double)angle,
                   (double)torque);
            fflush(stdout);
        }

        const ptrdiff_t base = (ptrdiff_t)control_node * block_size;
        fprintf(history,
                "%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g\n",
                (double)t,
                (double)angle,
                released ? 1 : 0,
                (double)torque,
                (double)u->data()[base + 0],
                (double)u->data()[base + 1],
                (double)u->data()[base + 2],
                newton_it,
                step_linear_it,
                (double)gnorm);
        fflush(history);

        if (step % c.output.export_freq == 0 || step == n_steps) {
            export_state(t, u->data(), v->data(), newmark ? a->data() : nullptr);
        }

        blas->copy(ndofs, u_n->data(), u_nm1->data());
        blas->copy(ndofs, u->data(), u_n->data());
        blas->copy(ndofs, v_n->data(), v_nm1->data());
        blas->copy(ndofs, v->data(), v_n->data());
        if (newmark) {
            blas->copy(ndofs, a->data(), a_n->data());
        }
    }

    if (!comm->rank()) {
        printf("Total linear iterations: %td\n", total_linear_iterations);
        printf("Final displacement norm: %.12e\n", (double)blas->norm2(ndofs, u_n->data()));
        printf("History written to %s\n", c.output.history_csv.c_str());
    }

    fclose(history);
    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize_serial(argc, argv);
    return solve_prony_visco_torsion(ctx->communicator(), argc, argv);
}
