#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BDF2InertiaPotential.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_NeumannConditions.hpp"
#include "sfem_Rotate.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"

#include "sfem_ssgmg.hpp"

namespace {

    void set_material_parameter_if_present(const std::shared_ptr<sfem::Op>   &op,
                                           const std::shared_ptr<sfem::Mesh> &mesh,
                                           const char *const                  env_name,
                                           const char *const                  parameter_name) {
        if (!std::getenv(env_name)) {
            return;
        }

        const real_t value = smesh::Env::read(env_name, 1.0);
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), parameter_name, value);
        }
    }

    void set_material_parameters_from_env(const std::shared_ptr<sfem::Op> &op, const std::shared_ptr<sfem::Mesh> &mesh) {
        if (std::getenv("SFEM_MU")) {
            set_material_parameter_if_present(op, mesh, "SFEM_MU", "mu");
        }

        if (std::getenv("SFEM_LAMBDA")) {
            set_material_parameter_if_present(op, mesh, "SFEM_LAMBDA", "lmbda");
        }

        if (std::getenv("SFEM_C1")) {
            set_material_parameter_if_present(op, mesh, "SFEM_C1", "c1");
        }

        if (std::getenv("SFEM_C2")) {
            set_material_parameter_if_present(op, mesh, "SFEM_C2", "c2");
        }

        if (std::getenv("SFEM_KAPPA")) {
            set_material_parameter_if_present(op, mesh, "SFEM_KAPPA", "kappa");
        }
    }

    real_t load_factor(const real_t t) {
        const real_t scale      = smesh::Env::read("SFEM_LOAD_SCALE", 1.0);
        const real_t ramp_time  = smesh::Env::read("SFEM_LOAD_RAMP_TIME", 0.0);
        const real_t ramp_value = ramp_time > 0 ? std::min<real_t>(t / ramp_time, 1) : 1;
        return scale * ramp_value;
    }

    void scale_neumann_values(const std::shared_ptr<sfem::NeumannConditions> &neumann,
                              const std::vector<real_t>                      &base_values,
                              const real_t                                    scale) {
        if (!neumann) {
            return;
        }

        auto &conditions = neumann->conditions();
        for (size_t i = 0; i < conditions.size(); ++i) {
            conditions[i].value = scale * base_values[i];
        }
    }

    real_t dot_host(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (ptrdiff_t i = 0; i < n; ++i) {
            ret += a[i] * b[i];
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

    void update_acceleration(const ptrdiff_t     n,
                             const real_t        inv_dt,
                             const real_t *const v_np1,
                             const real_t *const v_n,
                             real_t *const       a_np1) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n; ++i) {
            a_np1[i] = inv_dt * (v_np1[i] - v_n[i]);
        }
    }

    idx_t nearest_node(const std::shared_ptr<sfem::Mesh> &mesh, const geom_t target[3]) {
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

    void write_control_point(FILE *const   file,
                             const real_t  t,
                             const int     block_size,
                             const idx_t   node,
                             const real_t *u,
                             const real_t *v,
                             const real_t *a) {
        if (!file) {
            return;
        }

        const ptrdiff_t base = node * block_size;
        const real_t    uz   = block_size > 2 ? u[base + 2] : 0;
        const real_t    vz   = block_size > 2 ? v[base + 2] : 0;
        const real_t    az   = block_size > 2 ? a[base + 2] : 0;
        fprintf(file,
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                (double)t,
                (double)u[base + 0],
                (double)u[base + 1],
                (double)uz,
                (double)v[base + 0],
                (double)v[base + 1],
                (double)vz,
                (double)a[base + 0],
                (double)a[base + 1],
                (double)az);
        fflush(file);
    }

}  // namespace

int solve_hyperelasticity_bdf2(const std::shared_ptr<sfem::Communicator> &comm, int argc, char *argv[]) {
    SFEM_TRACE_SCOPE("solve_hyperelasticity_bdf2");

    if (argc != 5) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <mesh> <dirichlet.yaml|NONE> <neumann.yaml|NONE> <output>\n", argv[0]);
        }
        return SFEM_FAILURE;
    }

    if (comm->size() > 1) {
        SFEM_ERROR("MPI runtimes are not supported by hyperelasticity_bdf2!\n");
    }

    const smesh::Path mesh_path{argv[1]};
    const smesh::Path dirichlet_path{argv[2]};
    const smesh::Path neumann_path{argv[3]};
    const smesh::Path output_path{argv[4]};

    int         refine_level  = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
    const bool  promote_to_p2 = smesh::Env::read("SFEM_PROMOTE_TO_P2", false);
    const bool  verbose       = smesh::Env::read("SFEM_VERBOSE", false);
    const char *op_name       = "GeneratedNeoHookeanOgden";
    const char *SFEM_OPERATOR = nullptr;
    SFEM_READ_ENV(SFEM_OPERATOR, );
    if (SFEM_OPERATOR) {
        op_name = SFEM_OPERATOR;
    }

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);
    if (promote_to_p2) {
        if (mesh->spatial_dimension() == 3) {
            mesh = smesh::promote_to(smesh::TET10, mesh);
        } else {
            mesh = smesh::promote_to(smesh::TRI6, mesh);
        }
    } else if (refine_level > 0) {
        mesh = smesh::to_semistructured(refine_level, mesh, true, false);
    }

    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);
    auto      f          = sfem::Function::create(fs);

    auto elastic_op = sfem::create_op(fs, op_name, sfem::EXECUTION_SPACE_HOST);
    if (!elastic_op) {
        SFEM_ERROR("Failed to create operator %s\n", op_name);
        return SFEM_FAILURE;
    }

    if (elastic_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    set_material_parameters_from_env(elastic_op, mesh);
    f->add_operator(elastic_op);

    auto inertia_op = std::make_shared<sfem::BDF2InertiaPotential>(fs);
    inertia_op->set_density(smesh::Env::read("SFEM_RHO", 1.0));
    if (inertia_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    f->add_operator(inertia_op);

    if (dirichlet_path.to_string() != "NONE") {
        auto dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
        f->add_constraint(dirichlet_conditions);
    }

    auto rotate_conds = sfem::RotateYZ::create_from_env(fs, sfem::EXECUTION_SPACE_HOST);
    if (rotate_conds) {
        f->add_constraint(rotate_conds->create_constraint());
    }

    std::shared_ptr<sfem::NeumannConditions> neumann_conditions;
    std::vector<real_t>                      neumann_base_values;
    if (neumann_path.to_string() != "NONE") {
        neumann_conditions = sfem::NeumannConditions::create_from_file(fs, neumann_path);
        neumann_base_values.reserve(neumann_conditions->conditions().size());
        for (const auto &condition : neumann_conditions->conditions()) {
            neumann_base_values.push_back(condition.value);
        }
        f->add_operator(neumann_conditions);
    }

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    auto u_nm1 = sfem::create_host_buffer<real_t>(ndofs);
    auto u_n   = sfem::create_host_buffer<real_t>(ndofs);
    auto u     = sfem::create_host_buffer<real_t>(ndofs);
    auto v_nm1 = sfem::create_host_buffer<real_t>(ndofs);
    auto v_n   = sfem::create_host_buffer<real_t>(ndofs);
    auto v     = sfem::create_host_buffer<real_t>(ndofs);
    auto a     = sfem::create_host_buffer<real_t>(ndofs);
    auto rhs   = sfem::create_host_buffer<real_t>(ndofs);
    auto incr  = sfem::create_host_buffer<real_t>(ndofs);

    auto u_hat = inertia_op->u_hat();
    f->apply_constraints(u_n->data());
    f->apply_constraints(u_nm1->data());
    f->apply_constraints(u->data());

    const real_t dt          = smesh::Env::read("SFEM_DT", 0.01);
    const real_t t_end       = smesh::Env::read("SFEM_T_END", 1.0);
    const int    n_steps     = std::max<int>(1, smesh::Env::read("SFEM_STEPS", (int)std::ceil(t_end / dt)));
    const int    export_freq = std::max<int>(1, smesh::Env::read("SFEM_EXPORT_FREQ", 1));

    const int    nl_max_it          = smesh::Env::read("SFEM_NL_MAX_IT", 30);
    const real_t nl_tol             = smesh::Env::read("SFEM_NL_TOL", 1e-9);
    const real_t lsolve_rtol        = smesh::Env::read("SFEM_LSOLVE_RTOL", 1e-3);
    const real_t newton_alpha       = smesh::Env::read("SFEM_NL_ALPHA", 1.0);
    const bool   enable_line_search = smesh::Env::read("SFEM_ENABLE_LINE_SEARCH", true);

    const std::string linear_op_type = smesh::Env::read_string("SFEM_LINEAR_OP_TYPE", sfem::op_type::BSR);
    auto              linear_op      = sfem::create_linear_operator(linear_op_type, f, u, sfem::EXECUTION_SPACE_HOST);
    auto              cg             = sfem::create_cg<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
    cg->verbose                      = verbose;
    cg->set_max_it(smesh::Env::read("SFEM_LSOLVE_MAX_IT", 20000));
    cg->set_rtol(lsolve_rtol);
    cg->set_atol(smesh::Env::read("SFEM_LSOLVE_ATOL", 1e-12));

    const bool use_preconditioner = smesh::Env::read("SFEM_USE_PRECONDITIONER", false);
    auto       diag               = sfem::create_host_buffer<real_t>(ndofs);
    auto       jacobi             = sfem::create_shiftable_jacobi(diag, sfem::EXECUTION_SPACE_HOST);
    if (use_preconditioner) {
        cg->set_preconditioner_op(jacobi);
    }

    smesh::create_directory(output_path);
    smesh::create_directory(output_path / "out");
    if (fs->has_semi_structured_mesh()) {
        smesh::semistructured_export_as_standard(fs->mesh_ptr(), output_path / "mesh");
        fs->mesh_ptr()->write(output_path / "coarse_mesh");
    } else {
        fs->mesh_ptr()->write(output_path / "mesh");
    }

    auto out = f->output();
    out->set_output_dir(output_path / "out");
    out->enable_AoS_to_SoA(true);
    out->write_time_step("disp", 0, u_n->data());
    out->write_time_step("velocity", 0, v_n->data());
    out->write_time_step("acceleration", 0, a->data());
    out->log_time(0);

    FILE             *control_csv      = nullptr;
    idx_t             control_node     = 0;
    const std::string control_csv_path = smesh::Env::read_string("SFEM_CONTROL_POINT_CSV", "");
    if (!control_csv_path.empty()) {
        geom_t target[3] = {(geom_t)smesh::Env::read("SFEM_CONTROL_POINT_X", 0.0),
                            (geom_t)smesh::Env::read("SFEM_CONTROL_POINT_Y", 0.0),
                            (geom_t)smesh::Env::read("SFEM_CONTROL_POINT_Z", 0.0)};
        control_node     = nearest_node(mesh, target);
        control_csv      = fopen(control_csv_path.c_str(), "w");
        if (!control_csv) {
            SFEM_ERROR("Unable to open control point CSV %s\n", control_csv_path.c_str());
            return SFEM_FAILURE;
        }

        fprintf(control_csv, "time,ux,uy,uz,vx,vy,vz,ax,ay,az\n");
        write_control_point(control_csv, 0, block_size, control_node, u_n->data(), v_n->data(), a->data());
        if (!comm->rank()) {
            printf("Writing control point CSV %s at node %d\n", control_csv_path.c_str(), (int)control_node);
        }
    }

    if (!comm->rank()) {
        printf("Solving BDF2 hyperelasticity: op=%s, ndofs=%td, dt=%g, steps=%d\n", op_name, ndofs, (double)dt, n_steps);
        printf("%-8s %-10s %-5s %-14s %-14s %-10s\n", "step", "newton", "cg", "gnorm", "energy", "alpha");
    }

    ptrdiff_t total_linear_iterations = 0;
    int       last_iterations         = 0;

    for (int step = 1; step <= n_steps; ++step) {
        const real_t t = step * dt;

        if (rotate_conds) {
            rotate_conds->update(step);
        }

        scale_neumann_values(neumann_conditions, neumann_base_values, load_factor(t));

        if (step == 1) {
            inertia_op->set_alpha(1 / (dt * dt));
            bdf2_predictor_be(ndofs, dt, u_n->data(), v_n->data(), u_hat->data());
        } else {
            inertia_op->set_alpha(real_t(9.0 / 4.0) / (dt * dt));
            bdf2_predictor(ndofs, dt, u_n->data(), u_nm1->data(), v_n->data(), v_nm1->data(), u_hat->data());
        }

        blas->copy(ndofs, u_n->data(), u->data());
        f->apply_constraints(u->data());

        real_t energy = 0;
        f->value(u->data(), &energy);

        for (int it = 0; it < nl_max_it; ++it) {
            f->update(u->data());
            if (use_preconditioner) {
                blas->zeros(ndofs, diag->data());
                if (f->hessian_diag(u->data(), diag->data()) != SFEM_SUCCESS) {
                    return SFEM_FAILURE;
                }
                f->set_value_to_constrained_dofs(1, diag->data());
                jacobi->set_diag(diag);
            }

            blas->zeros(ndofs, rhs->data());
            f->gradient(u->data(), rhs->data());
            f->set_value_to_constrained_dofs(0, rhs->data());

            const real_t gnorm = blas->norm2(ndofs, rhs->data());
            if (gnorm < nl_tol) {
                if (!comm->rank()) {
                    printf("%-8d %-10d %-5d %-14.4e %-14.4e %-10.4g\n", step, it, 0, (double)gnorm, (double)energy, 0.0);
                }
                break;
            }

            blas->zeros(ndofs, incr->data());
            f->copy_constrained_dofs(rhs->data(), incr->data());
            cg->set_op(linear_op);
            cg->apply(rhs->data(), incr->data());
            last_iterations = cg->iterations();
            total_linear_iterations += last_iterations;

            real_t selected_alpha = -newton_alpha;
            bool   no_progress    = false;
            if (enable_line_search) {
                std::vector<real_t> alphas{-2 * newton_alpha,
                                           -newton_alpha,
                                           real_t(-0.9) * newton_alpha,
                                           real_t(-2.0 / 3.0) * newton_alpha,
                                           real_t(-0.5) * newton_alpha,
                                           real_t(-0.25) * newton_alpha,
                                           real_t(-0.125) * newton_alpha,
                                           real_t(-1.0 / 32.0) * newton_alpha,
                                           real_t(-1.0 / 128.0) * newton_alpha,
                                           real_t(-1.0 / 256.0) * newton_alpha,
                                           real_t(-1.0 / 512.0) * newton_alpha,
                                           0};
                std::vector<real_t> energies(alphas.size(), 0);
                if (f->value_steps(u->data(), incr->data(), (int)alphas.size(), alphas.data(), energies.data()) != SFEM_SUCCESS) {
                    return SFEM_FAILURE;
                }

                const int best = std::distance(energies.begin(), std::min_element(energies.begin(), energies.end()));
                selected_alpha = alphas[best];
                energy         = energies[best];

                if (selected_alpha == 0) {
                    no_progress = true;
                }
            }

            blas->axpy(ndofs, selected_alpha, incr->data(), u->data());
            f->apply_constraints(u->data());

            if (!enable_line_search) {
                energy = 0;
                f->value(u->data(), &energy);
            }

            if (!comm->rank()) {
                printf("%-8d %-10d %-5d %-14.4e %-14.4e %-10.4g\n",
                       step,
                       it,
                       last_iterations,
                       (double)gnorm,
                       (double)energy,
                       (double)selected_alpha);
            }

            if (no_progress) break;
        }

        if (step == 1) {
            update_velocity_be(ndofs, 1 / dt, u->data(), u_n->data(), v->data());
        } else {
            update_velocity_bdf2(ndofs, 1 / (2 * dt), u->data(), u_n->data(), u_nm1->data(), v->data());
        }
        update_acceleration(ndofs, 1 / dt, v->data(), v_n->data(), a->data());

        if (step % export_freq == 0 || step == n_steps) {
            out->write_time_step("disp", t, u->data());
            out->write_time_step("velocity", t, v->data());
            out->write_time_step("acceleration", t, a->data());
            out->log_time(t);
        }

        write_control_point(control_csv, t, block_size, control_node, u->data(), v->data(), a->data());

        blas->copy(ndofs, u_n->data(), u_nm1->data());
        blas->copy(ndofs, u->data(), u_n->data());
        blas->copy(ndofs, v_n->data(), v_nm1->data());
        blas->copy(ndofs, v->data(), v_n->data());
    }

    if (!comm->rank()) {
        printf("Total linear iterations: %td\n", total_linear_iterations);
        printf("Final displacement norm: %.12e\n", (double)blas->norm2(ndofs, u_n->data()));
        printf("Final velocity norm: %.12e\n", (double)blas->norm2(ndofs, v_n->data()));
    }

    if (control_csv) {
        fclose(control_csv);
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize_serial(argc, argv);
    return solve_hyperelasticity_bdf2(ctx->communicator(), argc, argv);
}
