#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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
#include "sfem_StateField.hpp"
#include "sfem_defs.hpp"
#include "smesh_env.hpp"

#include "sfem_ssgmg.hpp"

namespace {

    struct EnvOptions {
        int         refine_level;
        bool        promote_to_p2;
        bool        verbose;
        std::string operator_name;

        bool   has_mu;
        bool   has_lambda;
        bool   has_c1;
        bool   has_c2;
        bool   has_kappa;
        real_t mu;
        real_t lambda;
        real_t c1;
        real_t c2;
        real_t kappa;

        real_t rho;
        real_t dt;
        real_t t_end;
        int    n_steps;
        int    export_freq;

        int    nl_max_it;
        real_t nl_tol;
        real_t lsolve_rtol;
        real_t newton_alpha;
        bool   enable_line_search;

        std::string linear_op_type;
        int         lsolve_max_it;
        real_t      lsolve_atol;
        bool        use_preconditioner;

        real_t      load_scale;
        real_t      load_ramp_time;
        std::string initial_displacement;
        std::string initial_displacement_components;
        std::string initial_velocity;
        std::string initial_velocity_components;

        std::string control_point_csv;
        geom_t      control_point_x;
        geom_t      control_point_y;
        geom_t      control_point_z;

        real_t      rotate_angle;
        std::string rotate_sideset;
        int         rotate_steps;
        bool        rotate_verbose;
        real_t      rotate_rcenter[3];

        static EnvOptions read() {
            EnvOptions ret{};

            ret.refine_level = smesh::Env::read("SFEM_ELEMENT_REFINE_LEVEL", 0);
            ret.promote_to_p2 = smesh::Env::read("SFEM_PROMOTE_TO_P2", false);
            ret.verbose       = smesh::Env::read("SFEM_VERBOSE", false);
            ret.operator_name = smesh::Env::read_string("SFEM_OPERATOR", "GeneratedNeoHookeanOgden");

            ret.has_mu     = std::getenv("SFEM_MU") != nullptr;
            ret.has_lambda = std::getenv("SFEM_LAMBDA") != nullptr;
            ret.has_c1     = std::getenv("SFEM_C1") != nullptr;
            ret.has_c2     = std::getenv("SFEM_C2") != nullptr;
            ret.has_kappa  = std::getenv("SFEM_KAPPA") != nullptr;
            ret.mu         = smesh::Env::read("SFEM_MU", 1.0);
            ret.lambda     = smesh::Env::read("SFEM_LAMBDA", 1.0);
            ret.c1         = smesh::Env::read("SFEM_C1", 1.0);
            ret.c2         = smesh::Env::read("SFEM_C2", 1.0);
            ret.kappa      = smesh::Env::read("SFEM_KAPPA", 1.0);

            ret.rho         = smesh::Env::read("SFEM_RHO", 1.0);
            ret.dt          = smesh::Env::read("SFEM_DT", 0.01);
            ret.t_end       = smesh::Env::read("SFEM_T_END", 1.0);
            ret.n_steps     = std::max<int>(1, smesh::Env::read("SFEM_STEPS", (int)std::ceil(ret.t_end / ret.dt)));
            ret.export_freq = std::max<int>(1, smesh::Env::read("SFEM_EXPORT_FREQ", 1));

            ret.nl_max_it          = smesh::Env::read("SFEM_NL_MAX_IT", 30);
            ret.nl_tol             = smesh::Env::read("SFEM_NL_TOL", 1e-9);
            ret.lsolve_rtol        = smesh::Env::read("SFEM_LSOLVE_RTOL", 1e-3);
            ret.newton_alpha       = smesh::Env::read("SFEM_NL_ALPHA", 1.0);
            ret.enable_line_search = smesh::Env::read("SFEM_ENABLE_LINE_SEARCH", true);

            ret.linear_op_type     = smesh::Env::read_string("SFEM_LINEAR_OP_TYPE", sfem::op_type::BSR);
            ret.lsolve_max_it      = smesh::Env::read("SFEM_LSOLVE_MAX_IT", 20000);
            ret.lsolve_atol        = smesh::Env::read("SFEM_LSOLVE_ATOL", 1e-12);
            ret.use_preconditioner = smesh::Env::read("SFEM_USE_PRECONDITIONER", false);

            ret.load_scale                      = smesh::Env::read("SFEM_LOAD_SCALE", 1.0);
            ret.load_ramp_time                  = smesh::Env::read("SFEM_LOAD_RAMP_TIME", 0.0);
            ret.initial_displacement            = smesh::Env::read_string("SFEM_INITIAL_DISPLACEMENT", "");
            ret.initial_displacement_components = smesh::Env::read_string("SFEM_INITIAL_DISPLACEMENT_COMPONENTS", "");
            ret.initial_velocity                = smesh::Env::read_string("SFEM_INITIAL_VELOCITY", "");
            ret.initial_velocity_components     = smesh::Env::read_string("SFEM_INITIAL_VELOCITY_COMPONENTS", "");

            ret.control_point_csv = smesh::Env::read_string("SFEM_CONTROL_POINT_CSV", "");
            ret.control_point_x   = (geom_t)smesh::Env::read("SFEM_CONTROL_POINT_X", 0.0);
            ret.control_point_y   = (geom_t)smesh::Env::read("SFEM_CONTROL_POINT_Y", 0.0);
            ret.control_point_z   = (geom_t)smesh::Env::read("SFEM_CONTROL_POINT_Z", 0.0);

            ret.rotate_angle      = smesh::Env::read("SFEM_ROTATE_ANGLE", 0.0);
            ret.rotate_sideset    = smesh::Env::read_string("SFEM_ROTATE_SIDESET", "");
            ret.rotate_steps      = smesh::Env::read("SFEM_ROTATE_STEPS", 10);
            ret.rotate_verbose    = smesh::Env::read("SFEM_ROTATE_VERBOSE", false);
            ret.rotate_rcenter[0] = smesh::Env::read("SFEM_ROTATE_RCENTER_X", 0.0);
            ret.rotate_rcenter[1] = smesh::Env::read("SFEM_ROTATE_RCENTER_Y", 0.0);
            ret.rotate_rcenter[2] = smesh::Env::read("SFEM_ROTATE_RCENTER_Z", 0.0);

            return ret;
        }

        void print(std::ostream &os) const {
            os << "EnvOptions:" << std::endl;
            os << "  refine_level: " << refine_level << std::endl;
            os << "  promote_to_p2: " << promote_to_p2 << std::endl;
            os << "  verbose: " << verbose << std::endl;
            os << "  operator_name: " << operator_name << std::endl;
            os << "  has_mu: " << has_mu << ", mu: " << mu << std::endl;
            os << "  has_lambda: " << has_lambda << ", lambda: " << lambda << std::endl;
            os << "  has_c1: " << has_c1 << ", c1: " << c1 << std::endl;
            os << "  has_c2: " << has_c2 << ", c2: " << c2 << std::endl;
            os << "  has_kappa: " << has_kappa << ", kappa: " << kappa << std::endl;
            os << "  rho: " << rho << std::endl;
            os << "  dt: " << dt << std::endl;
            os << "  t_end: " << t_end << std::endl;
            os << "  n_steps: " << n_steps << std::endl;
            os << "  export_freq: " << export_freq << std::endl;
            os << "  nl_max_it: " << nl_max_it << std::endl;
            os << "  nl_tol: " << nl_tol << std::endl;
            os << "  lsolve_rtol: " << lsolve_rtol << std::endl;
            os << "  newton_alpha: " << newton_alpha << std::endl;
            os << "  enable_line_search: " << enable_line_search << std::endl;
            os << "  linear_op_type: " << linear_op_type << std::endl;
            os << "  lsolve_max_it: " << lsolve_max_it << std::endl;
            os << "  lsolve_atol: " << lsolve_atol << std::endl;
            os << "  use_preconditioner: " << use_preconditioner << std::endl;
            os << "  load_scale: " << load_scale << std::endl;
            os << "  load_ramp_time: " << load_ramp_time << std::endl;
            os << "  control_point_csv: " << control_point_csv << std::endl;
            os << "  control_point: " << control_point_x << ", " << control_point_y << ", " << control_point_z << std::endl;
            os << "  rotate_angle: " << rotate_angle << std::endl;
            os << "  rotate_sideset: " << rotate_sideset << std::endl;
            os << "  rotate_steps: " << rotate_steps << std::endl;
            os << "  rotate_verbose: " << rotate_verbose << std::endl;
            os << "  rotate_rcenter: " << rotate_rcenter[0] << ", " << rotate_rcenter[1] << ", " << rotate_rcenter[2]
               << std::endl;
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

    void set_material_parameters(const EnvOptions                        &env,
                                 const std::shared_ptr<sfem::Op>        &op,
                                 const std::shared_ptr<sfem::Mesh>      &mesh) {
        if (env.has_mu) {
            set_material_parameter(op, mesh, "mu", env.mu);
        }

        if (env.has_lambda) {
            set_material_parameter(op, mesh, "lmbda", env.lambda);
        }

        if (env.has_c1) {
            set_material_parameter(op, mesh, "c1", env.c1);
        }

        if (env.has_c2) {
            set_material_parameter(op, mesh, "c2", env.c2);
        }

        if (env.has_kappa) {
            set_material_parameter(op, mesh, "kappa", env.kappa);
        }
    }

    real_t load_factor(const EnvOptions &env, const real_t t) {
        const real_t ramp_value = env.load_ramp_time > 0 ? std::min<real_t>(t / env.load_ramp_time, 1) : 1;
        return env.load_scale * ramp_value;
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

    std::shared_ptr<sfem::RotateYZ> create_rotate_conditions(const std::shared_ptr<sfem::FunctionSpace> &fs,
                                                             const EnvOptions                           &env) {
        if (env.rotate_sideset.empty()) {
            return nullptr;
        }

        if (env.rotate_verbose) {
            printf("Rotating sideset %s with angle %g\n", env.rotate_sideset.c_str(), (double)env.rotate_angle);
        }

        auto sideset = sfem::Sideset::create_from_file(fs->mesh_ptr()->comm(), smesh::Path(env.rotate_sideset));
        auto ret     = sfem::RotateYZ::create(fs, sideset, env.rotate_steps, env.rotate_angle, sfem::EXECUTION_SPACE_HOST);
        ret->verbose = env.rotate_verbose;
        ret->rcenter[0] = env.rotate_rcenter[0];
        ret->rcenter[1] = env.rotate_rcenter[1];
        ret->rcenter[2] = env.rotate_rcenter[2];
        ret->create_constraint();
        return ret;
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

    const EnvOptions env = EnvOptions::read();
    if (env.verbose && !comm->rank()) {
        env.print(std::cout);
    }

    auto mesh = sfem::Mesh::create_from_file(comm, mesh_path);
    if (env.promote_to_p2) {
        if (mesh->spatial_dimension() == 3) {
            mesh = smesh::promote_to(smesh::TET10, mesh);
        } else {
            mesh = smesh::promote_to(smesh::TRI6, mesh);
        }
    } else if (env.refine_level > 0) {
        mesh = smesh::to_semistructured(env.refine_level, mesh, true, false);
    }

    const int block_size = mesh->spatial_dimension();
    auto      fs         = sfem::FunctionSpace::create(mesh, block_size);
    auto      f          = sfem::Function::create(fs);

    auto elastic_op = sfem::create_op(fs, env.operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
    if (!elastic_op) {
        SFEM_ERROR("Failed to create operator %s\n", env.operator_name.c_str());
        return SFEM_FAILURE;
    }

    if (elastic_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    set_material_parameters(env, elastic_op, mesh);
    f->add_operator(elastic_op);

    auto inertia_op = std::make_shared<sfem::BDF2InertiaPotential>(fs);
    inertia_op->set_density(env.rho);
    if (inertia_op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    f->add_operator(inertia_op);

    std::shared_ptr<sfem::DirichletConditions> dirichlet_conditions;
    if (dirichlet_path.to_string() != "NONE") {
        dirichlet_conditions = sfem::DirichletConditions::create_from_file(fs, dirichlet_path);
        if (!dirichlet_conditions) return SFEM_FAILURE;
        f->add_constraint(dirichlet_conditions);
    }

    auto rotate_conds = create_rotate_conditions(fs, env);
    if (rotate_conds) {
        f->add_constraint(rotate_conds->create_constraint());
    }

    std::shared_ptr<sfem::NeumannConditions> neumann_conditions;
    if (neumann_path.to_string() != "NONE") {
        neumann_conditions = sfem::NeumannConditions::create_from_file(fs, neumann_path);
        if (!neumann_conditions) return SFEM_FAILURE;
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
    blas->zeros(ndofs, u_nm1->data());
    blas->zeros(ndofs, u_n->data());
    blas->zeros(ndofs, u->data());
    blas->zeros(ndofs, v_nm1->data());
    blas->zeros(ndofs, v_n->data());
    blas->zeros(ndofs, v->data());
    blas->zeros(ndofs, a->data());
    if (read_initial_state(mesh, env.initial_displacement, env.initial_displacement_components, u_n->data()) != SFEM_SUCCESS)
        return SFEM_FAILURE;
    if (read_initial_state(mesh, env.initial_velocity, env.initial_velocity_components, v_n->data()) != SFEM_SUCCESS)
        return SFEM_FAILURE;
    blas->copy(ndofs, u_n->data(), u_nm1->data());
    blas->copy(ndofs, u_n->data(), u->data());
    blas->copy(ndofs, v_n->data(), v_nm1->data());
    if (dirichlet_conditions && dirichlet_conditions->set_time(0) != SFEM_SUCCESS) return SFEM_FAILURE;
    f->apply_constraints(u_n->data());
    f->apply_constraints(u_nm1->data());
    f->apply_constraints(u->data());

    auto              linear_op      = sfem::create_linear_operator(env.linear_op_type, f, u, sfem::EXECUTION_SPACE_HOST);
    auto              cg             = sfem::create_cg<real_t>(linear_op, sfem::EXECUTION_SPACE_HOST);
    cg->verbose                      = env.verbose;
    cg->set_max_it(env.lsolve_max_it);
    cg->set_rtol(env.lsolve_rtol);
    cg->set_atol(env.lsolve_atol);

    auto diag   = sfem::create_host_buffer<real_t>(ndofs);
    auto jacobi = sfem::create_shiftable_jacobi(diag, sfem::EXECUTION_SPACE_HOST);
    if (env.use_preconditioner) {
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
    if (!env.control_point_csv.empty()) {
        geom_t target[3] = {env.control_point_x, env.control_point_y, env.control_point_z};
        control_node     = nearest_node(mesh, target);
        control_csv      = fopen(env.control_point_csv.c_str(), "w");
        if (!control_csv) {
            SFEM_ERROR("Unable to open control point CSV %s\n", env.control_point_csv.c_str());
            return SFEM_FAILURE;
        }

        fprintf(control_csv, "time,ux,uy,uz,vx,vy,vz,ax,ay,az\n");
        write_control_point(control_csv, 0, block_size, control_node, u_n->data(), v_n->data(), a->data());
        if (!comm->rank()) {
            printf("Writing control point CSV %s at node %d\n", env.control_point_csv.c_str(), (int)control_node);
        }
    }

    if (!comm->rank()) {
        printf("Solving BDF2 hyperelasticity: op=%s, ndofs=%td, dt=%g, steps=%d\n",
               env.operator_name.c_str(),
               ndofs,
               (double)env.dt,
               env.n_steps);
        printf("%-8s %-10s %-5s %-14s %-14s %-10s\n", "step", "newton", "cg", "gnorm", "energy", "alpha");
    }

    ptrdiff_t total_linear_iterations = 0;
    int       last_iterations         = 0;

    for (int step = 1; step <= env.n_steps; ++step) {
        const real_t t = step * env.dt;

        if (rotate_conds) {
            rotate_conds->update(step);
        }

        if (dirichlet_conditions && dirichlet_conditions->set_time(t) != SFEM_SUCCESS) return SFEM_FAILURE;
        if (neumann_conditions && neumann_conditions->set_time(t, load_factor(env, t)) != SFEM_SUCCESS) return SFEM_FAILURE;

        if (step == 1) {
            inertia_op->set_alpha(1 / (env.dt * env.dt));
            bdf2_predictor_be(ndofs, env.dt, u_n->data(), v_n->data(), u_hat->data());
        } else {
            inertia_op->set_alpha(real_t(9.0 / 4.0) / (env.dt * env.dt));
            bdf2_predictor(ndofs, env.dt, u_n->data(), u_nm1->data(), v_n->data(), v_nm1->data(), u_hat->data());
        }

        blas->copy(ndofs, u_n->data(), u->data());
        f->apply_constraints(u->data());

        real_t energy = 0;
        f->value(u->data(), &energy);

        for (int it = 0; it < env.nl_max_it; ++it) {
            f->update(u->data());
            if (env.use_preconditioner) {
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
            if (gnorm < env.nl_tol) {
                if (!comm->rank()) {
                    printf("%-8d %-10d %-5d %-14.4e %-14.4e %-10.4g (converged)\n",
                           step,
                           it,
                           0,
                           (double)gnorm,
                           (double)energy,
                           0.0);
                }
                break;
            }

            if (env.linear_op_type != sfem::op_type::MATRIX_FREE) {
                linear_op = sfem::create_linear_operator(env.linear_op_type, f, u, sfem::EXECUTION_SPACE_HOST);
                if (!linear_op) {
                    SFEM_ERROR("Failed to update linear operator %s\n", env.linear_op_type.c_str());
                    return SFEM_FAILURE;
                }
            }

            blas->zeros(ndofs, incr->data());
            f->copy_constrained_dofs(rhs->data(), incr->data());
            cg->set_op(linear_op);
            cg->apply(rhs->data(), incr->data());
            last_iterations = cg->iterations();
            total_linear_iterations += last_iterations;

            real_t selected_alpha = -env.newton_alpha;
            bool   no_progress    = false;
            if (env.enable_line_search) {
                std::vector<real_t> alphas{-2 * env.newton_alpha,
                                           -env.newton_alpha,
                                           real_t(-0.9) * env.newton_alpha,
                                           real_t(-2.0 / 3.0) * env.newton_alpha,
                                           real_t(-0.5) * env.newton_alpha,
                                           real_t(-0.25) * env.newton_alpha,
                                           real_t(-0.125) * env.newton_alpha,
                                           real_t(-1.0 / 32.0) * env.newton_alpha,
                                           real_t(-1.0 / 128.0) * env.newton_alpha,
                                           real_t(-1.0 / 256.0) * env.newton_alpha,
                                           real_t(-1.0 / 512.0) * env.newton_alpha,
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

            if (!env.enable_line_search) {
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

            if (no_progress) {
                fprintf(stderr, "No progress made, stopping Newton iteration\n");
                break;
            }
        }

        if (step == 1) {
            update_velocity_be(ndofs, 1 / env.dt, u->data(), u_n->data(), v->data());
        } else {
            update_velocity_bdf2(ndofs, 1 / (2 * env.dt), u->data(), u_n->data(), u_nm1->data(), v->data());
        }
        update_acceleration(ndofs, 1 / env.dt, v->data(), v_n->data(), a->data());

        if (step % env.export_freq == 0 || step == env.n_steps) {
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
