#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_DirichletConditions.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

namespace {

    struct BoundaryNodes {
        sfem::SharedBuffer<idx_t> left;
        sfem::SharedBuffer<idx_t> right;
    };

    BoundaryNodes create_x_boundary_nodes(const std::shared_ptr<sfem::Mesh> &mesh) {
        const ptrdiff_t nnodes = mesh->n_nodes();
        geom_t **const  points = mesh->points()->data();

        geom_t xmin = points[0][0];
        geom_t xmax = points[0][0];

#pragma omp parallel for reduction(min : xmin) reduction(max : xmax)
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            xmin = std::min(xmin, points[0][i]);
            xmax = std::max(xmax, points[0][i]);
        }

        const geom_t tolerance = std::max<geom_t>((xmax - xmin) * 1e-8, 1e-12);
        ptrdiff_t    nleft     = 0;
        ptrdiff_t    nright    = 0;

#pragma omp parallel for reduction(+ : nleft, nright)
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            nleft += std::abs(points[0][i] - xmin) <= tolerance;
            nright += std::abs(points[0][i] - xmax) <= tolerance;
        }

        auto left  = sfem::create_host_buffer<idx_t>(nleft);
        auto right = sfem::create_host_buffer<idx_t>(nright);

        ptrdiff_t left_offset  = 0;
        ptrdiff_t right_offset = 0;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            const geom_t x = points[0][i];
            if (std::abs(x - xmin) <= tolerance) {
                left->data()[left_offset++] = i;
            }

            if (std::abs(x - xmax) <= tolerance) {
                right->data()[right_offset++] = i;
            }
        }

        return {left, right};
    }

    void print_rate(const char *name, const double elapsed, const ptrdiff_t nelements, const ptrdiff_t ndofs, const int repeat) {
        const double seconds_per_call = elapsed / repeat;
        const double melements_per_s  = 1e-6 * static_cast<double>(nelements) / seconds_per_call;
        const double mdofs_per_s      = 1e-6 * static_cast<double>(ndofs) / seconds_per_call;

        printf("%-72s %12.6e %16.3f %13.3f %10s %13s\n", name, seconds_per_call, melements_per_s, mdofs_per_s, "-", "-");
    }

    bool generated_neohookean_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
            case smesh::HEX27:
            case smesh::PROTEUS_HEX8:
            case smesh::PROTEUS_HEX27:
            case smesh::PROTEUS_HEX64:
            case smesh::PROTEUS_HEX125:
            case smesh::PROTEUS_HEX729:
                return true;
            default:
                return false;
        }
    }

    bool baseline_neohookean_supported(const smesh::ElemType element_type) {
        switch (element_type) {
            case smesh::TET4:
            case smesh::TET10:
            case smesh::HEX8:
                return true;
            default:
                return false;
        }
    }

    void set_neohookean_geometry_options(const std::shared_ptr<sfem::Op> &op, const bool assume_affine) {
        op->set_option("assume_affine", assume_affine);
        op->set_option("ASSUME_AFFINE", assume_affine);
    }

    void set_neohookean_material(const std::shared_ptr<sfem::Op> &op, const real_t mu, const real_t lambda) {
        op->set_value_in_block("default", "mu", mu);
        op->set_value_in_block("default", "lambda", lambda);
        op->set_value_in_block("default", "lmbda", lambda);
    }

    void configure_initialized_neohookean_op(const std::shared_ptr<sfem::Op> &op,
                                             const real_t                     mu,
                                             const real_t                     lambda,
                                             const bool                       assume_affine) {
        set_neohookean_geometry_options(op, assume_affine);
        set_neohookean_material(op, mu, lambda);
    }

    double time_gradient(const std::shared_ptr<sfem::Function>         &f,
                         const real_t *const                            x,
                         real_t *const                                  out,
                         const ptrdiff_t                                ndofs,
                         const int                                      repeat,
                         const std::shared_ptr<sfem::BLAS_Tpl<real_t>> &blas) {
        blas->zeros(ndofs, out);
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            f->gradient(x, out);
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

    double time_apply(const std::shared_ptr<sfem::Operator<real_t>> &op,
                      const real_t *const                            direction,
                      real_t *const                                  out,
                      const ptrdiff_t                                ndofs,
                      const int                                      repeat,
                      const std::shared_ptr<sfem::BLAS_Tpl<real_t>> &blas) {
        blas->zeros(ndofs, out);
        sfem::device_synchronize();
        const double t0 = MPI_Wtime();
        for (int i = 0; i < repeat; ++i) {
            op->apply(direction, out);
        }
        sfem::device_synchronize();
        return MPI_Wtime() - t0;
    }

}  // namespace

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    auto          comm = context.communicator();

    if (comm->size() != 1) {
        SFEM_ERROR("bench_hyperelasticity.exe supports one MPI rank\n");
    }

    const int         resolution              = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int         warmup                  = smesh::Env::read("SFEM_WARMUP", 3);
    const int         repeat                  = smesh::Env::read("SFEM_REPEAT", 10);
    const int         nl_max_it               = smesh::Env::read("SFEM_NL_MAX_IT", 10);
    const int         linear_max_it           = smesh::Env::read("SFEM_LSOLVE_MAX_IT", 1000);
    const real_t      linear_rtol             = smesh::Env::read("SFEM_LSOLVE_RTOL", 1e-6);
    const real_t      nonlinear_tol           = smesh::Env::read("SFEM_NL_TOL", 1e-9);
    const real_t      displacement_value      = smesh::Env::read("SFEM_DISPLACEMENT", 0.05);
    const real_t      damping                 = smesh::Env::read("SFEM_NL_ALPHA", 1.0);
    const int         line_search_steps       = smesh::Env::read("SFEM_NL_LINE_SEARCH_STEPS", 20);
    const real_t      mu                      = smesh::Env::read("SFEM_SHEAR_MODULUS", 1.0);
    const real_t      lambda                  = smesh::Env::read("SFEM_FIRST_LAME_PARAMETER", 1.0);
    const std::string generated_operator_name = smesh::Env::read_string("SFEM_GENERATED_OPERATOR", "GeneratedNeoHookeanOgden");
    const std::string baseline_operator_name  = smesh::Env::read_string("SFEM_BASELINE_OPERATOR", "NeoHookeanOgden");
    const std::string codegen_geometry        = smesh::Env::read_string("SFEM_CODEGEN_GEOMETRY", "isoparametric");
    const std::string output_path             = smesh::Env::read_string("SFEM_OUTPUT_PATH", "");
    const bool        run_baseline_requested  = smesh::Env::read("SFEM_RUN_BASELINE", true);

    const auto element_type = smesh::type_from_string(smesh::Env::read_string("SFEM_ELEM_TYPE", "HEX8").c_str());
    auto       mesh         = sfem::Mesh::create_cube(
            comm, static_cast<smesh::ElemType>(element_type), resolution, resolution, resolution, 0, 0, 0, 1, 1, 1);

    const int block_size = mesh->spatial_dimension();
    if (block_size != 3) {
        SFEM_ERROR("bench_hyperelasticity.exe requires a three-dimensional mesh\n");
    }
    if (!generated_neohookean_supported(mesh->element_type(0))) {
        SFEM_ERROR("generated solve path does not support SFEM_ELEM_TYPE=%s\n", type_to_string(mesh->element_type(0)));
    }
    if (codegen_geometry != "affine" && codegen_geometry != "isoparametric") {
        SFEM_ERROR("SFEM_CODEGEN_GEOMETRY must be affine or isoparametric\n");
    }
    const bool assume_affine = codegen_geometry == "affine";

    auto                      fs          = sfem::FunctionSpace::create(mesh, block_size);
    auto                      generated_f = sfem::Function::create(fs);
    std::shared_ptr<sfem::Op> generated_op;
    if (generated_operator_name == "GeneratedNeoHookeanOgden") {
        generated_op = std::make_shared<sfem::GeneratedNeoHookeanOgden>(fs);
        set_neohookean_geometry_options(generated_op, assume_affine);
        if (generated_op->initialize() != SFEM_SUCCESS) {
            SFEM_ERROR("Unable to initialize generated operator %s\n", generated_operator_name.c_str());
        }
        set_neohookean_material(generated_op, mu, lambda);
    } else {
        generated_op = sfem::create_op(fs, generated_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!generated_op) {
            SFEM_ERROR("Unable to create generated operator %s\n", generated_operator_name.c_str());
        }
        configure_initialized_neohookean_op(generated_op, mu, lambda, assume_affine);
    }
    generated_f->add_operator(generated_op);

    const bool                      run_baseline = run_baseline_requested && baseline_neohookean_supported(mesh->element_type(0));
    std::shared_ptr<sfem::Function> baseline_f;
    if (run_baseline) {
        baseline_f       = sfem::Function::create(fs);
        auto baseline_op = sfem::create_op(fs, baseline_operator_name.c_str(), sfem::EXECUTION_SPACE_HOST);
        if (!baseline_op) {
            SFEM_ERROR("Unable to create baseline operator %s\n", baseline_operator_name.c_str());
        }
        configure_initialized_neohookean_op(baseline_op, mu, lambda, assume_affine);
        baseline_f->add_operator(baseline_op);
    }

    const BoundaryNodes                               boundary = create_x_boundary_nodes(mesh);
    std::vector<sfem::DirichletConditions::Condition> conditions;
    conditions.reserve(4);
    for (int component = 0; component < block_size; ++component) {
        conditions.push_back({.nodeset = boundary.left, .value = 0, .component = component});
    }
    conditions.push_back({.nodeset = boundary.right, .value = displacement_value, .component = 0});
    auto dirichlet = sfem::DirichletConditions::create(fs, conditions);
    generated_f->add_constraint(dirichlet);
    if (baseline_f) {
        baseline_f->add_constraint(dirichlet);
    }

    const ptrdiff_t     nelements = mesh->n_elements();
    const ptrdiff_t     ndofs     = fs->n_dofs();
    auto                x         = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto                rhs       = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto                increment = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto                trial     = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto                output    = sfem::create_buffer<real_t>(ndofs, sfem::EXECUTION_SPACE_HOST);
    auto                blas      = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
    std::vector<real_t> line_search_alphas(std::max(line_search_steps, 1));
    std::vector<real_t> line_search_values(std::max(line_search_steps, 1));
    if (line_search_steps > 0) {
        real_t alpha = -damping;
        for (int s = 0; s < line_search_steps; ++s) {
            line_search_alphas[s] = alpha;
            alpha *= static_cast<real_t>(0.5);
        }
    }

    blas->zeros(ndofs, x->data());
    generated_f->apply_constraints(x->data());

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        trial->data()[i] = static_cast<real_t>((i % 97) + 1) / 97;
    }
    generated_f->apply_zero_constraints(trial->data());

    generated_f->update(x->data());
    auto generated_linear_op = sfem::create_linear_operator("MF", generated_f, x, sfem::EXECUTION_SPACE_HOST);

    std::shared_ptr<sfem::Operator<real_t>> baseline_linear_op;
    if (baseline_f) {
        baseline_f->update(x->data());
        baseline_linear_op = sfem::create_linear_operator("MF", baseline_f, x, sfem::EXECUTION_SPACE_HOST);
    }

    for (int i = 0; i < warmup; ++i) {
        blas->zeros(ndofs, rhs->data());
        generated_f->gradient(x->data(), rhs->data());
        blas->zeros(ndofs, output->data());
        generated_linear_op->apply(trial->data(), output->data());
        if (baseline_f) {
            blas->zeros(ndofs, rhs->data());
            baseline_f->gradient(x->data(), rhs->data());
            blas->zeros(ndofs, output->data());
            baseline_linear_op->apply(trial->data(), output->data());
        }
    }

    const double generated_gradient_elapsed = time_gradient(generated_f, x->data(), rhs->data(), ndofs, repeat, blas);
    const double generated_apply_elapsed    = time_apply(generated_linear_op, trial->data(), output->data(), ndofs, repeat, blas);

    double baseline_gradient_elapsed = 0;
    double baseline_apply_elapsed    = 0;
    if (baseline_f) {
        baseline_gradient_elapsed = time_gradient(baseline_f, x->data(), rhs->data(), ndofs, repeat, blas);
        baseline_apply_elapsed    = time_apply(baseline_linear_op, trial->data(), output->data(), ndofs, repeat, blas);
    }

    printf("generated_operator %s\n", generated_operator_name.c_str());
    printf("baseline_operator %s\n", baseline_f ? baseline_operator_name.c_str() : "disabled");
    printf("geometry %s\n", codegen_geometry.c_str());
    printf("element_type %s\n", type_to_string(mesh->element_type(0)));
    printf("#elements %ld\n", static_cast<long>(nelements));
    printf("#nodes %ld\n", static_cast<long>(mesh->n_nodes()));
    printf("#dofs %ld\n", static_cast<long>(ndofs));
    printf("#left_nodes %ld\n", static_cast<long>(boundary.left->size()));
    printf("#right_nodes %ld\n", static_cast<long>(boundary.right->size()));
    printf("\n%-72s %12s %16s %13s %10s %13s\n",
           "Operation",
           "Time [s]",
           "Rate [MElem/s]",
           "Rate [MDOF/s]",
           "AI",
           "Rate [GFLOP/s]");
    printf("---------------------------------------------------------------------------------------------------------------------"
           "-------------------------\n");
    print_rate("generated_gradient", generated_gradient_elapsed, nelements, ndofs, repeat);
    print_rate("generated_hessian_apply", generated_apply_elapsed, nelements, ndofs, repeat);
    if (baseline_f) {
        print_rate("baseline_gradient", baseline_gradient_elapsed, nelements, ndofs, repeat);
        print_rate("baseline_hessian_apply", baseline_apply_elapsed, nelements, ndofs, repeat);
    } else if (run_baseline_requested) {
        printf("baseline_skipped unsupported_element %s\n", type_to_string(mesh->element_type(0)));
    }

    const auto generated_gradient = [&](const real_t *const state, real_t *const out) {
        blas->zeros(ndofs, out);
        generated_f->gradient(state, out);
    };

    auto cg = sfem::create_cg<real_t>(generated_linear_op, sfem::EXECUTION_SPACE_HOST);
    cg->set_max_it(linear_max_it);
    cg->set_rtol(linear_rtol);
    cg->set_atol(1e-12);
    cg->verbose = false;

    printf("\n%-10s %-8s %-14s %-12s %-14s %-12s\n", "Newton", "CG", "Residual", "Time [s]", "Rate [MDOF/s]", "Step");
    printf("------------------------------------------------------------------------------\n");
    printf("solve_operator %s_%s\n", generated_operator_name.c_str(), codegen_geometry.c_str());

    int       completed_newton = 0;
    ptrdiff_t total_cg_it      = 0;
    double    solve_t0         = MPI_Wtime();
    for (int i = 0; i < nl_max_it; ++i) {
        const double iteration_t0 = MPI_Wtime();
        generated_gradient(x->data(), rhs->data());

        const real_t residual = blas->norm2(ndofs, rhs->data());
        if (residual < nonlinear_tol) {
            printf("%-10d %-8d %-14.4e %-12.4e %-14.3f %-12.4e\n", i, 0, residual, 0.0, 0.0, 0.0);
            completed_newton = i;
            break;
        }

        blas->zeros(ndofs, increment->data());
        generated_f->copy_constrained_dofs(rhs->data(), increment->data());
        cg->apply(rhs->data(), increment->data());

        const int cg_it = cg->iterations();
        total_cg_it += cg_it;

        real_t step = -damping;
        if (line_search_steps > 0) {
            std::fill(line_search_values.begin(), line_search_values.begin() + line_search_steps, real_t(0));
            if (generated_f->value_steps(
                        x->data(), increment->data(), line_search_steps, line_search_alphas.data(), line_search_values.data()) !=
                SFEM_SUCCESS) {
                SFEM_ERROR("Generated value_steps failed during Newton line search\n");
            }

            real_t best_value = std::numeric_limits<real_t>::infinity();
            int    best_step  = -1;
            for (int s = 0; s < line_search_steps; ++s) {
                const real_t value = line_search_values[s];
                if (std::isfinite(value) && value < best_value) {
                    best_value = value;
                    best_step  = s;
                }
            }

            if (best_step < 0) {
                SFEM_ERROR("Generated value_steps returned no finite Newton line-search candidate\n");
            }

            step = line_search_alphas[best_step];
        }

        blas->axpy(ndofs, step, increment->data(), x->data());

        const double iteration_time = MPI_Wtime() - iteration_t0;
        const double rate_m         = 1e-6 * static_cast<double>(ndofs) / iteration_time;
        printf("%-10d %-8d %-14.4e %-12.4e %-14.3f %-12.4e\n", i, cg_it, residual, iteration_time, rate_m, step);
        completed_newton = i + 1;
    }
    const double solve_time = MPI_Wtime() - solve_t0;

    generated_gradient(x->data(), rhs->data());
    const real_t final_residual = blas->norm2(ndofs, rhs->data());

    printf("\nnewton_iterations %d\n", completed_newton);
    printf("cg_iterations %ld\n", static_cast<long>(total_cg_it));
    printf("solve_time %g [s]\n", solve_time);
    printf("solve_rate %g [MDOF/s]\n", 1e-6 * static_cast<double>(ndofs) / solve_time);
    printf("final_residual %g\n", static_cast<double>(final_residual));

    if (!output_path.empty()) {
        const smesh::Path path(output_path);
        smesh::create_directory(path);
        if (fs->has_semi_structured_mesh()) {
            smesh::semistructured_export_as_standard(mesh, path / "mesh");
            mesh->write(path / "coarse_mesh");
        } else {
            mesh->write(path / "mesh");
        }

        auto out = generated_f->output();
        smesh::create_directory(path / "out");
        out->set_output_dir(path / "out");
        out->enable_AoS_to_SoA(true);
        out->write("disp", smesh::to_host(x)->data());
        out->write("rhs", smesh::to_host(rhs)->data());
    }

    return std::isfinite(final_residual) ? SFEM_SUCCESS : SFEM_FAILURE;
}
