#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BSRBlockGaussSeidel.hpp"
#include "sfem_Function.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_mask.hpp"
#include "sfem_openmp_blas.hpp"
#include "smesh_env.hpp"
#include "smesh_mesh_reorder.hpp"

namespace {

    using namespace sfem;

    constexpr int BS = 3;

    struct SolverResult {
        const char* name{nullptr};
        int         iterations{0};
        double      time_s{0};
        double      mdofs_per_s{0};
        real_t      r0{0};
        real_t      r_final{0};
        int         ok{0};
    };

    real_t residual_norm(const std::shared_ptr<Operator<real_t>>& A,
                         const real_t* const                      b,
                         const real_t* const                      x,
                         real_t* const                            r_work,
                         real_t* const                            ax_work) {
        auto            blas = make_openmp_blas<real_t>();
        const ptrdiff_t n    = A->rows();
        blas->zeros(n, ax_work);
        A->apply(x, ax_work);
        blas->zaxpby(n, 1, b, -1, ax_work, r_work);
        return blas->norm2(n, r_work);
    }

    SolverResult time_solver(const char* const                                 name,
                             const ptrdiff_t                                   ndofs,
                             const int                                         repeat,
                             const real_t                                      r0,
                             const real_t* const                               b,
                             real_t* const                                     x,
                             real_t* const                                     r_work,
                             real_t* const                                     ax_work,
                             const std::shared_ptr<Operator<real_t>>&          A,
                             const std::function<int(const real_t*, real_t*)>& solve_once,
                             const std::function<int()>&                       get_iterations) {
        auto blas = make_openmp_blas<real_t>();

        // Warmup
        for (int w = 0; w < 1; w++) {
            blas->zeros(ndofs, x);
            solve_once(b, x);
        }

        double total = 0;
        int    it    = 0;
        int    ok    = SFEM_FAILURE;
        real_t rfin  = 0;

        for (int rr = 0; rr < repeat; rr++) {
            blas->zeros(ndofs, x);
            const double tick = smesh::time_seconds();
            ok                = solve_once(b, x);
            const double tock = smesh::time_seconds();
            total += (tock - tick);
            it   = get_iterations();
            rfin = residual_norm(A, b, x, r_work, ax_work);
        }

        const double time_s      = total / double(repeat);
        const double mdofs_per_s = (time_s > 0) ? (1e-6 * double(ndofs) * double(std::max(it, 1)) / time_s) : 0;

        return SolverResult{name, it, time_s, mdofs_per_s, r0, rfin, ok};
    }

    bool solver_enabled(const std::string& filter, const char* name) {
        if (filter.empty() || filter == "all") {
            return true;
        }
        // Exact comma-separated token match (avoid "cg" matching "cg_jacobi").
        const std::string token = name;
        std::string       rest  = filter;
        while (!rest.empty()) {
            const auto comma = rest.find(',');
            const std::string part = (comma == std::string::npos) ? rest : rest.substr(0, comma);
            if (part == token) {
                return true;
            }
            if (comma == std::string::npos) {
                break;
            }
            rest = rest.substr(comma + 1);
        }
        return false;
    }

}  // namespace

int main(int argc, char** argv) {
    auto ctx = sfem::initialize(argc, argv);

    const auto es = EXECUTION_SPACE_HOST;

    const int             SFEM_BASE_RESOLUTION = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int             SFEM_REPEAT          = smesh::Env::read("SFEM_REPEAT", 3);
    const int             SFEM_MAX_IT          = smesh::Env::read("SFEM_MAX_IT", 20000);
    const int             SFEM_SMOOTH_IT       = smesh::Env::read("SFEM_SMOOTH_IT", 40);
    const int             SFEM_BENCH_SMOOTHERS = smesh::Env::read("SFEM_BENCH_SMOOTHERS", 1);
    const real_t          SFEM_RTOL            = smesh::Env::read("SFEM_RTOL", real_t(1e-8));
    const real_t          SFEM_ATOL            = smesh::Env::read("SFEM_ATOL", real_t(1e-12));
    const smesh::ElemType SFEM_ELEM_TYPE       = smesh::Env::read("SFEM_ELEM_TYPE", smesh::ElemType::HEX8);
    auto                  SFEM_OPERATOR        = smesh::Env::read_string("SFEM_OPERATOR", std::string("LinearElasticity"));
    auto                  SFEM_OP_FORMAT       = smesh::Env::read_string("SFEM_OP_FORMAT", std::string(op_type::MATRIX_FREE));
    auto                  SFEM_SOLVERS         = smesh::Env::read_string("SFEM_SOLVERS", std::string("all"));

    const geom_t Lx   = 1;
    auto         mesh = Mesh::create_cube(Communicator::world(),
                                  SFEM_ELEM_TYPE,
                                  SFEM_BASE_RESOLUTION,
                                  SFEM_BASE_RESOLUTION,
                                  SFEM_BASE_RESOLUTION,
                                  0,
                                  0,
                                  0,
                                  Lx,
                                  1,
                                  1);

    auto sfc = smesh::SFC::create_from_env();
    sfc->reorder(*mesh);

    auto fs = FunctionSpace::create(mesh, BS);
    auto f  = Function::create(fs);
    auto op = create_op(fs, SFEM_OPERATOR, es);
    op->initialize();
    f->add_operator(op);

    auto left_ss = Sideset::create_from_selector(
            mesh, [](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool { return x > -1e-5 && x < 1e-5; });
    auto right_ss = Sideset::create_from_selector(mesh, [=](const geom_t x, const geom_t /*y*/, const geom_t /*z*/) -> bool {
        return x > (Lx - 1e-5) && x < (Lx + 1e-5);
    });

    DirichletConditions::Condition left{.sidesets = left_ss, .value = -1, .component = 0};
    DirichletConditions::Condition right0{.sidesets = right_ss, .value = 1, .component = 0};
    DirichletConditions::Condition right1{.sidesets = right_ss, .value = 0, .component = 1};
    DirichletConditions::Condition right2{.sidesets = right_ss, .value = 0, .component = 2};
    f->add_constraint(create_dirichlet_conditions(fs, {left, right0, right1, right2}, es));

    const ptrdiff_t ndofs  = fs->n_dofs();
    const ptrdiff_t nnodes = mesh->n_nodes();

    auto blas = make_openmp_blas<real_t>();
    auto x0   = create_host_buffer<real_t>(ndofs);
    blas->zeros(ndofs, x0->data());
    f->apply_constraints(x0->data());

    auto mask = create_host_buffer<mask_t>(mask_count(ndofs));
    std::memset(mask->data(), 0, size_t(mask_count(ndofs)) * sizeof(mask_t));
    f->constraints_mask(mask->data());

    // Linear operator (MF or assembled BSR/…)
    auto A = create_linear_operator(SFEM_OP_FORMAT, f, x0, es);
    if (!A) {
        fprintf(stderr, "[Error] create_linear_operator(%s) failed\n", SFEM_OP_FORMAT.c_str());
        return EXIT_FAILURE;
    }

    // Diagonals for preconditioners
    auto diag = create_host_buffer<real_t>(ndofs);
    blas->zeros(ndofs, diag->data());
    if (f->hessian_diag(x0->data(), diag->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "[Error] hessian_diag failed\n");
        return EXIT_FAILURE;
    }
    f->set_value_to_constrained_dofs(1, diag->data());

    auto B_sym6 = create_host_buffer<real_t>(nnodes * 6);
    if (f->hessian_block_diag_sym(x0->data(), B_sym6->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "[Error] hessian_block_diag_sym failed\n");
        return EXIT_FAILURE;
    }

    // Krylov PCs: overwrite inverse scaling. ShiftableJacobi defaults to damped accumulate
    // apply intended for smoothers.
    auto jacobi                   = create_inverse_diagonal_scaling(diag, es);
    auto bjacobi                  = create_shiftable_block_sym_jacobi(BS, B_sym6, mask, es);
    bjacobi->relaxation_parameter = 1;
    bjacobi->set_diag(B_sym6);

    // Separate damped copies for stationary smoothers
    auto jacobi_smooth  = create_shiftable_jacobi(diag, es);
    auto bjacobi_smooth = create_shiftable_block_sym_jacobi(BS, B_sym6, mask, es);

    // RHS: constrained lift residual b = -g(x0) with constraints applied
    auto b  = create_host_buffer<real_t>(ndofs);
    auto x  = create_host_buffer<real_t>(ndofs);
    auto r  = create_host_buffer<real_t>(ndofs);
    auto ax = create_host_buffer<real_t>(ndofs);
    blas->zeros(ndofs, b->data());
    f->gradient(x0->data(), b->data());
    blas->scal(ndofs, -1, b->data());
    f->apply_constraints(b->data());

    const real_t r0 = blas->norm2(ndofs, b->data());

    // SpMV reference
    {
        for (int w = 0; w < 2; w++) {
            blas->zeros(ndofs, ax->data());
            A->apply(b->data(), ax->data());
        }
        const double tick = smesh::time_seconds();
        for (int rr = 0; rr < SFEM_REPEAT; rr++) {
            blas->zeros(ndofs, ax->data());
            A->apply(b->data(), ax->data());
        }
        const double tock          = smesh::time_seconds();
        const double time_per_call = (tock - tick) / double(SFEM_REPEAT);
        const double mdofs_per_s   = 1e-6 * double(ndofs) / time_per_call;

        printf("\n");
        printf("Linear elasticity solver benchmark\n");
        printf("==================================\n");

        printf("\nRun setup\n");
        printf("+------------------------+--------------------------------+\n");
        printf("| %-22s | %-30s |\n", "field", "value");
        printf("+------------------------+--------------------------------+\n");
        printf("| %-22s | %-30s |\n", "operator", SFEM_OPERATOR.c_str());
        printf("| %-22s | %-30s |\n", "op_format", SFEM_OP_FORMAT.c_str());
        printf("| %-22s | %-30s |\n", "element", type_to_string(mesh->element_type(0)));
        printf("| %-22s | %30d |\n", "base_resolution", SFEM_BASE_RESOLUTION);
        printf("| %-22s | %30d |\n", "block_size", BS);
        printf("| %-22s | %30d |\n", "repeat", SFEM_REPEAT);
        printf("| %-22s | %30d |\n", "max_it", SFEM_MAX_IT);
        printf("| %-22s | %30.6e |\n", "rtol", double(SFEM_RTOL));
        printf("| %-22s | %30.6e |\n", "atol", double(SFEM_ATOL));
        printf("| %-22s | %-30s |\n", "solvers", SFEM_SOLVERS.c_str());
        printf("+------------------------+--------------------------------+\n");

        printf("\nProblem\n");
        printf("+------------------------+--------------------------------+\n");
        printf("| %-22s | %-30s |\n", "field", "value");
        printf("+------------------------+--------------------------------+\n");
        printf("| %-22s | %30td |\n", "nodes", nnodes);
        printf("| %-22s | %30td |\n", "dofs", ndofs);
        printf("| %-22s | %30.6e |\n", "r0", double(r0));
        printf("+------------------------+--------------------------------+\n");

        printf("\nOperator apply\n");
        printf("+------------+--------------+--------------+\n");
        printf("| %-10s | %12s | %12s |\n", "operator", "time_s", "MDOF/s");
        printf("+------------+--------------+--------------+\n");
        printf("| %-10s | %12.6e | %12.3f |\n", "A", time_per_call, mdofs_per_s);
        printf("+------------+--------------+--------------+\n");
    }

    std::vector<SolverResult> results;

    auto run_cg = [&](const char* name, const std::shared_ptr<Operator<real_t>>& precond) {
        if (!solver_enabled(SFEM_SOLVERS, name)) {
            return;
        }
        auto cg = create_cg<real_t>(A, es);
        cg->set_max_it(SFEM_MAX_IT);
        cg->set_rtol(SFEM_RTOL);
        cg->set_atol(SFEM_ATOL);
        cg->verbose = false;
        if (SFEM_OP_FORMAT == op_type::BSR || SFEM_OP_FORMAT == op_type::BSR_SYM) {
            cg->set_apply_overwrites_output(true);
        }
        if (precond) {
            cg->set_preconditioner_op(precond);
        }

        results.push_back(time_solver(
                name,
                ndofs,
                SFEM_REPEAT,
                r0,
                b->data(),
                x->data(),
                r->data(),
                ax->data(),
                A,
                [&](const real_t* bb, real_t* xx) { return cg->apply(bb, xx); },
                [&]() { return cg->iterations(); }));
    };

    auto run_bcgs = [&](const char* name, const std::shared_ptr<Operator<real_t>>& precond) {
        if (!solver_enabled(SFEM_SOLVERS, name)) {
            return;
        }
        auto bcgs = create_bcgs<real_t>(A, es);
        bcgs->set_max_it(SFEM_MAX_IT);
        bcgs->set_rtol(SFEM_RTOL);
        bcgs->set_atol(SFEM_ATOL);
        bcgs->verbose = false;
        if (precond) {
            bcgs->set_preconditioner_op(precond);
        }

        results.push_back(time_solver(
                name,
                ndofs,
                SFEM_REPEAT,
                r0,
                b->data(),
                x->data(),
                r->data(),
                ax->data(),
                A,
                [&](const real_t* bb, real_t* xx) { return bcgs->apply(bb, xx); },
                [&]() { return bcgs->iterations(); }));
    };

    run_cg("cg", nullptr);
    run_cg("cg_jacobi", jacobi);
    run_cg("cg_bjacobi", bjacobi);
    run_bcgs("bcgs", nullptr);
    run_bcgs("bcgs_jacobi", jacobi);

    if (SFEM_BENCH_SMOOTHERS) {
        auto run_stationary = [&](const char* name, const std::shared_ptr<Operator<real_t>>& smoother) {
            if (!solver_enabled(SFEM_SOLVERS, name)) {
                return;
            }
            auto st = create_stationary<real_t>(A, smoother, es);
            st->set_max_it(SFEM_SMOOTH_IT);
            st->verbose = false;

            results.push_back(time_solver(
                    name,
                    ndofs,
                    SFEM_REPEAT,
                    r0,
                    b->data(),
                    x->data(),
                    r->data(),
                    ax->data(),
                    A,
                    [&](const real_t* bb, real_t* xx) { return st->apply(bb, xx); },
                    [&]() { return st->iterations(); }));
        };

        run_stationary("jacobi", jacobi_smooth);
        run_stationary("bjacobi", bjacobi_smooth);

        // BGS on assembled BSR (raw), as a fixed-iteration smoother/solver
        if (solver_enabled(SFEM_SOLVERS, "bgs") && (SFEM_OP_FORMAT == op_type::BSR || SFEM_OP_FORMAT == op_type::MATRIX_FREE)) {
            auto graph  = fs->node_to_node_graph();
            auto values = create_host_buffer<real_t>(graph->nnz() * BS * BS);
            if (f->hessian_bsr(x0->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data()) == SFEM_SUCCESS) {
                auto Abr = h_bsr_spmv<count_t, idx_t, real_t>(
                        nnodes, nnodes, BS, graph->rowptr(), graph->colidx(), values, static_cast<real_t>(0));
                auto bgs = h_bsr_block_gauss_seidel(Abr);
                bgs->set_max_it(SFEM_SMOOTH_IT);
                bgs->set_symmetric(false);

                results.push_back(time_solver(
                        "bgs",
                        ndofs,
                        SFEM_REPEAT,
                        r0,
                        b->data(),
                        x->data(),
                        r->data(),
                        ax->data(),
                        A,
                        [&](const real_t* bb, real_t* xx) { return bgs->apply(bb, xx); },
                        [&]() { return SFEM_SMOOTH_IT; }));
            }
        }
    }

    printf("\nSolvers\n");
    printf("+---------------+------+--------------+--------------+--------------+--------------+--------+\n");
    printf("| %-13s | %4s | %12s | %12s | %12s | %12s | %6s |\n",
           "solver",
           "it",
           "time_s",
           "MDOF/s",
           "r_final",
           "r_final/r0",
           "ok");
    printf("+---------------+------+--------------+--------------+--------------+--------------+--------+\n");
    for (const auto& res : results) {
        const double ratio = (res.r0 > 0) ? double(res.r_final / res.r0) : 0;
        printf("| %-13s | %4d | %12.6e | %12.3f | %12.6e | %12.6e | %6s |\n",
               res.name,
               res.iterations,
               res.time_s,
               res.mdofs_per_s,
               double(res.r_final),
               ratio,
               res.ok == SFEM_SUCCESS ? "yes" : "no");
    }
    printf("+---------------+------+--------------+--------------+--------------+--------------+--------+\n");
    printf("# MDOF/s = ndofs * iterations / time  (work rate, not time-to-solution)\n");
    printf("# Krylov stop: rtol=%g atol=%g; smoothers use fixed SFEM_SMOOTH_IT=%d\n",
           double(SFEM_RTOL),
           double(SFEM_ATOL),
           SFEM_SMOOTH_IT);
    printf("# Filter with SFEM_SOLVERS=cg,cg_jacobi,... or all\n");

    return EXIT_SUCCESS;
}
