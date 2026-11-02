// The solvers in one region (sfem_SolverExec.hpp): a run over a team operator against the same run
// vector by vector (the operator behind a plain one), on a linear elasticity cube with Dirichlet
// conditions -- the same iterations and, but for the order of the sums, the same solution. Each
// run's time is printed: SFEM_TEAM_SOLVE_CELLS sizes the cube for a timing on a full socket.

#include <chrono>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_TeamOperator.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_Stationary.hpp"
#include "sfem_cg.hpp"
#include "sfem_test.hpp"
#include "smesh_env.hpp"

namespace {
    struct Problem {
        std::shared_ptr<sfem::Function>          f;
        std::shared_ptr<sfem::Operator<real_t>>  A;
        std::shared_ptr<sfem::Buffer<real_t>>    rhs;
        ptrdiff_t                                n{0};
    };

    Problem elasticity_cube(const int cells) {
        auto es   = sfem::EXECUTION_SPACE_HOST;
        auto mesh = sfem::Mesh::create_tet4_cube(sfem::Communicator::self(), cells, cells, cells, 0, 0, 0, 1, 1, 1);
        auto fs   = sfem::FunctionSpace::create(mesh, 3);
        fs->initialize_packed_mesh();
        auto f    = sfem::Function::create(fs);

        // x = 0 clamped, x = 1 pulled along x.
        const ptrdiff_t      nnodes = mesh->n_nodes();
        const geom_t *const  x      = mesh->points()->data()[0];
        std::vector<idx_t>   left, right;
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            if (x[i] < 1e-8) left.push_back((idx_t)i);
            if (x[i] > 1 - 1e-8) right.push_back((idx_t)i);
        }
        auto to_buffer = [](const std::vector<idx_t> &v) {
            auto b = sfem::create_host_buffer<idx_t>((ptrdiff_t)v.size());
            std::copy(v.begin(), v.end(), b->data());
            return b;
        };
        std::vector<sfem::DirichletConditions::Condition> conds;
        for (int c = 0; c < 3; ++c) {
            sfem::DirichletConditions::Condition clamp;
            clamp.nodeset   = to_buffer(left);
            clamp.component = c;
            conds.push_back(clamp);
        }
        sfem::DirichletConditions::Condition pull;
        pull.nodeset   = to_buffer(right);
        pull.component = 0;
        pull.value     = 0.1;
        conds.push_back(pull);
        f->add_constraint(sfem::create_dirichlet_conditions(fs, conds, es));

        auto op = sfem::create_op(fs, "GeneratedLinearElasticity", es);
        op->initialize();
        f->add_operator(op);

        Problem p;
        p.f   = f;
        p.n   = fs->n_dofs();
        p.A   = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, nullptr, es);
        p.rhs = sfem::create_host_buffer<real_t>(p.n);
        f->apply_constraints(p.rhs->data());
        return p;
    }

    std::shared_ptr<sfem::Operator<real_t>> plain(const std::shared_ptr<sfem::Operator<real_t>> &op) {
        return sfem::make_op<real_t>(
                op->rows(), op->cols(), [op](const real_t *const x, real_t *const y) { op->apply(x, y); }, sfem::EXECUTION_SPACE_HOST);
    }

    std::vector<real_t> solve(const Problem                                 &p,
                              const std::shared_ptr<sfem::Operator<real_t>> &A,
                              int                                           *iterations,
                              double                                        *seconds,
                              const std::shared_ptr<sfem::Operator<real_t>> &M = nullptr) {
        auto cg     = sfem::create_cg<real_t>(A, sfem::EXECUTION_SPACE_HOST);
        if (M) cg->set_preconditioner_op(M);
        cg->verbose = false;
        cg->set_rtol(1e-10);
        cg->set_atol(0);
        cg->set_max_it(5000);
        std::vector<real_t> x((size_t)p.n, 0);
        p.f->apply_constraints(x.data());
        const auto t0 = std::chrono::steady_clock::now();
        cg->apply(p.rhs->data(), x.data());
        *seconds    = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        *iterations = cg->iterations();
        return x;
    }
}  // namespace

int test_team_cg_matches_the_vector_by_vector_run() {
    const Problem p = elasticity_cube(smesh::Env::read<int>("SFEM_TEAM_SOLVE_CELLS", 8));
    SFEM_TEST_ASSERT(std::dynamic_pointer_cast<sfem::TeamOperator<real_t>>(p.A) != nullptr);

    // The same product behind a plain operator: CG takes its BLAS route.
    auto plain_A = plain(p.A);

    int        it_team = 0, it_plain = 0;
    double     t_team = 0, t_plain = 0;
    const auto x_plain = solve(p, plain_A, &it_plain, &t_plain);
    const auto x_team  = solve(p, p.A, &it_team, &t_team);
    SFEM_TEST_ASSERT(it_team > 10);
    SFEM_TEST_ASSERT(std::abs(it_team - it_plain) <= 1);

    double num = 0, den = 0;
    for (size_t i = 0; i < x_team.size(); ++i) {
        num += (x_team[i] - x_plain[i]) * (x_team[i] - x_plain[i]);
        den += x_plain[i] * x_plain[i];
    }
    printf("  %ld dofs: team CG %d iterations %.4f s (%.1f us/it), plain %d %.4f s (%.1f us/it), |x_team - x_plain| / |x_plain| = %.2e\n",
           (long)p.n, it_team, t_team, 1e6 * t_team / it_team, it_plain, t_plain, 1e6 * t_plain / it_plain, std::sqrt(num / den));
    SFEM_TEST_ASSERT(std::sqrt(num / den) < 1e-8);
    return SFEM_TEST_SUCCESS;
}

namespace {
    int compare(const char *what, const Problem &p, const std::shared_ptr<sfem::Operator<real_t>> &M) {
        SFEM_TEST_ASSERT(std::dynamic_pointer_cast<sfem::TeamOperator<real_t>>(M) != nullptr);
        int        it_team = 0, it_plain = 0;
        double     t_team = 0, t_plain = 0;
        const auto x_plain = solve(p, plain(p.A), &it_plain, &t_plain, plain(M));
        const auto x_team  = solve(p, p.A, &it_team, &t_team, M);
        double     num = 0, den = 0;
        for (size_t i = 0; i < x_team.size(); ++i) {
            num += (x_team[i] - x_plain[i]) * (x_team[i] - x_plain[i]);
            den += x_plain[i] * x_plain[i];
        }
        printf("  %s, %ld dofs: team PCG %d iterations %.4f s, plain %d %.4f s, |x_team - x_plain| / |x_plain| = %.2e\n",
               what, (long)p.n, it_team, t_team, it_plain, t_plain, std::sqrt(num / den));
        SFEM_TEST_ASSERT(it_team > 2);
        SFEM_TEST_ASSERT(std::abs(it_team - it_plain) <= 1);
        SFEM_TEST_ASSERT(std::sqrt(num / den) < 1e-8);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

// Preconditioned: Jacobi and a Chebyshev smoother as team operators in CG's region.
int test_team_preconditioned_cg() {
    const Problem p    = elasticity_cube(smesh::Env::read<int>("SFEM_TEAM_SOLVE_CELLS", 8));
    // The generated materials have no hessian_diag: the stored tangent's (exact in f64).
    auto diag = sfem::create_host_buffer<real_t>(p.n);
    auto zero = sfem::create_host_buffer<real_t>(p.n);
    p.f->set_inexact_options(sfem::InexactOptions::from_strings("points", "f64"));
    SFEM_TEST_ASSERT(p.f->inexact_update(zero->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(p.f->inexact_diag(zero->data(), diag->data()) == SFEM_SUCCESS);
    auto jacobi = sfem::h_shiftable_jacobi<real_t>(diag);
    if (compare("Jacobi", p, jacobi) != SFEM_TEST_SUCCESS) return SFEM_TEST_FAILURE;

    auto cheb = sfem::create_cheb3<real_t>(p.A, sfem::EXECUTION_SPACE_HOST);
    cheb->verbose = false;
    cheb->set_max_it(3);
    cheb->set_initial_guess_zero(true);
    cheb->init_with_ones();
    return compare("Chebyshev(3)", p, cheb);
}

// A Jacobi smoother's sweeps (StationaryIteration) in one region against the same sweeps over
// plain operators: no sums, so the same bits.
int test_team_stationary_iteration() {
    const Problem p    = elasticity_cube(6);
    auto          diag = sfem::create_host_buffer<real_t>(p.n);
    auto          zero = sfem::create_host_buffer<real_t>(p.n);
    p.f->set_inexact_options(sfem::InexactOptions::from_strings("points", "f64"));
    SFEM_TEST_ASSERT(p.f->inexact_update(zero->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(p.f->inexact_diag(zero->data(), diag->data()) == SFEM_SUCCESS);
    auto jacobi = sfem::h_shiftable_jacobi<real_t>(diag);
    jacobi->set_relaxation_parameter(real_t(2) / 3);

    std::vector<std::vector<real_t>> x(2, std::vector<real_t>((size_t)p.n, 0));
    for (int k = 0; k < 2; ++k) {
        auto smoother = sfem::create_stationary<real_t>(k ? p.A : plain(p.A), k ? jacobi : plain(jacobi), sfem::EXECUTION_SPACE_HOST);
        smoother->set_max_it(30);
        SFEM_TEST_ASSERT((std::dynamic_pointer_cast<sfem::TeamOperator<real_t>>(smoother)->team_capable()) == (k == 1));
        p.f->apply_constraints(x[k].data());
        smoother->apply(p.rhs->data(), x[k].data());
    }
    for (size_t i = 0; i < x[0].size(); ++i) SFEM_TEST_ASSERT(x[0][i] == x[1][i]);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_team_cg_matches_the_vector_by_vector_run);
    SFEM_RUN_TEST(test_team_preconditioned_cg);
    SFEM_RUN_TEST(test_team_stationary_iteration);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
