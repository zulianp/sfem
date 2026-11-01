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

    std::vector<real_t> solve(const Problem &p, const std::shared_ptr<sfem::Operator<real_t>> &A, int *iterations, double *seconds) {
        auto cg     = sfem::create_cg<real_t>(A, sfem::EXECUTION_SPACE_HOST);
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
    auto A     = p.A;
    auto plain = sfem::make_op<real_t>(
            p.n, p.n, [A](const real_t *const x, real_t *const y) { A->apply(x, y); }, sfem::EXECUTION_SPACE_HOST);

    int        it_team = 0, it_plain = 0;
    double     t_team = 0, t_plain = 0;
    const auto x_plain = solve(p, plain, &it_plain, &t_plain);
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

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_team_cg_matches_the_vector_by_vector_run);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
