#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.hpp"

#include "smesh_env.hpp"

using namespace sfem;

namespace {

    std::shared_ptr<FunctionSpace> create_sshex8_space(const int base_resolution, const int refine_level) {
        auto mesh = Mesh::create_hex8_cube(
                Communicator::world(), base_resolution, base_resolution, base_resolution, 0, 0, 0, 1, 1, 1);
        mesh = smesh::to_semistructured(refine_level, mesh, true, false);
        return FunctionSpace::create(mesh, 3);
    }

    std::shared_ptr<Buffer<real_t>> make_deterministic_device_vector(const std::shared_ptr<FunctionSpace> &fs) {
        auto h = create_host_buffer<real_t>(fs->n_dofs());
        {
            geom_t **const  pts = fs->mesh().points()->data();
            const ptrdiff_t nn  = fs->n_dofs() / fs->block_size();
            for (ptrdiff_t i = 0; i < nn; ++i) {
                const real_t x       = (real_t)pts[0][i];
                const real_t y       = (real_t)pts[1][i];
                const real_t z       = (real_t)pts[2][i];
                h->data()[i * 3 + 0] = (real_t)(0.1 + x + 0.2 * y);
                h->data()[i * 3 + 1] = (real_t)(0.2 + y + 0.3 * z);
                h->data()[i * 3 + 2] = (real_t)(0.3 + z + 0.4 * x);
            }
        }
        return smesh::to_device(h);
    }

    void device_zeros(const std::shared_ptr<Buffer<real_t>> &v) { d_memset(v->data(), 0, v->size() * sizeof(real_t)); }

    int check_finite(const char *label, const std::shared_ptr<Buffer<real_t>> &v) {
        auto h = smesh::to_host(v);
        for (ptrdiff_t i = 0; i < (ptrdiff_t)h->size(); ++i) {
            if (!std::isfinite(h->data()[i])) {
                fprintf(stderr, "[Error] %s: non-finite at i=%ld value=%g\n", label, (long)i, (double)h->data()[i]);
                return SFEM_TEST_FAILURE;
            }
        }
        return SFEM_TEST_SUCCESS;
    }

    int compare_device_vectors(const char                            *label,
                               const std::shared_ptr<Buffer<real_t>> &a,
                               const std::shared_ptr<Buffer<real_t>> &b,
                               const real_t                           tol) {
        auto ha = smesh::to_host(a);
        auto hb = smesh::to_host(b);
        SFEM_TEST_ASSERT(ha->size() == hb->size());

        // Pass per entry if abs <= tol OR rel <= tol. Do not combine global max_abs
        // with global max_rel — those often come from different DOFs (near-zero inflate rel).
        real_t    max_abs   = 0;
        real_t    max_rel   = 0;
        real_t    worst_abs = 0;
        real_t    worst_rel = 0;
        ptrdiff_t worst_i   = -1;
        ptrdiff_t n_fail    = 0;

        for (ptrdiff_t i = 0; i < (ptrdiff_t)ha->size(); ++i) {
            const real_t ai = ha->data()[i];
            const real_t bi = hb->data()[i];
            if (!std::isfinite(ai) || !std::isfinite(bi)) {
                fprintf(stderr, "[Error] %s: non-finite at i=%ld a=%g b=%g\n", label, (long)i, (double)ai, (double)bi);
                return SFEM_TEST_FAILURE;
            }
            const real_t abs = std::fabs(ai - bi);
            const real_t den = std::max(std::fabs(ai), std::fabs(bi));
            const real_t rel = (den > (real_t)1e-14) ? abs / den : abs;
            max_abs          = std::max(max_abs, abs);
            max_rel          = std::max(max_rel, rel);

            if (abs > tol && rel > tol) {
                ++n_fail;
                if (abs > worst_abs) {
                    worst_abs = abs;
                    worst_rel = rel;
                    worst_i   = i;
                }
            }
        }

        if (n_fail > 0) {
            fprintf(stderr,
                    "[Error] %s: %ld entries fail abs>tol && rel>tol (tol=%g); "
                    "worst i=%ld abs=%g rel=%g; global max_abs=%g max_rel=%g\n",
                    label,
                    (long)n_fail,
                    (double)tol,
                    (long)worst_i,
                    (double)worst_abs,
                    (double)worst_rel,
                    (double)max_abs,
                    (double)max_rel);
            return SFEM_TEST_FAILURE;
        }
        return SFEM_TEST_SUCCESS;
    }

    // Match CudaContactTest / MGSDFContactTest Dirichlet setup (top-face displacement).
    int make_top_bcs(const std::shared_ptr<FunctionSpace> &fs, std::vector<DirichletConditions::Condition> &bcs) {
        auto m      = fs->mesh_ptr();
        auto top_ss = Sideset::create_from_selector(m, [](const geom_t /*x*/, const geom_t y, const geom_t /*z*/) -> bool {
            return y > (1 - 1e-5) && y < (1 + 1e-5);
        });

        if (top_ss.empty() || !top_ss[0]->parent() || top_ss[0]->parent()->size() == 0) {
            fprintf(stderr, "[Error] empty top sideset\n");
            return SFEM_TEST_FAILURE;
        }

        bcs = {{.sidesets = top_ss, .value = 0, .component = 0},
               {.sidesets = top_ss, .value = -0.05, .component = 1},
               {.sidesets = top_ss, .value = 0, .component = 2}};
        return SFEM_TEST_SUCCESS;
    }

    int solve_cg(const std::shared_ptr<FunctionSpace>              &fs,
                 const char                                        *op_name,
                 const std::vector<DirichletConditions::Condition> &bcs,
                 std::shared_ptr<Buffer<real_t>>                   &x_out) {
        SFEM_TRACE_SCOPE(op_name);
        const auto es = EXECUTION_SPACE_DEVICE;

        auto f  = Function::create(fs);
        auto op = create_op(fs, op_name, es);
        SFEM_TEST_ASSERT(op != nullptr);
        SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);
        f->add_operator(op);

        auto conds = create_dirichlet_conditions(fs, bcs, es);
        f->add_constraint(conds);

        auto x   = create_buffer<real_t>(fs->n_dofs(), es);
        auto rhs = create_buffer<real_t>(fs->n_dofs(), es);
        f->apply_constraints(x->data());
        f->apply_constraints(rhs->data());

        SFEM_TEST_ASSERT(check_finite("x after apply_constraints", x) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(check_finite("rhs after apply_constraints", rhs) == SFEM_TEST_SUCCESS);

        // Smoke-check operator apply before CG (catches NaN jacobians / kernels early).
        {
            auto y = create_buffer<real_t>(fs->n_dofs(), es);
            device_zeros(y);
            SFEM_TEST_ASSERT(f->apply(nullptr, x->data(), y->data()) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(check_finite("A*x before CG", y) == SFEM_TEST_SUCCESS);
        }

        auto linear_op = create_linear_operator(op_type::MATRIX_FREE, f, nullptr, es);
        auto cg        = create_cg(linear_op, es);
        cg->set_max_it(2000);
        cg->set_rtol(1e-4);
        cg->set_atol(1e-10);
        cg->verbose = smesh::Env::read<bool>("SFEM_SOLVER_VERBOSE", false);
        SFEM_TEST_ASSERT(cg->apply(rhs->data(), x->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(check_finite("x after CG", x) == SFEM_TEST_SUCCESS);

        x_out = x;
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_em_vector_warp_apply_vs_linear_elasticity() {
    const auto es = EXECUTION_SPACE_DEVICE;

    const int base_resolution = smesh::Env::read<int>("SFEM_BASE_RESOLUTION", 10);
    const int refine_level    = smesh::Env::read<int>("SFEM_ELEMENT_REFINE_LEVEL", 2);
    // Float GPU apply: ~1e-7 is appropriate (CudaContactTest); override with SFEM_TEST_TOL.
    const real_t tol = (real_t)smesh::Env::read<double>("SFEM_TEST_TOL", 1e-7);

    SFEM_TEST_ASSERT(refine_level >= 2);
    auto fs = create_sshex8_space(base_resolution, refine_level);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_ASSERT(fs->block_size() == 3);

    auto op_le = create_op(fs, "LinearElasticity", es);
    auto op_em = create_op(fs, "EMVectorWarpOp", es);

    SFEM_TEST_ASSERT(op_em != nullptr);
    SFEM_TEST_ASSERT(op_le != nullptr);
    SFEM_TEST_ASSERT(op_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_le->initialize() == SFEM_SUCCESS);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            x     = make_deterministic_device_vector(fs);
    auto            y_em  = create_buffer<real_t>(ndofs, es);
    auto            y_le  = create_buffer<real_t>(ndofs, es);
    device_zeros(y_em);
    device_zeros(y_le);

    SFEM_TEST_ASSERT(op_em->apply(nullptr, x->data(), y_em->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_le->apply(nullptr, x->data(), y_le->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(check_finite("apply EMVectorWarpOp", y_em) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_finite("apply LinearElasticity", y_le) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compare_device_vectors("apply EMVectorWarpOp vs LinearElasticity", y_em, y_le, tol) == SFEM_TEST_SUCCESS);

    auto diag_em = create_buffer<real_t>(ndofs, es);
    auto diag_le = create_buffer<real_t>(ndofs, es);
    device_zeros(diag_em);
    device_zeros(diag_le);
    SFEM_TEST_ASSERT(op_em->hessian_diag(nullptr, diag_em->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_le->hessian_diag(nullptr, diag_le->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(check_finite("hessian_diag EMVectorWarpOp", diag_em) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_finite("hessian_diag LinearElasticity", diag_le) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compare_device_vectors("hessian_diag EMVectorWarpOp vs LinearElasticity", diag_em, diag_le, tol) ==
                     SFEM_TEST_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_em_vector_warp_cg_vs_linear_elasticity() {
    const int    base_resolution = smesh::Env::read<int>("SFEM_BASE_RESOLUTION", 1);
    const int    refine_level    = smesh::Env::read<int>("SFEM_ELEMENT_REFINE_LEVEL", 2);
    const real_t tol             = (real_t)smesh::Env::read<double>("SFEM_TEST_TOL", 1e-6);

    SFEM_TEST_ASSERT(refine_level >= 2);
    auto                                        fs = create_sshex8_space(base_resolution, refine_level);
    std::vector<DirichletConditions::Condition> bcs;
    SFEM_TEST_ASSERT(make_top_bcs(fs, bcs) == SFEM_TEST_SUCCESS);

    std::shared_ptr<Buffer<real_t>> x_em;
    std::shared_ptr<Buffer<real_t>> x_le;
    // Solve LinearElasticity first so a setup/BC failure is not blamed on EMVectorWarpOp.
    SFEM_TEST_ASSERT(solve_cg(fs, "LinearElasticity", bcs, x_le) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(solve_cg(fs, "EMVectorWarpOp", bcs, x_em) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(compare_device_vectors("CG solution EMVectorWarpOp vs LinearElasticity", x_em, x_le, tol) ==
                     SFEM_TEST_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_em_vector_warp_apply_vs_linear_elasticity);
    SFEM_RUN_TEST(test_em_vector_warp_cg_vs_linear_elasticity);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
