#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BSRBlockGaussSeidel.hpp"
#include "sfem_Chebyshev3.hpp"
#include "sfem_Function.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_mask.hpp"
#include "sfem_openmp_blas.hpp"
#include "smesh_env.hpp"

namespace {

    using namespace sfem;

    constexpr int BS = 3;

    SFEM_INLINE void sym6_to_full9(const real_t* const SFEM_RESTRICT s6, real_t* const SFEM_RESTRICT a9) {
        // Packed upper triangle: [d00, d01, d02, d11, d12, d22]
        a9[0] = s6[0];
        a9[1] = s6[1];
        a9[2] = s6[2];
        a9[3] = s6[1];
        a9[4] = s6[3];
        a9[5] = s6[4];
        a9[6] = s6[2];
        a9[7] = s6[4];
        a9[8] = s6[5];
    }

    SFEM_INLINE void matvec3(const real_t* const SFEM_RESTRICT a,
                             const real_t* const SFEM_RESTRICT x,
                             real_t* const SFEM_RESTRICT       y) {
        for (int d1 = 0; d1 < BS; d1++) {
            real_t acc = 0;
            for (int d2 = 0; d2 < BS; d2++) {
                acc += a[d1 * BS + d2] * x[d2];
            }
            y[d1] = acc;
        }
    }

    // Jacobi eigenvalue decomposition for SPD 3x3: A = Q diag(w) Q^T, Q stored row-major.
    SFEM_INLINE void eigen_sym3(const real_t A_in[9], real_t Q[9], real_t w[3]) {
        real_t A[9];
        for (int i = 0; i < 9; i++) {
            A[i] = A_in[i];
        }

        // Q = I
        for (int i = 0; i < 9; i++) {
            Q[i] = 0;
        }
        Q[0] = Q[4] = Q[8] = 1;

        for (int iter = 0; iter < 32; iter++) {
            // Find largest off-diagonal
            int          p = 0, q = 1;
            real_t       max_abs = std::fabs(A[1]);
            const real_t a02     = std::fabs(A[2]);
            const real_t a12     = std::fabs(A[5]);
            if (a02 > max_abs) {
                max_abs = a02;
                p       = 0;
                q       = 2;
            }
            if (a12 > max_abs) {
                max_abs = a12;
                p       = 1;
                q       = 2;
            }

            if (max_abs < real_t(1e-14) * (std::fabs(A[0]) + std::fabs(A[4]) + std::fabs(A[8]) + real_t(1))) {
                break;
            }

            const real_t app = A[p * BS + p];
            const real_t aqq = A[q * BS + q];
            const real_t apq = A[p * BS + q];

            real_t c, s;
            if (std::fabs(apq) < real_t(1e-30)) {
                c = 1;
                s = 0;
            } else {
                const real_t tau = (aqq - app) / (real_t(2) * apq);
                const real_t t   = (tau >= 0) ? (real_t(1) / (tau + std::sqrt(real_t(1) + tau * tau)))
                                              : (-real_t(1) / (-tau + std::sqrt(real_t(1) + tau * tau)));
                c                = real_t(1) / std::sqrt(real_t(1) + t * t);
                s                = t * c;
            }

            // Rotate A
            for (int k = 0; k < BS; k++) {
                if (k == p || k == q) continue;
                const real_t aik = A[k * BS + p];
                const real_t aiq = A[k * BS + q];
                A[k * BS + p] = A[p * BS + k] = c * aik - s * aiq;
                A[k * BS + q] = A[q * BS + k] = s * aik + c * aiq;
            }
            A[p * BS + p] = c * c * app - real_t(2) * s * c * apq + s * s * aqq;
            A[q * BS + q] = s * s * app + real_t(2) * s * c * apq + c * c * aqq;
            A[p * BS + q] = A[q * BS + p] = 0;

            // Accumulate Q (columns are eigenvectors): Q := Q R
            for (int k = 0; k < BS; k++) {
                const real_t qkp = Q[k * BS + p];
                const real_t qkq = Q[k * BS + q];
                Q[k * BS + p]    = c * qkp - s * qkq;
                Q[k * BS + q]    = s * qkp + c * qkq;
            }
        }

        w[0] = A[0];
        w[1] = A[4];
        w[2] = A[8];
    }

    SFEM_INLINE void inv_sqrt_sym3(const real_t B[9], real_t S[9]) {
        real_t Q[9], w[3];
        eigen_sym3(B, Q, w);

        real_t inv_sqrt_w[3];
        for (int i = 0; i < BS; i++) {
            const real_t wi = std::max(w[i], real_t(1e-30));
            inv_sqrt_w[i]   = real_t(1) / std::sqrt(wi);
        }

        // S = Q diag(λ^{-1/2}) Q^T
        for (int i = 0; i < BS; i++) {
            for (int j = 0; j < BS; j++) {
                real_t acc = 0;
                for (int k = 0; k < BS; k++) {
                    acc += Q[i * BS + k] * inv_sqrt_w[k] * Q[j * BS + k];
                }
                S[i * BS + j] = acc;
            }
        }
    }

    SharedBuffer<real_t> build_block_inv_sqrt(const SharedBuffer<real_t>& sym6, const ptrdiff_t nnodes) {
        auto S = create_host_buffer<real_t>(nnodes * BS * BS);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            real_t B[9];
            sym6_to_full9(&sym6->data()[i * 6], B);
            inv_sqrt_sym3(B, &S->data()[i * 9]);
        }
        return S;
    }

    // Avoid mixing constrained and free components under the block scaling.
    void set_S_identity_on_constrained(const SharedBuffer<real_t>& S_blocks, const mask_t* const mask, const ptrdiff_t nnodes) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            real_t* const Si = &S_blocks->data()[i * 9];
            for (int d = 0; d < BS; d++) {
                if (!mask_get(i * BS + d, mask)) continue;
                for (int k = 0; k < BS; k++) {
                    Si[d * BS + k] = 0;
                    Si[k * BS + d] = 0;
                }
                Si[d * BS + d] = 1;
            }
        }
    }

    void apply_block_diag(const ptrdiff_t                   nnodes,
                          const real_t* const SFEM_RESTRICT blocks,
                          const real_t* const SFEM_RESTRICT x,
                          real_t* const SFEM_RESTRICT       y) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            matvec3(&blocks[i * 9], &x[i * BS], &y[i * BS]);
        }
    }

    std::shared_ptr<Operator<real_t>> make_sas_op(const std::shared_ptr<Operator<real_t>>& A,
                                                  const SharedBuffer<real_t>&              S_blocks,
                                                  const SharedBuffer<real_t>&              t1,
                                                  const SharedBuffer<real_t>&              t2,
                                                  const ptrdiff_t                          nnodes) {
        const ptrdiff_t n = nnodes * BS;
        return make_op<real_t>(
                n,
                n,
                [=](const real_t* const x, real_t* const y) {
                    apply_block_diag(nnodes, S_blocks->data(), x, t1->data());
                    A->apply(t1->data(), y);  // BSR with scale_output=0 zeros y then accumulates
                    apply_block_diag(nnodes, S_blocks->data(), y, t2->data());
                    std::memcpy(y, t2->data(), size_t(n) * sizeof(real_t));
                },
                EXECUTION_SPACE_HOST);
    }

    struct TimedResult {
        double time_per_call{0};
        double mdofs_per_s{0};
        real_t r_norm{0};
        int    it{0};
    };

    real_t free_gradient_norm(const std::shared_ptr<Function>& f, const real_t* const x, real_t* const g_work) {
        auto            blas = make_openmp_blas<real_t>();
        const ptrdiff_t n    = f->space()->n_dofs();
        blas->zeros(n, g_work);
        f->gradient(x, g_work);
        f->apply_zero_constraints(g_work);
        return blas->norm2(n, g_work);
    }

    template <typename ApplyFn>
    TimedResult time_smoother(const char*                              name,
                              const int                                it,
                              const ptrdiff_t                          ndofs,
                              const int                                repeat,
                              const int                                smooth_sweeps,
                              const std::shared_ptr<Function>&         f,
                              const std::shared_ptr<Operator<real_t>>& A,
                              const real_t* const                      x0,
                              const real_t* const                      b,
                              real_t* const                            delta_work,
                              real_t* const                            r_work,
                              real_t* const                            tmp_work,
                              real_t* const                            x_phys,
                              real_t* const                            g_work,
                              ApplyFn&&                                apply_once) {
        auto blas = make_openmp_blas<real_t>();

        auto residual_correct = [&](real_t* const delta) {
            // r = b - A delta (free dofs); delta += M^{-1} r
            A->apply(delta, tmp_work);
            blas->zaxpby(ndofs, 1, b, -1, tmp_work, r_work);
            f->apply_zero_constraints(r_work);
            apply_once(r_work, delta);
            f->apply_zero_constraints(delta);
        };

        // Warmup
        for (int w = 0; w < 2; w++) {
            blas->zeros(ndofs, delta_work);
            apply_once(b, delta_work);
        }

        const double tick = smesh::time_seconds();
        for (int rr = 0; rr < repeat; rr++) {
            blas->zeros(ndofs, delta_work);
            apply_once(b, delta_work);
        }
        const double tock          = smesh::time_seconds();
        const double time_per_call = (tock - tick) / double(repeat);
        const double mdofs_per_s   = 1e-6 * double(ndofs) / time_per_call;

        // Outer residual-correcting sweeps, then recover physical x = x0 + delta
        blas->zeros(ndofs, delta_work);
        for (int s = 0; s < smooth_sweeps; s++) {
            residual_correct(delta_work);
        }
        blas->copy(ndofs, x0, x_phys);
        blas->axpy(ndofs, 1, delta_work, x_phys);
        f->apply_constraints(x_phys);

        const real_t r_norm = free_gradient_norm(f, x_phys, g_work);

        printf("| %-8s | %3d | %10.6e | %10.3f | %10.6e |\n", name, it, time_per_call, mdofs_per_s, double(r_norm));

        return TimedResult{time_per_call, mdofs_per_s, r_norm, it};
    }

}  // namespace

int main(int argc, char** argv) {
    auto ctx = sfem::initialize(argc, argv);

    const auto es = EXECUTION_SPACE_HOST;

    const int SFEM_BASE_RESOLUTION = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int SFEM_REPEAT          = smesh::Env::read("SFEM_REPEAT", 10);
    const int SFEM_CHEB_IT         = smesh::Env::read("SFEM_CHEB_IT", 40);
    const int SFEM_BGS_IT          = smesh::Env::read("SFEM_BGS_IT", 40);
    const int SFEM_BGS_SYMMETRIC   = smesh::Env::read("SFEM_BGS_SYMMETRIC", 0);
    const int SFEM_SMOOTH_SWEEPS   = smesh::Env::read("SFEM_SMOOTH_SWEEPS", 5);

    const geom_t Lx   = 1;
    auto         mesh = Mesh::create_hex8_cube(
            Communicator::world(), SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, SFEM_BASE_RESOLUTION, 0, 0, 0, Lx, 1, 1);

    auto fs = FunctionSpace::create(mesh, BS);
    auto f  = Function::create(fs);
    auto op = create_op(fs, "LinearElasticity", es);
    op->initialize();
    f->add_operator(op);

    // Dirichlet: left/right faces (same as SSGMG LE cube)
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

    // Assemble BSR at constrained state
    auto graph  = fs->node_to_node_graph();
    auto values = create_host_buffer<real_t>(graph->nnz() * BS * BS);
    if (f->hessian_bsr(x0->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "[Error] hessian_bsr failed\n");
        return EXIT_FAILURE;
    }

    auto A_raw = h_bsr_spmv<count_t, idx_t, real_t>(
            nnodes, nnodes, BS, graph->rowptr(), graph->colidx(), values, static_cast<real_t>(0));
    auto A = compose_constraints_op(f, A_raw);

    // Block-diagonal B (sym-6) → S = B^{-1/2}, identity on constrained dofs
    auto B_sym6 = create_host_buffer<real_t>(nnodes * 6);
    if (f->hessian_block_diag_sym(x0->data(), B_sym6->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "[Error] hessian_block_diag_sym failed\n");
        return EXIT_FAILURE;
    }
    auto S_blocks = build_block_inv_sqrt(B_sym6, nnodes);
    set_S_identity_on_constrained(S_blocks, mask->data(), nnodes);

    auto t1  = create_host_buffer<real_t>(ndofs);
    auto t2  = create_host_buffer<real_t>(ndofs);
    auto SAS = make_sas_op(A, S_blocks, t1, t2, nnodes);

    // BGS on raw BSR (corrections zeroed on constrained dofs after apply)
    auto bgs = h_bsr_block_gauss_seidel(A_raw);
    bgs->set_max_it(SFEM_BGS_IT);
    bgs->set_symmetric(SFEM_BGS_SYMMETRIC != 0);

    // Chebyshev on S A S
    auto cheb = create_cheb3<real_t>(SAS, es);
    cheb->set_max_it(SFEM_CHEB_IT);
    cheb->verbose = false;
    cheb->set_initial_guess_zero(true);
    cheb->init_with_ones();

    // Free residual at Dirichlet lift: b = -P g(x0)
    auto b      = create_host_buffer<real_t>(ndofs);
    auto delta  = create_host_buffer<real_t>(ndofs);
    auto x      = create_host_buffer<real_t>(ndofs);
    auto g      = create_host_buffer<real_t>(ndofs);
    auto xs     = create_host_buffer<real_t>(ndofs);
    auto rs     = create_host_buffer<real_t>(ndofs);
    auto corr   = create_host_buffer<real_t>(ndofs);
    auto r_free = create_host_buffer<real_t>(ndofs);
    auto a_tmp  = create_host_buffer<real_t>(ndofs);

    blas->zeros(ndofs, b->data());
    f->gradient(x0->data(), b->data());
    blas->scal(ndofs, -1, b->data());
    f->apply_zero_constraints(b->data());

    const real_t r0 = free_gradient_norm(f, x0->data(), g->data());

    printf("ndofs=%td nnodes=%td nnz_blocks=%td\n", ndofs, nnodes, (ptrdiff_t)graph->nnz());
    printf("r0_free_grad=%g  cheb_eig_max=%g (scaled SAS)\n", double(r0), double(cheb->eig_max));
    printf("| smoother |  it |     time_s |    MDOF/s | r_grad_free |\n");
    printf("|----------|-----|------------|-----------|-------------|\n");

    auto apply_bgs = [&](const real_t* const rhs, real_t* const xx) {
        bgs->apply(rhs, xx);
        f->apply_zero_constraints(xx);
    };

    auto apply_cheb = [&](const real_t* const rhs, real_t* const xx) {
        // r_s = S * rhs; solve SAS x_s = r_s; physical correction += S * x_s
        apply_block_diag(nnodes, S_blocks->data(), rhs, rs->data());
        f->apply_zero_constraints(rs->data());
        blas->zeros(ndofs, xs->data());
        cheb->apply(rs->data(), xs->data());
        f->apply_zero_constraints(xs->data());
        apply_block_diag(nnodes, S_blocks->data(), xs->data(), corr->data());
        f->apply_zero_constraints(corr->data());
        blas->axpy(ndofs, 1, corr->data(), xx);
    };

    const auto bgs_res = time_smoother("BGS",
                                       SFEM_BGS_IT,
                                       ndofs,
                                       SFEM_REPEAT,
                                       SFEM_SMOOTH_SWEEPS,
                                       f,
                                       A,
                                       x0->data(),
                                       b->data(),
                                       delta->data(),
                                       r_free->data(),
                                       a_tmp->data(),
                                       x->data(),
                                       g->data(),
                                       apply_bgs);

    const auto cheb_res = time_smoother("ChebSAS",
                                        SFEM_CHEB_IT,
                                        ndofs,
                                        SFEM_REPEAT,
                                        SFEM_SMOOTH_SWEEPS,
                                        f,
                                        A,
                                        x0->data(),
                                        b->data(),
                                        delta->data(),
                                        r_free->data(),
                                        a_tmp->data(),
                                        x->data(),
                                        g->data(),
                                        apply_cheb);

    const double speedup = cheb_res.time_per_call / bgs_res.time_per_call;
    printf("speedup_bgs_vs_cheb=%.4f  ( >1 means BGS faster )\n", speedup);
    printf("r_grad_bgs=%g r_grad_cheb=%g (free dofs via f->gradient)\n", double(bgs_res.r_norm), double(cheb_res.r_norm));

    return EXIT_SUCCESS;
}
