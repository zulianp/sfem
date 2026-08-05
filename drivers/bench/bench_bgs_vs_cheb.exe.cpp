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
        const real_t x0 = x[0];
        const real_t x1 = x[1];
        const real_t x2 = x[2];
        y[0]            = a[0] * x0 + a[1] * x1 + a[2] * x2;
        y[1]            = a[3] * x0 + a[4] * x1 + a[5] * x2;
        y[2]            = a[6] * x0 + a[7] * x1 + a[8] * x2;
    }

    // C = A * B  (3x3 row-major)
    SFEM_INLINE void matmul3(const real_t* const SFEM_RESTRICT a,
                             const real_t* const SFEM_RESTRICT b,
                             real_t* const SFEM_RESTRICT       c) {
        for (int i = 0; i < BS; i++) {
            const real_t* const ai = &a[i * BS];
            for (int j = 0; j < BS; j++) {
                c[i * BS + j] = ai[0] * b[j] + ai[1] * b[BS + j] + ai[2] * b[2 * BS + j];
            }
        }
    }

    // In-place Ã_ij ← S_i A_ij S_j  (symmetric Jacobi scaling absorbed into BSR values).
    // Leaves A available for BGS; call on a copy of the value array.
    void absorb_s_into_bsr(const ptrdiff_t                   nnodes,
                           const count_t* const SFEM_RESTRICT rowptr,
                           const idx_t* const SFEM_RESTRICT   colidx,
                           const real_t* const SFEM_RESTRICT  S_blocks,
                           real_t* const SFEM_RESTRICT        values) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const real_t* const Si = &S_blocks[i * 9];
            const count_t       begin = rowptr[i];
            const count_t       end   = rowptr[i + 1];
            for (count_t k = begin; k < end; k++) {
                const ptrdiff_t     j  = colidx[k];
                const real_t* const Sj = &S_blocks[j * 9];
                real_t* const       a  = &values[k * 9];

                real_t asj[9];
                matmul3(a, Sj, asj);   // Aij * Sj
                matmul3(Si, asj, a);   // Si * (Aij * Sj)
            }
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

    // Symmetric Dirichlet clamp: for each constrained component c, set row/col c of B
    // to e_c so B = blkdiag(I_c, B_ff). Then B^{-1/2} = blkdiag(I_c, B_ff^{-1/2}).
    SFEM_INLINE void clamp_constrained_components_sym(real_t B[9], const bool constrained[BS]) {
        for (int c = 0; c < BS; c++) {
            if (!constrained[c]) continue;
            for (int k = 0; k < BS; k++) {
                B[c * BS + k] = 0;
                B[k * BS + c] = 0;
            }
            B[c * BS + c] = 1;
        }
    }

    SharedBuffer<real_t> build_block_inv_sqrt(const SharedBuffer<real_t>& sym6,
                                              const mask_t* const         mask,
                                              const ptrdiff_t             nnodes) {
        auto S = create_host_buffer<real_t>(nnodes * BS * BS);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            real_t B[9];
            sym6_to_full9(&sym6->data()[i * 6], B);

            bool constrained[BS];
            for (int d = 0; d < BS; d++) {
                constrained[d] = mask_get(i * BS + d, mask) != 0;
            }
            clamp_constrained_components_sym(B, constrained);
            inv_sqrt_sym3(B, &S->data()[i * 9]);
        }
        return S;
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
                                                  const SharedBuffer<real_t>&              temp,
                                                  const ptrdiff_t                          nnodes) {
        const ptrdiff_t n = nnodes * BS;
        return make_op<real_t>(
                n,
                n,
                [=](const real_t* const x, real_t* const y) {
                    SFEM_TRACE_SCOPE("SASOp::apply");
                    apply_block_diag(nnodes, S_blocks->data(), x, y);
                    A->apply(y, temp->data());
                    apply_block_diag(nnodes, S_blocks->data(), temp->data(), y);
                },
                EXECUTION_SPACE_HOST);
    }

    struct TimedResult {
        double time_per_call{0};
        double mdofs_per_s{0};
        real_t r_norm{0};
        int    it{0};
    };

    struct TimedApply {
        double time_per_call{0};
        double mdofs_per_s{0};
    };

    real_t free_gradient_norm(const std::shared_ptr<Function>& f, const real_t* const x, real_t* const g_work) {
        auto            blas = make_openmp_blas<real_t>();
        const ptrdiff_t n    = f->space()->n_dofs();
        blas->zeros(n, g_work);
        f->gradient(x, g_work);
        return blas->norm2(n, g_work);
    }

    template <typename ApplyFn>
    TimedResult time_smoother(const int                                it,
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
            // r = b - A delta; delta += M^{-1} r
            A->apply(delta, tmp_work);
            blas->zaxpby(ndofs, 1, b, -1, tmp_work, r_work);
            apply_once(r_work, delta);
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

        return TimedResult{time_per_call, mdofs_per_s, r_norm, it};
    }

    TimedApply time_apply(const ptrdiff_t                          ndofs,
                          const int                                repeat,
                          const std::shared_ptr<Operator<real_t>>& A,
                          const real_t* const                      x,
                          real_t* const                            y) {
        for (int w = 0; w < 2; w++) {
            A->apply(x, y);
        }

        const double tick = smesh::time_seconds();
        for (int rr = 0; rr < repeat; rr++) {
            A->apply(x, y);
        }
        const double tock          = smesh::time_seconds();
        const double time_per_call = (tock - tick) / double(repeat);
        const double mdofs_per_s   = 1e-6 * double(ndofs) / time_per_call;
        return TimedApply{time_per_call, mdofs_per_s};
    }

    TimedApply time_block_diag(const ptrdiff_t                   ndofs,
                               const ptrdiff_t                   nnodes,
                               const int                         repeat,
                               const real_t* const SFEM_RESTRICT S_blocks,
                               const real_t* const SFEM_RESTRICT x,
                               real_t* const SFEM_RESTRICT       y) {
        for (int w = 0; w < 2; w++) {
            apply_block_diag(nnodes, S_blocks, x, y);
        }
        const double tick = smesh::time_seconds();
        for (int rr = 0; rr < repeat; rr++) {
            apply_block_diag(nnodes, S_blocks, x, y);
        }
        const double tock          = smesh::time_seconds();
        const double time_per_call = (tock - tick) / double(repeat);
        const double mdofs_per_s   = 1e-6 * double(ndofs) / time_per_call;
        return TimedApply{time_per_call, mdofs_per_s};
    }

}  // namespace

int main(int argc, char** argv) {
    auto ctx = sfem::initialize(argc, argv);

    const auto es = EXECUTION_SPACE_HOST;

    const int             SFEM_BASE_RESOLUTION = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
    const int             SFEM_REPEAT          = smesh::Env::read("SFEM_REPEAT", 10);
    const int             SFEM_CHEB_IT         = smesh::Env::read("SFEM_CHEB_IT", 40);
    const int             SFEM_BGS_IT          = smesh::Env::read("SFEM_BGS_IT", 40);
    const int             SFEM_BGS_SYMMETRIC   = smesh::Env::read("SFEM_BGS_SYMMETRIC", 0);
    const int             SFEM_SMOOTH_SWEEPS   = smesh::Env::read("SFEM_SMOOTH_SWEEPS", 5);
    const smesh::ElemType SFEM_ELEM_TYPE       = smesh::Env::read("SFEM_ELEM_TYPE", smesh::ElemType::HEX8);
    auto                  SFEM_OPERATOR        = smesh::Env::read_string("SFEM_OPERATOR", std::string("LinearElasticity"));

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

    auto fs = FunctionSpace::create(mesh, BS);
    auto f  = Function::create(fs);
    auto op = create_op(fs, SFEM_OPERATOR, es);
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

    auto A = h_bsr_spmv<count_t, idx_t, real_t>(
            nnodes, nnodes, BS, graph->rowptr(), graph->colidx(), values, static_cast<real_t>(0));
    // auto A = compose_constraints_op(f, A_raw);

    // Block-diagonal B (sym-6) → clamp BC components → S = B^{-1/2}
    auto B_sym6 = create_host_buffer<real_t>(nnodes * 6);
    if (f->hessian_block_diag_sym(x0->data(), B_sym6->data()) != SFEM_SUCCESS) {
        fprintf(stderr, "[Error] hessian_block_diag_sym failed\n");
        return EXIT_FAILURE;
    }
    auto S_blocks = build_block_inv_sqrt(B_sym6, mask->data(), nnodes);

    // Absorbed Ã = S A S (copy of BSR values). Chebyshev then pays one SpMV per apply_op,
    // instead of two block-diag passes + SpMV. BGS keeps the unscaled A.
    const int SFEM_SAS_ABSORB = smesh::Env::read("SFEM_SAS_ABSORB", 1);
    auto      values_sas      = create_host_buffer<real_t>(graph->nnz() * BS * BS);
    std::memcpy(values_sas->data(), values->data(), size_t(graph->nnz() * BS * BS) * sizeof(real_t));
    const double absorb_tick = smesh::time_seconds();
    absorb_s_into_bsr(nnodes, graph->rowptr()->data(), graph->colidx()->data(), S_blocks->data(), values_sas->data());
    const double absorb_time = smesh::time_seconds() - absorb_tick;

    auto A_sas = h_bsr_spmv<count_t, idx_t, real_t>(
            nnodes, nnodes, BS, graph->rowptr(), graph->colidx(), values_sas, static_cast<real_t>(0));

    auto temp     = create_host_buffer<real_t>(ndofs);
    auto SAS_comp = make_sas_op(A, S_blocks, temp, nnodes);

    // BGS on raw BSR (corrections zeroed on constrained dofs after apply)
    auto bgs = h_bsr_block_gauss_seidel(A);
    bgs->set_max_it(SFEM_BGS_IT);
    bgs->set_symmetric(SFEM_BGS_SYMMETRIC != 0);

    // Chebyshev on Ã (absorbed) or composed S A S
    auto cheb_op = SFEM_SAS_ABSORB ? A_sas : SAS_comp;
    auto cheb    = create_cheb3<real_t>(cheb_op, es);
    cheb->set_max_it(SFEM_CHEB_IT);
    cheb->verbose = false;
    cheb->set_initial_guess_zero(true);
    cheb->set_apply_overwrites_output(true);  // BSR SpMV scale_output==0
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

    const real_t r0 = free_gradient_norm(f, x0->data(), g->data());

    const auto bsr_spmv   = time_apply(ndofs, SFEM_REPEAT, A, b->data(), a_tmp->data());
    const auto s_diag     = time_block_diag(ndofs, nnodes, SFEM_REPEAT, S_blocks->data(), b->data(), a_tmp->data());
    const auto sas_comp   = time_apply(ndofs, SFEM_REPEAT, SAS_comp, b->data(), a_tmp->data());
    const auto sas_absorb = time_apply(ndofs, SFEM_REPEAT, A_sas, b->data(), a_tmp->data());

    auto apply_bgs = [&](const real_t* const rhs, real_t* const xx) { bgs->apply(rhs, xx); };

    auto apply_cheb = [&](const real_t* const rhs, real_t* const xx) {
        // r_s = S * rhs; solve Ã x_s = r_s (or S A S); physical correction += S * x_s
        apply_block_diag(nnodes, S_blocks->data(), rhs, rs->data());
        blas->zeros(ndofs, xs->data());
        cheb->apply(rs->data(), xs->data());
        apply_block_diag(nnodes, S_blocks->data(), xs->data(), corr->data());
        blas->axpy(ndofs, 1, corr->data(), xx);
    };

    const auto bgs_res = time_smoother(SFEM_BGS_IT,
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

    const auto cheb_res = time_smoother(SFEM_CHEB_IT,
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

    printf("\n");
    printf("BSR Block Gauss-Seidel vs Chebyshev/SAS\n");
    printf("=======================================\n");

    printf("\nRun setup\n");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %-30s |\n", "field", "value");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %-30s |\n", "operator", SFEM_OPERATOR.c_str());
    printf("| %-22s | %-30s |\n", "element", type_to_string(mesh->element_type(0)));
    printf("| %-22s | %30d |\n", "base_resolution", SFEM_BASE_RESOLUTION);
    printf("| %-22s | %30d |\n", "block_size", BS);
    printf("| %-22s | %30d |\n", "repeat", SFEM_REPEAT);
    printf("| %-22s | %30d |\n", "smooth_sweeps", SFEM_SMOOTH_SWEEPS);
    printf("| %-22s | %30s |\n", "bgs_symmetric", SFEM_BGS_SYMMETRIC ? "yes" : "no");
    printf("| %-22s | %30s |\n", "sas_absorb", SFEM_SAS_ABSORB ? "yes (Atilde=SAS)" : "no (compose S*A*S)");
    printf("| %-22s | %30.6e |\n", "absorb_setup_s", absorb_time);
    printf("+------------------------+--------------------------------+\n");

    printf("\nProblem\n");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %-30s |\n", "field", "value");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %30td |\n", "nodes", nnodes);
    printf("| %-22s | %30td |\n", "dofs", ndofs);
    printf("| %-22s | %30td |\n", "nnz_blocks", (ptrdiff_t)graph->nnz());
    printf("| %-22s | %30.6e |\n", "r0_free_grad", double(r0));
    printf("| %-22s | %30.6e |\n", "cheb_eig_max", double(cheb->eig_max));
    printf("+------------------------+--------------------------------+\n");

    printf("\nOperator apply\n");
    printf("+---------------+--------------+--------------+------------------+\n");
    printf("| %-13s | %12s | %12s | %-16s |\n", "operator", "time_s", "MDOF/s", "speedup_vs_SpMV");
    printf("+---------------+--------------+--------------+------------------+\n");
    printf("| %-13s | %12.6e | %12.3f | %16s |\n", "BSR SpMV", bsr_spmv.time_per_call, bsr_spmv.mdofs_per_s, "1.00x");
    printf("| %-13s | %12.6e | %12.3f | %15.2fx |\n",
           "S block-diag",
           s_diag.time_per_call,
           s_diag.mdofs_per_s,
           bsr_spmv.time_per_call / s_diag.time_per_call);
    printf("| %-13s | %12.6e | %12.3f | %15.2fx |\n",
           "SAS compose",
           sas_comp.time_per_call,
           sas_comp.mdofs_per_s,
           bsr_spmv.time_per_call / sas_comp.time_per_call);
    printf("| %-13s | %12.6e | %12.3f | %15.2fx |\n",
           "SAS absorb",
           sas_absorb.time_per_call,
           sas_absorb.mdofs_per_s,
           bsr_spmv.time_per_call / sas_absorb.time_per_call);
    printf("+---------------+--------------+--------------+------------------+\n");
    printf("# sas_compose = S*(A*(S*x)); sas_absorb = Ãx with Ãij=Si Aij Sj (one SpMV)\n");
    printf("# speedup_vs_SpMV = t_SpMV / t_op  (>1 faster than SpMV)\n");
    printf("# cheb uses %s\n", SFEM_SAS_ABSORB ? "sas_absorb" : "sas_compose");

    printf("\nSmoothers\n");
    printf("+----------+------+--------------+--------------+--------------+\n");
    printf("| %-8s | %4s | %12s | %12s | %12s |\n", "smoother", "it", "time_s", "MDOF/s", "r_grad_free");
    printf("+----------+------+--------------+--------------+--------------+\n");
    printf("| %-8s | %4d | %12.6e | %12.3f | %12.6e |\n",
           "BGS",
           bgs_res.it,
           bgs_res.time_per_call,
           bgs_res.mdofs_per_s,
           double(bgs_res.r_norm));
    printf("| %-8s | %4d | %12.6e | %12.3f | %12.6e |\n",
           "ChebSAS",
           cheb_res.it,
           cheb_res.time_per_call,
           cheb_res.mdofs_per_s,
           double(cheb_res.r_norm));
    printf("+----------+------+--------------+--------------+--------------+\n");

    printf("\nComparison\n");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %-30s |\n", "metric", "value");
    printf("+------------------------+--------------------------------+\n");
    printf("| %-22s | %30.4f |\n", "speedup_bgs_vs_cheb", speedup);
    printf("| %-22s | %-30s |\n", "speedup_meaning", ">1 means BGS faster");
    printf("+------------------------+--------------------------------+\n");

    return EXIT_SUCCESS;
}

