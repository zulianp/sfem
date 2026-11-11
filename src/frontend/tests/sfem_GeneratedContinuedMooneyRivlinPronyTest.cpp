// The Mooney-Rivlin-Prony operator whose powers of J are continued below Jc.
//
// `GeneratedContinuedMooneyRivlinProny` is `GeneratedMooneyRivlinProny` with J^(-2/3) and
// J^(-4/3) replaced by h_m(J) (`sfem_continued_power` in the kernel math header): J^(-m) for
// J >= Jc, a quartic C2 continuation for Jmin < J < Jc, constant below Jmin.  The gates:
//
//   1. h_m itself: J^(-m) on J >= Jc, C2 at Jc and at Jmin, its derivatives those of its value.
//   2. Where nothing is compressed below Jc the two operators are one law: value, gradient,
//      apply and assembled tangent agree to round-off after the same committed history, on
//      every 3D element they are generated for.
//   3. Through inversion: on a mesh with elements at J < 0 and between 0 and Jc, the original's
//      energy is not finite and the continued one is, and its value, gradient, apply and tangent
//      stay consistent with each other.
//   4. A node pushed through the opposite face, inverting its elements, returns to the stress
//      free state under Newton's method, where the original cannot evaluate the start.

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BSR.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedContinuedMooneyRivlinProny.hpp"
#include "sfem_GeneratedMooneyRivlinProny.hpp"
#include "sfem_test.hpp"

#include "kernel_math.hpp"

namespace {

    const real_t test_C10  = 0.3;
    const real_t test_C01  = 0.05;
    const real_t test_K    = 50.0;
    const real_t test_dt   = 0.1;
    const real_t test_Jc   = 0.2;
    const real_t test_Jmin = -1.0;

    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x, uint64_t state = 0x9e3779b97f4a7c15ull) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x[i] = scale * (real_t)((double)(state >> 11) / (double)(1ull << 53) - 0.5);
        }
    }

    real_t relative_difference(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t diff = 0, norm = 0;
        for (ptrdiff_t i = 0; i < n; i++) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
            norm = std::max(norm, std::abs(b[i]));
        }
        return norm > 0 ? diff / norm : diff;
    }

    bool all_finite(const ptrdiff_t n, const real_t *const a) {
        for (ptrdiff_t i = 0; i < n; i++) {
            if (!std::isfinite(a[i])) return false;
        }
        return true;
    }

    /// The unit cube in `n` cells per side, its interior distorted; TET10 and HEX27 promoted from
    /// the distorted linear mesh, so their elements stay straight sided.
    std::shared_ptr<sfem::Mesh> unit_cube(const smesh::ElemType element, const int n, const real_t distortion) {
        auto       comm = sfem::Communicator::world();
        const bool hex  = element == smesh::HEX8 || element == smesh::HEX27;
        auto       mesh = hex ? sfem::Mesh::create_hex8_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1)
                              : sfem::Mesh::create_tet4_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
        if (distortion > 0) {
            const ptrdiff_t     nnodes = mesh->n_nodes();
            auto                points = mesh->points()->data();
            std::vector<real_t> offset(nnodes * 3);
            fill_pseudo_random((ptrdiff_t)offset.size(), distortion / n, offset.data());
            for (ptrdiff_t i = 0; i < nnodes; ++i) {
                bool interior = true;
                for (int d = 0; d < 3; ++d) {
                    const geom_t c = points[d][i];
                    if (c <= geom_t(1e-6) || c >= geom_t(1 - 1e-6)) interior = false;
                }
                if (interior) {
                    for (int d = 0; d < 3; ++d) points[d][i] += (geom_t)offset[i * 3 + d];
                }
            }
        }
        if (element == smesh::TET10 || element == smesh::HEX27) return smesh::promote_to(element, mesh);
        return mesh;
    }

    template <class OpType>
    void configure(OpType &op, const std::shared_ptr<sfem::Mesh> &mesh, const int n_terms) {
        for (const auto &block : mesh->blocks()) {
            op.set_value_in_block(block->name(), "C10", test_C10);
            op.set_value_in_block(block->name(), "C01", test_C01);
            op.set_value_in_block(block->name(), "K", test_K);
        }
        if (n_terms > 0) {
            const std::vector<real_t> g{0.40, 0.30}, tau{1.0, 10.0};
            op.relaxation_groups().set_series(
                    "", "iso", sfem::PronySeries({g.begin(), g.begin() + n_terms}, {tau.begin(), tau.begin() + n_terms}));
        }
    }

    std::shared_ptr<sfem::GeneratedContinuedMooneyRivlinProny> make_continued(const std::shared_ptr<sfem::FunctionSpace> &fs,
                                                                             const int n_terms) {
        // Through the factory, by the name a case file gives as `material.type`.
        auto op = std::dynamic_pointer_cast<sfem::GeneratedContinuedMooneyRivlinProny>(
                sfem::create_op(fs, "GeneratedContinuedMooneyRivlinProny", sfem::EXECUTION_SPACE_HOST));
        if (!op) return nullptr;
        configure(*op, fs->mesh(), n_terms);
        for (const auto &block : fs->mesh()->blocks()) {
            op->set_value_in_block(block->name(), "Jc", test_Jc);
            op->set_value_in_block(block->name(), "Jmin", test_Jmin);
        }
        return op;
    }

    std::shared_ptr<sfem::GeneratedMooneyRivlinProny> make_original(const std::shared_ptr<sfem::FunctionSpace> &fs, const int n_terms) {
        auto op = std::dynamic_pointer_cast<sfem::GeneratedMooneyRivlinProny>(
                sfem::create_op(fs, "GeneratedMooneyRivlinProny", sfem::EXECUTION_SPACE_HOST));
        if (!op) return nullptr;
        configure(*op, fs->mesh(), n_terms);
        return op;
    }

    struct Evaluation {
        real_t                     value{0};
        sfem::SharedBuffer<real_t> gradient, apply, bsr;
    };

    template <class OpType>
    int evaluate(OpType                                   &op,
                 const std::shared_ptr<sfem::FunctionSpace> &fs,
                 const real_t *const                        u,
                 const real_t *const                        d,
                 Evaluation                                &e) {
        const ptrdiff_t ndofs = fs->n_dofs();
        auto            graph = fs->node_to_node_graph();
        e.gradient            = sfem::create_host_buffer<real_t>(ndofs);
        e.apply               = sfem::create_host_buffer<real_t>(ndofs);
        e.bsr                 = sfem::create_host_buffer<real_t>(graph->nnz() * 9);
        e.value               = 0;
        if (op.value(u, &e.value) != SFEM_SUCCESS) return SFEM_FAILURE;
        if (op.gradient(u, e.gradient->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        if (op.apply(u, d, e.apply->data()) != SFEM_SUCCESS) return SFEM_FAILURE;
        return op.hessian_bsr(u, graph->rowptr()->data(), graph->colidx()->data(), e.bsr->data());
    }

    /// J of every TET4 element at displacement u (affine: one per element).
    std::vector<real_t> tet4_jacobians(const sfem::Mesh &mesh, const real_t *const u) {
        auto                block = mesh.block(0);
        auto                elems = block->elements()->data();
        auto                X     = mesh.points()->data();
        std::vector<real_t> J(block->n_elements());
        for (ptrdiff_t e = 0; e < block->n_elements(); ++e) {
            real_t A[3][3], a[3][3];
            for (int c = 0; c < 3; ++c) {
                const idx_t n0 = elems[0][e], n1 = elems[c + 1][e];
                for (int d = 0; d < 3; ++d) {
                    A[d][c] = X[d][n1] - X[d][n0];
                    a[d][c] = A[d][c] + u[n1 * 3 + d] - u[n0 * 3 + d];
                }
            }
            auto det = [](real_t m[3][3]) {
                return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            };
            J[e] = det(a) / det(A);
        }
        return J;
    }

}  // namespace

// 1. h_m: J^(-m) where J >= Jc, C2 at Jc and Jmin, derivatives consistent, finite everywhere.
int test_continued_power() {
    using sfem::codegen::sfem_continued_power;
    using sfem::codegen::sfem_continued_power_d1;
    using sfem::codegen::sfem_continued_power_d2;
    const double Jc = test_Jc, Jmin = test_Jmin;
    for (const double m : {2.0 / 3.0, 4.0 / 3.0, 1.0}) {
        auto h  = [&](double J) { return sfem_continued_power<double>(J, m, Jc, Jmin); };
        auto h1 = [&](double J) { return sfem_continued_power_d1<double>(J, m, Jc, Jmin); };
        auto h2 = [&](double J) { return sfem_continued_power_d2<double>(J, m, Jc, Jmin); };

        // The physical branch is the power itself.
        for (const double J : {Jc, 0.5, 1.0, 1.7, 40.0}) {
            SFEM_TEST_ASSERT(h(J) == pow(J, -m));
            SFEM_TEST_ASSERT(std::abs(h1(J) - (-m * pow(J, -m - 1))) <= 1e-14 * std::abs(h1(J)));
            SFEM_TEST_ASSERT(std::abs(h2(J) - (m * (m + 1) * pow(J, -m - 2))) <= 1e-14 * std::abs(h2(J)));
        }

        // C2 at Jc and at Jmin: the jumps across each vanish with the distance.
        const double scale = h2(Jc);
        for (const double J0 : {Jc, Jmin}) {
            const double e = 1e-7;
            const double jump0 = std::abs(h(J0 + e) - h(J0 - e)), jump1 = std::abs(h1(J0 + e) - h1(J0 - e)),
                         jump2 = std::abs(h2(J0 + e) - h2(J0 - e));
            printf("  m=%.4f J0=%g: jumps %.2e %.2e %.2e (h'' at Jc %.3e)\n", m, J0, jump0, jump1, jump2, scale);
            SFEM_TEST_ASSERT(jump0 < 1e-5 * scale);
            SFEM_TEST_ASSERT(jump1 < 1e-5 * scale);
            SFEM_TEST_ASSERT(jump2 < 1e-4 * scale);
        }
        SFEM_TEST_ASSERT(h1(Jmin) == 0 && h2(Jmin) == 0 && h1(-3.0) == 0 && h2(-3.0) == 0);
        SFEM_TEST_ASSERT(h(-3.0) == h(Jmin));

        // The derivatives are those of the value, inside each piece.
        for (const double J : {-0.9, -0.5, 0.0, 0.1, 0.19, 0.3, 2.0}) {
            const double e   = 1e-6;
            const double fd1 = (h(J + e) - h(J - e)) / (2 * e), fd2 = (h1(J + e) - h1(J - e)) / (2 * e);
            SFEM_TEST_ASSERT(std::abs(fd1 - h1(J)) < 1e-7 * scale);
            SFEM_TEST_ASSERT(std::abs(fd2 - h2(J)) < 1e-6 * scale);
        }

        for (double J = -5; J <= 5; J += 0.01) {
            SFEM_TEST_ASSERT(std::isfinite(h(J)) && std::isfinite(h1(J)) && std::isfinite(h2(J)));
        }
    }
    return SFEM_TEST_SUCCESS;
}

// 2. One law where nothing is compressed below Jc.
int equals_original(const smesh::ElemType element) {
    auto mesh = unit_cube(element, 3, 0.15);
    auto fs   = sfem::FunctionSpace::create(mesh, 3);
    auto cont = make_continued(fs, 2);
    auto orig = make_original(fs, 2);
    SFEM_TEST_ASSERT(cont && orig);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_host_buffer<real_t>(ndofs);
    auto            d     = sfem::create_host_buffer<real_t>(ndofs);
    const bool      fine  = element == smesh::TET10 || element == smesh::HEX27;
    fill_pseudo_random(ndofs, fine ? 0.025 : 0.05, u->data());
    fill_pseudo_random(ndofs, 1, d->data(), 0x2545f4914f6cdd1dull);

    // The same committed history in both, so the relaxation terms are not zero.
    real_t t = 0;
    for (int i = 0; i < 5; ++i) {
        for (ptrdiff_t k = 0; k < ndofs; ++k) u->data()[k] *= real_t(1.1);
        t += test_dt;
        cont->begin_step(t, test_dt);
        orig->begin_step(t, test_dt);
        cont->advance(u->data());
        orig->advance(u->data());
    }
    cont->begin_step(t + test_dt, test_dt);
    orig->begin_step(t + test_dt, test_dt);

    Evaluation a, b;
    SFEM_TEST_ASSERT(evaluate(*cont, fs, u->data(), d->data(), a) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(evaluate(*orig, fs, u->data(), d->data(), b) == SFEM_SUCCESS);
    const real_t value_error    = std::abs(a.value - b.value) / std::abs(b.value);
    const real_t gradient_error = relative_difference(ndofs, a.gradient->data(), b.gradient->data());
    const real_t apply_error    = relative_difference(ndofs, a.apply->data(), b.apply->data());
    const real_t bsr_error      = relative_difference(a.bsr->size(), a.bsr->data(), b.bsr->data());
    printf("  %s ndof=%td: value %.3e, gradient %.3e, apply %.3e, bsr %.3e\n",
           smesh::type_to_string(element),
           ndofs,
           (double)value_error,
           (double)gradient_error,
           (double)apply_error,
           (double)bsr_error);
    // J^(-m) is evaluated as pow(J, -m) rather than 1 / pow(J, m): the same law in another order.
    SFEM_TEST_ASSERT(value_error < 1e-12);
    SFEM_TEST_ASSERT(gradient_error < 1e-12);
    SFEM_TEST_ASSERT(apply_error < 1e-12);
    SFEM_TEST_ASSERT(bsr_error < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int test_equals_original_tet4() { return equals_original(smesh::TET4); }
int test_equals_original_tet10() { return equals_original(smesh::TET10); }
int test_equals_original_hex8() { return equals_original(smesh::HEX8); }
int test_equals_original_hex27() { return equals_original(smesh::HEX27); }

// 3. Consistent through inversion.
int test_consistent_through_inversion() {
    auto mesh = unit_cube(smesh::TET4, 3, 0.15);
    auto fs   = sfem::FunctionSpace::create(mesh, 3);
    auto cont = make_continued(fs, 2);
    auto orig = make_original(fs, 2);
    SFEM_TEST_ASSERT(cont && orig);
    cont->begin_step(test_dt, test_dt);
    orig->begin_step(test_dt, test_dt);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_host_buffer<real_t>(ndofs);
    auto            d     = sfem::create_host_buffer<real_t>(ndofs);
    fill_pseudo_random(ndofs, 0.5, u->data());
    fill_pseudo_random(ndofs, 1, d->data(), 0x2545f4914f6cdd1dull);

    const auto J        = tet4_jacobians(*mesh, u->data());
    const auto inverted = std::count_if(J.begin(), J.end(), [](real_t j) { return j <= 0; });
    const auto squeezed = std::count_if(J.begin(), J.end(), [](real_t j) { return j > 0 && j < test_Jc; });
    printf("  %td elements: %td at J <= 0, %td at 0 < J < Jc, min J %.3f\n",
           J.size(),
           (ptrdiff_t)inverted,
           (ptrdiff_t)squeezed,
           (double)*std::min_element(J.begin(), J.end()));
    SFEM_TEST_ASSERT(inverted > 0 && squeezed > 0);

    real_t original_value = 0;
    orig->value(u->data(), &original_value);
    SFEM_TEST_ASSERT(!std::isfinite(original_value));

    Evaluation e;
    SFEM_TEST_ASSERT(evaluate(*cont, fs, u->data(), d->data(), e) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::isfinite(e.value));
    SFEM_TEST_ASSERT(all_finite(ndofs, e.gradient->data()) && all_finite(ndofs, e.apply->data()) &&
                     all_finite(e.bsr->size(), e.bsr->data()));

    // value -> gradient and gradient -> apply along d, by central differences.
    auto         blas    = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
    const real_t g_dot_d = blas->dot(ndofs, e.gradient->data(), d->data());
    auto         trial   = sfem::create_host_buffer<real_t>(ndofs);
    auto         gp      = sfem::create_host_buffer<real_t>(ndofs);
    auto         gm      = sfem::create_host_buffer<real_t>(ndofs);
    real_t       best_value = 1e30, best_apply = 1e30;
    for (int k = 3; k <= 8; ++k) {
        const real_t eps = pow(real_t(10), -real_t(k));
        real_t       vp = 0, vm = 0;
        blas->copy(ndofs, u->data(), trial->data());
        blas->axpy(ndofs, eps, d->data(), trial->data());
        cont->value(trial->data(), &vp);
        blas->zeros(ndofs, gp->data());
        cont->gradient(trial->data(), gp->data());
        blas->copy(ndofs, u->data(), trial->data());
        blas->axpy(ndofs, -eps, d->data(), trial->data());
        cont->value(trial->data(), &vm);
        blas->zeros(ndofs, gm->data());
        cont->gradient(trial->data(), gm->data());
        best_value = std::min(best_value, std::abs((vp - vm) / (2 * eps) - g_dot_d) / std::abs(g_dot_d));
        blas->axpy(ndofs, real_t(-1), gm->data(), gp->data());
        blas->scal(ndofs, 1 / (2 * eps), gp->data());
        best_apply = std::min(best_apply, relative_difference(ndofs, gp->data(), e.apply->data()));
    }

    // The assembled tangent applies like the apply.
    auto graph    = fs->node_to_node_graph();
    auto from_bsr = sfem::create_host_buffer<real_t>(ndofs);
    sfem::bsr_spmv<count_t, idx_t, real_t>(
            mesh->n_nodes(), mesh->n_nodes(), 3, graph->rowptr()->data(), graph->colidx()->data(), e.bsr->data(), real_t(0), d->data(), from_bsr->data());
    const real_t bsr_error = relative_difference(ndofs, from_bsr->data(), e.apply->data());

    printf("  value %.3e, apply %.3e, bsr %.3e\n", (double)best_value, (double)best_apply, (double)bsr_error);
    SFEM_TEST_ASSERT(best_value < 1e-7);
    SFEM_TEST_ASSERT(best_apply < 1e-6);
    SFEM_TEST_ASSERT(bsr_error < 1e-12);
    return SFEM_TEST_SUCCESS;
}

// 4. An inverted start recovers. One cube corner pushed through the cube's centre inverts the
// tetrahedra it belongs to; every other node is held, so the stress-free state is the minimum.
int test_inverted_start_recovers() {
    auto mesh = unit_cube(smesh::TET4, 1, 0);
    auto fs   = sfem::FunctionSpace::create(mesh, 3);
    auto cont = make_continued(fs, 0);
    auto orig = make_original(fs, 0);
    SFEM_TEST_ASSERT(cont && orig);
    cont->begin_step(test_dt, test_dt);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            X     = mesh->points()->data();
    ptrdiff_t       node  = -1;
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        if (X[0][i] > 0.5 && X[1][i] > 0.5 && X[2][i] > 0.5) node = i;
    }
    SFEM_TEST_ASSERT(node >= 0);

    auto u = sfem::create_host_buffer<real_t>(ndofs);
    std::fill(u->data(), u->data() + ndofs, real_t(0));
    for (int d = 0; d < 3; ++d) u->data()[node * 3 + d] = real_t(-0.9);

    const auto J0 = tet4_jacobians(*mesh, u->data());
    SFEM_TEST_ASSERT(*std::min_element(J0.begin(), J0.end()) < 0);
    real_t original_value = 0;
    orig->value(u->data(), &original_value);
    SFEM_TEST_ASSERT(!std::isfinite(original_value));

    // Newton on the free node's three unknowns, its 3x3 tangent from three applies, an Armijo
    // backtracking on the energy, steepest descent where the tangent gives no descent.
    auto g   = sfem::create_host_buffer<real_t>(ndofs);
    auto e_k = sfem::create_host_buffer<real_t>(ndofs);
    auto Hk  = sfem::create_host_buffer<real_t>(ndofs);
    auto energy = [&](const real_t *x) {
        real_t v = 0;
        cont->value(x, &v);
        return v;
    };
    int    it   = 0;
    real_t norm = 0;
    for (; it < 100; ++it) {
        std::fill(g->data(), g->data() + ndofs, real_t(0));
        cont->gradient(u->data(), g->data());
        const real_t *gi = g->data() + node * 3;
        norm             = sqrt(gi[0] * gi[0] + gi[1] * gi[1] + gi[2] * gi[2]);
        if (norm < 1e-10) break;

        real_t H[3][3];
        for (int c = 0; c < 3; ++c) {
            std::fill(e_k->data(), e_k->data() + ndofs, real_t(0));
            std::fill(Hk->data(), Hk->data() + ndofs, real_t(0));
            e_k->data()[node * 3 + c] = 1;
            cont->apply(u->data(), e_k->data(), Hk->data());
            for (int r = 0; r < 3; ++r) H[r][c] = Hk->data()[node * 3 + r];
        }
        const real_t det = H[0][0] * (H[1][1] * H[2][2] - H[1][2] * H[2][1]) - H[0][1] * (H[1][0] * H[2][2] - H[1][2] * H[2][0]) +
                           H[0][2] * (H[1][0] * H[2][1] - H[1][1] * H[2][0]);
        real_t step[3] = {-gi[0], -gi[1], -gi[2]};
        if (det != 0) {
            real_t inv[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const int r1 = (c + 1) % 3, r2 = (c + 2) % 3, c1 = (r + 1) % 3, c2 = (r + 2) % 3;
                    inv[r][c] = (H[r1][c1] * H[r2][c2] - H[r1][c2] * H[r2][c1]) / det;
                }
            }
            real_t newton[3];
            for (int r = 0; r < 3; ++r) newton[r] = -(inv[r][0] * gi[0] + inv[r][1] * gi[1] + inv[r][2] * gi[2]);
            if (newton[0] * gi[0] + newton[1] * gi[1] + newton[2] * gi[2] < 0) std::copy(newton, newton + 3, step);
        }

        const real_t slope = step[0] * gi[0] + step[1] * gi[1] + step[2] * gi[2];
        const real_t f0    = energy(u->data());
        real_t       alpha = 1;
        auto         trial = sfem::create_host_buffer<real_t>(ndofs);
        for (;; alpha /= 2) {
            std::copy(u->data(), u->data() + ndofs, trial->data());
            for (int d = 0; d < 3; ++d) trial->data()[node * 3 + d] += alpha * step[d];
            const real_t f = energy(trial->data());
            if ((std::isfinite(f) && f <= f0 + 1e-4 * alpha * slope) || alpha < 1e-12) break;
        }
        std::copy(trial->data(), trial->data() + ndofs, u->data());
    }

    const auto   J1   = tet4_jacobians(*mesh, u->data());
    const real_t minJ = *std::min_element(J1.begin(), J1.end());
    const real_t umax = std::max({std::abs(u->data()[node * 3]), std::abs(u->data()[node * 3 + 1]), std::abs(u->data()[node * 3 + 2])});
    printf("  from min J %.3f: %d iterations, |g| %.2e, min J %.6f, |u| %.2e\n",
           (double)*std::min_element(J0.begin(), J0.end()),
           it,
           (double)norm,
           (double)minJ,
           (double)umax);
    SFEM_TEST_ASSERT(norm < 1e-10);
    SFEM_TEST_ASSERT(minJ > test_Jc);
    SFEM_TEST_ASSERT(umax < 1e-8);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_continued_power);
    SFEM_RUN_TEST(test_equals_original_tet4);
    SFEM_RUN_TEST(test_equals_original_tet10);
    SFEM_RUN_TEST(test_equals_original_hex8);
    SFEM_RUN_TEST(test_equals_original_hex27);
    SFEM_RUN_TEST(test_consistent_through_inversion);
    SFEM_RUN_TEST(test_inverted_start_recovers);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
