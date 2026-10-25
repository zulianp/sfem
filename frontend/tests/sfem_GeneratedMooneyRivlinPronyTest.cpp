// The generated Prony-series Mooney-Rivlin operator: against the hand-written one on HEX8, and
// against a closed form on every element it is generated for.
//
// `GeneratedMooneyRivlinProny` is the code generator's Mooney-Rivlin energy with its isochoric
// part handed to `PronySeries.relax`: within a step it minimises
//
//     psi_vol + gamma psi_iso + 1/2 S_hist : (C - I),
//
// with gamma and S_hist supplied per step by the material-independent `PronyRelaxation` it
// holds.  `MooneyRivlinVisco` is the hand-written operator for the same law and the same
// recursion, kept as the reference.  The gates, in the order they matter:
//
//   1. Parity: after the same committed history, the two operators' gradients and assembled
//      tangents agree to round-off -- on an axis-aligned and on a distorted mesh, with and
//      without Prony terms.  Both integrate with 2x2x2 Gauss, so nothing but the order of
//      evaluation separates them.
//   2. Consistency: the step energy's value differentiates to the gradient, the gradient to
//      the matrix-free apply, and the assembled BSR tangent applies like the apply.  Swept over
//      the step size, because a single step cannot tell a wrong derivative from a bad step.
//   3. Purity: gradient and apply mutate no history, so a rejected trial leaves no trace.
//   4. Relaxation on every element: the face force of a held homogeneous deformation, which
//      linear elements represent exactly, against the hereditary integral in closed form.

#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_BDF2Scheme.hpp"
#include "sfem_BSR.hpp"
#include "sfem_BackwardEulerScheme.hpp"
#include "sfem_ContactStabilisedNewmarkScheme.hpp"
#include "sfem_NewmarkScheme.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedMooneyRivlinProny.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_test.hpp"

namespace {

    const real_t test_C10 = 0.3;
    const real_t test_C01 = 0.05;
    const real_t test_K   = 50.0;
    const real_t test_dt  = 0.1;

    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x, uint64_t state = 0x9e3779b97f4a7c15ull) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x[i] = scale * (real_t)((double)(state >> 11) / (double)(1ull << 53) - 0.5);
        }
    }

    void distort_interior(const std::shared_ptr<sfem::Mesh> &mesh, const int n, const real_t amplitude) {
        if (amplitude <= 0) return;

        const ptrdiff_t     nnodes = mesh->n_nodes();
        auto                points = mesh->points()->data();
        std::vector<real_t> offset(nnodes * 3);
        fill_pseudo_random((ptrdiff_t)offset.size(), amplitude / n, offset.data());

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

    struct Setup {
        std::shared_ptr<sfem::Mesh>                        mesh;
        std::shared_ptr<sfem::FunctionSpace>               fs;
        std::shared_ptr<sfem::GeneratedMooneyRivlinProny>  generated;
        std::shared_ptr<sfem::MooneyRivlinVisco>           reference;
        sfem::SharedBuffer<real_t>                         u;
        ptrdiff_t                                          ndofs{0};
    };

    const std::vector<real_t> &prony_g() {
        static const std::vector<real_t> g{0.40, 0.30};
        return g;
    }

    const std::vector<real_t> &prony_tau() {
        static const std::vector<real_t> tau{1.0, 10.0};
        return tau;
    }

    /// The unit cube in `n` cells per side, of HEX8, TET4 or TET10 elements, its interior
    /// distorted.  TET10 is the distorted TET4 mesh promoted, so its elements stay straight
    /// sided: moving its edge nodes on their own curves elements this coarse until some of
    /// their points invert.
    std::shared_ptr<sfem::Mesh> unit_cube(const smesh::ElemType element, const int n, const real_t distortion) {
        auto comm = sfem::Communicator::world();
        auto mesh = element == smesh::HEX8 ? sfem::Mesh::create_hex8_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1)
                                           : sfem::Mesh::create_tet4_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
        distort_interior(mesh, n, distortion);
        return element == smesh::TET10 ? smesh::promote_to(smesh::TET10, mesh) : mesh;
    }

    /// Both operators on one mesh, driven through `n_history_steps` committed steps of a
    /// growing displacement.  The history is what makes the relaxation terms non-zero, so a
    /// check run without it cannot see a defect in them.  The hand-written reference exists
    /// for HEX8 only, so on another element the setup holds the generated operator alone.
    /// A material other than the test's own: its moduli, its whole series and the step length
    /// (the generated operator alone).
    struct Material {
        real_t              C10, C01, K, dt;
        std::vector<real_t> g, tau;
    };

    Setup make_setup(const int                  n,
                     const int                  n_terms,
                     const real_t               distortion,
                     const int                  n_history_steps,
                     const smesh::ElemType      element  = smesh::HEX8,
                     const Material *const      material = nullptr) {
        const real_t C10 = material ? material->C10 : test_C10;
        const real_t C01 = material ? material->C01 : test_C01;
        const real_t K   = material ? material->K : test_K;
        const real_t dt  = material ? material->dt : test_dt;
        Setup s;
        s.mesh = unit_cube(element, n, distortion);
        s.fs = sfem::FunctionSpace::create(s.mesh, 3);

        s.generated = std::make_shared<sfem::GeneratedMooneyRivlinProny>(s.fs);
        s.generated->initialize();
        for (const auto &block : s.mesh->blocks()) {
            s.generated->set_value_in_block(block->name(), "C10", C10);
            s.generated->set_value_in_block(block->name(), "C01", C01);
            s.generated->set_value_in_block(block->name(), "K", K);
        }

        if (element == smesh::HEX8 && !material) {
            s.reference = std::make_shared<sfem::MooneyRivlinVisco>(s.fs);
            s.reference->set_C10(test_C10);
            s.reference->set_C01(test_C01);
            s.reference->set_K(test_K);
            s.reference->set_dt(test_dt);
        }

        if (material) {
            s.generated->relaxation_groups().set_series("", "iso", sfem::PronySeries(material->g, material->tau));
        } else if (n_terms > 0) {
            const std::vector<real_t> g(prony_g().begin(), prony_g().begin() + n_terms);
            const std::vector<real_t> tau(prony_tau().begin(), prony_tau().begin() + n_terms);
            s.generated->relaxation_groups().set_series("", "iso", sfem::PronySeries(g, tau));
            if (s.reference) s.reference->set_prony_terms(n_terms, g.data(), tau.data());
        }

        if (s.reference) {
            s.reference->initialize();
            s.reference->initialize_history();
        }

        s.ndofs = s.fs->n_dofs();
        s.u     = sfem::create_host_buffer<real_t>(s.ndofs);
        // TET10's nodes are twice as dense, so the same random field would invert its elements.
        fill_pseudo_random(s.ndofs, element == smesh::TET10 ? 0.025 : 0.05, s.u->data());

        auto   blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        real_t t    = 0;
        for (int i = 0; i < n_history_steps; ++i) {
            blas->scal(s.ndofs, real_t(1.1), s.u->data());
            t += dt;
            s.generated->begin_step(t, dt);
            s.generated->advance(s.u->data());
            if (s.reference) s.reference->update_history(s.u->data());
        }

        // The step the comparison is made in.
        s.generated->begin_step(t + dt, dt);
        return s;
    }

    real_t relative_difference(const ptrdiff_t n, const real_t *const a, const real_t *const b) {
        real_t diff = 0, norm = 0;
        for (ptrdiff_t i = 0; i < n; i++) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
            norm = std::max(norm, std::abs(b[i]));
        }
        return norm > 0 ? diff / norm : diff;
    }

    int assemble_bsr(sfem::Op &op, const Setup &s, const std::shared_ptr<sfem::CRSGraph> &graph, const sfem::SharedBuffer<real_t> &values) {
        auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        blas->zeros(graph->nnz() * 9, values->data());
        return op.hessian_bsr(s.u->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data());
    }

    void bsr_apply(const Setup                              &s,
                   const std::shared_ptr<sfem::CRSGraph>    &graph,
                   const sfem::SharedBuffer<real_t>         &values,
                   const real_t *const                       x,
                   real_t *const                             y) {
        sfem::bsr_spmv<count_t, idx_t, real_t>(s.mesh->n_nodes(),
                                               s.mesh->n_nodes(),
                                               3,
                                               graph->rowptr()->data(),
                                               graph->colidx()->data(),
                                               values->data(),
                                               real_t(0),
                                               x,
                                               y);
    }

    /// Parity of gradient and tangent with the hand-written operator.
    int parity(const int n_terms, const real_t distortion) {
        Setup s    = make_setup(3, n_terms, distortion, 5);
        auto  blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

        auto g_gen = sfem::create_host_buffer<real_t>(s.ndofs);
        auto g_ref = sfem::create_host_buffer<real_t>(s.ndofs);
        blas->zeros(s.ndofs, g_gen->data());
        blas->zeros(s.ndofs, g_ref->data());
        SFEM_TEST_ASSERT(s.generated->gradient(s.u->data(), g_gen->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(s.reference->gradient(s.u->data(), g_ref->data()) == SFEM_SUCCESS);
        const real_t gradient_error = relative_difference(s.ndofs, g_gen->data(), g_ref->data());

        auto graph  = s.fs->node_to_node_graph();
        auto v_gen  = sfem::create_host_buffer<real_t>(graph->nnz() * 9);
        auto v_ref  = sfem::create_host_buffer<real_t>(graph->nnz() * 9);
        SFEM_TEST_ASSERT(assemble_bsr(*s.generated, s, graph, v_gen) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(assemble_bsr(*s.reference, s, graph, v_ref) == SFEM_SUCCESS);
        const real_t tangent_error = relative_difference(graph->nnz() * 9, v_gen->data(), v_ref->data());

        printf("  parity terms=%d distortion=%g ndof=%td: gradient %.3e, tangent %.3e\n",
               n_terms,
               (double)distortion,
               s.ndofs,
               (double)gradient_error,
               (double)tangent_error);
        // The floor is the two codes evaluating one law in different orders: 1.4e-11 for the
        // gradient and 1.9e-11 for the tangent, the same with no Prony terms as with two and
        // on a distorted mesh as on a straight one, so none of it is the history.  A modelling
        // difference -- another recursion, another rule, a missing term -- shows up orders
        // above it.
        SFEM_TEST_ASSERT(gradient_error < 1e-10);
        SFEM_TEST_ASSERT(tangent_error < 1e-10);
        return SFEM_TEST_SUCCESS;
    }

    /// Best relative central-difference error of `derivative` along `d`, over an eps sweep.
    template <class F, class DF>
    real_t fd_sweep(const Setup &s, const real_t *const d, F &&f, DF &&df, const ptrdiff_t n_out) {
        auto blas  = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        auto exact = sfem::create_host_buffer<real_t>(n_out);
        auto plus  = sfem::create_host_buffer<real_t>(n_out);
        auto minus = sfem::create_host_buffer<real_t>(n_out);
        auto trial = sfem::create_host_buffer<real_t>(s.ndofs);

        blas->zeros(n_out, exact->data());
        df(s.u->data(), exact->data());
        const real_t norm = blas->norm2(n_out, exact->data());

        real_t best = 1e30;
        for (int k = 2; k <= 9; ++k) {
            const real_t eps = pow(real_t(10), -real_t(k));
            blas->copy(s.ndofs, s.u->data(), trial->data());
            blas->axpy(s.ndofs, eps, d, trial->data());
            blas->zeros(n_out, plus->data());
            f(trial->data(), plus->data());

            blas->copy(s.ndofs, s.u->data(), trial->data());
            blas->axpy(s.ndofs, -eps, d, trial->data());
            blas->zeros(n_out, minus->data());
            f(trial->data(), minus->data());

            blas->axpy(n_out, real_t(-1), minus->data(), plus->data());
            blas->scal(n_out, real_t(1) / (2 * eps), plus->data());
            blas->axpy(n_out, real_t(-1), exact->data(), plus->data());
            best = std::min(best, blas->norm2(n_out, plus->data()) / norm);
        }
        return best;
    }

    int consistency(const int              n_terms,
                    const real_t           distortion,
                    const smesh::ElemType  element  = smesh::HEX8,
                    const Material *const  material = nullptr) {
        Setup s    = make_setup(3, n_terms, distortion, 5, element, material);
        auto  blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        auto  op   = s.generated;

        auto d = sfem::create_host_buffer<real_t>(s.ndofs);
        fill_pseudo_random(s.ndofs, 1.0, d->data(), 0x2545f4914f6cdd1dull);
        blas->scal(s.ndofs, 1 / blas->norm2(s.ndofs, d->data()), d->data());

        // value -> gradient, as a directional derivative: one number.
        const real_t value_error = fd_sweep(
                s,
                d->data(),
                [&](const real_t *x, real_t *out) { op->value(x, out); },
                [&](const real_t *x, real_t *out) {
                    auto g = sfem::create_host_buffer<real_t>(s.ndofs);
                    blas->zeros(s.ndofs, g->data());
                    op->gradient(x, g->data());
                    *out = blas->dot(s.ndofs, g->data(), d->data());
                },
                1);

        // gradient -> apply.
        const real_t apply_error = fd_sweep(
                s,
                d->data(),
                [&](const real_t *x, real_t *out) { op->gradient(x, out); },
                [&](const real_t *x, real_t *out) { op->apply(x, d->data(), out); },
                s.ndofs);

        // assembled tangent -> apply.
        auto graph  = s.fs->node_to_node_graph();
        auto values = sfem::create_host_buffer<real_t>(graph->nnz() * 9);
        SFEM_TEST_ASSERT(assemble_bsr(*op, s, graph, values) == SFEM_SUCCESS);
        auto from_bsr = sfem::create_host_buffer<real_t>(s.ndofs);
        auto from_op  = sfem::create_host_buffer<real_t>(s.ndofs);
        blas->zeros(s.ndofs, from_bsr->data());
        blas->zeros(s.ndofs, from_op->data());
        bsr_apply(s, graph, values, d->data(), from_bsr->data());
        op->apply(s.u->data(), d->data(), from_op->data());
        const real_t bsr_error = relative_difference(s.ndofs, from_bsr->data(), from_op->data());

        printf("  consistency %s terms=%d distortion=%g ndof=%td: value %.3e, apply %.3e, bsr %.3e\n",
               smesh::type_to_string(element),
               n_terms,
               (double)distortion,
               s.ndofs,
               (double)value_error,
               (double)apply_error,
               (double)bsr_error);
        // Central differences bottom out at round-off near 1e-10 here.
        SFEM_TEST_ASSERT(value_error < 1e-7);
        SFEM_TEST_ASSERT(apply_error < 1e-7);
        SFEM_TEST_ASSERT(bsr_error < 1e-12);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_parity_elastic() { return parity(0, 0); }
int test_parity_prony() { return parity(2, 0); }
int test_parity_prony_distorted() { return parity(2, 0.15); }

// The broadband EPDM of the LDP pump strokes: Felhos et al.'s 15 terms (tau 1.8e-27 to 1610 s,
// g_inf = 2.24e-3) on its scaled Mooney-Rivlin backbone, at a step of 1e4 s, where all but the
// slowest term relax within it.
const Material &broadband_epdm(const real_t dt) {
    static Material m{5.850e8, 1.463e8, 1.529e7, 0,
                      {2.83e-04, 3.12e-04, 4.45e-04, 1.03e-03, 6.09e-03, 6.74e-02, 2.22e-01, 1.90e-01, 1.12e-01, 5.07e-02,
                       6.91e-02, 4.59e-02, 8.62e-02, 6.76e-02, 7.87e-02},
                      {1.61e+03, 1.29e+01, 1.53e-01, 9.89e-04, 1.18e-06, 8.51e-09, 1.42e-10, 1.80e-12, 1.27e-14, 2.00e-16,
                       1.10e-19, 1.38e-22, 1.62e-24, 6.62e-26, 1.82e-27}};
    m.dt = dt;
    return m;
}
int test_consistency_broadband_tet4() { return consistency(15, 0.15, smesh::TET4, &broadband_epdm(1e4)); }
int test_consistency_broadband_hex8() { return consistency(15, 0.15, smesh::HEX8, &broadband_epdm(1e4)); }
int test_consistency_broadband_tet4_short() { return consistency(15, 0.15, smesh::TET4, &broadband_epdm(1e-4)); }
int test_consistency_elastic() { return consistency(0, 0.15); }
int test_consistency_prony_distorted() { return consistency(2, 0.15); }
int test_consistency_prony_tet4() { return consistency(2, 0.15, smesh::TET4); }
int test_consistency_prony_tet10() { return consistency(2, 0.15, smesh::TET10); }



namespace {
    std::shared_ptr<sfem::GeneratedMooneyRivlinProny> material_on(const std::shared_ptr<sfem::FunctionSpace> &fs,
                                                                  const real_t                               C10,
                                                                  const real_t                               C01,
                                                                  const real_t                               K,
                                                                  const bool                                 relaxing,
                                                                  const bool                                 affine) {
        auto op = std::make_shared<sfem::GeneratedMooneyRivlinProny>(fs);
        if (affine) op->set_option("ASSUME_AFFINE", true);
        op->initialize();
        for (const auto &block : fs->mesh_ptr()->blocks()) {
            op->set_value_in_block(block->name(), "C10", C10);
            op->set_value_in_block(block->name(), "C01", C01);
            op->set_value_in_block(block->name(), "K", K);
        }
        if (relaxing) op->relaxation_groups().set_series("", "iso", sfem::PronySeries(prony_g(), prony_tau()));
        return op;
    }

    /// The force the face x = 1 carries: the gradient summed over its nodes, per component.
    std::vector<real_t> face_force(const sfem::Mesh &mesh, const int dim, const real_t *const gradient) {
        std::vector<real_t> force(dim, 0);
        const auto          x = mesh.points()->data()[0];
        for (ptrdiff_t i = 0; i < mesh.n_nodes(); ++i) {
            if (std::abs(x[i] - geom_t(1)) > geom_t(1e-6)) continue;
            for (int d = 0; d < dim; ++d) force[d] += gradient[i * dim + d];
        }
        return force;
    }

    /// The force on the face x = 1 at the start of every step of a homogeneous deformation
    /// u = A X held from the first step on, for an operator stepped `n_steps` times.
    std::vector<std::vector<real_t>> held_face_forces(sfem::GeneratedMooneyRivlinProny &op,
                                                      const sfem::FunctionSpace       &fs,
                                                      const real_t *const              u,
                                                      const int                        n_steps,
                                                      const real_t                     dt) {
        const auto                       &mesh = *fs.mesh_ptr();
        const int                         dim  = mesh.spatial_dimension();
        std::vector<real_t>               gradient(fs.n_dofs());
        std::vector<std::vector<real_t>>  forces;
        for (int n = 1; n <= n_steps; ++n) {
            op.begin_step(n * dt, dt);
            std::fill(gradient.begin(), gradient.end(), real_t(0));
            if (op.gradient(u, gradient.data()) != SFEM_SUCCESS) return {};
            forces.push_back(face_force(mesh, dim, gradient.data()));
            op.advance(u);
        }
        return forces;
    }
}  // namespace

/// A homogeneous deformation u = A X, applied over the first step and held after it.  Linear
/// elements represent it exactly and integrate its constant density exactly, so the force on the
/// face x = 1 has a closed form on every element: the elastic volumetric part plus the isochoric
/// part scaled by the relaxation modulus of that ramp and hold,
///
///     m(t_n) = g_inf + sum_i g_i tau_i / dt (1 - exp(-dt / tau_i)) exp(-(t_n - dt) / tau_i),
///
/// which the linear-interpolation recursion reproduces exactly.  The two elastic parts are the
/// same operator with K = 0 and with C10 = C01 = 0, both without terms.  `affine` takes TET10's
/// affine route, whose rule is pinned to the isoparametric one's points.
int homogeneous_relaxation(const smesh::ElemType element, const bool affine = false) {
    auto comm = sfem::Communicator::world();
    std::shared_ptr<sfem::Mesh> mesh;
    if (element == smesh::TRI3) {
        mesh = sfem::Mesh::create_tri3_square(comm, 2, 2);
    } else if (element == smesh::QUAD4) {
        mesh = sfem::Mesh::create_quad4_square(comm, 2, 2);
    } else {
        mesh = unit_cube(element, 2, 0);
    }
    const int  dim = mesh->spatial_dimension();
    auto       fs  = sfem::FunctionSpace::create(mesh, dim);

    const real_t A[3][3] = {{0.04, 0.01, -0.02}, {0.0, -0.015, 0.01}, {0.02, 0.0, 0.03}};
    std::vector<real_t> u(fs->n_dofs(), 0);
    const auto          X = mesh->points()->data();
    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        for (int d = 0; d < dim; ++d) {
            for (int k = 0; k < dim; ++k) u[i * dim + d] += A[d][k] * X[k][i];
        }
    }

    const int    n_steps = 12;
    const real_t dt      = 0.25;
    auto prony = material_on(fs, test_C10, test_C01, test_K, true, affine);
    auto iso   = material_on(fs, test_C10, test_C01, 0, false, affine);
    auto vol   = material_on(fs, 0, 0, test_K, false, affine);
    const auto relaxed = held_face_forces(*prony, *fs, u.data(), n_steps, dt);
    const auto R_iso   = held_face_forces(*iso, *fs, u.data(), 1, dt);
    const auto R_vol   = held_face_forces(*vol, *fs, u.data(), 1, dt);
    SFEM_TEST_ASSERT((int)relaxed.size() == n_steps && R_iso.size() == 1 && R_vol.size() == 1);

    real_t g_inf = 1;
    for (const real_t g : prony_g()) g_inf -= g;
    real_t worst = 0, scale = 0;
    for (int n = 1; n <= n_steps; ++n) {
        real_t m = g_inf;
        for (size_t i = 0; i < prony_g().size(); ++i) {
            const real_t tau = prony_tau()[i];
            m += prony_g()[i] * tau / dt * (1 - std::exp(-dt / tau)) * std::exp(-(n - 1) * dt / tau);
        }
        for (int d = 0; d < dim; ++d) {
            const real_t want = R_vol[0][d] + m * R_iso[0][d];
            worst             = std::max(worst, std::abs(relaxed[n - 1][d] - want));
            scale             = std::max(scale, std::abs(want));
        }
    }
    printf("  homogeneous relaxation %s%s, %td dofs, %d steps: worst force error %.3e of %.3e\n",
           smesh::type_to_string(element),
           affine ? " affine" : "",
           fs->n_dofs(),
           n_steps,
           (double)worst,
           (double)scale);
    SFEM_TEST_ASSERT(scale > 0);
    SFEM_TEST_ASSERT(worst <= real_t(1e-11) * scale);
    return SFEM_TEST_SUCCESS;
}

int test_homogeneous_relaxation_hex8() { return homogeneous_relaxation(smesh::HEX8); }
int test_homogeneous_relaxation_tet4() { return homogeneous_relaxation(smesh::TET4); }
int test_homogeneous_relaxation_tet10() { return homogeneous_relaxation(smesh::TET10); }
int test_homogeneous_relaxation_tet10_affine() { return homogeneous_relaxation(smesh::TET10, true); }
int test_homogeneous_relaxation_tri3() { return homogeneous_relaxation(smesh::TRI3); }
int test_homogeneous_relaxation_quad4() { return homogeneous_relaxation(smesh::QUAD4); }

/// Nothing but `advance` moves the history: a trial gradient elsewhere, a repeated
/// `begin_step`, and an apply leave the next gradient bitwise unchanged.
int test_evaluation_is_pure() {
    sfem::test::SerialScope serial;
    Setup s    = make_setup(3, 2, 0.15, 3);
    auto  blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    auto before = sfem::create_host_buffer<real_t>(s.ndofs);
    auto after  = sfem::create_host_buffer<real_t>(s.ndofs);
    auto other  = sfem::create_host_buffer<real_t>(s.ndofs);
    auto scrap  = sfem::create_host_buffer<real_t>(s.ndofs);
    blas->zeros(s.ndofs, before->data());
    SFEM_TEST_ASSERT(s.generated->gradient(s.u->data(), before->data()) == SFEM_SUCCESS);

    fill_pseudo_random(s.ndofs, 0.2, other->data(), 12345);
    SFEM_TEST_ASSERT(s.generated->gradient(other->data(), scrap->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(s.generated->apply(other->data(), s.u->data(), scrap->data()) == SFEM_SUCCESS);
    s.generated->begin_step(1, test_dt);

    blas->zeros(s.ndofs, after->data());
    SFEM_TEST_ASSERT(s.generated->gradient(s.u->data(), after->data()) == SFEM_SUCCESS);
    for (ptrdiff_t i = 0; i < s.ndofs; i++) SFEM_TEST_ASSERT(before->data()[i] == after->data()[i]);
    return SFEM_TEST_SUCCESS;
}

/// The relaxation and the time scheme are two policies the operator holds, and they compose:
/// with a scheme attached, value and gradient are the relaxing material's plus the scheme's
/// separable term, and the gradient still differentiates to the apply.  Run under every scheme
/// the library has, with a non-trivial history, so neither policy can mask the other.
namespace {
    /// A contact that never touches: no reaction, nothing to project.
    class NoContact final : public sfem::ContactStabilisedNewmarkScheme::Contact {
    public:
        explicit NoContact(const ptrdiff_t ndofs) : ndofs_(ndofs) {}
        void reaction(real_t *const out) const override { std::fill(out, out + ndofs_, real_t(0)); }
        void project_velocity(const real_t *const, real_t *const, const real_t) override {}

    private:
        ptrdiff_t ndofs_;
    };
}  // namespace

int test_relaxation_composes_with_every_time_scheme() {
    for (int which = 0; which < 4; which++) {
        Setup s    = make_setup(3, 2, 0.15, 3);
        auto  blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);
        auto  op   = s.generated;

        std::shared_ptr<sfem::TimeScheme> scheme;
        const char                       *name = "";
        if (which == 0) {
            auto be = std::make_shared<sfem::BackwardEulerScheme>(s.fs);
            SFEM_TEST_ASSERT(be->initialize() == SFEM_SUCCESS);
            scheme = be;
            name   = "backward Euler";
        } else if (which == 1) {
            auto bdf2 = std::make_shared<sfem::BDF2Scheme>(s.fs);
            bdf2->set_density(real_t(2));
            SFEM_TEST_ASSERT(bdf2->initialize() == SFEM_SUCCESS);
            scheme = bdf2;
            name   = "BDF2";
        } else {
            auto newmark = std::make_shared<sfem::NewmarkScheme>(s.fs);
            newmark->set_density(real_t(2));
            SFEM_TEST_ASSERT(newmark->initialize() == SFEM_SUCCESS);
            fill_pseudo_random(s.ndofs, 0.01, newmark->velocity()->data(), 77);
            if (which == 2) {
                scheme = newmark;
                name   = "Newmark";
            } else {
                scheme = std::make_shared<sfem::ContactStabilisedNewmarkScheme>(newmark, std::make_shared<NoContact>(s.ndofs));
                name   = "contact-stabilised Newmark";
            }
        }
        scheme->begin_step(test_dt, test_dt);

        real_t bare_value = 0;
        auto   bare_grad  = sfem::create_host_buffer<real_t>(s.ndofs);
        SFEM_TEST_ASSERT(op->value(s.u->data(), &bare_value) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->gradient(s.u->data(), bare_grad->data()) == SFEM_SUCCESS);

        real_t term_value = 0;
        auto   term_grad  = sfem::create_host_buffer<real_t>(s.ndofs);
        if (auto inertia = scheme->inertia_op()) {
            SFEM_TEST_ASSERT(inertia->value(s.u->data(), &term_value) == SFEM_SUCCESS);
            SFEM_TEST_ASSERT(inertia->gradient(s.u->data(), term_grad->data()) == SFEM_SUCCESS);
        }

        op->set_time_scheme(scheme);
        real_t held_value = 0;
        auto   held_grad  = sfem::create_host_buffer<real_t>(s.ndofs);
        SFEM_TEST_ASSERT(op->value(s.u->data(), &held_value) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op->gradient(s.u->data(), held_grad->data()) == SFEM_SUCCESS);

        SFEM_TEST_ASSERT(std::abs(held_value - (bare_value + term_value)) <= real_t(1e-12) * (1 + std::abs(held_value)));
        for (ptrdiff_t i = 0; i < s.ndofs; ++i) {
            const real_t want = bare_grad->data()[i] + term_grad->data()[i];
            SFEM_TEST_ASSERT(std::abs(held_grad->data()[i] - want) <= real_t(1e-12) * (1 + std::abs(want)));
        }

        auto d = sfem::create_host_buffer<real_t>(s.ndofs);
        fill_pseudo_random(s.ndofs, 1.0, d->data(), 99);
        blas->scal(s.ndofs, 1 / blas->norm2(s.ndofs, d->data()), d->data());
        const real_t apply_error = fd_sweep(
                s,
                d->data(),
                [&](const real_t *x, real_t *out) { op->gradient(x, out); },
                [&](const real_t *x, real_t *out) { op->apply(x, d->data(), out); },
                s.ndofs);
        printf("  %s: gradient vs apply %.3e\n", name, (double)apply_error);
        SFEM_TEST_ASSERT(apply_error < 1e-7);
        op->set_time_scheme(nullptr);
    }
    return SFEM_TEST_SUCCESS;
}

/// `reset_history` forgets everything: an operator with a history, reset at x0, steps exactly as
/// a fresh one reset at x0.
int test_reset_forgets_the_history() {
    sfem::test::SerialScope serial;
    Setup s     = make_setup(3, 2, 0.15, 4);
    auto  fresh = std::make_shared<sfem::GeneratedMooneyRivlinProny>(s.fs);
    fresh->initialize();
    for (const auto &block : s.mesh->blocks()) {
        fresh->set_value_in_block(block->name(), "C10", test_C10);
        fresh->set_value_in_block(block->name(), "C01", test_C01);
        fresh->set_value_in_block(block->name(), "K", test_K);
    }
    fresh->relaxation_groups().set_series("", "iso", sfem::PronySeries(prony_g(), prony_tau()));

    auto x0 = sfem::create_host_buffer<real_t>(s.ndofs);
    fill_pseudo_random(s.ndofs, 0.03, x0->data(), 31337);
    s.generated->reset_history(x0->data());
    fresh->reset_history(x0->data());
    s.generated->begin_step(1, test_dt);
    fresh->begin_step(1, test_dt);

    auto g0 = sfem::create_host_buffer<real_t>(s.ndofs);
    auto g1 = sfem::create_host_buffer<real_t>(s.ndofs);
    std::fill(g0->data(), g0->data() + s.ndofs, real_t(0));
    std::fill(g1->data(), g1->data() + s.ndofs, real_t(0));
    SFEM_TEST_ASSERT(s.generated->gradient(s.u->data(), g0->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(fresh->gradient(s.u->data(), g1->data()) == SFEM_SUCCESS);
    for (ptrdiff_t i = 0; i < s.ndofs; i++) SFEM_TEST_ASSERT(g0->data()[i] == g1->data()[i]);
    return SFEM_TEST_SUCCESS;
}

#ifdef SFEM_ENABLE_RYAML
/// YAML: a top-level relaxation for every block, and a block entry that only changes the
/// temperature, which must keep the spectrum and shift it inherited.  The result must be the
/// operator the same configuration builds in C++.
int test_yaml_block_entry_overrides_only_what_it_names() {
    sfem::test::SerialScope serial;
    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(), 3, 3, 3, 0, 0, 0, 1, 1, 1);
    auto fs    = sfem::FunctionSpace::create(mesh, 3);
    auto block = mesh->blocks()[0]->name();

    const std::string yaml = "type: GeneratedMooneyRivlinProny\n"
                             "C10: 0.3\nC01: 0.05\nK: 50\n"
                             "relaxation:\n"
                             "  iso:\n"
                             "    prony: [{g: 0.4, tau: 1.0}, {g: 0.3, tau: 10.0}]\n"
                             "    wlf: {C1: 8.86, C2: 101.6, T_ref: 0}\n"
                             "    temperature: 0\n"
                             "blocks:\n"
                             "  - name: " + block + "\n"
                             "    relaxation:\n"
                             "      iso: {temperature: 20}\n";
    auto from_yaml = std::dynamic_pointer_cast<sfem::GeneratedMooneyRivlinProny>(
            sfem::create_op_from_yaml(fs, yaml, sfem::EXECUTION_SPACE_HOST));
    SFEM_TEST_ASSERT(from_yaml != nullptr);

    const auto &relaxation = from_yaml->relaxation_groups().relaxation(block, "iso");
    SFEM_TEST_ASSERT(relaxation.series().n_terms() == 2);
    SFEM_TEST_ASSERT(relaxation.temperature() == 20);

    auto by_hand = std::make_shared<sfem::GeneratedMooneyRivlinProny>(fs);
    by_hand->initialize();
    by_hand->set_value_in_block(block, "C10", 0.3);
    by_hand->set_value_in_block(block, "C01", 0.05);
    by_hand->set_value_in_block(block, "K", 50);
    sfem::PronyRelaxation::Options options;
    options.shift       = std::make_shared<sfem::WLFShift>(8.86, 101.6, 0);
    options.temperature = 20;
    by_hand->relaxation_groups().set_series("", "iso", sfem::PronySeries({0.4, 0.3}, {1.0, 10.0}), options);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_host_buffer<real_t>(ndofs);
    auto            g0    = sfem::create_host_buffer<real_t>(ndofs);
    auto            g1    = sfem::create_host_buffer<real_t>(ndofs);
    fill_pseudo_random(ndofs, 0.05, u->data(), 99);
    for (int step = 1; step <= 3; step++) {
        for (ptrdiff_t i = 0; i < ndofs; i++) u->data()[i] *= real_t(1.1);
        from_yaml->begin_step(step * test_dt, test_dt);
        by_hand->begin_step(step * test_dt, test_dt);
        std::fill(g0->data(), g0->data() + ndofs, real_t(0));
        std::fill(g1->data(), g1->data() + ndofs, real_t(0));
        SFEM_TEST_ASSERT(from_yaml->gradient(u->data(), g0->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(by_hand->gradient(u->data(), g1->data()) == SFEM_SUCCESS);
        for (ptrdiff_t i = 0; i < ndofs; i++) SFEM_TEST_ASSERT(g0->data()[i] == g1->data()[i]);
        from_yaml->advance(u->data());
        by_hand->advance(u->data());
    }
    return SFEM_TEST_SUCCESS;
}
#endif  // SFEM_ENABLE_RYAML

/// The affine kernels -- quadrature_output included -- against the isoparametric ones, on an
/// axis-aligned mesh, over committed steps.
///
/// The affine route reads the Jacobian cached in geom_t, single precision in this build, so the
/// two routes differ by that rounding whatever the material.  The run with no Prony terms is the
/// control that measures it; the history must add nothing above it.
real_t affine_vs_isoparametric(const int n_terms) {
    Setup iso    = make_setup(3, n_terms, 0, 0);
    Setup affine = make_setup(3, n_terms, 0, 0);
    affine.generated->set_option("ASSUME_AFFINE", true);

    auto   g0    = sfem::create_host_buffer<real_t>(iso.ndofs);
    auto   g1    = sfem::create_host_buffer<real_t>(iso.ndofs);
    real_t worst = 0;
    for (int step = 1; step <= 4; step++) {
        for (ptrdiff_t i = 0; i < iso.ndofs; i++) {
            iso.u->data()[i] *= real_t(1.1);
            affine.u->data()[i] = iso.u->data()[i];
        }
        iso.generated->begin_step(step * test_dt, test_dt);
        affine.generated->begin_step(step * test_dt, test_dt);
        std::fill(g0->data(), g0->data() + iso.ndofs, real_t(0));
        std::fill(g1->data(), g1->data() + iso.ndofs, real_t(0));
        if (iso.generated->gradient(iso.u->data(), g0->data()) != SFEM_SUCCESS) return 1e30;
        if (affine.generated->gradient(affine.u->data(), g1->data()) != SFEM_SUCCESS) return 1e30;
        worst = std::max(worst, relative_difference(iso.ndofs, g1->data(), g0->data()));
        iso.generated->advance(iso.u->data());
        affine.generated->advance(affine.u->data());
    }
    return worst;
}

int test_affine_route_matches_isoparametric() {
    const real_t control = affine_vs_isoparametric(0);
    const real_t prony   = affine_vs_isoparametric(2);
    printf("  affine vs isoparametric over 4 steps: no terms %.3e, two terms %.3e\n", (double)control, (double)prony);
    SFEM_TEST_ASSERT(control < real_t(sizeof(geom_t) == sizeof(double) ? 1e-12 : 1e-6));
    SFEM_TEST_ASSERT(prony < 2 * control + 1e-12);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_parity_elastic);
    SFEM_RUN_TEST(test_consistency_broadband_tet4);
    SFEM_RUN_TEST(test_consistency_broadband_hex8);
    SFEM_RUN_TEST(test_consistency_broadband_tet4_short);
    SFEM_RUN_TEST(test_parity_prony);
    SFEM_RUN_TEST(test_parity_prony_distorted);
    SFEM_RUN_TEST(test_consistency_elastic);
    SFEM_RUN_TEST(test_consistency_prony_distorted);
    SFEM_RUN_TEST(test_consistency_prony_tet4);
    SFEM_RUN_TEST(test_consistency_prony_tet10);
    SFEM_RUN_TEST(test_homogeneous_relaxation_hex8);
    SFEM_RUN_TEST(test_homogeneous_relaxation_tet4);
    SFEM_RUN_TEST(test_homogeneous_relaxation_tet10);
    SFEM_RUN_TEST(test_homogeneous_relaxation_tet10_affine);
    SFEM_RUN_TEST(test_homogeneous_relaxation_tri3);
    SFEM_RUN_TEST(test_homogeneous_relaxation_quad4);
    SFEM_RUN_TEST(test_evaluation_is_pure);
    SFEM_RUN_TEST(test_relaxation_composes_with_every_time_scheme);
    SFEM_RUN_TEST(test_reset_forgets_the_history);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_yaml_block_entry_overrides_only_what_it_names);
#endif
    SFEM_RUN_TEST(test_affine_route_matches_isoparametric);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
