// Does the assembled viscoelastic tangent match the derivative of the assembled gradient?
//
// The question is not academic here. In the torsion-with-release scenario the Newton steps
// taken while the twist is ramping converge quadratically, and the ones taken after the
// release converge linearly at a rate near 0.9 -- a full Newton step removing only a tenth of
// the residual is the signature of a Jacobian that is not the derivative of the residual it is
// paired with. The difference between the two phases is the accumulated Prony history, so this
// gate checks the tangent at a state that has some.
//
// The check is a central difference along a direction:
//
//     J d  ==  ( g(u + eps d) - g(u - eps d) ) / (2 eps) + O(eps^2)
//
// swept over eps, because a single eps cannot distinguish a wrong tangent from a badly chosen
// step: the error falls as eps^2 until round-off in the differenced gradient takes over and it
// rises as 1/eps. A correct tangent shows that V; a wrong one shows a floor it never goes below.
//
// The operator is exercised on its own here -- no mass term, no constraints, no Function. The
// reference usages in frontend/tests (sfem_MooneyRivlinGravityTest.cpp,
// sfem_MooneyRivlinViscoTest.cpp) are dynamic and add c0 M to the tangent diagonal, with
// c0 = 1/(beta dt^2) reaching several hundred, which dominates the material tangent and hides
// an error of this size. Anything added here would hide it in the same way.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_MooneyRivlinVisco.hpp"
#include "sfem_defs.hpp"

namespace {

    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x);

    /// Perturbs the interior nodes of a structured cube by a deterministic offset, leaving the
    /// boundary where it is so the domain stays a cube. Defect B in the geometric-stiffness
    /// kernel -- the transposed Jacobian -- is invisible on an axis-aligned mesh, because there
    /// J is diagonal and J^-1 is its own transpose. It only becomes observable once the elements
    /// are distorted, so the mesh has to be distorted before the fix can be said to have done
    /// anything.
    void distort_interior(const std::shared_ptr<sfem::Mesh> &mesh, const int n, const real_t amplitude) {
        if (amplitude <= 0) {
            return;
        }

        const ptrdiff_t nnodes = mesh->n_nodes();
        auto            points = mesh->points()->data();
        const real_t    h      = real_t(1) / n;

        std::vector<real_t> offset(nnodes * 3);
        fill_pseudo_random((ptrdiff_t)offset.size(), amplitude * h, offset.data());

        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            bool interior = true;
            for (int d = 0; d < 3; ++d) {
                const geom_t c = points[d][i];
                if (c <= geom_t(1e-6) || c >= geom_t(1 - 1e-6)) {
                    interior = false;
                    break;
                }
            }

            if (!interior) {
                continue;
            }

            for (int d = 0; d < 3; ++d) {
                points[d][i] += (geom_t)offset[i * 3 + d];
            }
        }
    }

    /// A deterministic pseudo-random field, so a failure is reproducible.
    void fill_pseudo_random(const ptrdiff_t n, const real_t scale, real_t *const x) {
        uint64_t state = 0x9e3779b97f4a7c15ull;
        for (ptrdiff_t i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            x[i] = scale * (real_t)((double)(state >> 11) / (double)(1ull << 53) - 0.5);
        }
    }

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx  = sfem::initialize_serial(argc, argv);
    auto comm = ctx->communicator();

    const int n = smesh::Env::read("PRONY_CHECK_RESOLUTION", 3);

    auto mesh = sfem::Mesh::create_hex8_cube(comm, n, n, n, 0, 0, 0, 1, 1, 1);
    distort_interior(mesh, n, smesh::Env::read("PRONY_CHECK_DISTORTION", 0.0));

    auto fs = sfem::FunctionSpace::create(mesh, 3);

    auto op = std::make_shared<sfem::MooneyRivlinVisco>(fs);
    op->set_C10(smesh::Env::read("PRONY_CHECK_C10", 0.3));
    op->set_C01(smesh::Env::read("PRONY_CHECK_C01", 0.05));
    op->set_K(smesh::Env::read("PRONY_CHECK_K", 50.0));

    const real_t dt = smesh::Env::read("PRONY_CHECK_DT", 0.1);
    op->set_dt(dt);

    const int n_terms = smesh::Env::read("PRONY_CHECK_TERMS", 2);
    if (n_terms > 0) {
        const std::vector<real_t> g{0.40, 0.30};
        const std::vector<real_t> tau{1.0, 10.0};
        op->set_prony_terms(std::min<int>(n_terms, 2), g.data(), tau.data());
    }

    if (op->initialize() != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }
    op->initialize_history();

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            blas  = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    auto u     = sfem::create_host_buffer<real_t>(ndofs);
    auto d     = sfem::create_host_buffer<real_t>(ndofs);
    auto jd    = sfem::create_host_buffer<real_t>(ndofs);
    auto trial = sfem::create_host_buffer<real_t>(ndofs);
    auto gp    = sfem::create_host_buffer<real_t>(ndofs);
    auto gm    = sfem::create_host_buffer<real_t>(ndofs);

    fill_pseudo_random(ndofs, smesh::Env::read("PRONY_CHECK_AMPLITUDE", 0.05), u->data());
    fill_pseudo_random(ndofs, 1.0, d->data());
    blas->scal(ndofs, 1 / blas->norm2(ndofs, d->data()), d->data());

    // Build up history the way the driver does: a few closed steps at a deforming state, so
    // that the H_i are not all zero when the tangent is checked.
    const int n_history_steps = smesh::Env::read("PRONY_CHECK_HISTORY_STEPS", 5);
    for (int i = 0; i < n_history_steps; ++i) {
        blas->scal(ndofs, real_t(1) + real_t(0.1), u->data());
        if (op->update_history(u->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
    }

    printf("ndof: %td  history steps: %d  prony terms: %d  active: %d  gamma: %g  distortion: %g\n",
           ndofs,
           n_history_steps,
           n_terms,
           op->get_num_active_terms(),
           (double)op->get_gamma(),
           (double)smesh::Env::read("PRONY_CHECK_DISTORTION", 0.0));

    bool diag_ok = true;

    // J d, through the same BSR path the driver's linear operator uses.
    auto      graph      = fs->node_to_node_graph();
    const int block_size = 3;
    auto      values     = sfem::create_host_buffer<real_t>(graph->nnz() * block_size * block_size);
    blas->zeros(graph->nnz() * block_size * block_size, values->data());

    if (op->hessian_bsr(u->data(), graph->rowptr()->data(), graph->colidx()->data(), values->data()) != SFEM_SUCCESS) {
        return SFEM_FAILURE;
    }

    // ---------------------------------------------------------------- hessian_diag gate
    // Independent of the tangent question: MooneyRivlinVisco::hessian_diag must return the
    // diagonal of the matrix MooneyRivlinVisco::hessian_bsr assembles. Nothing in SFEM's own
    // suite checks this -- the two frontend tests that use this material extract the diagonal
    // out of the assembled BSR by hand rather than calling hessian_diag -- and it is the only
    // defect here that is reachable with no Prony terms at all.
    {
        auto diag_op  = sfem::create_host_buffer<real_t>(ndofs);
        auto diag_bsr = sfem::create_host_buffer<real_t>(ndofs);
        blas->zeros(ndofs, diag_op->data());
        blas->zeros(ndofs, diag_bsr->data());

        if (op->hessian_diag(u->data(), diag_op->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        const auto rowptr = graph->rowptr()->data();
        const auto colidx = graph->colidx()->data();
        const auto vals   = values->data();
        for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
            for (count_t k = rowptr[node]; k < rowptr[node + 1]; ++k) {
                if (colidx[k] != (idx_t)node) {
                    continue;
                }
                for (int d = 0; d < block_size; ++d) {
                    diag_bsr->data()[node * block_size + d] = vals[k * block_size * block_size + d * block_size + d];
                }
                break;
            }
        }

        const real_t diag_norm = blas->norm2(ndofs, diag_bsr->data());
        blas->axpy(ndofs, real_t(-1), diag_bsr->data(), diag_op->data());
        const real_t diag_err = blas->norm2(ndofs, diag_op->data());
        const real_t diag_rel = diag_norm > 0 ? diag_err / diag_norm : diag_err;

        printf("hessian_diag vs diag(hessian_bsr): |diff| %.6e  relative %.6e\n", (double)diag_err, (double)diag_rel);
        if (!(diag_rel < 1e-12)) {
            fprintf(stderr,
                    "[prony] hessian_diag does not return the diagonal of hessian_bsr: "
                    "relative difference %.6e\n",
                    (double)diag_rel);
            diag_ok = false;
        }
    }

    blas->zeros(ndofs, jd->data());
    sfem::bsr_spmv<count_t, idx_t, real_t>(mesh->n_nodes(),
                                           mesh->n_nodes(),
                                           block_size,
                                           graph->rowptr()->data(),
                                           graph->colidx()->data(),
                                           values->data(),
                                           real_t(0),
                                           d->data(),
                                           jd->data());

    const real_t jd_norm = blas->norm2(ndofs, jd->data());

    printf("|J d|: %.6e\n", (double)jd_norm);
    printf("%-12s %-16s %-16s %-14s %-16s\n", "eps", "|fd - Jd|", "relative", "best scalar c", "rel after fit");

    real_t best_relative = std::numeric_limits<real_t>::max();
    for (int k = 2; k <= 9; ++k) {
        const real_t eps = std::pow(real_t(10), -real_t(k));

        blas->copy(ndofs, u->data(), trial->data());
        blas->axpy(ndofs, eps, d->data(), trial->data());
        blas->zeros(ndofs, gp->data());
        if (op->gradient(trial->data(), gp->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        blas->copy(ndofs, u->data(), trial->data());
        blas->axpy(ndofs, -eps, d->data(), trial->data());
        blas->zeros(ndofs, gm->data());
        if (op->gradient(trial->data(), gm->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }

        // gp <- (gp - gm) / (2 eps) - J d
        blas->axpy(ndofs, real_t(-1), gm->data(), gp->data());
        blas->scal(ndofs, real_t(1) / (2 * eps), gp->data());
        blas->axpy(ndofs, real_t(-1), jd->data(), gp->data());

        const real_t err      = blas->norm2(ndofs, gp->data());
        const real_t relative = jd_norm > 0 ? err / jd_norm : err;
        best_relative         = std::min(best_relative, relative);

        // How much of the discrepancy is a single wrong scalar in front of an otherwise
        // correct tangent: fit fd ~ c (J d) in least squares and report both c and what is
        // left once it is removed. A coefficient bug leaves almost nothing behind; a
        // structurally wrong tangent leaves most of the error.
        //
        // gp currently holds fd - J d, so <fd, Jd> = <gp, Jd> + |Jd|^2.
        const real_t c         = jd_norm > 0 ? 1 + blas->dot(ndofs, gp->data(), jd->data()) / (jd_norm * jd_norm) : 0;
        blas->axpy(ndofs, c - 1, jd->data(), gp->data());
        const real_t after_fit = blas->norm2(ndofs, gp->data());

        printf("%-12.1e %-16.6e %-16.6e %-14.9f %-16.6e\n",
               (double)eps,
               (double)err,
               (double)relative,
               (double)c,
               (double)(jd_norm > 0 ? after_fit / jd_norm : after_fit));
    }

    const real_t tol         = smesh::Env::read("PRONY_CHECK_TOL", 1e-6);
    const bool   tangent_ok  = best_relative < tol;

    if (!tangent_ok) {
        fprintf(stderr,
                "[prony] the assembled tangent does not match the derivative of the gradient: "
                "best relative error %.6e over the eps sweep, tolerance %.6e\n",
                (double)best_relative,
                (double)tol);
    } else {
        printf("tangent consistent: best relative error %.6e < %.6e\n", (double)best_relative, (double)tol);
    }

    return (tangent_ok && diag_ok) ? SFEM_SUCCESS : SFEM_FAILURE;
}
