// Does the generated Op's inexact path actually work inside SFEM?
//
// Everything so far verified the emitted text.  This compiles a generated Op
// with the split turned on and drives it through the interface a solver would:
//
//     op->initialize()
//     op->apply(x, h, exact)                  the exact operator
//     op->inexact_update(x)                   assemble the tangent once
//     op->inexact_apply(h, inexact)           apply it, with no state
//
// On a linear simplex the projection loses nothing, so the two must agree to
// round-off.  It also checks the contract: an inexact_apply before any
// inexact_update must fail rather than return an empty or stale answer.
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>

#include "sfem_Function.hpp"
#include "sfem_API.hpp"
#include "sfem_base.hpp"
#include "smesh_env.hpp"
#include "sfem_GeneratedLinearElasticity.hpp"

// This drives the Op that libsfem actually carries, now that linear_elasticity
// is generated in-tree with the split enabled.
//
// This worktree registers a generated Op whose sources it does not carry:
// sfem_generated_ops_registration.cpp calls
// register_GeneratedMooneyRivlinKelvinVoigtNewmark_generated_op, but
// frontend/ops/generated/ has no mooney_rivlin_kelvin_voigt_newmark tree, so
// libsfem.a cannot link a binary on its own.  That is a pre-existing gap in the
// checkout rather than anything to do with the split, and a spike has no
// business fixing it, so it is stubbed here to get the check linked.  The stub
// registers nothing; this driver instantiates the Op it tests directly.
namespace sfem {
    void register_GeneratedMooneyRivlinKelvinVoigtNewmark_generated_op() {}
}

int main(int argc, char *argv[]) {
    sfem::Context context(argc, argv);
    int status = 0;
    {
        auto comm = context.communicator();
        const int n = smesh::Env::read("SFEM_BASE_RESOLUTION", 16);
        auto m = sfem::Mesh::create_cube(comm, smesh::TET4, n, n, n, 0, 0, 0, 1, 1, 1);
        auto fs = sfem::FunctionSpace::create(m, 3);

        auto op = sfem::GeneratedLinearElasticity::create(fs);
        if (!op) { std::printf("FAIL: could not create the Op\n"); return 1; }

        std::printf("inexact_supported() = %s\n", op->inexact_supported() ? "true" : "false");
        if (!op->inexact_supported()) { std::printf("FAIL: expected support\n"); return 1; }

        if (op->initialize() != SFEM_SUCCESS) { std::printf("FAIL: initialize\n"); return 1; }

        const ptrdiff_t ndof = op->n_dofs_domain();
        std::vector<real_t> x(ndof), h(ndof), exact(ndof, 0), inexact(ndof, 0);
        auto points = m->points()->data();
        const ptrdiff_t nnodes = m->n_nodes();
        for (ptrdiff_t v = 0; v < nnodes; ++v) {
            const double px = points[0][v], py = points[1][v], pz = points[2][v];
            x[3*v+0] = 0.02*std::sin(3*px + py + 0.5*pz);
            x[3*v+1] = 0.02*std::sin(px + 3*py + 1.5*pz);
            x[3*v+2] = 0.02*std::sin(0.5*px + 1.5*py + 3*pz);
            h[3*v+0] = 0.05*std::sin(2*px + 0.7*py + 1.1*pz);
            h[3*v+1] = 0.05*std::sin(0.7*px + 2*py + 1.3*pz);
            h[3*v+2] = 0.05*std::sin(1.1*px + 1.3*py + 2*pz);
        }

        // The contract: applying before assembling must not quietly succeed.
        // It aborts rather than returning, which is what the generated Ops do
        // for any unmet precondition, so it needs a process of its own:
        // SFEM_CHECK_PRECONDITION=1 runs it and is expected to die.
        if (smesh::Env::read("SFEM_CHECK_PRECONDITION", false)) {
            std::printf("calling inexact_apply with no tangent; expecting an abort\n");
            op->inexact_apply(h.data(), inexact.data());
            std::printf("FAIL: inexact_apply returned without a tangent\n");
            return 1;
        }

        if (op->apply(x.data(), h.data(), exact.data()) != SFEM_SUCCESS) {
            std::printf("FAIL: exact apply\n"); return 1;
        }
        if (op->inexact_update(x.data()) != SFEM_SUCCESS) {
            std::printf("FAIL: inexact_update\n"); return 1;
        }
        std::fill(inexact.begin(), inexact.end(), real_t(0));
        if (op->inexact_apply(h.data(), inexact.data()) != SFEM_SUCCESS) {
            std::printf("FAIL: inexact_apply\n"); return 1;
        }

        double num = 0, den = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) {
            num += std::fabs(exact[i] - inexact[i]);
            den += std::fabs(exact[i]);
        }
        const double rel = den > 0 ? num / den : num;
        std::printf("ndof %ld, exact |.|_1 = %.6e\n", (long)ndof, den);
        std::printf("relative difference exact vs inexact = %.3e\n", rel);

        // A linear simplex: the projection is exact, so this is round-off.
        if (!(rel < 1e-12)) {
            std::printf("FAIL: expected agreement to round-off on TET4\n");
            status = 1;
        } else {
            std::printf("PASS: the stored tangent reproduces the exact apply\n");
        }

        // A second apply from the same tangent must give the same answer, which
        // is the reuse the split exists for.  Not bitwise, though: the scatter
        // accumulates through an OpenMP atomic, so the summation order varies
        // between runs and two applies agree only to round-off.  That is a
        // property of the threaded scatter and the exact apply shares it, so the
        // test asks for agreement rather than for identical bits.
        std::vector<real_t> again(ndof, 0);
        op->inexact_apply(h.data(), again.data());
        double drift = 0;
        for (ptrdiff_t i = 0; i < ndof; ++i) drift += std::fabs(again[i] - inexact[i]);
        const double relative_drift = den > 0 ? drift / den : drift;
        std::printf("second apply from the same tangent, relative drift = %.3e\n",
                    relative_drift);
        if (!(relative_drift < 1e-12)) {
            std::printf("FAIL: repeated applies from one tangent disagree\n");
            status = 1;
        } else {
            std::printf("PASS: the tangent is reusable across applies\n");
        }
    }
    return status;
}
