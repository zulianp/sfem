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
        const auto element = smesh::type_from_string(
                smesh::Env::read_string("SFEM_ELEM_TYPE", "TET4").c_str());
        // The block size is the field's component count, which follows the
        // element's dimension: two in 2D, three in 3D.  The tangent store is
        // sized from the same dimension, so driving both is the only way to
        // exercise both sizes -- 10 numbers per element against 45.
        const int dim = (element == smesh::TRI3 || element == smesh::QUAD4) ? 2 : 3;
        auto m = (dim == 2)
                ? sfem::Mesh::create_square(comm, static_cast<smesh::ElemType>(element),
                                            n, n, 0, 0, 1, 1)
                : sfem::Mesh::create_cube(comm, static_cast<smesh::ElemType>(element),
                                          n, n, n, 0, 0, 0, 1, 1, 1);
        auto fs = sfem::FunctionSpace::create(m, dim);
        std::printf("element %s, dim %d\n", type_to_string(m->element_type(0)), dim);

        auto op = sfem::GeneratedLinearElasticity::create(fs);
        if (!op) { std::printf("FAIL: could not create the Op\n"); return 1; }

        if (op->initialize() != SFEM_SUCCESS) { std::printf("FAIL: initialize\n"); return 1; }

        // Asked after initialize, because it answers from the geometry cache
        // that initialize builds.  A 2D mesh has none -- smesh fills the
        // adjugate for 3D elements only -- so the operator says so instead of
        // offering a path that would abort.
        std::printf("inexact_supported() = %s\n", op->inexact_supported() ? "true" : "false");
        if (!op->inexact_supported()) {
            std::printf("SKIP: this element has no inexact path\n");
            return 0;
        }

        const ptrdiff_t ndof = op->n_dofs_domain();
        std::vector<real_t> x(ndof), h(ndof), exact(ndof, 0), inexact(ndof, 0);
        auto points = m->points()->data();
        const ptrdiff_t nnodes = m->n_nodes();
        for (ptrdiff_t v = 0; v < nnodes; ++v) {
            const double px = points[0][v], py = points[1][v];
            const double pz = (dim == 3) ? points[2][v] : 0.0;
            for (int c = 0; c < dim; ++c) {
                x[dim*v + c] = 0.02*std::sin((3.0 - c)*px + (1.0 + c)*py + 0.5*pz);
                h[dim*v + c] = 0.05*std::sin((2.0 - 0.5*c)*px + (0.7 + c)*py + 1.1*pz);
            }
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

        // Linear elasticity's tangent does not vary over an element, so the
        // projection loses nothing and what remains is the store's precision --
        // `metric_tensor_t` is float.
        //
        // Two things make that number less obvious than it sounds.
        //
        // On a mesh whose spacing is a power of two (resolution 8, 16, 32 on the
        // unit cube) the tangent entries are exactly representable in float and
        // the difference collapses to 1e-14, which looks like proof of something
        // far stronger than is true.  At resolution 10, 12 or 20 the same kernel
        // gives its real answer.  Measuring only at a power of two is how this
        // check first reported 1.6e-15 for a float store.
        //
        // And the real answer depends on the element.  Measured at resolution
        // 12: TET4 1.2e-07, HEX8 6.8e-07, TET10 1.0e-05.  TET10's quadratic
        // basis gives its tangent a wider dynamic range, so the same float store
        // costs it about two more digits.  That is a property worth knowing when
        // choosing a store precision, not a tolerance to widen away.
        const double tolerance =
                (m->element_type(0) == smesh::TET10) ? 1e-4 : 1e-6;
        if (!(rel < tolerance)) {
            std::printf("FAIL: %.3e exceeds the store's precision (%.0e)\n", rel, tolerance);
            status = 1;
        } else {
            std::printf("PASS: the stored tangent reproduces the exact apply "
                        "to the store's precision\n");
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
