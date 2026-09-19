// The reduction idiom's serial guarantee.
//
// cvfem_parallel.hpp exists so that every cross-rank scalar in this spike is reduced one way
// instead of growing an `if (comm->size() > 1)` at each of a dozen call sites. The property
// that makes that safe to adopt everywhere -- including on paths that run in the verification
// matrix, which is compared byte for byte -- is that at one rank each function returns its
// argument unchanged, with no MPI call.
//
// So this gates the serial identity, which is the thing a future edit is most likely to break
// by "simplifying" a short-circuit away. It deliberately does NOT test the multi-rank
// behaviour: this binary is registered as a one-rank ctest, and a test that silently passes
// because it never had a second rank would be worse than no test. The multi-rank side is
// exercised by smesh's own distributed tests and, once the driver is parallel, by the 1-vs-4
// comparisons the plan calls for.
#include "cvfem_parallel.hpp"

#include "sfem_context.hpp"

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void check(const bool ok, const char *const what) {
    std::printf("%-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) failures++;
}

// Exact comparison is correct here and is the point of the test: at one rank these must be
// the SAME value, not a value that survived a round trip through a reduction. A tolerance
// would hide precisely the regression this exists to catch.
void check_same(const double got, const double want, const char *const what) {
    check(got == want, what);
}

}  // namespace

int main(int argc, char **argv) {
    // MPI has to be up before a communicator can be asked anything. Communicator::self()
    // merely wraps MPI_COMM_SELF, which is harmless, but size() calls MPI_Comm_size and that
    // aborts the process when MPI was never initialised. Measured rather than assumed: without
    // this line the binary dies with "MPI_Comm_size() was called before MPI_INIT was invoked".
    //
    // initialize_serial rather than initialize, for the same reason the bench drivers use it:
    // this is registered as a one-rank ctest and means nothing on several, so it should refuse
    // a second rank through the guard smesh already provides instead of quietly running the
    // same assertions N times and reporting N passes.
    auto ctx = sfem::initialize_serial(argc, argv);

    // Both spellings of "not distributed" that the driver can produce: a communicator of one
    // rank, and no communicator at all (which is what a Domain default-constructs to, and
    // what any code path that never set one up will hold).
    cvfem::Domain self;
    self.comm    = smesh::Communicator::self();
    self.n_owned = 17;
    self.n_local = 17;

    cvfem::Domain none;
    none.n_owned = 17;
    none.n_local = 17;

    check(!self.distributed(), "a one-rank communicator is not distributed");
    check(!none.distributed(), "a null communicator is not distributed");
    check(self.size() == 1, "size() is 1 without MPI ranks");
    check(self.is_root(), "the only rank is the root");
    check(none.size() == 1, "size() is 1 with no communicator");
    check(none.is_root(), "no communicator still reports root");

    for (const cvfem::Domain *d : {&self, &none}) {
        const char *tag = (d == &self) ? "self" : "null";
        char        buf[128];

        std::snprintf(buf, sizeof(buf), "%s: sum returns its argument", tag);
        check_same(cvfem::sum(*d, 3.25), 3.25, buf);

        std::snprintf(buf, sizeof(buf), "%s: max returns its argument", tag);
        check_same(cvfem::max(*d, -7.5), -7.5, buf);

        // min is built from max by negating twice, so a serial short-circuit is the only
        // thing standing between this and a double sign flip. Negative and positive inputs
        // both, because a sign error survives one of them.
        std::snprintf(buf, sizeof(buf), "%s: min returns its argument (negative)", tag);
        check_same(cvfem::min(*d, -7.5), -7.5, buf);

        std::snprintf(buf, sizeof(buf), "%s: min returns its argument (positive)", tag);
        check_same(cvfem::min(*d, 2.0), 2.0, buf);

        std::snprintf(buf, sizeof(buf), "%s: any(true) is true", tag);
        check(cvfem::any(*d, true), buf);

        std::snprintf(buf, sizeof(buf), "%s: any(false) is false", tag);
        check(!cvfem::any(*d, false), buf);

        // all() is !any(!v), so it is the one most easily inverted by a careless edit.
        std::snprintf(buf, sizeof(buf), "%s: all(true) is true", tag);
        check(cvfem::all(*d, true), buf);

        std::snprintf(buf, sizeof(buf), "%s: all(false) is false", tag);
        check(!cvfem::all(*d, false), buf);

        std::snprintf(buf, sizeof(buf), "%s: argmin returns the value and this rank", tag);
        const auto am = cvfem::argmin(*d, 1.5, 0);
        check(am.first == 1.5 && am.second == 0, buf);

        std::snprintf(buf, sizeof(buf), "%s: sum_kahan narrows without reducing", tag);
        check_same(cvfem::sum_kahan(*d, (long double)0.1), (double)(long double)0.1, buf);
    }

    // Integer reductions go through the same path; the templates must not silently only work
    // for double, since counts (nodes, elements, iterations) are reduced as integers.
    check(cvfem::sum(self, (int)5) == 5, "integer sum returns its argument");
    check(cvfem::max(self, (int)-3) == -3, "integer max returns its argument");
    check(cvfem::min(self, (int)-3) == -3, "integer min returns its argument");

    std::printf("\n%s\n", failures == 0 ? "PASSED" : "FAILED");
    return failures == 0 ? 0 : 1;
}
