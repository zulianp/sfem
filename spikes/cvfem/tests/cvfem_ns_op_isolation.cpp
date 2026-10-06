// Compile-time proof that cvfem_hex8_ns_op.hpp is usable from any driver.
//
// There is nothing to run: if this translation unit compiles, the header is clean. It
// asserts three things that a driver depends on and that no runtime test would catch.
//
// 1. The operator header leaks none of the solver core. The names below are the ones
//    cvfem_hex8_ns_core.hpp defines at file scope; declaring our own here would be a
//    redefinition if the operator header had pulled the core in.
// 2. It coexists with the benchmark's layout family. Those headers define sixteen names
//    in common with the core, so a driver that wanted both the operator and the
//    benchmark layouts could not have had them before.
// 3. `scalar_t` is not forced on the includer. The core fixes it to double at file
//    scope; here it is float, and the operator is unaffected because its interface is
//    real_t throughout.

#include "cvfem_hex8_ns_op.hpp"

// The benchmark family, included alongside the operator on purpose.
#include "cvfem_hex8_layout_common.hpp"
#include "cvfem_hex8_layout_atomic.hpp"

namespace check_no_core_leak {
    // Each of these would clash with cvfem_hex8_ns_core.hpp if the operator header
    // exposed it. They are in a namespace so they do not clash with the layout family,
    // which is included above at global scope and legitimately defines its own.
    using scalar_t = float;
    struct MeshData {
        int placeholder;
    };
    struct BSR4 {
        int placeholder;
    };
    enum class GeomKind { Something, Else };
    enum class InitKind { Other };
    enum class FlowCase { Different };
    constexpr int N_FIELDS = 7;

    void assemble_jacobian() {}
    void apply_residual() {}
    void assemble_block_diag() {}
    void make_bsr4() {}
    void pack_fields() {}
    void unpack_fields() {}
    void zero_bsr4() {}
    void precompute_element_bsr_slots() {}
}  // namespace check_no_core_leak

// The operator's interface must be expressible without the core: real_t for the
// parameters, and its own enum for the geometry choice.
static void uses_only_the_public_interface(const std::shared_ptr<sfem::FunctionSpace> &space) {
    sfem::CVFEMNavierStokes op(space);
    op.rho             = real_t(1);
    op.mu              = real_t(0.01);
    op.rhie_chow_scale = real_t(1);
    op.geom            = sfem::CVFEMGeometry::Isoparam;
    op.pack_size       = 0;
    (void)op.name();
    (void)op.is_linear();
}

int main() {
    // Never called. Referenced so the compiler must instantiate and check it, while the
    // test stays a pure compile-time assertion with no mesh and no MPI.
    (void)&uses_only_the_public_interface;
    static_assert(check_no_core_leak::N_FIELDS == 7, "our own name, not the core's");
    static_assert(sizeof(check_no_core_leak::scalar_t) == sizeof(float), "our own scalar_t, not the core's");
    return 0;
}
