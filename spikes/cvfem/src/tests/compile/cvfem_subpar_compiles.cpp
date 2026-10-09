// EVERY QUARANTINED HEADER STILL COMPILES.
//
// subpar/ is where a measured loser goes instead of being deleted, and the reason it is kept
// rather than deleted is that the measurement which retired it has to stay reproducible. A
// header that no longer compiles is not reproducible, so the quarantine needs a gate of its
// own -- nothing else includes these files, which is exactly why they rot unnoticed.
//
// Two of them had already rotted when this was written, and both were invisible because the
// subpar build did not complete at all: cvfem_sympy_action_test had not followed the
// affine/isoparametric split or the limiter's promotion to a template parameter, and
// cvfem_ss_scatter_fixed_width.hpp had not followed the sweeps' conversion to ranges. Each
// break dated from the commit that made the change it missed.
//
// It takes two translation units because the benchmark and the solver operator families
// cannot be included together; see the note beside the includes.
//
// It is a COMPILE gate: there is nothing to assert at run time, and a quarantined kernel's
// numerical agreement with the one that replaced it is checked by the oracle that retired it,
// not here. Without -DCVFEM_ENABLE_SUBPAR the headers are not on the include path at all, so
// the test compiles to nothing and reports ctest's SKIP.
#include <cstdio>

#ifdef CVFEM_ENABLE_SUBPAR

// The same prologue the benchmark uses, because these headers are the benchmark family's and
// expect the aliases it supplies. cvfem_hex8_best_common.hpp pulls in the mesh and the types.
#include <cstdint>
#include <cstring>

#include "frontend/staging/cvfem_hex8_best_common.hpp"
#include "frontend/staging/cvfem_hex8_packed_launch.hpp"
#include "kernels/standard/cvfem_hex8_best_atomic.hpp"
#include "kernels/standard/affine/cvfem_hex8_best_atomic_affine.hpp"
#include "kernels/standard/isoparametric/cvfem_hex8_best_atomic_isoparam.hpp"

// The quarantined sweeps. cvfem_hex8_packed_defcor_scalar.hpp arrives through
// cvfem_hex8_packed_launch.hpp above, which includes it under this same flag.
#include "cvfem_hex8_atomic_defcor_scalar.hpp"
#include "cvfem_hex8_atomic_retired.hpp"

// The semi-structured pair -- cvfem_ss_scatter_fixed_width.hpp and cvfem_sshex8_em.hpp --
// cannot join them: they reach the SOLVER family through frontend/ss/, and
// frontend/op/cvfem_hex8_ns_core.hpp #errors on being included beside the benchmark family
// because the two define sixteen of the same names with different physics. They have their own
// translation unit, cvfem_subpar_ss_compiles.cpp, for exactly that reason.

int main() {
    // The addresses, so nothing above is discarded as unused before it has been checked.
    const void *const kept[] = {
            (const void *)&apply_residual_packed_defcor_scalar,
            (const void *)&apply_residual_atomic_sumfact_defcor,
            (const void *)&assemble_jacobian_atomic_fd,
            (const void *)&assemble_jacobian_atomic_sympy,
            (const void *)&assemble_jacobian_atomic_sympy_block,
            (const void *)&assemble_jacobian_atomic_sympy_row,
            (const void *)&assemble_jacobian_atomic_sympy_face,
            (const void *)&apply_residual_atomic_sympy,
            (const void *)&assemble_jacobian_atomic_fd_isoparam,
            (const void *)&assemble_jacobian_atomic_linear_isoparam,
            (const void *)&assemble_jacobian_atomic_nonlinear_isoparam,
    };
    std::printf("cvfem_subpar_compiles: %zu quarantined sweeps linked\n",
                sizeof(kept) / sizeof(kept[0]));
    return 0;
}

#else
int main() {
    std::printf("cvfem_subpar_compiles: subpar/ is not on the include path; rebuild with "
                "-DCVFEM_ENABLE_SUBPAR=ON\n");
    return 77;  // ctest SKIP
}
#endif
