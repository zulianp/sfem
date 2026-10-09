// EVERY QUARANTINED SEMI-STRUCTURED HEADER STILL COMPILES.
//
// The other half of cvfem_subpar_compiles -- see its header comment for why the quarantine
// needs a gate at all. These two headers reach the SOLVER operator family through
// frontend/ss/, and frontend/op/cvfem_hex8_ns_core.hpp refuses to be included beside the
// benchmark family, so they cannot share that translation unit.
//
// cvfem_ss_scatter_fixed_width.hpp is the reason this file exists: its wrapper had not
// followed the sweeps' conversion to ranges and had not compiled since that commit, which
// nothing noticed because nothing includes it.
#include <cstdio>

#ifdef CVFEM_ENABLE_SUBPAR

#include "frontend/ss/cvfem_sshex8_ns.hpp"

#include "cvfem_ss_scatter_fixed_width.hpp"
#include "cvfem_sshex8_em.hpp"

int main() {
    const void *const kept[] = {
            (const void *)&sscvfem_scatter_element_soa,
            (const void *)&sscvfem_reduce_shared_soa,
    };
    std::printf("cvfem_subpar_ss_compiles: %zu quarantined wrappers linked\n",
                sizeof(kept) / sizeof(kept[0]));
    return 0;
}

#else
int main() {
    std::printf("cvfem_subpar_ss_compiles: subpar/ is not on the include path; rebuild with "
                "-DCVFEM_ENABLE_SUBPAR=ON\n");
    return 77;  // ctest SKIP
}
#endif
