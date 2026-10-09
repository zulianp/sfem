// A WHOLE PACK SWEEP, INSTANTIATED AND RUN AT SINGLE PRECISION.
//
// DESIGN.md's correction asks for the kernels to be templated on their types so that single
// precision becomes reachable. cvfem_mixed_precision_packs holds the element-kernel half of
// that -- the lane width, the packs, the leaf kernels. This holds the SWEEP half, which is the
// part that was argued to be impossible: a sweep instantiated at `float` would have kept the
// lane width computed for `double`, so there was no point templating it.
//
// It runs the isoparametric packed residual over a one-pack mesh at both precisions and
// requires the answers to agree. That is a stronger statement than the element-kernel gate can
// make, because a sweep is where the staging lives: the pack's field gather, its coordinate
// fill, the lane-blocked loop over elements, the scatter into pack-local storage and the drain
// back out. Every one of those has to carry the precision, and a single one that does not --
// a gather with the build's lane stride, a drain sized by the build's field count -- shows up
// here as a disagreement rather than as a compile error.
//
// ONE PACK, ONE ELEMENT. The mesh is the smallest thing the sweep will accept, because what is
// under test is the type plumbing rather than the physics: a bigger mesh would exercise the
// same code paths and take longer to say so. The ghost machinery is present but empty -- one
// pack owns all its nodes -- which is the configuration that lets the drain run without a
// reduction graph.
//
// WHAT IS NOT CLAIMED: that single precision is accurate enough for this operator on a real
// mesh. That is a question for a verification case, and the tolerance here (1e-4 relative) is
// loose enough not to pretend otherwise. What is claimed is that the f32 instantiation exists,
// runs, and computes the same thing.
// THE INSTRUMENTATION IS COMPILED OUT, which this translation unit needs and incidentally
// tests. kernels/cvfem_phases.hpp names PhaseAcc, g_breakdown, wall_time, g_phase and the PH_
// enumerators without defining them -- a contract the including translation unit satisfies, and
// one a driver satisfies through the bench's staging header. A unit that wants the kernels and
// not the staging has the other option DESIGN.md asks for: "Phase instrumentation should be
// handled with non invasive macros that can be disabled in production". With CVFEM_PHASES 0
// every one of those macros expands to nothing and none of the contract names is needed.
//
// So this file is also the only check that the disabled expansion still compiles. Nothing else
// builds with it.
#define CVFEM_PHASES 0

#include "smesh_types.hpp"
#include "kernels/cvfem_portability.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

#include "support/cvfem_default_types.hpp"
using idx_t   = smesh::idx_t;
using count_t = smesh::count_t;

#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/isoparametric/cvfem_hex8_ns_upwind_isoparam.hpp"
#include "kernels/cvfem_range.hpp"
#include "kernels/cvfem_scatter.hpp"
#include "kernels/packed/cvfem_pack_scratch.hpp"
#include "kernels/packed/cvfem_hex8_pack_staging.hpp"
#include "kernels/packed/isoparametric/cvfem_hex8_best_packed_isoparam.hpp"

static constexpr int NE = 1;   // elements in the pack
static constexpr int NN = 8;   // nodes, all owned by the one pack

// One unit cube, one pack, no ghosts. The pack-local node id of element node `a` is `a`.
template <typename S, typename I, typename PI, typename G>
static void run(S out[NN * CVFEM_HEX8_N_FIELDS]) {
    std::vector<G> xs(NN), ys(NN), zs(NN);
    for (int a = 0; a < NN; ++a) {
        xs[a] = G(CVFEM_HEX8_UNIT_CUBE[a][0]);
        ys[a] = G(CVFEM_HEX8_UNIT_CUBE[a][1]);
        zs[a] = G(CVFEM_HEX8_UNIT_CUBE[a][2]);
    }
    G *points[3] = {xs.data(), ys.data(), zs.data()};

    // A state off the upwind switch, as in the element-kernel gate: a zero velocity field sits
    // exactly on its non-differentiable point.
    std::vector<S> ux(NN), uy(NN), uz(NN), pr(NN);
    for (int a = 0; a < NN; ++a) {
        const S t = S(1) + S(a) / S(16);
        ux[a] = t;
        uy[a] = S(0.5) * t;
        uz[a] = S(0.25) * t;
        pr[a] = S(0.1) * S(a);
    }
    std::vector<S> rx(NN, S(0)), ry(NN, S(0)), rz(NN, S(0)), rc(NN, S(0));

    std::vector<PI> e0(NE * NN);
    PI             *pack_elems[NN];
    for (int a = 0; a < NN; ++a) {
        e0[a]         = PI(a);
        pack_elems[a] = &e0[a];
    }
    // owned_nodes_ptr[p] is the first global node of pack p; ghost_ptr is empty for both packs.
    const ptrdiff_t owned_nodes_ptr[2] = {0, NN};
    const ptrdiff_t ghost_ptr[2]       = {0, 0};
    std::vector<I>  ghost_idx(1, I(0));
    std::vector<S>  ghost_buf(4, S(0));

    apply_residual_packed_isoparam_range<S, I, PI, G>(
            cvfem_range{0, 1}, NE, pr.data(), points, ux.data(), uy.data(), uz.data(),
            pack_elems, ghost_buf.data(), ghost_idx.data(), ghost_ptr,
            /*max_actual_nodes_per_pack=*/NN, /*n_elements_per_pack=*/NE,
            /*n_ghost_entries=*/1, owned_nodes_ptr, S(1), S(1),
            rx.data(), ry.data(), rz.data(), rc.data(), packed_scratch_n(NN));

    for (int a = 0; a < NN; ++a) {
        out[a * CVFEM_HEX8_N_FIELDS + 0] = rx[a];
        out[a * CVFEM_HEX8_N_FIELDS + 1] = ry[a];
        out[a * CVFEM_HEX8_N_FIELDS + 2] = rz[a];
        out[a * CVFEM_HEX8_N_FIELDS + 3] = rc[a];
    }
}

int main() {
    double r64[NN * CVFEM_HEX8_N_FIELDS];
    float  r32[NN * CVFEM_HEX8_N_FIELDS];
    run<double, std::int32_t, std::uint16_t, double>(r64);
    run<float, std::int32_t, std::uint16_t, float>(r32);

    double scale = 0, worst = 0;
    int    worst_i = -1;
    for (int i = 0; i < NN * CVFEM_HEX8_N_FIELDS; ++i) scale = std::max(scale, std::fabs(r64[i]));
    if (scale == 0) {
        std::fprintf(stderr, "the f64 sweep produced an all-zero residual; the state is wrong\n");
        return 1;
    }
    for (int i = 0; i < NN * CVFEM_HEX8_N_FIELDS; ++i) {
        const double d = std::fabs((double)r32[i] - r64[i]) / scale;
        if (d > worst) { worst = d; worst_i = i; }
    }
    if (!(worst < 1.0e-4)) {
        std::fprintf(stderr,
                     "the f32 and f64 sweeps disagree by %.3e relative at node %d field %d "
                     "(bound 1e-4) -- at this tolerance that is a type-plumbing fault in the "
                     "staging, not round-off\n",
                     worst, worst_i / CVFEM_HEX8_N_FIELDS, worst_i % CVFEM_HEX8_N_FIELDS);
        return 1;
    }
    std::printf("cvfem_mixed_precision_sweep: the isoparametric packed residual agrees to "
                "%.2e relative between f32 (%d lanes) and f64 (%d lanes)\n",
                worst, cvfem_hex8_vec_size<float>, cvfem_hex8_vec_size<double>);
    return 0;
}
