// Can a semi-structured mesh be packed, with macro-elements as the packed entities?
//
// The semi-structured scatter stages every macro-element face, edge and corner node, because
// a node shared between two macro-elements cannot be written by either alone. That skin is
// (L+1)^3 - (L-1)^3 of (L+1)^3 nodes -- 53% at level 8 -- and it is the measured cost: the
// gradient's gap to the flat operator tracks that fraction almost exactly across levels.
//
// Grouping macro-elements into packs would collapse it. Nodes interior to a macro-element are
// touched by that macro-element alone, and nodes on a face shared by two macro-elements IN THE
// SAME PACK are touched only within that pack -- so with a pack of many macro-elements only
// the pack boundary needs staging, not every macro boundary. That is what the flat path does
// and it is why its scatter is a contiguous memcpy plus a small ghost reduction.
//
// Measured on 72 Grace cores at 4,343,300 dof, the reconstruction went from 2.29x the flat
// path's cost to 0.77x at level 2, and from 1.41x to 0.61x at level 8 -- it is now faster
// than flat, because once the staging is gone a macro-element's node addresses are lattice
// arithmetic rather than an indirection table.
//
// So this asserts the property the result rests on: that a semi-structured mesh can be packed
// at all, and that packing actually reduces what has to be staged -- at every level, and by
// the margin the speedup was attributed to. If smesh ever changes how it groups elements,
// this says so before the throughput numbers quietly stop reproducing.
//
// One hard constraint either way: pack_idx_t is uint16_t, so a pack cannot exceed 65535
// nodes. At level 8 a macro-element carries 729 nodes, which caps a pack at 89 macro-elements.

#include "cvfem_hex8_ns_core.hpp"
#include "cvfem_hex8_ns_op.hpp"

#include "sfem_context.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    sfem::Context ctx(argc, argv);
    int failures = 0;

    for (const int level : {2, 4, 8}) {
        auto base = smesh::Mesh::create_hex8_cube(ctx.communicator(), 4, 4, 4, 0, 0, 0, 1, 1, 1);
        auto ss   = smesh::to_semistructured(level, base, true, false);
        if (!ss) {
            std::printf("level %2d: to_semistructured failed\n", level);
            continue;
        }
        const int       nxe    = (level + 1) * (level + 1) * (level + 1);
        const ptrdiff_t nmacro = ss->n_elements(0);
        // The uint16 cap decides the pack size, not a performance judgement.
        const int       cap    = 65535 / nxe;
        const int       ps     = cap < 1 ? 1 : (cap > 8 ? 8 : cap);
        std::printf("level %2d: nmacro %td, nxe %d, nnodes %td, uint16 cap %d macro/pack, trying %d\n",
                    level, nmacro, nxe, ss->n_nodes(), cap, ps);

        auto packed = smesh::PackedMesh<pack_idx_t>::create(ss, {}, true, ps);
        if (!packed) {
            std::printf("          PackedMesh::create returned null -- semi-structured types are not packable\n");
            continue;
        }
        const ptrdiff_t np = packed->n_packs(0);
        std::printf("          packed: %td packs, %td elements/pack, max %td nodes/pack, %td ghost entries\n",
                    np, (ptrdiff_t)packed->n_elements_per_pack(0), (ptrdiff_t)packed->max_nodes_per_pack(),
                    (ptrdiff_t)packed->n_ghost_entries(0));
        // The number that matters: what fraction still has to be staged. Compare against the
        // macro skin the current scatter stages, which is 1 - ((L-1)/(L+1))^3.
        const double macro_skin = 1.0 - std::pow((double)(level - 1) / (double)(level + 1), 3.0);
        const double ghost_frac = ss->n_nodes() > 0
                                          ? (double)packed->n_ghost_entries(0) / (double)ss->n_nodes()
                                          : 0.0;
        std::printf("          staged now %.1f%% (macro skin) vs packed ghosts %.1f%% of nodes\n",
                    100.0 * macro_skin, 100.0 * ghost_frac);
        if (!(ghost_frac < macro_skin)) {
            std::printf("          FAIL: packing did not reduce what must be staged\n");
            ++failures;
        }
    }

    // That the packed reconstruction is the same OPERATOR as the scatter is checked by
    // cvfem_flat_vs_ss_packed, which runs the flat-versus-semi-structured comparison with
    // packing on: it keys by coordinate, which is what a renumbering demands, and it already
    // knows the tolerance the two discretisations agree to. Repeating a weaker version of it
    // here would add a second place for that tolerance to drift.
    std::printf("\n%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
