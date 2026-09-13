// The benchmark's mesh in the layout the packed kernels require.
//
// The partition itself comes from `python/codegen/framework/tools/packed_layout.inc`,
// shared with the reproducibility harness -- that harness drives the generated packed
// kernels through this exact layout and checks their answers against the unpacked
// ones, so it is a contract something already verifies rather than one written twice
// and hoped about.  What is here is only the half that harness cannot share: rebuilding
// *this* mesh under the packed numbering.
//
// Nothing links smesh.  `smesh::PackedMesh` is the real thing and lives behind
// `SFEM::sfem`; see the shared file for why borrowing it would cost more than it saves.
#pragma once

#include "element_mesh.inc"
#include "packed_layout.inc"

#ifndef PACK_SIZE
#define PACK_SIZE 512
#endif

struct PackedMeshView {
    PackedLayout layout;
    Mesh mesh;  //: the same mesh, renumbered so each pack owns a contiguous node range
    //: `layout.element_ptrs` in the order the templated micro-kernel is written
    //: against.  The two are the same unless this element forwards its kernel to
    //: a lexicographic twin; see `finalise_element_pointers`.
    std::vector<uint16_t *> kernel_element_ptrs;
};

// Geometry is per element and packs are contiguous element ranges, so the adjugate,
// the determinant and the per-element tangent store are all untouched by packing --
// only the connectivity and the node coordinates move.
static PackedMeshView build_packed_mesh(const Mesh &source, ptrdiff_t elements_per_pack) {
    PackedMeshView p;
    p.layout = build_packed_layout(source.ev, source.nelements, source.nnodes,
                                   elements_per_pack);
    // The pack-local connectivity in the kernel's node order, for the same
    // reason `finalise_element_pointers` builds `kevp`.  The *layout* stays the
    // library's -- packing renumbers nodes, and doing that from permuted rows
    // would give a different numbering from the one SFEM produces, which the
    // packed `extern "C"` apply is entitled to.  Only the pointer view moves.
    p.kernel_element_ptrs = p.layout.element_ptrs;
#ifdef KERNEL_SHAPE_ORDER
    {
        static const int kernel_shape_order[] = {KERNEL_SHAPE_ORDER};
        for (int node = 0; node < NXE; ++node)
            p.kernel_element_ptrs[node] = p.layout.element_ptrs[kernel_shape_order[node]];
    }
#endif

    p.mesh.nelements = source.nelements;
    p.mesh.nnodes = source.nnodes;
    p.mesh.adj = source.adj;
    p.mesh.det = source.det;
    p.mesh.ev = p.layout.renumbered;
    p.mesh.px.resize((size_t)source.nnodes);
    p.mesh.py.resize((size_t)source.nnodes);
    p.mesh.pz.resize((size_t)source.nnodes);
    for (ptrdiff_t node = 0; node < source.nnodes; ++node) {
        const idx_t to = p.layout.to_new[node];
        p.mesh.px[to] = source.px[node];
        p.mesh.py[to] = source.py[node];
        p.mesh.pz[to] = source.pz[node];
    }
    for (auto &row : p.mesh.ev) p.mesh.evp.push_back(row.data());
    return p;
}

// A node-indexed field, seeded through the packed permutation so the packed run and
// the standard run are the same problem relabelled.  Seeding by index instead would
// make them different problems and any comparison of their answers meaningless.
static void place_packed(const std::vector<double> &source, const PackedLayout &layout,
                         std::vector<double> &out) {
    out.resize(source.size());
    for (size_t node = 0; node < source.size(); ++node) out[layout.to_new[node]] = source[node];
}
