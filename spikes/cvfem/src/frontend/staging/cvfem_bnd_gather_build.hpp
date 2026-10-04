#pragma once

// The boundary shell's node gather map, built once per mesh.
//
// This was in kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp, where it did not belong on
// either of DESIGN.md's counts: it takes a staging object -- templated on MeshT only so as not
// to name which family's -- and it builds its table with a std::vector of std::pair. Both
// operator families call it from their own staging header, which is what this header is for.
//
// The map and the reason it exists are unchanged; only its address is.

// cvfem_hex8_boundary_scs.hpp names CVFEM_HEX8_N_DOF and the other element constants and
// relies on its includer having brought them in, so this header brings them in itself
// rather than depending on where it is included from.
#include "kernels/microkernels/hex8/cvfem_hex8_ns_upwind_kernels.hpp"
#include "kernels/microkernels/hex8/cvfem_hex8_boundary_scs.hpp"

#include <algorithm>
#include <utility>
#include <vector>

// ------------------------------------------------- the shell's node gather map
//
// A node-indexed CSR over the boundary shell, so the closure can be summed in an order
// fixed by the index array instead of by thread timing.
//
// The two boundary passes used to scatter their element contribution into the shared node
// arrays with `atomic_add` under `schedule(static)`. Static scheduling fixes WHICH thread
// owns a face, but not the order in which two threads holding faces that meet at a node
// commit to it, and floating-point addition is not associative -- so the operator was not
// bit-reproducible with itself across threads. Measured on one case at 33,124 dof: five
// runs at one thread all took 922 linear iterations and printed the same residual to every
// digit at iteration 100, while five at 72 threads took 900, 839, 1595, 943 and 742.
//
// The map turns the scatter into a gather. Each entry is a slot `i * 8 + a` naming the
// local node `a` of the i-th boundary element, grouped by the global node it lands on, so
// a pass can stage its per-element contributions and then have each node sum the slots
// that belong to it, alone and in a fixed order. That is deterministic for any thread
// count, and the summation order is the same one a serial run would use.
//
// Built serially. It is O(boundary shell), which is a surface rather than a volume, and a
// parallel build would have to be sorted afterwards to be reproducible anyway -- which is
// the property the whole map exists to provide.
template <typename MeshT>
static void cvfem_hex8_build_bnd_gather(MeshT &d) {
    const ptrdiff_t n_bnd = (ptrdiff_t)d.bnd_elems.size();
    if (d.bnd_gather_valid && d.bnd_gather_n_bnd == n_bnd) return;

    std::vector<std::pair<smesh::idx_t, int32_t>> pairs;
    pairs.reserve((size_t)n_bnd * CVFEM_HEX8_N_NODES);
    for (ptrdiff_t i = 0; i < n_bnd; ++i) {
        const ptrdiff_t e = d.bnd_elems[(size_t)i];
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a)
            pairs.emplace_back(d.elems[a][e], (int32_t)(i * CVFEM_HEX8_N_NODES + a));
    }
    // By node first, then by slot, so the summation order within a node is the order the
    // boundary list gives -- the order a serial sweep would have produced.
    std::sort(pairs.begin(), pairs.end());

    d.bnd_gather_dest.clear();
    d.bnd_gather_ptr.clear();
    d.bnd_gather_slot.clear();
    d.bnd_gather_slot.reserve(pairs.size());
    d.bnd_gather_ptr.push_back(0);
    for (size_t j = 0; j < pairs.size();) {
        const smesh::idx_t node = pairs[j].first;
        d.bnd_gather_dest.push_back(node);
        for (; j < pairs.size() && pairs[j].first == node; ++j) d.bnd_gather_slot.push_back(pairs[j].second);
        d.bnd_gather_ptr.push_back((ptrdiff_t)d.bnd_gather_slot.size());
    }
    d.bnd_r.assign((size_t)n_bnd * CVFEM_HEX8_N_DOF, scalar_t(0));
    d.bnd_gather_n_bnd = n_bnd;
    d.bnd_gather_valid = true;
}
