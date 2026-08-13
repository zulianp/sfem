/**
 * @file sfem_ElementScope.hpp
 * @brief Block-local element ranges for phased assembly / compute-comm overlap
 */

#pragma once

#include "sfem_defs.hpp"
#include "smesh_mesh.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sfem {

    enum class ElementScope {
        ALL              = 0,
        OWNED_NOT_SHARED = 1,
        SHARED_AND_AURA  = 2,
    };

    /// Half-open [begin, end) indices into a single mesh block's local element array.
    struct ElementRange {
        ptrdiff_t begin{0};
        ptrdiff_t end{0};

        ptrdiff_t size() const { return end > begin ? end - begin : 0; }
        bool      empty() const { return size() == 0; }
    };

    /// Block index plus a block-local element subrange.
    struct BlockElementSlice {
        smesh::block_idx_t block{0};
        ElementRange       range;
    };

    inline bool mesh_is_distributed(const smesh::Mesh &mesh) {
        return mesh.comm() && mesh.comm()->size() > 1;
    }

    /// Distributed ElementScope requires smesh single-block layout (owned/shared counts are mesh-global).
    inline bool mesh_supports_distributed_element_scopes(const smesh::Mesh &mesh) {
        if (!mesh_is_distributed(mesh)) {
            return true;
        }
        return mesh.n_blocks() == 1 && mesh.n_elements(0) == mesh.distributed()->n_elements_local();
    }

    /// Abort if MPI mesh cannot use phased ElementScope (multi-block distributed not implemented in smesh).
    inline void assert_mesh_supports_distributed_element_scopes(const smesh::Mesh &mesh) {
        if (!mesh_is_distributed(mesh)) {
            return;
        }
        if (mesh.n_blocks() != 1) {
            SFEM_ERROR(
                    "ElementScope on distributed meshes requires exactly one block; smesh does not yet provide "
                    "per-block owned/shared/aura element partitioning.\n");
        }
        const ptrdiff_t n_block = mesh.n_elements(0);
        const ptrdiff_t n_local = mesh.distributed()->n_elements_local();
        if (n_block != n_local) {
            SFEM_ERROR(
                    "ElementScope on distributed mesh: block 0 has %ld elements but distributed layout has %ld local "
                    "elements; block layout must match smesh distributed element ordering.\n",
                    (long)n_block,
                    (long)n_local);
        }
    }

    inline ptrdiff_t block_element_offset(const smesh::Mesh &mesh, const smesh::block_idx_t block_idx) {
        ptrdiff_t off = 0;
        for (smesh::block_idx_t b = 0; b < block_idx; ++b) {
            off += mesh.n_elements(b);
        }
        return off;
    }

    /// Concatenated local element span [begin, end) for @p scope (single-block distributed or serial sum of blocks).
    inline ElementRange mesh_element_range(const smesh::Mesh &mesh, const ElementScope scope) {
        if (!mesh_is_distributed(mesh)) {
            const ptrdiff_t n = mesh.n_elements();
            switch (scope) {
                case ElementScope::ALL:
                case ElementScope::OWNED_NOT_SHARED:
                    return {0, n};
                case ElementScope::SHARED_AND_AURA:
                    return {0, 0};
            }
            return {0, 0};
        }

        assert_mesh_supports_distributed_element_scopes(mesh);

        const auto        dist    = mesh.distributed();
        const ptrdiff_t n_ons   = dist->n_elements_owned_not_shared();
        const ptrdiff_t n_local = dist->n_elements_local();

        switch (scope) {
            case ElementScope::ALL:
                return {0, n_local};
            case ElementScope::OWNED_NOT_SHARED:
                return {0, n_ons};
            case ElementScope::SHARED_AND_AURA:
                return {n_ons, n_local};
        }
        return {0, 0};
    }

    /// Block-local [begin, end) for @p scope. Serial: per-block ALL/OWNED; distributed: only block 0 (single-block mesh).
    inline ElementRange element_range(const smesh::Mesh &mesh, const smesh::block_idx_t block_idx, const ElementScope scope) {
        if (mesh_is_distributed(mesh)) {
            if (block_idx != 0) {
                SFEM_ERROR(
                        "ElementScope on distributed meshes is only defined for block 0 until smesh supports "
                        "multi-block distributed element layout.\n");
            }
            assert_mesh_supports_distributed_element_scopes(mesh);
            const ElementRange mesh_rng = mesh_element_range(mesh, scope);
            const ptrdiff_t    n_block  = mesh.n_elements(0);
            const ptrdiff_t    begin    = std::max(mesh_rng.begin, ptrdiff_t(0));
            const ptrdiff_t    end      = std::min(mesh_rng.end, n_block);
            if (begin >= end) {
                return {0, 0};
            }
            return {begin, end};
        }

        const ptrdiff_t block_off = block_element_offset(mesh, block_idx);
        const ptrdiff_t block_end = block_off + mesh.n_elements(block_idx);
        const ElementRange mesh_rng = mesh_element_range(mesh, scope);

        const ptrdiff_t begin = std::max(mesh_rng.begin, block_off);
        const ptrdiff_t end     = std::min(mesh_rng.end, block_end);
        if (begin >= end) {
            return {0, 0};
        }
        return {begin - block_off, end - block_off};
    }

    inline std::vector<BlockElementSlice> block_element_slices(const smesh::Mesh &mesh, const ElementScope scope) {
        std::vector<BlockElementSlice> slices;
        if (mesh_is_distributed(mesh)) {
            assert_mesh_supports_distributed_element_scopes(mesh);
            const ElementRange r = element_range(mesh, 0, scope);
            if (!r.empty()) {
                slices.push_back({0, r});
            }
            return slices;
        }

        const size_t n_blocks = mesh.n_blocks();
        slices.reserve(n_blocks);
        for (smesh::block_idx_t b = 0; b < static_cast<smesh::block_idx_t>(n_blocks); ++b) {
            const ElementRange r = element_range(mesh, b, scope);
            if (!r.empty()) {
                slices.push_back({b, r});
            }
        }
        return slices;
    }

    inline ptrdiff_t count_block_elements(const smesh::Mesh &mesh, const ElementScope scope) {
        ptrdiff_t n = 0;
        for (const auto &slice : block_element_slices(mesh, scope)) {
            n += slice.range.size();
        }
        return n;
    }

    /// Contiguous static partition of [0, n) across n_workers workers.
    inline ElementRange static_chunk(const ptrdiff_t n, const int worker_index, const int n_workers) {
        if (n_workers <= 0 || worker_index < 0 || worker_index >= n_workers || n <= 0) {
            return {0, 0};
        }
        const ptrdiff_t base = n / n_workers;
        const ptrdiff_t rem  = n % n_workers;
        const ptrdiff_t begin =
                static_cast<ptrdiff_t>(worker_index) * base + static_cast<ptrdiff_t>(std::min(worker_index, static_cast<int>(rem)));
        const ptrdiff_t len = base + (worker_index < static_cast<int>(rem) ? 1 : 0);
        return {begin, begin + len};
    }

    /// Half-open flat chunk [flat_begin, flat_end) as block-local slices (may split one block).
    inline std::vector<BlockElementSlice> flat_block_element_chunks(const smesh::Mesh &mesh,
                                                                    const ElementScope scope,
                                                                    const ptrdiff_t   flat_begin,
                                                                    const ptrdiff_t   flat_end) {
        std::vector<BlockElementSlice> out;
        if (flat_end <= flat_begin) {
            return out;
        }
        ptrdiff_t cursor = 0;
        for (const auto &slice : block_element_slices(mesh, scope)) {
            const ptrdiff_t n = slice.range.size();
            const ptrdiff_t slice_flat_begin = cursor;
            const ptrdiff_t slice_flat_end   = cursor + n;
            cursor                           = slice_flat_end;

            const ptrdiff_t begin = std::max(flat_begin, slice_flat_begin);
            const ptrdiff_t end     = std::min(flat_end, slice_flat_end);
            if (begin >= end) {
                continue;
            }
            const ptrdiff_t local_begin = slice.range.begin + (begin - slice_flat_begin);
            const ptrdiff_t local_end   = slice.range.begin + (end - slice_flat_begin);
            out.push_back({slice.block, {local_begin, local_end}});
        }
        return out;
    }

}  // namespace sfem
