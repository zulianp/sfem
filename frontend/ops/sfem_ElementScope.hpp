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

    /// True when each block has owned+ghosts matching its local SoA (serial always).
    inline bool mesh_supports_distributed_element_scopes(const smesh::Mesh &mesh) {
        if (!mesh_is_distributed(mesh)) {
            return true;
        }
        if (!mesh.distributed()) {
            return false;
        }
        for (smesh::block_idx_t b = 0; b < static_cast<smesh::block_idx_t>(mesh.n_blocks()); ++b) {
            const auto block = mesh.block(b);
            if (!block) {
                return false;
            }
            if (block->n_elements_owned() < block->n_elements_shared()) {
                return false;
            }
            if (block->n_elements() != block->n_elements_owned() + block->n_elements_ghosts()) {
                return false;
            }
        }
        return true;
    }

    /// Abort if a distributed mesh is missing per-block owned/shared/aura layout.
    inline void assert_mesh_supports_distributed_element_scopes(const smesh::Mesh &mesh) {
        if (!mesh_is_distributed(mesh)) {
            return;
        }
        if (!mesh.distributed()) {
            SFEM_ERROR("ElementScope on distributed mesh: Mesh::distributed() is unset.\n");
        }
        for (smesh::block_idx_t b = 0; b < static_cast<smesh::block_idx_t>(mesh.n_blocks()); ++b) {
            const auto block = mesh.block(b);
            if (!block) {
                SFEM_ERROR("ElementScope on distributed mesh: block %d is null.\n", (int)b);
            }
            if (block->n_elements_owned() < block->n_elements_shared()) {
                SFEM_ERROR(
                        "ElementScope on distributed mesh: block %d n_shared (%ld) > n_owned (%ld).\n",
                        (int)b,
                        (long)block->n_elements_shared(),
                        (long)block->n_elements_owned());
            }
            const ptrdiff_t n_block = block->n_elements();
            const ptrdiff_t n_local = block->n_elements_owned() + block->n_elements_ghosts();
            if (n_block != n_local) {
                SFEM_ERROR(
                        "ElementScope on distributed mesh: block %d has %ld elements but owned+ghosts is %ld; "
                        "block layout must match smesh per-block element ordering.\n",
                        (int)b,
                        (long)n_block,
                        (long)n_local);
            }
        }
    }

    inline ptrdiff_t block_element_offset(const smesh::Mesh &mesh, const smesh::block_idx_t block_idx) {
        ptrdiff_t off = 0;
        for (smesh::block_idx_t b = 0; b < block_idx; ++b) {
            off += mesh.n_elements(b);
        }
        return off;
    }

    /// Block-local [begin, end) for @p scope. Distributed ranges come from that block's
    /// owned-not-shared / owned / local counts (SoA order: ONS | shared | aura).
    inline ElementRange element_range(const smesh::Mesh &mesh, const smesh::block_idx_t block_idx, const ElementScope scope) {
        if (static_cast<size_t>(block_idx) >= mesh.n_blocks()) {
            SFEM_ERROR("element_range: block index %d out of range (n_blocks=%ld).\n",
                       (int)block_idx,
                       (long)mesh.n_blocks());
        }

        const ptrdiff_t n = mesh.n_elements(block_idx);

        if (!mesh_is_distributed(mesh)) {
            switch (scope) {
                case ElementScope::ALL:
                case ElementScope::OWNED_NOT_SHARED:
                    return {0, n};
                case ElementScope::SHARED_AND_AURA:
                    return {0, 0};
            }
            return {0, 0};
        }

        const auto      block   = mesh.block(block_idx);
        const ptrdiff_t n_ons   = block->n_elements_owned_not_shared();
        const ptrdiff_t n_local = n;
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

    inline std::vector<BlockElementSlice> block_element_slices(const smesh::Mesh &mesh, const ElementScope scope) {
        std::vector<BlockElementSlice> slices;
        const size_t                   n_blocks = mesh.n_blocks();
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

    /// Concatenated flat span [0, count) of per-block slices for @p scope (not SoA indices).
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

        return {0, count_block_elements(mesh, scope)};
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
