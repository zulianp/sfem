/**
 * @file sfem_ElementScope.hpp
 * @brief Block-local element ranges for phased assembly / compute-comm overlap
 */

#pragma once

#include "sfem_defs.hpp"
#include "smesh_mesh.hpp"

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

    /// Block-local [begin, end) for @p scope. Distributed ranges come from that block's
    /// owned-not-shared / owned / local counts (SoA order: ONS | shared | aura).
    inline ElementRange element_range(const smesh::Mesh &mesh, const smesh::Mesh::Block &block, const ElementScope scope) {
        const ptrdiff_t n = block.n_elements();

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

        const ptrdiff_t n_ons   = block.n_elements_owned_not_shared();
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

    inline ElementRange element_range(const smesh::Mesh &mesh, const smesh::block_idx_t block_idx, const ElementScope scope) {
        if (static_cast<size_t>(block_idx) >= mesh.n_blocks()) {
            SFEM_ERROR("element_range: block index %d out of range (n_blocks=%ld).\n",
                       (int)block_idx,
                       (long)mesh.n_blocks());
        }
        return element_range(mesh, *mesh.block(block_idx), scope);
    }

    /// A block's connectivity as raw pointers. Taken outside any region, it spares the threads of a
    /// team body the shared reference counts that Mesh::block and Mesh::elements copy.
    struct BlockElements {
        ptrdiff_t n_elements{0};
        idx_t   **elements{nullptr};
    };

    inline std::vector<BlockElements> block_elements(const smesh::Mesh &mesh) {
        std::vector<BlockElements> ret(mesh.n_blocks());
        for (size_t b = 0; b < ret.size(); ++b) {
            auto block = mesh.block(static_cast<smesh::block_idx_t>(b));
            ret[b]     = {block->n_elements(), block->elements()->data()};
        }
        return ret;
    }

}  // namespace sfem
