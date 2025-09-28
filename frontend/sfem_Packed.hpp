#ifndef SFEM_PACKED_HPP
#define SFEM_PACKED_HPP

#include "sfem_base.h"

#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Mesh.hpp"

#include <memory>

namespace sfem {
    template <typename pack_idx_t>
    class Packed {
    public:
        Packed();
        ~Packed();
        static std::shared_ptr<Packed> create(const std::shared_ptr<Mesh>    &mesh,
                                              const std::vector<std::string> &block_names = {},
                                              const bool                      modify_mesh = false);

        void map_to_packed(const real_t *const SFEM_RESTRICT values,
                           real_t *const SFEM_RESTRICT       out_values,
                           const int                         block_size = 1) const;
        //

        void                   map_to_unpacked(const real_t *const SFEM_RESTRICT values,
                                               real_t *const SFEM_RESTRICT       out_values,
                                               const int                         block_size = 1) const;
        SharedBuffer<geom_t *> points();

        ptrdiff_t                  n_blocks() const;
        std::string                block_name(const int block_idx) const;
        SharedBuffer<pack_idx_t *> elements(const int block_idx) const;
        SharedBuffer<ptrdiff_t>    owned_nodes_ptr(const int block_idx) const;
        SharedBuffer<ptrdiff_t>    n_shared(const int block_idx) const;
        SharedBuffer<ptrdiff_t>    ghost_ptr(const int block_idx) const;
        SharedBuffer<idx_t>        ghost_idx(const int block_idx) const;
        ptrdiff_t                  n_packs(const int block_idx) const;
        ptrdiff_t                  n_elements_per_pack(const int block_idx) const;

        const ptrdiff_t max_nodes_per_pack() const;

    private:
        class Block;
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem

#endif  // SFEM_PACKED_HPP
