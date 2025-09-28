#include "sfem_Packed.hpp"

#include "sfem_macros.h"
#include "sfem_mask.h"

#include "sfem_Env.hpp"
#include "sfem_Mesh.hpp"

#include <cstdint>
#include <limits>

namespace sfem {

    template <typename pack_idx_t>
    class Packed<pack_idx_t>::Block {
    public:
        ptrdiff_t n_packs;
        ptrdiff_t elements_per_pack;

        std::shared_ptr<Mesh::Block> block;
        SharedBuffer<pack_idx_t *>   packed_elements;

        SharedBuffer<ptrdiff_t> owned_nodes_ptr;
        SharedBuffer<ptrdiff_t> shared_nodes_ptr;
        SharedBuffer<count_t>   ghost_ptr;
        SharedBuffer<idx_t>     ghost_idx;

        size_t nbytes() const {
            return packed_elements->nbytes() + owned_nodes_ptr->nbytes() + ghost_ptr->nbytes() + ghost_idx->nbytes();
        }

        void print(std::ostream &os = std::cout) const {
            os << "--------------------" << std::endl;
            os << "| Packed |" << std::endl;
            os << "n_packs: " << n_packs << std::endl;
            os << "elements_per_pack: " << elements_per_pack << std::endl;
            os << "Memory Packed: " << (nbytes() / 1204.) << " KB" << std::endl;
            os << "Original:      " << (block->elements()->nbytes() / 1204.) << " KB" << std::endl;
            os << "--------------------" << std::endl;
        }

        void identify_owner(const ptrdiff_t             pack_offset,
                            const SharedBuffer<idx_t>  &node_owner,
                            const SharedBuffer<mask_t> &selected) {
            auto            d_elements = block->elements()->data();
            const int       nxe        = block->n_nodes_per_element();
            const ptrdiff_t nelements  = block->n_elements();

            auto d_node_owner = node_owner->data();
            auto d_selected   = selected->data();

            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t start = p * elements_per_pack;
                const ptrdiff_t end   = MIN(nelements, (p + 1) * elements_per_pack);

                for (ptrdiff_t e = start; e < end; e++) {
                    for (int v = 0; v < nxe; v++) {
                        const idx_t node = d_elements[v][e];
                        if (!mask_get(node, d_selected)) {
                            d_node_owner[node] = p + pack_offset;
                            mask_set(node, d_selected);
                        }
                    }
                }
            }
        }

        void identify_shared(const ptrdiff_t             pack_offset,
                             const SharedBuffer<idx_t>  &node_owner,
                             const SharedBuffer<mask_t> &shared) {
            auto d_node_owner = node_owner->data();
            auto d_shared     = shared->data();
            memset(d_shared, 0, shared->nbytes());
            auto d_elements = block->elements()->data();
            const int nxe = block->n_nodes_per_element();
            const ptrdiff_t nelements = block->n_elements();

            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t start = p * elements_per_pack;
                const ptrdiff_t end   = MIN(nelements, (p + 1) * elements_per_pack);

                for (ptrdiff_t e = start; e < end; e++) {
                    for (int v = 0; v < nxe; v++) {
                        const idx_t node = d_elements[v][e];

                        if (d_node_owner[node] != p + pack_offset) {
                            mask_set(node, d_shared);
                        }
                    }
                }
            }
        }

        void construct_owned_shared_ghost_and_map(const ptrdiff_t             pack_offset,
                                                  const SharedBuffer<idx_t>  &node_map,
                                                  const SharedBuffer<idx_t>  &node_owner,
                                                  const SharedBuffer<mask_t> &shared) {
            auto d_node_map         = node_map->data();
            auto d_elements         = block->elements()->data();
            auto d_node_owner       = node_owner->data();
            auto d_shared_nodes_ptr = shared_nodes_ptr->data();
            auto d_shared           = shared->data();
            auto d_owned_nodes_ptr  = owned_nodes_ptr->data();
            auto d_ghost_ptr        = ghost_ptr->data();


            const int       nxe        = block->n_nodes_per_element();
            const ptrdiff_t nelements  = block->n_elements();
            const ptrdiff_t nnodes     = node_map->size();

            SharedBuffer<mask_t> selected = sfem::create_host_buffer<mask_t>(mask_count(nnodes));
            auto d_selected = selected->data();

            ptrdiff_t next_id = 0;
            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t start = p * elements_per_pack;
                const ptrdiff_t end   = MIN(nelements, (p + 1) * elements_per_pack);

                ptrdiff_t nowned  = 0;
                ptrdiff_t nshared = 0;

                for (ptrdiff_t e = start; e < end; e++) {
                    for (int v = 0; v < nxe; v++) {
                        const idx_t node = d_elements[v][e];

                        if(!mask_get(node, d_selected)) {
                            bool is_owned  = d_node_owner[node] == p + pack_offset;
                            bool is_shared = is_owned && mask_get(node, d_shared);

                            nowned += is_owned && !is_shared;
                            nshared += is_shared;
                            mask_set(node, d_selected);
                        }
                    }
                }

                for (ptrdiff_t e = start; e < end; e++) {
                    for (int v = 0; v < nxe; v++) {
                        const idx_t node = d_elements[v][e];
                        mask_unset(node, d_selected);
                    }
                }

                ptrdiff_t owned_idx  = 0;
                ptrdiff_t shared_idx = nowned;
                ptrdiff_t ghost_idx  = nowned + nshared;

                for (ptrdiff_t e = start; e < end; e++) {
                    for (int v = 0; v < nxe; v++) {
                        const idx_t node = d_elements[v][e];
                        bool is_selected = mask_get(node, d_selected);
                        if(is_selected) continue;
                        mask_set(node, d_selected);

                        bool is_owned  = d_node_owner[node] == p + pack_offset;
                        bool is_shared = is_owned && mask_get(node, d_shared);
                        bool is_ghost  = !is_owned;

                        if (is_shared) {
                            d_shared_nodes_ptr[p + 1]++;
                            d_node_map[node] = shared_idx++;
                            assert(d_node_map[node] < nnodes);
                            assert(d_node_map[node] >= 0);
                        } else if (is_owned) {
                            d_owned_nodes_ptr[p + 1]++;
                            d_node_map[node] = owned_idx++;
                            assert(d_node_map[node] < nnodes);
                            assert(d_node_map[node] >= 0);
                        } else {
                            d_ghost_ptr[p + 1]++;
                        }
                    }
                }

                next_id += nowned + nshared;

                // Accumulate
                d_owned_nodes_ptr[p + 1] += d_owned_nodes_ptr[p];
                d_shared_nodes_ptr[p + 1] += d_shared_nodes_ptr[p];
                d_ghost_ptr[p + 1] += d_ghost_ptr[p];
            }
        }

        void pack(const std::shared_ptr<Mesh> &mesh,
                  const ptrdiff_t              pack_offset,
                  const SharedBuffer<idx_t>   &node_map,
                  const SharedBuffer<idx_t>   &node_owner,
                  const SharedBuffer<mask_t>  &shared) {
            auto d_elements         = block->elements()->data();
            auto d_packed_elements  = this->packed_elements->data();
            auto d_owned_nodes_ptr  = this->owned_nodes_ptr->data();
            auto d_shared_nodes_ptr = this->shared_nodes_ptr->data();
            auto d_ghost_ptr        = this->ghost_ptr->data();
            auto d_node_map         = node_map->data();
            auto d_node_owner       = node_owner->data();
            auto d_shared           = shared->data();

            const int       nxe       = block->n_nodes_per_element();
            const ptrdiff_t nelements = block->n_elements();

            this->ghost_idx  = sfem::create_host_buffer<idx_t>(d_ghost_ptr[n_packs]);
            auto d_ghost_idx = this->ghost_idx->data();

            for (ptrdiff_t p = 0; p < n_packs; p++) {
                const ptrdiff_t start = p * elements_per_pack;
                const ptrdiff_t end   = MIN(nelements, (p + 1) * elements_per_pack);

                const ptrdiff_t nowned       = d_owned_nodes_ptr[p + 1] - d_owned_nodes_ptr[p];
                const ptrdiff_t nshared      = d_shared_nodes_ptr[p + 1] - d_shared_nodes_ptr[p];
                const ptrdiff_t ghost_offset = nowned + nshared;

                pack_idx_t bkghosts = 0;
                for (int v = 0; v < nxe; v++) {
                    for (ptrdiff_t e = start; e < end; e++) {
                        const ptrdiff_t node     = d_elements[v][e];
                        bool            is_owned = d_node_owner[node] == p + pack_offset;

                        assert(d_node_map[node] < mesh->n_nodes());

                        if (is_owned) {
                            d_packed_elements[v][e] = d_node_map[d_elements[v][e]] - d_owned_nodes_ptr[p];
                            assert(d_packed_elements[v][e] + d_owned_nodes_ptr[p] == d_node_map[d_elements[v][e]]);
                        } else {
                            d_ghost_idx[d_ghost_ptr[p] + bkghosts] = d_node_map[node];
                            d_packed_elements[v][e]                = ghost_offset + bkghosts++;
                        }
                    }
                }
            }
        }
    };

    template <typename pack_idx_t>
    class Packed<pack_idx_t>::Impl {
    public:
        static const ptrdiff_t max_nodes_per_pack = std::numeric_limits<pack_idx_t>::max() + 1l;

        std::shared_ptr<Mesh>  mesh;
        SharedBuffer<idx_t>    node_map;
        SharedBuffer<geom_t *> reordered_points;
        bool                   synched_with_mesh{false};

        // New mesh data
        std::vector<std::shared_ptr<Block>> blocks;

        void init(const std::shared_ptr<Mesh> &mesh, const std::vector<std::string> &block_names, const bool modify_mesh) {
            this->mesh = mesh;

            const ptrdiff_t SFEM_ELEMENTS_PER_PACK = sfem::Env::read("SFEM_ELEMENTS_PER_PACK", 0);

            node_map = sfem::create_host_buffer<idx_t>(mesh->n_nodes());

            auto node_owner = sfem::create_host_buffer<idx_t>(mesh->n_nodes());
            auto selected   = sfem::create_host_buffer<mask_t>(mask_count(mesh->n_nodes()));

            // Construct packed blocks storage
            ptrdiff_t pack_offset = 0;
            for (auto &block : mesh->blocks(block_names)) {
                auto packed_block   = std::make_shared<Block>();
                packed_block->block = block;

                packed_block->n_packs =
                        (block->n_elements() * block->n_nodes_per_element() + max_nodes_per_pack - 1) / max_nodes_per_pack;
                packed_block->elements_per_pack = (block->n_elements() + packed_block->n_packs - 1) / packed_block->n_packs;

                if (SFEM_ELEMENTS_PER_PACK) {
                    packed_block->elements_per_pack = MIN(packed_block->elements_per_pack, SFEM_ELEMENTS_PER_PACK);
                    packed_block->n_packs =
                            (block->n_elements() + packed_block->elements_per_pack - 1) / packed_block->elements_per_pack;
                }

                assert(packed_block->elements_per_pack * block->n_nodes_per_element() <= max_nodes_per_pack);

                packed_block->packed_elements =
                        sfem::create_host_buffer<pack_idx_t>(block->n_nodes_per_element(), block->n_elements());
                packed_block->owned_nodes_ptr  = sfem::create_host_buffer<ptrdiff_t>(packed_block->n_packs + 1);
                packed_block->shared_nodes_ptr = sfem::create_host_buffer<ptrdiff_t>(packed_block->n_packs + 1);
                packed_block->ghost_ptr        = sfem::create_host_buffer<idx_t>(packed_block->n_packs + 1);

                blocks.push_back(packed_block);
                pack_offset += packed_block->n_packs;
            }

            pack_offset = 0;
            for (auto &packed_block : blocks) {
                packed_block->identify_owner(pack_offset, node_owner, selected);
                pack_offset += packed_block->n_packs;
            }

            pack_offset = 0;
            for (auto &packed_block : blocks) {
                packed_block->identify_shared(pack_offset, node_owner, selected);
                pack_offset += packed_block->n_packs;
            }

            pack_offset = 0;
            for (auto &packed_block : blocks) {
                packed_block->construct_owned_shared_ghost_and_map(pack_offset, node_map, node_owner, selected);
                pack_offset += packed_block->n_packs;
            }

            pack_offset = 0;
            for (auto &packed_block : blocks) {
                packed_block->pack(mesh, pack_offset, node_map, node_owner, selected);
                pack_offset += packed_block->n_packs;
            }

            if (modify_mesh) {
                mesh->renumber_nodes(node_map);
                reordered_points  = mesh->points();
                synched_with_mesh = true;
                node_map          = nullptr;
            }
        }

        void init_reordered_points() {
            if (!reordered_points) {
                reordered_points                   = sfem::zeros_like(mesh->points());
                auto            d_reordered_points = reordered_points->data();
                auto            d_points           = mesh->points()->data();
                auto            d_node_map         = node_map->data();
                const ptrdiff_t nnodes             = mesh->n_nodes();
                const int       dim                = mesh->spatial_dimension();

                for (int d = 0; d < dim; d++) {
                    for (ptrdiff_t node = 0; node < nnodes; node++) {
                        d_reordered_points[d][d_node_map[node]] = d_points[d][node];
                    }
                }
            }
        }
    };

    template <typename pack_idx_t>
    ptrdiff_t Packed<pack_idx_t>::n_blocks() const {
        return impl_->blocks.size();
    }
    template <typename pack_idx_t>
    SharedBuffer<pack_idx_t *> Packed<pack_idx_t>::elements(const int block_idx) const {
        return impl_->blocks[block_idx]->packed_elements;
    }

    template <typename pack_idx_t>
    SharedBuffer<ptrdiff_t> Packed<pack_idx_t>::owned_nodes_ptr(const int block_idx) const {
        return impl_->blocks[block_idx]->owned_nodes_ptr;
    }

    template <typename pack_idx_t>
    SharedBuffer<ptrdiff_t> Packed<pack_idx_t>::shared_nodes_ptr(const int block_idx) const {
        return impl_->blocks[block_idx]->shared_nodes_ptr;
    }

    template <typename pack_idx_t>
    SharedBuffer<count_t> Packed<pack_idx_t>::ghost_ptr(const int block_idx) const {
        return impl_->blocks[block_idx]->ghost_ptr;
    }
    template <typename pack_idx_t>
    SharedBuffer<idx_t> Packed<pack_idx_t>::ghost_idx(const int block_idx) const {
        return impl_->blocks[block_idx]->ghost_idx;
    }

    template <typename pack_idx_t>
    std::string Packed<pack_idx_t>::block_name(const int block_idx) const {
        return impl_->blocks[block_idx]->block->name();
    }

    template <typename pack_idx_t>
    Packed<pack_idx_t>::Packed() : impl_(std::make_unique<Impl>()) {}

    template <typename pack_idx_t>
    Packed<pack_idx_t>::~Packed() = default;

    template <typename pack_idx_t>
    std::shared_ptr<Packed<pack_idx_t>> Packed<pack_idx_t>::create(const std::shared_ptr<Mesh>    &mesh,
                                                                   const std::vector<std::string> &block_names,
                                                                   const bool                      modify_mesh) {
        auto packed = std::make_shared<Packed<pack_idx_t>>();
        packed->impl_->init(mesh, block_names, modify_mesh);
        return packed;
    }

    template <typename pack_idx_t>
    void Packed<pack_idx_t>::map_to_packed(const real_t *const SFEM_RESTRICT values,
                                           real_t *const SFEM_RESTRICT       out_values,
                                           const int                         block_size) const {
        if (!impl_->synched_with_mesh) SFEM_ERROR("Mesh is not synched! Cannot call this!\n");

        auto            d_node_map = impl_->node_map->data();
        const ptrdiff_t nnodes     = impl_->mesh->n_nodes();
        for (ptrdiff_t node = 0; node < nnodes; node++) {
            out_values[d_node_map[node]] = values[node];
        }
    }

    template <typename pack_idx_t>
    void Packed<pack_idx_t>::map_to_unpacked(const real_t *const SFEM_RESTRICT values,
                                             real_t *const SFEM_RESTRICT       out_values,
                                             const int                         block_size) const {
        if (!impl_->synched_with_mesh) SFEM_ERROR("Mesh is not synched! Cannot call this!\n");

        auto            d_node_map = impl_->node_map->data();
        const ptrdiff_t nnodes     = impl_->mesh->n_nodes();
        for (ptrdiff_t node = 0; node < nnodes; node++) {
            out_values[node] = values[d_node_map[node]];
        }
    }

    template <typename pack_idx_t>
    SharedBuffer<geom_t *> Packed<pack_idx_t>::points() {
        impl_->init_reordered_points();

        return impl_->reordered_points;
    }

    template <typename pack_idx_t>
    const ptrdiff_t Packed<pack_idx_t>::max_nodes_per_pack() const {
        return Impl::max_nodes_per_pack;
    }

    template <typename pack_idx_t>
    ptrdiff_t Packed<pack_idx_t>::n_packs(const int block_idx) const {
        return impl_->blocks[block_idx]->n_packs;
    }

    template <typename pack_idx_t>
    ptrdiff_t Packed<pack_idx_t>::n_elements_per_pack(const int block_idx) const {
        return impl_->blocks[block_idx]->elements_per_pack;
    }

    template class Packed<uint8_t>;
    template class Packed<int16_t>;
    template class Packed<uint16_t>;

}  // namespace sfem
