#ifndef SFEM_ELASTICITY_ASSEMBLY_DATA_HPP
#define SFEM_ELASTICITY_ASSEMBLY_DATA_HPP

#include "sfem_config.h"

#include "sfem_aliases.hpp"
#include "sfem_MultiDomainOp.hpp"

#include <vector>

namespace sfem {

    struct ElasticityAssemblyData {
        SharedBuffer<metric_tensor_t> partial_assembly_buffer;
        SharedBuffer<scaling_t>       compression_scaling;
        SharedBuffer<compressed_t>    partial_assembly_compressed;
        SharedBuffer<idx_t *>         elements;
        ptrdiff_t                     elements_stride{1};

        // `elements` (with elements_stride) and the block's own columns from the first element of
        // ElementScope::SHARED_AND_AURA on, made at initialize on a distributed mesh: a scoped product
        // then reads them as they are, and builds no view inside a region.
        ptrdiff_t            interface_begin{0};
        std::vector<idx_t *> interface_elements;
        std::vector<idx_t *> interface_block_elements;

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};

        int compress_partial_assembly(const OpDomain &domain);
    };

}  // namespace sfem

#endif  // SFEM_ELASTICITY_ASSEMBLY_DATA_HPP
