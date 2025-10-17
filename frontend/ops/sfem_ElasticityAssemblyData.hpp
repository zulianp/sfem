#ifndef SFEM_ELASTICITY_ASSEMBLY_DATA_HPP
#define SFEM_ELASTICITY_ASSEMBLY_DATA_HPP

#include "sfem_config.h"

#include "sfem_Buffer.hpp"
#include "sfem_MultiDomainOp.hpp"

namespace sfem {

    struct ElasticityAssemblyData {
        SharedBuffer<metric_tensor_t> partial_assembly_buffer;
        SharedBuffer<scaling_t>       compression_scaling;
        SharedBuffer<compressed_t>    partial_assembly_compressed;
        SharedBuffer<idx_t *>         elements;
        ptrdiff_t                     elements_stride{1};

        bool use_partial_assembly{false};
        bool use_compression{false};
        bool use_AoS{false};

        int compress_partial_assembly(OpDomain &domain);
    };

}  // namespace sfem

#endif  // SFEM_ELASTICITY_ASSEMBLY_DATA_HPP
