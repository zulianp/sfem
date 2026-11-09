#pragma once

#include <memory>

#include "sfem_InexactInterface.hpp"
#include "smesh_crs_graph.hpp"

namespace sfem {
    namespace inexact {
        namespace cuda {

            /// The `bsr` store on the device (sfem_cuda_QuantBSRStore.cu): the host store's encoding
            /// (sfem_QuantBSRStore.hpp) in the node graph's own order -- the full difference form, a
            /// record per block, applied by a few threads per row as the BSR product is -- with the
            /// same row corrections for a term that does not annihilate translations.  `graph` is the
            /// host graph; the store keeps its own device copy.
            std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                       const bool                       int8,
                                                       const bool                       symmetric,
                                                       void *const                      stream);

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem
