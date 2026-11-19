#pragma once

#include <memory>

#include "sfem_InexactInterface.hpp"
#include "sfem_InexactOptions.hpp"
#include "smesh_crs_graph.hpp"

namespace sfem {
    namespace inexact {
        namespace cuda {

            /// The `bsr` store on the device (sfem_cuda_QuantBSRStore.cu): the host store's encoding
            /// (sfem_QuantBSRStore.hpp) in the node graph's own order -- the full difference form, a
            /// record per block, applied by a few threads per row as the BSR product is -- with the
            /// same row corrections for a term that does not annihilate translations; with `sell` the
            /// same records in SELL-32-1024 order, a thread per row (the faster on regular rows).
            /// f32, f16 or i8. `graph` is the host graph; the store keeps its own device copy.
            std::unique_ptr<BsrStore> create_bsr_store(const std::shared_ptr<CRSGraph> &graph,
                                                       const InexactOptions::Encoding   encoding,
                                                       const bool                       symmetric,
                                                       void *const                      stream,
                                                       const bool                       sell);

        }  // namespace cuda
    }  // namespace inexact
}  // namespace sfem
