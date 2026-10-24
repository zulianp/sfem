#include "sfem_execution_context.hpp"

#include "sfem_config.h"

#ifdef SFEM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sfem {

    void* default_device_stream() {
#ifdef SFEM_ENABLE_CUDA
        return (void*)cudaStreamPerThread;
#else
        return nullptr;
#endif
    }

}  // namespace sfem
