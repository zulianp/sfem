#include "sfem_context.hpp"

#include "sfem_config.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace sfem {

    std::shared_ptr<smesh::Context> initialize(int argc, char *argv[]) {
#ifdef SFEM_ENABLE_CUDA
        sfem::register_device_ops();
#endif

        return smesh::initialize(argc, argv);
    }

    std::shared_ptr<smesh::Context> initialize_serial(int argc, char *argv[]) {
#ifdef SFEM_ENABLE_CUDA
        sfem::register_device_ops();
#endif
        return smesh::initialize_serial(argc, argv);
    }

}  // namespace sfem