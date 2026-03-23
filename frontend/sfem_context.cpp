#include "sfem_context.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace sfem {

    std::shared_ptr<smesh::Context> initialize(int argc, char *argv[]) {
#ifdef SMESH_ENABLE_CUDA
        register_device_ops();
#endif

        return smesh::initialize(argc, argv);
    }

    std::shared_ptr<smesh::Context> initialize_serial(int argc, char *argv[]) {
#ifdef SMESH_ENABLE_CUDA
        register_device_ops();
#endif
        return smesh::initialize_serial(argc, argv);
    }

}  // namespace sfem