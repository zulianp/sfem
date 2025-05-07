#include "sfem_Context.hpp"

#include "sfem_base.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace sfem {

    class Context::Impl {
    public:
        MPI_Comm comm;
        bool     owns_mpi_context{false};

        Impl() {
#ifdef SFEM_ENABLE_CUDA
            sfem::register_device_ops();
#endif
        }
    };

    Context::Context(int argc, char *argv[]) : impl_(std::make_unique<Impl>()) {
        MPI_Init(&argc, &argv);
        impl_->comm             = MPI_COMM_WORLD;
        impl_->owns_mpi_context = true;
    }

    Context::Context(int argc, char *argv[], MPI_Comm comm) : impl_(std::make_unique<Impl>()) { impl_->comm = comm; }

    Context::~Context() {
        if (impl_->owns_mpi_context) {
            MPI_Finalize();
        }
    }
    MPI_Comm Context::comm() { return impl_->comm; }

}  // namespace sfem
