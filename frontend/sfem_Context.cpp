#include "sfem_Context.hpp"
#include "sfem_Communicator.hpp"

#include "sfem_base.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#endif

namespace sfem {

    class Context::Impl {
    public:
        MPI_Comm comm;
        bool     owns_mpi_context{false};
        std::shared_ptr<Communicator> communicator;

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
        impl_->communicator     = Communicator::world();
    }

    Context::Context(int argc, char *argv[], MPI_Comm comm) : impl_(std::make_unique<Impl>()) { 
        impl_->comm = comm; 
        impl_->communicator = Communicator::wrap(comm);
    }

    Context::~Context() {
        if (impl_->owns_mpi_context) {
            MPI_Finalize();
        }
    }
    MPI_Comm Context::comm() { return impl_->comm; }
    std::shared_ptr<Communicator> Context::communicator() { return impl_->communicator; }

}  // namespace sfem
