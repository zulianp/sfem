#include "sfem_Communicator.hpp"

class Communicator::Impl {
public:
    Impl();
    ~Impl();

#ifdef SFEM_ENABLE_MPI
    MPI_Comm comm;
    Impl(MPI_Comm comm) : comm(comm) {}
#endif
};

Communicator::Communicator() : impl_(std::make_unique<Impl>()) {}
Communicator::~Communicator() {}

std::shared_ptr<Communicator> Communicator::world() {
#ifdef SFEM_ENABLE_MPI
    return std::make_shared<Communicator>(MPI_COMM_WORLD);
#else
    return std::make_shared<Communicator>();
#endif
}

std::shared_ptr<Communicator> Communicator::null() {
#ifdef SFEM_ENABLE_MPI
    return std::make_shared<Communicator>(MPI_COMM_NULL);
#else
    return std::make_shared<Communicator>();
#endif
}

#ifdef SFEM_ENABLE_MPI
Communicator::Communicator(MPI_Comm comm) : impl_(std::make_unique<Impl>(comm)) {}
MPI_Comm &Communicator::comm() { return impl_->comm; }
#endif

#ifdef SFEM_ENABLE_MPI
std::shared_ptr<Communicator> Communicator::wrap(MPI_Comm comm) {
    return std::make_shared<Communicator>(comm);
}
#endif