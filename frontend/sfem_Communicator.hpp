#ifndef SFEM_COMMUNICATOR_HPP
#define SFEM_COMMUNICATOR_HPP

#include "sfem_base.h"

#ifdef SFEM_ENABLE_MPI
#include <mpi.h>
#endif

#include <memory>

namespace sfem {

class Communicator {
public:
    Communicator();
    ~Communicator();
    static std::shared_ptr<Communicator> world();
    static std::shared_ptr<Communicator> null();
    static std::shared_ptr<Communicator> self();

    int rank() const;
    int size() const;

#ifdef SFEM_ENABLE_MPI
    static std::shared_ptr<Communicator> wrap(MPI_Comm comm);
    Communicator(MPI_Comm comm);
    MPI_Comm &comm();
#endif

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfem

#endif  // SFEM_COMMUNICATOR_HPP
