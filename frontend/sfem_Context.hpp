#ifndef CONTEXT_INIT_HPP
#define CONTEXT_INIT_HPP

#include <mpi.h>
#include <memory>

namespace sfem {
    class Context {
    public:
 
        Context(int argc, char *argv[]);
        Context(int argc, char *argv[], MPI_Comm comm);
        ~Context();
        MPI_Comm comm();

        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // CONTEXT_INIT_HPP
