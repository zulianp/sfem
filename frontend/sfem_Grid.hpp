#ifndef SFEM_GRID_HPP
#define SFEM_GRID_HPP

#include "sfem_Buffer.hpp"

#include <string>
#include <cstddef>
#include <memory>

#include <mpi.h>

#include "sfem_defs.h"

namespace sfem {
    template<class T>
    class Grid {
    public:
        static std::unique_ptr<Grid> create_from_file(MPI_Comm comm, const std::string &path);

        Grid(MPI_Comm comm);
        ~Grid();
        
        ptrdiff_t stride(int dim) const;
        ptrdiff_t extent(int dim) const;
        ptrdiff_t size() const;
        int spatial_dimension() const;
        int block_size() const;

        const ptrdiff_t * const nlocal() const;
        const ptrdiff_t * const nglobal() const;
        const ptrdiff_t * const stride() const;

        const geom_t *const origin() const;
        const geom_t *const delta() const;

        std::shared_ptr<Buffer<T>> buffer();
        T *data();

        MPI_Datatype mpi_data_type() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}

#endif //SFEM_GRID_HPP
