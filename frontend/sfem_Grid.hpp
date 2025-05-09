#ifndef SFEM_GRID_HPP
#define SFEM_GRID_HPP

#include "sfem_Buffer.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include <mpi.h>

#include "sfem_defs.h"

namespace sfem {
    template <class T>
    class Grid {
    public:
        static std::shared_ptr<Grid> create_from_file(MPI_Comm comm, const std::string &path);

        static std::shared_ptr<Grid> create(MPI_Comm        comm,
                                            const ptrdiff_t nx,
                                            const ptrdiff_t ny,
                                            const ptrdiff_t nz,
                                            const geom_t    xmin,
                                            const geom_t    ymin,
                                            const geom_t    zmin,
                                            const geom_t xmax,
                                            const geom_t ymax,
                                            const geom_t zmax);

        int to_file(const std::string &folder);

        Grid(MPI_Comm comm);
        ~Grid();

        ptrdiff_t stride(int dim) const;
        ptrdiff_t extent(int dim) const;
        ptrdiff_t size() const;
        int       spatial_dimension() const;
        int       block_size() const;

        const ptrdiff_t *const nlocal() const;
        const ptrdiff_t *const nglobal() const;
        const ptrdiff_t *const stride() const;

        const geom_t *const origin() const;
        const geom_t *const delta() const;

        std::shared_ptr<Buffer<T>> buffer();
        T                         *data();

        MPI_Datatype mpi_data_type() const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<Grid<geom_t>> create_sdf(MPI_Comm                                                        comm,
                                             const ptrdiff_t                                                 nx,
                                             const ptrdiff_t                                                 ny,
                                             const ptrdiff_t                                                 nz,
                                             const geom_t                                                    xmin,
                                             const geom_t                                                    ymin,
                                             const geom_t                                                    zmin,
                                             const geom_t                                                    xmax,
                                             const geom_t                                                    ymax,
                                             const geom_t                                                    zmax,
                                             std::function<geom_t(const geom_t, const geom_t, const geom_t)> f);
}  // namespace sfem

#endif  // SFEM_GRID_HPP
