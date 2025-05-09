#include "sfem_Grid.hpp"
#include "sfem_Input.hpp"

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

#include "sfem_glob.hpp"
#include "sfem_Tracer.hpp"

#include <fstream>
#include <sstream>

namespace sfem {

    template <class T>
    class Grid<T>::Impl {
    public:
        MPI_Comm                   comm;
        std::shared_ptr<Buffer<T>> field;
        int                        block_size{1};
        int                        spatial_dimension{3};

        ptrdiff_t nlocal[3];
        ptrdiff_t nglobal[3];
        ptrdiff_t stride[3];

        // Grid geometry
        geom_t origin[3];
        geom_t delta[3];
    };

    template <class T>
    MPI_Datatype Grid<T>::mpi_data_type() const {
        if (std::is_same<T, float>::value) {
            return MPI_FLOAT;
        }

        if (std::is_same<T, double>::value) {
            return MPI_DOUBLE;
        }

        if (std::is_same<T, int>::value) {
            return MPI_INT;
        }

        if (std::is_same<T, long>::value) {
            return MPI_LONG;
        }

        SFEM_ERROR("Unsupported type request in Grid::mpi_data_type()\n");
        return MPI_DATATYPE_NULL;
    }

    template <class T>
    Grid<T>::Grid(MPI_Comm comm) : impl_(std::make_unique<Impl>()) {
        impl_->comm = comm;
    }

    template <class T>
    Grid<T>::~Grid() = default;

    template <class T>
    std::shared_ptr<Grid<T>> Grid<T>::create_from_file(MPI_Comm comm, const std::string &path) {
        auto ret = std::make_unique<Grid<T>>(comm);
        auto in  = YAMLNoIndent::create_from_file(path + "/meta.yaml");

        std::string field_path;
        in->require("path", field_path);

        in->require("spatial_dimension", ret->impl_->spatial_dimension);

        assert(ret->impl_->spatial_dimension > 0);
        assert(ret->impl_->spatial_dimension <= 3);

        in->require("nx", ret->impl_->nglobal[0]);
        in->require("ox", ret->impl_->origin[0]);
        in->require("dx", ret->impl_->delta[0]);

        if (ret->impl_->spatial_dimension > 1) {
            in->require("ny", ret->impl_->nglobal[1]);
            in->require("oy", ret->impl_->origin[1]);
            in->require("dy", ret->impl_->delta[1]);
        }

        if (ret->impl_->spatial_dimension > 2) {
            in->require("nz", ret->impl_->nglobal[2]);
            in->require("oz", ret->impl_->origin[2]);
            in->require("dz", ret->impl_->delta[2]);
        }

        in->get("block_size", ret->impl_->block_size);

        T *data;
        if (ndarray_create_from_file(ret->impl_->comm,
                                     field_path.c_str(),
                                     ret->mpi_data_type(),
                                     ret->impl_->spatial_dimension,
                                     (void **)&data,
                                     ret->impl_->nlocal,
                                     ret->impl_->nglobal) != SFEM_SUCCESS) {
            SFEM_ERROR("Grid::create_from_file: Unable to read %s\n", field_path.c_str());
            return nullptr;
        }

        ret->impl_->stride[0] = 1;
        ret->impl_->stride[1] = ret->impl_->nlocal[0];
        ret->impl_->stride[2] = ret->impl_->nlocal[0] * ret->impl_->nlocal[1];

        ptrdiff_t size    = ret->impl_->stride[2] * ret->impl_->nlocal[2];
        ret->impl_->field = Buffer<T>::own(size, data, free, MEMORY_SPACE_HOST);

        geom_t SFEM_GRID_SHIFT = 0;
        geom_t SFEM_GRID_SCALE = 1;

        SFEM_READ_ENV(SFEM_GRID_SHIFT, atof);
        SFEM_READ_ENV(SFEM_GRID_SCALE, atof);

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < size; i++) {
            data[i] = SFEM_GRID_SCALE * (data[i] + SFEM_GRID_SHIFT);
        }
        return ret;
    }

    template <class T>
    int Grid<T>::to_file(const std::string &folder) {
        std::stringstream ss;

        std::string field_path;
        ss << "path: " << "sdf.raw\n";
        ss << "spatial_dimension: " << impl_->spatial_dimension << "\n";
        ss << "nx: " << impl_->nglobal[0] << "\n";
        ss << "ox: " << impl_->origin[0] << "\n";
        ss << "dx: " << impl_->delta[0] << "\n";

        if (impl_->spatial_dimension > 1) {
            ss << "ny: " << impl_->nglobal[1] << "\n";
            ss << "oy: " << impl_->origin[1] << "\n";
            ss << "dy: " << impl_->delta[1] << "\n";
        }

        if (impl_->spatial_dimension > 2) {
            ss << "nz: " << impl_->nglobal[2] << "\n";
            ss << "oz: " << impl_->origin[2] << "\n";
            ss << "dz: " << impl_->delta[2] << "\n";
        }

        ss << "block_size: " << impl_->block_size << "\n";

        sfem::create_directory(folder.c_str());
        std::ofstream os(folder + "/meta.yaml");
        if (!os.good()) return SFEM_FAILURE;
        os << ss.str();
        os.close();

        return impl_->field->to_file((folder + "/sdf.raw").c_str());
    }

    template <class T>
    ptrdiff_t Grid<T>::stride(int dim) const {
        return impl_->stride[dim];
    }

    template <class T>
    ptrdiff_t Grid<T>::extent(int dim) const {
        return impl_->nlocal[dim];
    }

    template <class T>
    ptrdiff_t Grid<T>::size() const {
        return impl_->field->size();
    }

    template <class T>
    int Grid<T>::block_size() const {
        return impl_->block_size;
    }

    template <class T>
    const ptrdiff_t *const Grid<T>::nlocal() const {
        return impl_->nlocal;
    }

    template <class T>
    const ptrdiff_t *const Grid<T>::nglobal() const {
        return impl_->nglobal;
    }

    template <class T>
    const ptrdiff_t *const Grid<T>::stride() const {
        return impl_->stride;
    }

    template <class T>
    int Grid<T>::spatial_dimension() const {
        return impl_->spatial_dimension;
    }

    template <class T>
    std::shared_ptr<Buffer<T>> Grid<T>::buffer() {
        return impl_->field;
    }

    template <class T>
    T *Grid<T>::data() {
        return impl_->field->data();
    }

    template <class T>
    const geom_t *const Grid<T>::origin() const {
        return impl_->origin;
    }

    template <class T>
    const geom_t *const Grid<T>::delta() const {
        return impl_->delta;
    }

    template <class T>
    std::shared_ptr<Grid<T>> Grid<T>::create(MPI_Comm        comm,
                                             const ptrdiff_t nx,
                                             const ptrdiff_t ny,
                                             const ptrdiff_t nz,
                                             const geom_t    xmin,
                                             const geom_t    ymin,
                                             const geom_t    zmin,
                                             const geom_t    xmax,
                                             const geom_t    ymax,
                                             const geom_t    zmax) {
        auto ret = std::make_unique<Grid<T>>(comm);

        auto &impl = ret->impl_;

        impl->block_size        = 1;
        impl->spatial_dimension = 3;

        // FIXME
        impl->nlocal[0] = nx;
        impl->nlocal[1] = ny;
        impl->nlocal[2] = nz;

        // FIXME
        impl->stride[0] = 1;
        impl->stride[1] = nx;
        impl->stride[2] = nx * ny;

        // FIXME
        impl->nglobal[0] = nx;
        impl->nglobal[1] = ny;
        impl->nglobal[2] = nz;

        // Grid geometry
        impl->origin[0] = xmin;
        impl->origin[1] = ymin;
        impl->origin[2] = zmin;

        impl->delta[0] = (xmax - xmin) / (nx - 1);
        impl->delta[1] = (ymax - ymin) / (ny - 1);
        impl->delta[2] = (zmax - zmin) / (nz - 1);

        impl->field = create_host_buffer<T>(nx * ny * nz);
        return ret;
    }

    // Explicit instantiation
    template class Grid<float>;
    template class Grid<double>;

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
                                             std::function<geom_t(const geom_t, const geom_t, const geom_t)> f) {
        SFEM_TRACE_SCOPE("Grid::create_sdf");

        auto g = Grid<geom_t>::create(comm, nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax);

        const geom_t hx = (xmax - xmin) / (nx - 1);
        const geom_t hy = (ymax - ymin) / (ny - 1);
        const geom_t hz = (zmax - zmin) / (nz - 1);

        geom_t *const field  = g->buffer()->data();
        auto          stride = g->stride();

#pragma omp parallel for
        for (ptrdiff_t zi = 0; zi < nz; zi++) {
            for (ptrdiff_t yi = 0; yi < ny; yi++) {
                for (ptrdiff_t xi = 0; xi < nx; xi++) {
                    const geom_t x                                          = xmin + xi * hx;
                    const geom_t y                                          = ymin + yi * hy;
                    const geom_t z                                          = zmin + zi * hz;
                    const geom_t fx                                         = f(x, y, z);
                    field[xi * stride[0] + yi * stride[1] + zi * stride[2]] = fx;
                }
            }
        }

        return g;
    }
}  // namespace sfem
