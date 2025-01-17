#include "sfem_Grid.hpp"
#include "sfem_Input.hpp"

#include "matrixio_array.h"
#include "matrixio_ndarray.h"

namespace sfem {

    template <class T>
    class Grid<T>::Impl {
    public:
        MPI_Comm comm;
        std::shared_ptr<Buffer<T>> field;
        int block_size{1};
        int spatial_dimension{3};

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
    std::unique_ptr<Grid<T>> Grid<T>::create_from_file(MPI_Comm comm, const std::string &path) {
        auto ret = std::make_unique<Grid<T>>(comm);
        auto in = YAMLNoIndent::create_from_file(path + "/meta.yaml");

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

        ptrdiff_t size = ret->impl_->stride[2] * ret->impl_->nlocal[2];
        ret->impl_->field = Buffer<T>::own(size, data, free, MEMORY_SPACE_HOST);
        return ret;
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

    // Explicit instantiation
    template class Grid<float>;
    template class Grid<double>;
}  // namespace sfem
