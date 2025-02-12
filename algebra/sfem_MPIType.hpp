#ifndef SFEM_MPI_TYPE_HPP
#define SFEM_MPI_TYPE_HPP

#include <stdint.h>
#include <mpi.h>

namespace sfem {

    template <typename T>
    struct MPIType {};

#define SFEM_DEFINE_MPI_TYPE(type__, mpi_type__)                 \
    template <>                                                  \
    struct MPIType<type__> {                                     \
        static const MPI_Datatype value() { return mpi_type__; } \
    };

    SFEM_DEFINE_MPI_TYPE(int8_t, MPI_INT8_T);
    SFEM_DEFINE_MPI_TYPE(uint8_t, MPI_UINT8_T);
    SFEM_DEFINE_MPI_TYPE(int16_t, MPI_INT16_T);
    SFEM_DEFINE_MPI_TYPE(uint16_t, MPI_UINT16_T);
    SFEM_DEFINE_MPI_TYPE(int32_t, MPI_INT32_T);
    SFEM_DEFINE_MPI_TYPE(uint32_t, MPI_UINT32_T);
    SFEM_DEFINE_MPI_TYPE(int64_t, MPI_INT64_T);
    SFEM_DEFINE_MPI_TYPE(uint64_t, MPI_UINT64_T);
    SFEM_DEFINE_MPI_TYPE(char, MPI_CHAR);
    SFEM_DEFINE_MPI_TYPE(float, MPI_FLOAT);
    SFEM_DEFINE_MPI_TYPE(double, MPI_DOUBLE);

}  // namespace sfem

#endif  // SFEM_MPI_TYPE_HPP