#include "sfem_base.h"
#ifdef SFEM_ENABLE_CUDA

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include "coo_weight_sort.h"

void sort_weights(PartitionerWorkspace* ws, count_t nweights) {
    sort_sparse_matrix_cuda(ws->ptr_j, ws->ptr_i, ws->weights, nweights);
}

struct CompareTuples {
    __host__ __device__ bool operator()(const thrust::tuple<idx_t, idx_t, real_t>& a,
                                        const thrust::tuple<idx_t, idx_t, real_t>& b) const {
        return thrust::get<2>(a) > thrust::get<2>(b);  // Descending order
    }
};

void sort_sparse_matrix_cuda(idx_t* col_indices, idx_t* row_indices, real_t* values, count_t N) {
    printf("Hi from thrust sort\n");
    // Create device_vectors from your host arrays
    thrust::device_vector<idx_t> d_col_indices(col_indices, col_indices + N);
    thrust::device_vector<idx_t> d_row_indices(row_indices, row_indices + N);
    thrust::device_vector<real_t> d_values(values, values + N);

    // Create zip iterators over the device_vectors
    auto first = thrust::make_zip_iterator(
            thrust::make_tuple(d_col_indices.begin(), d_row_indices.begin(), d_values.begin()));
    auto last = thrust::make_zip_iterator(
            thrust::make_tuple(d_col_indices.end(), d_row_indices.end(), d_values.end()));

    // Perform the sort on the device
    thrust::sort(first, last, CompareTuples());

    // Copy the sorted data back to the host arrays
    thrust::copy(d_col_indices.begin(), d_col_indices.end(), col_indices);
    thrust::copy(d_row_indices.begin(), d_row_indices.end(), row_indices);
    thrust::copy(d_values.begin(), d_values.end(), values);
}
#endif
