// Micro kernels
{MK_FILE_CU}

#include <stddef.h>
#include "sfem_vec.h"

#ifdef __NVCC__
#include "sfem_cuda_base.h"
#define SFEM_DEVICE_KERNEL static __global__
#else
#define SFEM_DEVICE_KERNEL static
#define SFEM_NVTX_SCOPE(...)
#endif

SFEM_DEVICE_KERNEL void {NAME}_jacobian_kernel(
const ptrdiff_t nelements,
const geom_t *const SFEM_RESTRICT xyz,
real_t *const SFEM_RESTRICT jacobian)
{{
#ifdef __NVCC__
for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
for (ptrdiff_t e = 0; e < nelements; e++)
#endif
{{
// The element coordinates and jacobian
const geom_t *const this_xyz = &xyz[e];
real_t *const this_jacobian = &jacobian[e];

{COORDINATES_READ}

{NAME}_mk_jacobian
(
{COORDINATES}// arrays
nelements,
this_jacobian
);

}}
}}

SFEM_DEVICE_KERNEL void {NAME}_jacobian_inverse_kernel(
const ptrdiff_t nelements,
const geom_t *const SFEM_RESTRICT xyz,
real_t *const SFEM_RESTRICT jacobian)
{{
#ifdef __NVCC__
for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
for (ptrdiff_t e = 0; e < nelements; e++)
#endif
{{
// The element coordinates and jacobian
const geom_t *const this_xyz = &xyz[e];
real_t *const this_jacobian = &jacobian[e];

{COORDINATES_READ}

{NAME}_mk_jacobian_inverse
(
{COORDINATES}// arrays
nelements,
this_jacobian
);

}}
}}

static void {NAME}_host_pack_elements
(
const ptrdiff_t n,
const ptrdiff_t element_offset,
idx_t **const SFEM_RESTRICT elems,
geom_t **const SFEM_RESTRICT xyz,
geom_t * const SFEM_RESTRICT he_xyz
)
{{
    SFEM_NVTX_SCOPE("{NAME}_host_pack_elements");
    {{
        for (int d = 0; d < fe_spatial_dim; ++d) {{
            for (int e_node = 0; e_node < fe_subparam_n_nodes; e_node++) {{
                const geom_t *const x = xyz[d];
                ptrdiff_t offset = (d * fe_subparam_n_nodes + e_node) * n;
                const idx_t *const nodes = &elems[e_node][element_offset];

                geom_t *buff = &he_xyz[offset];

                #pragma omp parallel
                {{
                    #pragma omp for nowait
                    for (ptrdiff_t k = 0; k < n; k++) {{
                        buff[k] = x[nodes[k]];
                    }}
                }}
            }}
        }}
    }}
}}

static void {NAME}_host_pack_vector
(
const ptrdiff_t n,
const ptrdiff_t element_offset,
idx_t **const SFEM_RESTRICT elems,
const real_t *const SFEM_RESTRICT vec,
real_t * const SFEM_RESTRICT he_vec
)
{{
    SFEM_NVTX_SCOPE("{NAME}_host_pack_vector");
    {{
        
        for (int e_node = 0; e_node < fe_n_nodes; e_node++) 
        {{
            const idx_t *const nodes = &elems[e_node][element_offset];

            real_t *buff = &he_vec[e_node * n];

            #pragma omp parallel
            {{
                #pragma omp for nowait
                for (ptrdiff_t k = 0; k < n; k++) 
                {{
                    buff[k] = vec[nodes[k]];
                }}
            }}
        }}
    }}  
}}

template <typename T>
SFEM_DEVICE_KERNEL void {NAME}_pack_kernel(
const ptrdiff_t nelements,
const idx_t *const SFEM_RESTRICT node_idx,
const T *const SFEM_RESTRICT node_data,
T *const SFEM_RESTRICT per_element_data
) 
{{
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) 
    {{
        // coalesced write
        per_element_data[e * nelements] = node_data[node_idx[e]];
    }}
}}

SFEM_DEVICE_FUNCTION int {NAME}_linear_search
(
const idx_t target,
const idx_t *const arr,
const int size
)
{{
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE)
    {{
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }}
    for (; i < size; i++)
    {{
        if (arr[i] == target) return i;
    }}
    return -1;
}}

SFEM_DEVICE_FUNCTION int {NAME}_find_col
(
const idx_t key,
const idx_t *const row,
const int lenrow
)
{{
    return {NAME}_linear_search(key, row, lenrow);
}}

SFEM_DEVICE_FUNCTION void {NAME}_find_cols
(
const idx_t *targets,
const idx_t *const row,
const int lenrow,
int *ks
)
{{
    if (lenrow > 32)
    {{
        for (int d = 0; d < fe_n_nodes; ++d)
        {{
            ks[d] = {NAME}_find_col(targets[d], row, lenrow);
        }}
    }} else {{
#pragma unroll(fe_n_nodes)
        for (int d = 0; d < fe_n_nodes; ++d)
        {{
            ks[d] = 0;
        }}

        for (int i = 0; i < lenrow; ++i)
        {{
#pragma unroll(fe_n_nodes)
            for (int d = 0; d < fe_n_nodes; ++d)
            {{
                ks[d] += row[i] < targets[d];
            }}
        }}
    }}
}}


SFEM_DEVICE_KERNEL void {NAME}_vector_local_to_global_kernel
(
const ptrdiff_t nelements,
idx_t **const SFEM_RESTRICT elems,
const int stride_element_vector,
const real_t *const SFEM_RESTRICT element_vector,
real_t *const SFEM_RESTRICT values
) 
{{
    idx_t ev[fe_n_nodes];
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {{
#pragma unroll(fe_n_nodes)
        for (int v = 0; v < fe_n_nodes; ++v) 
        {{
            ev[v] = elems[v][e];
        }}

        // offsetted array for this element
        const real_t *const this_vector = &element_vector[e];

#pragma unroll(fe_n_nodes)
        for (int edof_i = 0; edof_i < fe_n_nodes; ++edof_i) 
        {{
            const idx_t dof_i = ev[edof_i];
            const real_t v = this_vector[edof_i * stride_element_vector];
#ifdef __NVCC__
            atomicAdd(&values[dof_i], v);
#else
            values[dof_i] += v;
#endif
        }}
        
    }}
}}

template<int block_size>
SFEM_DEVICE_KERNEL void {NAME}_block_vector_local_to_global_kernel
(
const ptrdiff_t nelements,
idx_t **const SFEM_RESTRICT elems,
const int stride_element_vector,
const real_t *const SFEM_RESTRICT element_vector,
real_t **const SFEM_RESTRICT values
) 
{{
    idx_t ev[fe_n_nodes];
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {{
#pragma unroll(fe_n_nodes)
        for (int v = 0; v < fe_n_nodes; ++v) 
        {{
            ev[v] = elems[v][e];
        }}

        // offsetted array for this element
        const real_t *const this_vector = &element_vector[e];

#pragma unroll(fe_n_nodes)
        for (int edof_i = 0; edof_i < fe_n_nodes; ++edof_i) 
        {{

            for(int b = 0; b < block_size; b++) 
            {{
                const idx_t dof_i = ev[edof_i];
                const real_t v = this_vector[(b * fe_n_nodes + edof_i) * stride_element_vector];
#ifdef __NVCC__
                atomicAdd(&values[dof_i], v);
#else
                values[b][dof_i] += v;
#endif
            }}
        }}
        
    }}
}}

SFEM_DEVICE_KERNEL void {NAME}_matrix_local_to_global_kernel
(
const ptrdiff_t nelements,
idx_t **const SFEM_RESTRICT elems,
const int stride_element_matrix,
const real_t *const SFEM_RESTRICT element_matrix,
const count_t *const SFEM_RESTRICT rowptr,
const idx_t *const SFEM_RESTRICT colidx,
real_t *const SFEM_RESTRICT values
) 
{{
    idx_t ev[fe_n_nodes];
    idx_t ks[fe_n_nodes];
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
        {{
#pragma unroll(fe_n_nodes)
        for (int v = 0; v < fe_n_nodes; ++v) 
        {{
            ev[v] = elems[v][e];
        }}

        // offsetted array for this element
        const real_t *const this_matrix = &element_matrix[e];

        for (int edof_i = 0; edof_i < fe_n_nodes; ++edof_i) 
        {{
            const idx_t dof_i = ev[edof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *const row = &colidx[rowptr[dof_i]];

            {NAME}_find_cols(ev, row, lenrow, ks);

            real_t *const rowvalues = &values[rowptr[dof_i]];

            for (int edof_j = 0; edof_j < fe_n_nodes; ++edof_j) 
            {{
                ptrdiff_t idx = (edof_i * fe_n_nodes + edof_j) * stride_element_matrix;
                const real_t v = this_matrix[idx];
#ifdef __NVCC__
                atomicAdd(&rowvalues[ks[edof_j]], v);
#else
                rowvalues[ks[edof_j]] += v;
#endif
            }}
        }}
    }}
}}


template<int block_rows, int block_cols>
SFEM_DEVICE_KERNEL void {NAME}_block_matrix_local_to_global_kernel
(
const ptrdiff_t nelements,
idx_t **const SFEM_RESTRICT elems,
const int stride_element_matrix,
const real_t *const SFEM_RESTRICT element_matrix,
const count_t *const SFEM_RESTRICT rowptr,
const idx_t *const SFEM_RESTRICT colidx,
real_t **const SFEM_RESTRICT values
) 
{{
    idx_t ev[fe_n_nodes];
    idx_t ks[fe_n_nodes];
#ifdef __NVCC__
    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x)
#else
    for (ptrdiff_t e = 0; e < nelements; e++)
#endif
    {{
#pragma unroll(fe_n_nodes)
        for (int v = 0; v < fe_n_nodes; ++v) 
        {{
            ev[v] = elems[v][e];
        }}

        // offsetted array for this element
        const real_t *const this_matrix = &element_matrix[e];

        static const int matrix_size = fe_n_nodes * fe_n_nodes;
        for (int edof_i = 0; edof_i < fe_n_nodes; ++edof_i) 
        {{
            const idx_t dof_i = ev[edof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *const row = &colidx[rowptr[dof_i]];

            {NAME}_find_cols(ev, row, lenrow, ks);

            for(int br = 0; br < block_rows; br++) 
            {{
                for(int bc = 0; bc < block_cols; bc++) 
                {{
                    int block_idx = br * block_cols + bc;
                    real_t *const rowvalues = &values[block_idx][rowptr[dof_i]];

                    for (int edof_j = 0; edof_j < fe_n_nodes; ++edof_j) 
                    {{
                        ptrdiff_t idx = (block_idx * matrix_size + (edof_i * fe_n_nodes + edof_j)) * stride_element_matrix;
                        const real_t v = this_matrix[idx];
#ifdef __NVCC__
                        atomicAdd(&rowvalues[ks[edof_j]], v);
#else
                        rowvalues[ks[edof_j]] += v;
#endif
                    }}
                }}
            }}
        }}
    }}
}}
