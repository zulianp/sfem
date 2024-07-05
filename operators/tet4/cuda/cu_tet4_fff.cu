#include "cu_tet4_fff.h"
#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "cu_tet4_inline.hpp"

extern int cu_tet4_fff_allocate(const ptrdiff_t nelements,
                                const enum RealType real_type,
                                void **const SFEM_RESTRICT fff) {
    switch (real_type) {
        case SFEM_FLOAT_DEFAULT: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(cu_jacobian_t)));
            return SFEM_SUCCESS;
        }
        case SFEM_FLOAT16: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(half)));
            return SFEM_SUCCESS;
        }
        case SFEM_FLOAT32: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(float)));
            return SFEM_SUCCESS;
        }
        case SFEM_FLOAT64: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(double)));
            return SFEM_SUCCESS;
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_fff_allocate: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            *fff = 0;
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
static int cu_tet4_fff_fill_tpl(const ptrdiff_t nelements,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                T *const SFEM_RESTRICT fff) {
    // Create FFF and store it on device
    T *h_fff = (T *)calloc(6 * nelements, sizeof(T));
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        cu_tet4_fff(points[0][elements[0][e]],
                    points[0][elements[1][e]],
                    points[0][elements[2][e]],
                    points[0][elements[3][e]],
                    points[1][elements[0][e]],
                    points[1][elements[1][e]],
                    points[1][elements[2][e]],
                    points[1][elements[3][e]],
                    points[2][elements[0][e]],
                    points[2][elements[1][e]],
                    points[2][elements[2][e]],
                    points[2][elements[3][e]],
                    nelements,
                    &h_fff[e]);
    }

    SFEM_CUDA_CHECK(cudaMemcpy(fff, h_fff, 6 * nelements * sizeof(T), cudaMemcpyHostToDevice));

    free(h_fff);
    return SFEM_SUCCESS;
}

extern int cu_tet4_fff_fill(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            const enum RealType real_type,
                            void *const SFEM_RESTRICT fff) {
    switch (real_type) {
        case SFEM_FLOAT_DEFAULT: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (cu_jacobian_t *)fff);
        }
        case SFEM_FLOAT16: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (half *)fff);
        }
        case SFEM_FLOAT32: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (float *)fff);
        }
        case SFEM_FLOAT64: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (double *)fff);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_fff_fill: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

int elements_to_device(const ptrdiff_t nelements,
                       const int num_nodes_x_element,
                       idx_t **const SFEM_RESTRICT h_elements,
                       idx_t **const SFEM_RESTRICT d_elements) {
    {  // Store elem indices on device
        SFEM_CUDA_CHECK(cudaMalloc(d_elements, num_nodes_x_element * nelements * sizeof(idx_t)));

        for (int d = 0; d < num_nodes_x_element; d++) {
            SFEM_CUDA_CHECK(cudaMemcpy((*d_elements) + d * nelements,
                                       h_elements[d],
                                       nelements * sizeof(idx_t),
                                       cudaMemcpyHostToDevice));
        }
    }
}
