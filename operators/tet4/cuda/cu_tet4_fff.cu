#include "cu_tet4_fff.hpp"
#include "sfem_cuda_base.hpp"
#include "sfem_defs.hpp"

#include "cu_tet4_inline.hpp"

static int cu_tet4_fff_allocate_generic(const ptrdiff_t nelements,
                                        const enum smesh::PrimitiveType real_type,
                                        void **const SFEM_RESTRICT fff) {
    switch (real_type) {
        case smesh::SMESH_DEFAULT: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(cu_jacobian_t)));
            return SFEM_SUCCESS;
        }
        case smesh::SMESH_FLOAT16: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(half)));
            return SFEM_SUCCESS;
        }
        case smesh::SMESH_FLOAT32: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(float)));
            return SFEM_SUCCESS;
        }
        case smesh::SMESH_FLOAT64: {
            SFEM_CUDA_CHECK(cudaMalloc(fff, 6 * nelements * sizeof(double)));
            return SFEM_SUCCESS;
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_fff_allocate: not implemented for type %s (code %d)\n",
                    smesh::to_string(real_type).data(),
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

    SFEM_DEBUG_SYNCHRONIZE(nullptr);
    return SFEM_SUCCESS;
}

static int cu_tet4_fff_fill_generic(const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    const enum smesh::PrimitiveType real_type,
                                    void *const SFEM_RESTRICT fff) {
    switch (real_type) {
        case smesh::SMESH_DEFAULT: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (cu_jacobian_t *)fff);
        }
        case smesh::SMESH_FLOAT16: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (half *)fff);
        }
        case smesh::SMESH_FLOAT32: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (float *)fff);
        }
        case smesh::SMESH_FLOAT64: {
            return cu_tet4_fff_fill_tpl(nelements, elements, points, (double *)fff);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_tet4_fff_fill: not implemented for type %s (code %d)\n",
                    smesh::to_string(real_type).data(),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_tet4_fff_allocate(const ptrdiff_t nelements, void **const SFEM_RESTRICT fff) {
    // Currently this is the only one supported
    return cu_tet4_fff_allocate_generic(nelements, smesh::SMESH_DEFAULT, fff);
}

extern int cu_tet4_fff_fill(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            void *const SFEM_RESTRICT fff) {
    // Currently this is the only one supported
    return cu_tet4_fff_fill_generic(nelements, elements, points, smesh::SMESH_DEFAULT, fff);
}
