#include "cu_hex8_fff.h"

#include "sfem_cuda_base.h"
#include "sfem_defs.h"

#include "hex8_laplacian_inline_cpu.h"

static int cu_hex8_fff_allocate_generic(const ptrdiff_t nelements,
                                        const enum RealType real_type,
                                        void **const SFEM_RESTRICT fff) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
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
                    "[Error] cu_hex8_fff_allocate: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            *fff = 0;
            return SFEM_FAILURE;
        }
    }
}

template <typename T>
static SFEM_INLINE __host__ __device__ void cu_hex8_fff(const scalar_t *const SFEM_RESTRICT x,
                                                        const scalar_t *const SFEM_RESTRICT y,
                                                        const scalar_t *const SFEM_RESTRICT z,
                                                        const scalar_t qx,
                                                        const scalar_t qy,
                                                        const scalar_t qz,
                                                        const ptrdiff_t stride,
                                                        T *const SFEM_RESTRICT fff) {
    const scalar_t x0 = qy * qz;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = qy * x1;
    const scalar_t x3 = 1 - qy;
    const scalar_t x4 = qz * x3;
    const scalar_t x5 = x1 * x3;
    const scalar_t x6 = x0 * z[6] - x0 * z[7] + x2 * z[2] - x2 * z[3] - x4 * z[4] + x4 * z[5] -
                        x5 * z[0] + x5 * z[1];
    const scalar_t x7 = qx * qy;
    const scalar_t x8 = qx * x3;
    const scalar_t x9 = 1 - qx;
    const scalar_t x10 = qy * x9;
    const scalar_t x11 = x3 * x9;
    const scalar_t x12 = qx * qy * x[6] + qx * x3 * x[5] + qy * x9 * x[7] - x10 * x[3] -
                         x11 * x[0] + x3 * x9 * x[4] - x7 * x[2] - x8 * x[1];
    const scalar_t x13 = qx * qz;
    const scalar_t x14 = qx * x1;
    const scalar_t x15 = qz * x9;
    const scalar_t x16 = x1 * x9;
    const scalar_t x17 = qx * qz * y[6] + qx * x1 * y[2] + qz * x9 * y[7] + x1 * x9 * y[3] -
                         x13 * y[5] - x14 * y[1] - x15 * y[4] - x16 * y[0];
    const scalar_t x18 = x12 * x17;
    const scalar_t x19 = x0 * x[6] - x0 * x[7] + x2 * x[2] - x2 * x[3] - x4 * x[4] + x4 * x[5] -
                         x5 * x[0] + x5 * x[1];
    const scalar_t x20 = qx * qy * y[6] + qx * x3 * y[5] + qy * x9 * y[7] - x10 * y[3] -
                         x11 * y[0] + x3 * x9 * y[4] - x7 * y[2] - x8 * y[1];
    const scalar_t x21 = qx * qz * z[6] + qx * x1 * z[2] + qz * x9 * z[7] + x1 * x9 * z[3] -
                         x13 * z[5] - x14 * z[1] - x15 * z[4] - x16 * z[0];
    const scalar_t x22 = x20 * x21;
    const scalar_t x23 = x0 * y[6] - x0 * y[7] + x2 * y[2] - x2 * y[3] - x4 * y[4] + x4 * y[5] -
                         x5 * y[0] + x5 * y[1];
    const scalar_t x24 = qx * qy * z[6] + qx * x3 * z[5] + qy * x9 * z[7] - x10 * z[3] -
                         x11 * z[0] + x3 * x9 * z[4] - x7 * z[2] - x8 * z[1];
    const scalar_t x25 = qx * qz * x[6] + qx * x1 * x[2] + qz * x9 * x[7] + x1 * x9 * x[3] -
                         x13 * x[5] - x14 * x[1] - x15 * x[4] - x16 * x[0];
    const scalar_t x26 = x24 * x25;
    const scalar_t x27 =
            x12 * x21 * x23 + x17 * x19 * x24 - x18 * x6 - x19 * x22 + x20 * x25 * x6 - x23 * x26;
    const scalar_t x28 = -x18 + x20 * x25;
    const scalar_t x29 = (1 / POW2(x27));
    const scalar_t x30 = x12 * x21 - x26;
    const scalar_t x31 = x17 * x24 - x22;
    const scalar_t x32 = x12 * x23 - x19 * x20;
    const scalar_t x33 = x28 * x29;
    const scalar_t x34 = -x12 * x6 + x19 * x24;
    const scalar_t x35 = x29 * x30;
    const scalar_t x36 = x20 * x6 - x23 * x24;
    const scalar_t x37 = x29 * x31;
    const scalar_t x38 = x17 * x19 - x23 * x25;
    const scalar_t x39 = -x19 * x21 + x25 * x6;
    const scalar_t x40 = -x17 * x6 + x21 * x23;
    fff[0 * stride] = x27 * (POW2(x28) * x29 + x29 * POW2(x30) + x29 * POW2(x31));
    fff[1 * stride] = x27 * (x32 * x33 + x34 * x35 + x36 * x37);
    fff[2 * stride] = x27 * (x33 * x38 + x35 * x39 + x37 * x40);
    fff[3 * stride] = x27 * (x29 * POW2(x32) + x29 * POW2(x34) + x29 * POW2(x36));
    fff[4 * stride] = x27 * (x29 * x32 * x38 + x29 * x34 * x39 + x29 * x36 * x40);
    fff[5 * stride] = x27 * (x29 * POW2(x38) + x29 * POW2(x39) + x29 * POW2(x40));
}

template <typename T>
static int cu_hex8_fff_fill_tpl(const ptrdiff_t nelements,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                T *const SFEM_RESTRICT fff) {
	const geom_t *const x = points[0];
	const geom_t *const y = points[1];
	const geom_t *const z = points[2];

    T *h_fff = (T *)calloc(6 * nelements, sizeof(T));
#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; e++) {
        idx_t ev[8];
        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][e];
        }

        const scalar_t lx[8] = {
                x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

        const scalar_t ly[8] = {
                y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

        const scalar_t lz[8] = {
                z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        cu_hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, nelements, &h_fff[e]);
    }

    SFEM_CUDA_CHECK(cudaMemcpy(fff, h_fff, 6 * nelements * sizeof(T), cudaMemcpyHostToDevice));

    free(h_fff);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

static int cu_hex8_fff_fill_generic(const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    const enum RealType real_type,
                                    void *const SFEM_RESTRICT fff) {
    switch (real_type) {
        case SFEM_REAL_DEFAULT: {
            return cu_hex8_fff_fill_tpl(nelements, elements, points, (cu_jacobian_t *)fff);
        }
        case SFEM_FLOAT16: {
            return cu_hex8_fff_fill_tpl(nelements, elements, points, (half *)fff);
        }
        case SFEM_FLOAT32: {
            return cu_hex8_fff_fill_tpl(nelements, elements, points, (float *)fff);
        }
        case SFEM_FLOAT64: {
            return cu_hex8_fff_fill_tpl(nelements, elements, points, (double *)fff);
        }
        default: {
            fprintf(stderr,
                    "[Error] cu_hex8_fff_fill: not implemented for type %s (code %d)\n",
                    real_type_to_string(real_type),
                    real_type);
            assert(0);
            return SFEM_FAILURE;
        }
    }
}

extern int cu_hex8_fff_allocate(const ptrdiff_t nelements, void **const SFEM_RESTRICT fff) {
    // Currently this is the only one supported
    return cu_hex8_fff_allocate_generic(nelements, SFEM_REAL_DEFAULT, fff);
}

extern int cu_hex8_fff_fill(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            void *const SFEM_RESTRICT fff) {
    // Currently this is the only one supported
    return cu_hex8_fff_fill_generic(nelements, elements, points, SFEM_REAL_DEFAULT, fff);
}
