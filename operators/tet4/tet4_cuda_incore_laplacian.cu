#include <cassert>
#include <cmath>
// #include <cstdio>
#include <algorithm>
#include <cstddef>

extern "C" {
#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"
}

#include "sfem_cuda_base.h"
#include "sfem_mesh.h"

#include "tet4_cuda_incore_laplacian.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define POW2(a) ((a) * (a))

static inline __device__ __host__ void fff_micro_kernel(const geom_t px0,
                                                        const geom_t px1,
                                                        const geom_t px2,
                                                        const geom_t px3,
                                                        const geom_t py0,
                                                        const geom_t py1,
                                                        const geom_t py2,
                                                        const geom_t py3,
                                                        const geom_t pz0,
                                                        const geom_t pz1,
                                                        const geom_t pz2,
                                                        const geom_t pz3,
                                                        const count_t stride,
                                                        geom_t *fff) {
    //      - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
    //      - Subexpressions: 4*ADD + 6*DIV + 28*MUL + NEG + POW + 24*SUB
    const geom_t x0 = -px0 + px1;
    const geom_t x1 = -py0 + py2;
    const geom_t x2 = -pz0 + pz3;
    const geom_t x3 = x1 * x2;
    const geom_t x4 = x0 * x3;
    const geom_t x5 = -py0 + py3;
    const geom_t x6 = -pz0 + pz2;
    const geom_t x7 = x5 * x6;
    const geom_t x8 = x0 * x7;
    const geom_t x9 = -py0 + py1;
    const geom_t x10 = -px0 + px2;
    const geom_t x11 = x10 * x2;
    const geom_t x12 = x11 * x9;
    const geom_t x13 = -pz0 + pz1;
    const geom_t x14 = x10 * x5;
    const geom_t x15 = x13 * x14;
    const geom_t x16 = -px0 + px3;
    const geom_t x17 = x16 * x6 * x9;
    const geom_t x18 = x1 * x16;
    const geom_t x19 = x13 * x18;
    const geom_t x20 = -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 +
                       (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const geom_t x21 = x14 - x18;
    const geom_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const geom_t x23 = -x11 + x16 * x6;
    const geom_t x24 = x3 - x7;
    const geom_t x25 = -x0 * x5 + x16 * x9;
    const geom_t x26 = x21 * x22;
    const geom_t x27 = x0 * x2 - x13 * x16;
    const geom_t x28 = x22 * x23;
    const geom_t x29 = x13 * x5 - x2 * x9;
    const geom_t x30 = x22 * x24;
    const geom_t x31 = x0 * x1 - x10 * x9;
    const geom_t x32 = -x0 * x6 + x10 * x13;
    const geom_t x33 = -x1 * x13 + x6 * x9;
    fff[0 * stride] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    fff[1 * stride] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    fff[2 * stride] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    fff[3 * stride] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    fff[4 * stride] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    fff[5 * stride] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

int tet4_cuda_incore_laplacian_init(cuda_incore_laplacian_t *ctx, mesh_t mesh) {

    { // Create FFF and store it on device
        geom_t *h_fff = (geom_t *)calloc(6 * mesh.nelements, sizeof(geom_t));

#pragma omp parallel
        {
#pragma omp for
            for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
                fff_micro_kernel(mesh.points[0][mesh.elements[0][e]],
                                 mesh.points[0][mesh.elements[1][e]],
                                 mesh.points[0][mesh.elements[2][e]],
                                 mesh.points[0][mesh.elements[3][e]],
                                 mesh.points[1][mesh.elements[0][e]],
                                 mesh.points[1][mesh.elements[1][e]],
                                 mesh.points[1][mesh.elements[2][e]],
                                 mesh.points[1][mesh.elements[3][e]],
                                 mesh.points[2][mesh.elements[0][e]],
                                 mesh.points[2][mesh.elements[1][e]],
                                 mesh.points[2][mesh.elements[2][e]],
                                 mesh.points[2][mesh.elements[3][e]],
                                 mesh.nelements,
                                 &h_fff[e]);
            }
        }

        SFEM_CUDA_CHECK(cudaMalloc(&ctx->d_fff, 6 * mesh.nelements * sizeof(geom_t)));
        SFEM_CUDA_CHECK(cudaMemcpy(
            ctx->d_fff, h_fff, 6 * mesh.nelements * sizeof(geom_t), cudaMemcpyHostToDevice));
        free(h_fff);
    }

    {
    	 // Store elem indices on device
    }
//
    return 0;
}

int tet4_cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                     const real_t *const d_x,
                                     real_t *const d_y) {
    // collect coeffs
    // apply operator
    // redistribute coeffs
    return 0;
}