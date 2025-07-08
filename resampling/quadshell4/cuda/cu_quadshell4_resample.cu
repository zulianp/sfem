#include "cu_quadshell4_resample.h"

#include "line_quadrature.h"
#include "sfem_macros.h"

#include "sfem_cuda_base.h"

static const scalar_t infty = 10000;

// #define SFEM_RESAMPLE_GAP_DUAL

static SFEM_INLINE __device__ real_t cu_put_inside(const real_t v) { return MIN(MAX(1e-7, v), 1 - 1e-7); }

SFEM_INLINE __device__ static real_t cu_quadshell4_measure(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        const real_t px2,
        const real_t px3,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        const real_t py2,
        const real_t py3,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        const real_t pz2,
        const real_t pz3,
        // Quadrature point
        const real_t qx,
        const real_t qy) {
    const scalar_t x0 = qx - 1;
    const scalar_t x1 = -x0;
    const scalar_t x2 = px0 * x0 - px1 * qx + px2 * qx + px3 * x1;
    const scalar_t x3 = qy - 1;
    const scalar_t x4 = -x3;
    const scalar_t x5 = px0 * x3 + px1 * x4 + px2 * qy - px3 * qy;
    const scalar_t x6 = py0 * x0 - py1 * qx + py2 * qx + py3 * x1;
    const scalar_t x7 = py0 * x3 + py1 * x4 + py2 * qy - py3 * qy;
    const scalar_t x8 = pz0 * x0 - pz1 * qx + pz2 * qx + pz3 * x1;
    const scalar_t x9 = pz0 * x3 + pz1 * x4 + pz2 * qy - pz3 * qy;
    return sqrt((POW2(x2) + POW2(x6) + POW2(x8)) * (POW2(x5) + POW2(x7) + POW2(x9)) - POW2(x2 * x5 + x6 * x7 + x8 * x9));
}

SFEM_INLINE __device__ static void cu_quadshell4_transform(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        const real_t px2,
        const real_t px3,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        const real_t py2,
        const real_t py3,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        const real_t pz2,
        const real_t pz3,
        // Quadrature point
        const real_t qx,
        const real_t qy,
        // Output
        real_t* const SFEM_RESTRICT out_x,
        real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    const scalar_t x0 = qx * qy;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = qx * x1;
    const scalar_t x3 = 1 - qx;
    const scalar_t x4 = qy * x3;
    const scalar_t x5 = x1 * x3;

    *out_x = px0 * x5 + px1 * x2 + px2 * x0 + px3 * x4;
    *out_y = py0 * x5 + py1 * x2 + py2 * x0 + py3 * x4;
    *out_z = pz0 * x5 + pz1 * x2 + pz2 * x0 + pz3 * x4;
}

SFEM_INLINE __device__ static void cu_hex_aa_8_eval_fun(
        // Quadrature point (local coordinates)
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT f) {
    f[0] = (1.0 - x) * (1.0 - y) * (1.0 - z);
    f[1] = x * (1.0 - y) * (1.0 - z);
    f[2] = x * y * (1.0 - z);
    f[3] = (1.0 - x) * y * (1.0 - z);
    f[4] = (1.0 - x) * (1.0 - y) * z;
    f[5] = x * (1.0 - y) * z;
    f[6] = x * y * z;
    f[7] = (1.0 - x) * y * z;
}

SFEM_INLINE __device__ static void cu_hex_aa_8_collect_coeffs(const ptrdiff_t* const SFEM_RESTRICT stride,
                                                              const ptrdiff_t                      i,
                                                              const ptrdiff_t                      j,
                                                              const ptrdiff_t                      k,
                                                              // Attention this is geometric data transformed to solver data!
                                                              const geom_t* const SFEM_RESTRICT data,
                                                              real_t* const SFEM_RESTRICT       out) {
    const ptrdiff_t i0 = i * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i1 = (i + 1) * stride[0] + j * stride[1] + k * stride[2];
    const ptrdiff_t i2 = (i + 1) * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i3 = i * stride[0] + (j + 1) * stride[1] + k * stride[2];
    const ptrdiff_t i4 = i * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i5 = (i + 1) * stride[0] + j * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i6 = (i + 1) * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];
    const ptrdiff_t i7 = i * stride[0] + (j + 1) * stride[1] + (k + 1) * stride[2];

    out[0] = data[i0];
    out[1] = data[i1];
    out[2] = data[i2];
    out[3] = data[i3];
    out[4] = data[i4];
    out[5] = data[i5];
    out[6] = data[i6];
    out[7] = data[i7];
}

SFEM_INLINE __device__ static void cu_hex_aa_8_eval_grad(
        // Quadrature point (local coordinates)
        const real_t x,
        const real_t y,
        const real_t z,
        // Output
        real_t* const SFEM_RESTRICT gx,
        real_t* const SFEM_RESTRICT gy,
        real_t* const SFEM_RESTRICT gz) {
    // Transformation to ref element
    gx[0] = -(1.0 - y) * (1.0 - z);
    gy[0] = -(1.0 - x) * (1.0 - z);
    gz[0] = -(1.0 - x) * (1.0 - y);

    gx[1] = (1.0 - y) * (1.0 - z);
    gy[1] = -x * (1.0 - z);
    gz[1] = -x * (1.0 - y);

    gx[2] = y * (1.0 - z);
    gy[2] = x * (1.0 - z);
    gz[2] = -x * y;

    gx[3] = -y * (1.0 - z);
    gy[3] = (1.0 - x) * (1.0 - z);
    gz[3] = -(1.0 - x) * y;

    gx[4] = -(1.0 - y) * z;
    gy[4] = -(1.0 - x) * z;
    gz[4] = (1.0 - x) * (1.0 - y);

    gx[5] = (1.0 - y) * z;
    gy[5] = -x * z;
    gz[5] = x * (1.0 - y);

    gx[6] = y * z;
    gy[6] = x * z;
    gz[6] = x * y;

    gx[7] = -y * z;
    gy[7] = (1.0 - x) * z;
    gz[7] = (1.0 - x) * y;
}

__global__ void cu_quadshell4_resample_gap_local_kernel(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg,
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t hex8_f[8];
        real_t hex8_grad_x[8];
        real_t hex8_grad_y[8];
        real_t hex8_grad_z[8];
        real_t coeffs[8];

        real_t quad4_f[4];
        real_t element_gap[4];
        real_t element_xnormal[4];
        real_t element_ynormal[4];
        real_t element_znormal[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            x[v] = xyz[0][ev[v]];
            y[v] = xyz[1][ev[v]];
            z[v] = xyz[2][ev[v]];
        }

        for (int v = 0; v < 4; v++) {
            element_gap[v]     = 0;
            element_xnormal[v] = 0;
            element_ynormal[v] = 0;
            element_znormal[v] = 0;
        }

        for (int q_ix = 0; q_ix < n_qp; q_ix++) {
            for (int q_iy = 0; q_iy < n_qp; q_iy++) {
                const real_t measure = cu_quadshell4_measure(
                        x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3], qx[q_ix], qx[q_iy]);

                assert(measure > 0);

                real_t g_qx, g_qy, g_qz;
                cu_quadshell4_transform(x[0],
                                        x[1],
                                        x[2],
                                        x[3],
                                        y[0],
                                        y[1],
                                        y[2],
                                        y[3],
                                        z[0],
                                        z[1],
                                        z[2],
                                        z[3],
                                        qx[q_ix],
                                        qx[q_iy],
                                        &g_qx,
                                        &g_qy,
                                        &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    quad4_f[0]        = x0 * x1;
                    quad4_f[1]        = qx[q_ix] * x1;
                    quad4_f[2]        = qx[q_ix] * qx[q_iy];
                    quad4_f[3]        = qx[q_iy] * x0;
                }
#else
                // DUAL basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    const scalar_t f0 = x0 * x1;
                    const scalar_t f1 = qx[q_ix] * x1;
                    const scalar_t f2 = qx[q_ix] * qx[q_iy];
                    const scalar_t f3 = qx[q_iy] * x0;

                    quad4_f[0] = 4 * f0 - 2 * f1 - f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 - f3;
                    quad4_f[2] = -f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 - f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside (potential thread divergence)
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += infty * quad4_f[edof_i] * dV;
                    }

                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_xnormal[edof_i] += 1 * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                // Get the reminder [0, 1]
                real_t l_x = (grid_x - i);
                real_t l_y = (grid_y - j);
                real_t l_z = (grid_z - k);

                assert(l_x >= -1e-8);
                assert(l_y >= -1e-8);
                assert(l_z >= -1e-8);

                assert(l_x <= 1 + 1e-8);
                assert(l_y <= 1 + 1e-8);
                assert(l_z <= 1 + 1e-8);

                cu_hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
                cu_hex_aa_8_eval_grad(
                        cu_put_inside(l_x), cu_put_inside(l_y), cu_put_inside(l_z), hex8_grad_x, hex8_grad_y, hex8_grad_z);
                cu_hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_gap = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_gap += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += eval_gap * quad4_f[edof_i] * dV;
                    }
                }

                {
                    real_t eval_xnormal = 0;
                    real_t eval_ynormal = 0;
                    real_t eval_znormal = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_xnormal += hex8_grad_x[edof_j] * coeffs[edof_j];
                        eval_ynormal += hex8_grad_y[edof_j] * coeffs[edof_j];
                        eval_znormal += hex8_grad_z[edof_j] * coeffs[edof_j];
                    }

                    {
                        // Normalize
                        const real_t denom = MAX(
                                1e-20,
                                sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal));

                        assert(denom != 0);

                        eval_xnormal /= denom;
                        eval_ynormal /= denom;
                        eval_znormal /= denom;
                    }

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_xnormal[edof_i] += eval_xnormal * quad4_f[edof_i] * dV;
                        element_ynormal[edof_i] += eval_ynormal * quad4_f[edof_i] * dV;
                        element_znormal[edof_i] += eval_znormal * quad4_f[edof_i] * dV;
                    }
                }
            }
        }

#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            // Invert sign since distance field is negative inside and positive outside

            atomicAdd(&wg[ev[v]], -element_gap[v]);
            atomicAdd(&xnormal[ev[v]], element_xnormal[v]);
            atomicAdd(&ynormal[ev[v]], element_ynormal[v]);
            atomicAdd(&znormal[ev[v]], element_znormal[v]);
        }
    }
}

extern "C" int cu_quadshell4_resample_gap_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg,
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_local_kernel<<<n_blocks, block_size, 0>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, xnormal, ynormal, znormal);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

__global__ void cu_quadshell4_resample_weight_local_kernel(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // Output
        real_t* const SFEM_RESTRICT w)

{
    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t quad4_f[4];
        real_t element_weight[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            x[v] = xyz[0][ev[v]];
            y[v] = xyz[1][ev[v]];
            z[v] = xyz[2][ev[v]];
        }

        for (int v = 0; v < 4; v++) {
            element_weight[v] = 0;
        }

        for (int q_ix = 0; q_ix < n_qp; q_ix++) {
            for (int q_iy = 0; q_iy < n_qp; q_iy++) {
                const real_t measure = cu_quadshell4_measure(
                        x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3], qx[q_ix], qx[q_iy]);

                assert(measure > 0);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    quad4_f[0]        = x0 * x1;
                    quad4_f[1]        = qx[q_ix] * x1;
                    quad4_f[2]        = qx[q_ix] * qx[q_iy];
                    quad4_f[3]        = qx[q_iy] * x0;
                }
#else
                // DUAL basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    const scalar_t f0 = x0 * x1;
                    const scalar_t f1 = qx[q_ix] * x1;
                    const scalar_t f2 = qx[q_ix] * qx[q_iy];
                    const scalar_t f3 = qx[q_iy] * x0;

                    quad4_f[0] = 4 * f0 - 2 * f1 - f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 - f3;
                    quad4_f[2] = -f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 - f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];

                // Integrate weight function
                {
#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_weight[edof_i] += quad4_f[edof_i] * dV;
                    }
                }
            }
        }

        for (int v = 0; v < 4; ++v) {
            atomicAdd(&w[ev[v]], element_weight[v]);
        }
    }
}

extern "C" int cu_quadshell4_resample_weight_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // Output
        real_t* const SFEM_RESTRICT w)

{
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_weight_local_kernel<<<n_blocks, block_size, 0>>>(nelements, nnodes, elems, xyz, w);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

__global__ void cu_quadshell4_resample_gap_value_local_kernel(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t hex8_f[8];
        real_t coeffs[8];

        real_t quad4_f[4];
        real_t element_gap[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            x[v] = xyz[0][ev[v]];
            y[v] = xyz[1][ev[v]];
            z[v] = xyz[2][ev[v]];
        }

        memset(element_gap, 0, 4 * sizeof(real_t));

        for (int q_ix = 0; q_ix < n_qp; q_ix++) {
            for (int q_iy = 0; q_iy < n_qp; q_iy++) {
                const real_t measure = cu_quadshell4_measure(
                        x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3], qx[q_ix], qx[q_iy]);

                assert(measure > 0);

                real_t g_qx, g_qy, g_qz;
                cu_quadshell4_transform(x[0],
                                        x[1],
                                        x[2],
                                        x[3],
                                        y[0],
                                        y[1],
                                        y[2],
                                        y[3],
                                        z[0],
                                        z[1],
                                        z[2],
                                        z[3],
                                        qx[q_ix],
                                        qx[q_iy],
                                        &g_qx,
                                        &g_qy,
                                        &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    quad4_f[0]        = x0 * x1;
                    quad4_f[1]        = qx[q_ix] * x1;
                    quad4_f[2]        = qx[q_ix] * qx[q_iy];
                    quad4_f[3]        = qx[q_iy] * x0;
                }
#else
                // DUAL basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    const scalar_t f0 = x0 * x1;
                    const scalar_t f1 = qx[q_ix] * x1;
                    const scalar_t f2 = qx[q_ix] * qx[q_iy];
                    const scalar_t f3 = qx[q_iy] * x0;

                    quad4_f[0] = 4 * f0 - 2 * f1 - f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 - f3;
                    quad4_f[2] = -f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 - f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += -infty * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                // Get the reminder [0, 1]
                real_t l_x = (grid_x - i);
                real_t l_y = (grid_y - j);
                real_t l_z = (grid_z - k);

                assert(l_x >= -1e-8);
                assert(l_y >= -1e-8);
                assert(l_z >= -1e-8);

                assert(l_x <= 1 + 1e-8);
                assert(l_y <= 1 + 1e-8);
                assert(l_z <= 1 + 1e-8);

                cu_hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);
                cu_hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                // Integrate gap function
                {
                    real_t eval_gap = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_gap += hex8_f[edof_j] * coeffs[edof_j];
                    }

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += eval_gap * quad4_f[edof_i] * dV;
                    }
                }
            }
        }

        for (int v = 0; v < 4; ++v) {
            // Invert sign since distance field is negative inside and positive outside
            atomicAdd(&wg[ev[v]], -element_gap[v]);
        }
    }
}

extern "C" int cu_quadshell4_resample_gap_value_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT wg) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_value_local_kernel<<<n_blocks, block_size, 0>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}

__global__ void cu_quadshell4_resample_gap_normals_local_kernel(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    const real_t ox = (real_t)origin[0];
    const real_t oy = (real_t)origin[1];
    const real_t oz = (real_t)origin[2];

    const real_t dx = (real_t)delta[0];
    const real_t dy = (real_t)delta[1];
    const real_t dz = (real_t)delta[2];

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t hex8_grad_x[8];
        real_t hex8_grad_y[8];
        real_t hex8_grad_z[8];
        real_t coeffs[8];

        real_t quad4_f[4];
        real_t element_xnormal[4];
        real_t element_ynormal[4];
        real_t element_znormal[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            x[v] = xyz[0][ev[v]];
            y[v] = xyz[1][ev[v]];
            z[v] = xyz[2][ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_xnormal[v] = 0;
            element_ynormal[v] = 0;
            element_znormal[v] = 0;
        }

        for (int q_ix = 0; q_ix < n_qp; q_ix++) {
            for (int q_iy = 0; q_iy < n_qp; q_iy++) {
                const real_t measure = cu_quadshell4_measure(
                        x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3], qx[q_ix], qx[q_iy]);

                assert(measure > 0);

                real_t g_qx, g_qy, g_qz;
                cu_quadshell4_transform(x[0],
                                        x[1],
                                        x[2],
                                        x[3],
                                        y[0],
                                        y[1],
                                        y[2],
                                        y[3],
                                        z[0],
                                        z[1],
                                        z[2],
                                        z[3],
                                        qx[q_ix],
                                        qx[q_iy],
                                        &g_qx,
                                        &g_qy,
                                        &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    quad4_f[0]        = x0 * x1;
                    quad4_f[1]        = qx[q_ix] * x1;
                    quad4_f[2]        = qx[q_ix] * qx[q_iy];
                    quad4_f[3]        = qx[q_iy] * x0;
                }
#else
                // DUAL basis function
                {
                    const scalar_t x0 = 1 - qx[q_ix];
                    const scalar_t x1 = 1 - qx[q_iy];
                    const scalar_t f0 = x0 * x1;
                    const scalar_t f1 = qx[q_ix] * x1;
                    const scalar_t f2 = qx[q_ix] * qx[q_iy];
                    const scalar_t f3 = qx[q_iy] * x0;

                    quad4_f[0] = 4 * f0 - 2 * f1 - f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 - f3;
                    quad4_f[2] = -f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 - f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];
                assert(dV > 0);

                const real_t grid_x = (g_qx - ox) / dx;
                const real_t grid_y = (g_qy - oy) / dy;
                const real_t grid_z = (g_qz - oz) / dz;

                const ptrdiff_t i = floor(grid_x);
                const ptrdiff_t j = floor(grid_y);
                const ptrdiff_t k = floor(grid_z);

                // If outside
                if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_xnormal[edof_i] += 1. * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                // Get the reminder [0, 1]
                real_t l_x = (grid_x - i);
                real_t l_y = (grid_y - j);
                real_t l_z = (grid_z - k);

                assert(l_x >= -1e-8);
                assert(l_y >= -1e-8);
                assert(l_z >= -1e-8);

                assert(l_x <= 1 + 1e-8);
                assert(l_y <= 1 + 1e-8);
                assert(l_z <= 1 + 1e-8);

                cu_hex_aa_8_eval_grad(
                        cu_put_inside(l_x), cu_put_inside(l_y), cu_put_inside(l_z), hex8_grad_x, hex8_grad_y, hex8_grad_z);
                cu_hex_aa_8_collect_coeffs(stride, i, j, k, data, coeffs);

                {
                    real_t eval_xnormal = 0;
                    real_t eval_ynormal = 0;
                    real_t eval_znormal = 0;

#pragma unroll(8)
                    for (int edof_j = 0; edof_j < 8; edof_j++) {
                        eval_xnormal += hex8_grad_x[edof_j] * coeffs[edof_j];
                        eval_ynormal += hex8_grad_y[edof_j] * coeffs[edof_j];
                        eval_znormal += hex8_grad_z[edof_j] * coeffs[edof_j];
                    }

                    {
                        // Normalize
                        const real_t denom = MAX(
                                1e-20,
                                sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal));

                        assert(denom != 0);

                        eval_xnormal /= denom;
                        eval_ynormal /= denom;
                        eval_znormal /= denom;
                    }

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_xnormal[edof_i] += eval_xnormal * quad4_f[edof_i] * dV;
                        element_ynormal[edof_i] += eval_ynormal * quad4_f[edof_i] * dV;
                        element_znormal[edof_i] += eval_znormal * quad4_f[edof_i] * dV;
                    }
                }
            }
        }

        for (int v = 0; v < 4; ++v) {
            atomicAdd(&xnormal[ev[v]], element_xnormal[v]);
            atomicAdd(&ynormal[ev[v]], element_ynormal[v]);
            atomicAdd(&znormal[ev[v]], element_znormal[v]);
        }
    }
}

extern "C" int cu_quadshell4_resample_gap_normals_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,
        const ptrdiff_t* const SFEM_RESTRICT stride,
        const geom_t* const SFEM_RESTRICT    origin,
        const geom_t* const SFEM_RESTRICT    delta,
        const geom_t* const SFEM_RESTRICT    data,
        // Output
        real_t* const SFEM_RESTRICT xnormal,
        real_t* const SFEM_RESTRICT ynormal,
        real_t* const SFEM_RESTRICT znormal) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE();

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_normals_local_kernel<<<n_blocks, block_size, 0>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, xnormal, ynormal, znormal);

    SFEM_DEBUG_SYNCHRONIZE();
    return SFEM_SUCCESS;
}
