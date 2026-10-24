#include "cu_quadshell4_resample.hpp"

#include "line_quadrature.hpp"
#include "sfem_sdf_sampling.hpp"
#include "sfem_macros.hpp"

#include "sfem_cuda_base.hpp"

static const scalar_t infty             = SFEM_SDF_UNREACHABLE;
static const real_t   SFEM_SDF_GRAD_EPS = 1e-12;

// The gap is weighted with the dual (biorthogonal) basis, as on TRISHELL3 and BEAM2: then
// int psi_i phi_j = delta_ij m_j, the weighted gap's linearisation is the nodal n . u the
// constraint operator applies, and re-linearising at the solution converges at once instead of
// like I - M_l^-1 M_c.
#define SFEM_RESAMPLE_GAP_DUAL

SFEM_INLINE __device__ static real_t cu_quadshell4_measure_and_normal(
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
        // Output unit face normal (mesh winding)
        real_t* const SFEM_RESTRICT nx,
        real_t* const SFEM_RESTRICT ny,
        real_t* const SFEM_RESTRICT nz) {
    const scalar_t x0 = qx - 1;
    const scalar_t x1 = -x0;
    // r_xi = (x2, x6, x8), r_eta = (x5, x7, x9)
    const scalar_t x2 = px0 * x0 - px1 * qx + px2 * qx + px3 * x1;
    const scalar_t x3 = qy - 1;
    const scalar_t x4 = -x3;
    const scalar_t x5 = px0 * x3 + px1 * x4 + px2 * qy - px3 * qy;
    const scalar_t x6 = py0 * x0 - py1 * qx + py2 * qx + py3 * x1;
    const scalar_t x7 = py0 * x3 + py1 * x4 + py2 * qy - py3 * qy;
    const scalar_t x8 = pz0 * x0 - pz1 * qx + pz2 * qx + pz3 * x1;
    const scalar_t x9 = pz0 * x3 + pz1 * x4 + pz2 * qy - pz3 * qy;

    const real_t cx      = x6 * x9 - x8 * x7;
    const real_t cy      = x8 * x5 - x2 * x9;
    const real_t cz      = x2 * x7 - x6 * x5;
    const real_t measure = sqrt(POW2(cx) + POW2(cy) + POW2(cz));

    if (measure > SFEM_SDF_GRAD_EPS) {
        *nx = cx / measure;
        *ny = cy / measure;
        *nz = cz / measure;
    } else {
        // Degenerate / collapsed quad at this QP
        *nx = 1;
        *ny = 0;
        *nz = 0;
    }
    return measure;
}

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
    real_t nx, ny, nz;
    return cu_quadshell4_measure_and_normal(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, qx, qy, &nx, &ny, &nz);
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
        real_t* const SFEM_RESTRICT  wg,
        real_t** const SFEM_RESTRICT normals) {

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t quad4_f[4];
        real_t element_gap[4];
        real_t element_xnormal[4];
        real_t element_ynormal[4];
        real_t element_znormal[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][e];
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
                real_t       face_nx, face_ny, face_nz;
                const real_t measure = cu_quadshell4_measure_and_normal(x[0],
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
                                                                        &face_nx,
                                                                        &face_ny,
                                                                        &face_nz);

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

                    quad4_f[0] = 4 * f0 - 2 * f1 + f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 + f3;
                    quad4_f[2] = f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 + f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];

                real_t eval_value, eval_grad[3];
                if (!sdf_sample<1>(n, stride, origin, delta, data, g_qx, g_qy, g_qz, &eval_value, eval_grad, nullptr)) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += -infty * quad4_f[edof_i] * dV;
                        element_xnormal[edof_i] += face_nx * quad4_f[edof_i] * dV;
                        element_ynormal[edof_i] += face_ny * quad4_f[edof_i] * dV;
                        element_znormal[edof_i] += face_nz * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                // Integrate gap function
                {
                    const real_t eval_gap = eval_value;

#pragma unroll(4)
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += eval_gap * quad4_f[edof_i] * dV;
                    }
                }

                {
                    real_t eval_xnormal = eval_grad[0];
                    real_t eval_ynormal = eval_grad[1];
                    real_t eval_znormal = eval_grad[2];

                    const real_t denom =
                            sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                    if (denom >= SFEM_SDF_GRAD_EPS) {
                        eval_xnormal /= denom;
                        eval_ynormal /= denom;
                        eval_znormal /= denom;
                    } else {
                        // Flat SDF cell: fall back to geometric face normal
                        eval_xnormal = face_nx;
                        eval_ynormal = face_ny;
                        eval_znormal = face_nz;
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
            atomicAdd(&normals[0][ev[v]], element_xnormal[v]);
            atomicAdd(&normals[1][ev[v]], element_ynormal[v]);
            atomicAdd(&normals[2][ev[v]], element_znormal[v]);
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
        real_t* const SFEM_RESTRICT  wg,
        real_t** const SFEM_RESTRICT normals,
        void* const                  stream) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE(stream);

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_local_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg, normals);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t quad4_f[4];
        real_t element_weight[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][e];
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

                    quad4_f[0] = 4 * f0 - 2 * f1 + f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 + f3;
                    quad4_f[2] = f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 + f1 - 2 * f2 + 4 * f3;
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
        real_t* const SFEM_RESTRICT w,
        void* const                 stream)

{
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE(stream);

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_weight_local_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            nelements, nnodes, elems, xyz, w);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t quad4_f[4];
        real_t element_gap[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][e];
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

                    quad4_f[0] = 4 * f0 - 2 * f1 + f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 + f3;
                    quad4_f[2] = f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 + f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];

                real_t eval_value;
                // If outside grid: inject -infty before wg -= so stored gap becomes +infty.
                // sample_value() feeds ShiftedPenalty upper bounds (n·u ≤ ub); +infty keeps the
                // constraint inactive. (gap_local uses +infty here instead → stored -infty for
                // signed-distance / viz semantics.)
                if (!sdf_sample<0>(n, stride, origin, delta, data, g_qx, g_qy, g_qz, &eval_value, nullptr, nullptr)) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_gap[edof_i] += -infty * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                // Integrate gap function
                {
                    const real_t eval_gap = eval_value;

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
        real_t* const SFEM_RESTRICT wg,
        void* const                 stream) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE(stream);

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_value_local_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, wg);

    SFEM_DEBUG_SYNCHRONIZE(stream);
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
        real_t** const SFEM_RESTRICT normals) {

    static const int      n_qp  = 6;
    static const scalar_t qx[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
    static const scalar_t qw[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};

    for (ptrdiff_t e = blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += blockDim.x * gridDim.x) {
        idx_t  ev[4];
        geom_t x[4], y[4], z[4];

        real_t quad4_f[4];
        real_t element_xnormal[4];
        real_t element_ynormal[4];
        real_t element_znormal[4];

        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][e];
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
                real_t       face_nx, face_ny, face_nz;
                const real_t measure = cu_quadshell4_measure_and_normal(x[0],
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
                                                                        &face_nx,
                                                                        &face_ny,
                                                                        &face_nz);

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

                    quad4_f[0] = 4 * f0 - 2 * f1 + f2 - 2 * f3;
                    quad4_f[1] = -2 * f0 + 4 * f1 - 2 * f2 + f3;
                    quad4_f[2] = f0 - 2 * f1 + 4 * f2 - 2 * f3;
                    quad4_f[3] = -2 * f0 + f1 - 2 * f2 + 4 * f3;
                }
#endif
                const real_t dV = measure * qw[q_ix] * qw[q_iy];
                assert(dV > 0);

                real_t eval_value, eval_grad[3];
                // If outside
                if (!sdf_sample<1>(n, stride, origin, delta, data, g_qx, g_qy, g_qz, &eval_value, eval_grad, nullptr)) {
                    for (int edof_i = 0; edof_i < 4; edof_i++) {
                        element_xnormal[edof_i] += face_nx * quad4_f[edof_i] * dV;
                        element_ynormal[edof_i] += face_ny * quad4_f[edof_i] * dV;
                        element_znormal[edof_i] += face_nz * quad4_f[edof_i] * dV;
                    }

                    continue;
                }

                {
                    real_t eval_xnormal = eval_grad[0];
                    real_t eval_ynormal = eval_grad[1];
                    real_t eval_znormal = eval_grad[2];

                    const real_t denom =
                            sqrt(eval_xnormal * eval_xnormal + eval_ynormal * eval_ynormal + eval_znormal * eval_znormal);

                    if (denom >= SFEM_SDF_GRAD_EPS) {
                        eval_xnormal /= denom;
                        eval_ynormal /= denom;
                        eval_znormal /= denom;
                    } else {
                        // Flat SDF cell: fall back to geometric face normal
                        eval_xnormal = face_nx;
                        eval_ynormal = face_ny;
                        eval_znormal = face_nz;
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
            atomicAdd(&normals[0][ev[v]], element_xnormal[v]);
            atomicAdd(&normals[1][ev[v]], element_ynormal[v]);
            atomicAdd(&normals[2][ev[v]], element_znormal[v]);
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
        real_t** const SFEM_RESTRICT normals,
        void* const                  stream) {
    if (!nelements) return 0;

    SFEM_DEBUG_SYNCHRONIZE(stream);

    int             block_size = 128;
    const ptrdiff_t n_blocks   = MAX(ptrdiff_t(1), (nelements + block_size - 1) / block_size);
    cu_quadshell4_resample_gap_normals_local_kernel<<<n_blocks, block_size, 0, (cudaStream_t)stream>>>(
            nelements, nnodes, elems, xyz, n, stride, origin, delta, data, normals);

    SFEM_DEBUG_SYNCHRONIZE(stream);
    return SFEM_SUCCESS;
}
