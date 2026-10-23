#include "trishell3_resample.hpp"
#include "trishell3_mass.hpp"

#include "sfem_sdf_sampling.hpp"

#include <math.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define SFEM_RESAMPLE_GAP_DUAL

// The gap where a point lies outside the SDF grid: large, so the constraint stays inactive,
// as on the other shells.
static const real_t infty = SFEM_SDF_UNREACHABLE;
static const real_t SFEM_SDF_GRAD_EPS = 1e-12;

// smesh::TRI3 6th order quadrature rule
#define TRI3_NQP 12
static real_t tri3_qw[TRI3_NQP] = {0.050844906370206816920936809106869,
                                   0.050844906370206816920936809106869,
                                   0.050844906370206816920936809106869,
                                   0.11678627572637936602528961138558,
                                   0.11678627572637936602528961138558,
                                   0.11678627572637936602528961138558,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442,
                                   0.082851075618373575193553456420442};

static real_t tri3_qx[TRI3_NQP] = {0.063089014491502228340331602870819,
                                   0.063089014491502228340331602870819,
                                   0.87382197101699554331933679425836,
                                   0.24928674517091042129163855310702,
                                   0.24928674517091042129163855310702,
                                   0.50142650965817915741672289378596,
                                   0.053145049844816947353249671631398,
                                   0.053145049844816947353249671631398,
                                   0.31035245103378440541660773395655,
                                   0.31035245103378440541660773395655,
                                   0.63650249912139864723014259441205,
                                   0.63650249912139864723014259441205};

static real_t tri3_qy[TRI3_NQP] = {0.063089014491502228340331602870819,
                                   0.87382197101699554331933679425836,
                                   0.063089014491502228340331602870819,
                                   0.24928674517091042129163855310702,
                                   0.50142650965817915741672289378596,
                                   0.24928674517091042129163855310702,
                                   0.31035245103378440541660773395655,
                                   0.63650249912139864723014259441205,
                                   0.053145049844816947353249671631398,
                                   0.63650249912139864723014259441205,
                                   0.053145049844816947353249671631398,
                                   0.31035245103378440541660773395655};

SFEM_INLINE static real_t tri_shell_3_measure(
        // X-coordinates
        const real_t px0,
        const real_t px1,
        const real_t px2,
        // Y-coordinates
        const real_t py0,
        const real_t py1,
        const real_t py2,
        // Z-coordinates
        const real_t pz0,
        const real_t pz1,
        const real_t pz2) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    return (1.0 / 2.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                              pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
}

SFEM_INLINE static void tri_shell_3_transform(
        // X-coordinates
        const real_t x0,
        const real_t x1,
        const real_t x2,
        // Y-coordinates
        const real_t y0,
        const real_t y1,
        const real_t y2,
        // Z-coordinates
        const real_t z0,
        const real_t z1,
        const real_t z2,
        // Quadrature point
        const real_t x,
        const real_t y,
        // Output
        real_t* const SFEM_RESTRICT out_x,
        real_t* const SFEM_RESTRICT out_y,
        real_t* const SFEM_RESTRICT out_z) {
    const real_t phi0 = 1 - x - y;
    const real_t phi1 = x;
    const real_t phi2 = y;

    *out_x = phi0 * x0 + phi1 * x1 + phi2 * x2;
    *out_y = phi0 * y0 + phi1 * y1 + phi2 * y2;
    *out_z = phi0 * z0 + phi1 * z1 + phi2 * z2;
}

int trishell3_resample_gap_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              /*nnodes*/,
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

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t  ev[3];
            geom_t x[3], y[3], z[3];

            real_t tri3_f[3];
            real_t element_gap[3];
            real_t element_xnormal[3];
            real_t element_ynormal[3];
            real_t element_znormal[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 3; ++v) {
                x[v] = xyz[0][ev[v]];
                y[v] = xyz[1][ev[v]];
                z[v] = xyz[2][ev[v]];
            }

            memset(element_gap, 0, 3 * sizeof(real_t));
            memset(element_xnormal, 0, 3 * sizeof(real_t));
            memset(element_ynormal, 0, 3 * sizeof(real_t));
            memset(element_znormal, 0, 3 * sizeof(real_t));

            const real_t measure = tri_shell_3_measure(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);

            assert(measure > 0);

            // The triangle's own normal, for points outside the grid and flat SDF cells.
            real_t face_nx, face_ny, face_nz;
            {
                const real_t ux = x[1] - x[0], uy = y[1] - y[0], uz = z[1] - z[0];
                const real_t vx = x[2] - x[0], vy = y[2] - y[0], vz = z[2] - z[0];
                face_nx = uy * vz - uz * vy;
                face_ny = uz * vx - ux * vz;
                face_nz = ux * vy - uy * vx;
                const real_t len = sqrt(face_nx * face_nx + face_ny * face_ny + face_nz * face_nz);
                face_nx /= len;
                face_ny /= len;
                face_nz /= len;
            }

            for (int q = 0; q < TRI3_NQP; q++) {
                real_t g_qx, g_qy, g_qz;
                tri_shell_3_transform(
                        x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2], tri3_qx[q], tri3_qy[q], &g_qx, &g_qy, &g_qz);

#ifndef SFEM_RESAMPLE_GAP_DUAL
                // Standard basis function
                {
                    tri3_f[0] = 1 - tri3_qx[q] - tri3_qy[q];
                    tri3_f[1] = tri3_qx[q];
                    tri3_f[2] = tri3_qy[q];
                }
#else
                // DUAL basis function
                {
                    const real_t f0 = 1 - tri3_qx[q] - tri3_qy[q];
                    const real_t f1 = tri3_qx[q];
                    const real_t f2 = tri3_qy[q];

                    tri3_f[0] = 3 * f0 - f1 - f2;
                    tri3_f[1] = -f0 + 3 * f1 - f2;
                    tri3_f[2] = -f0 - f1 + 3 * f2;
                }
#endif

                const real_t dV = measure * tri3_qw[q];

                real_t eval_value, eval_grad[3];
                // If outside
                if (!sdf_sample<1>(n, stride, origin, delta, data, g_qx, g_qy, g_qz, &eval_value, eval_grad, nullptr)) {
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_gap[edof_i] += -infty * tri3_f[edof_i] * dV;
                        element_xnormal[edof_i] += face_nx * tri3_f[edof_i] * dV;
                        element_ynormal[edof_i] += face_ny * tri3_f[edof_i] * dV;
                        element_znormal[edof_i] += face_nz * tri3_f[edof_i] * dV;
                    }
                    continue;
                }

                // Integrate gap function
                {
                    const real_t eval_gap = eval_value;

#pragma unroll(3)
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_gap[edof_i] += eval_gap * tri3_f[edof_i] * dV;
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

#pragma unroll(3)
                    for (int edof_i = 0; edof_i < 3; edof_i++) {
                        element_xnormal[edof_i] += eval_xnormal * tri3_f[edof_i] * dV;
                        element_ynormal[edof_i] += eval_ynormal * tri3_f[edof_i] * dV;
                        element_znormal[edof_i] += eval_znormal * tri3_f[edof_i] * dV;
                    }
                }
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                // Invert sign since distance field is negative insdide and positive outside
#pragma omp critical
                {
                    wg[ev[v]] -= element_gap[v];
                    xnormal[ev[v]] += element_xnormal[v];
                    ynormal[ev[v]] += element_ynormal[v];
                    znormal[ev[v]] += element_znormal[v];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int trishell3_resample_weight_local(
        // Mesh
        const ptrdiff_t              nelements,
        const ptrdiff_t              nnodes,
        idx_t** const SFEM_RESTRICT  elems,
        geom_t** const SFEM_RESTRICT xyz,
        // Output
        real_t* const SFEM_RESTRICT w)
{
    return trishell3_assemble_lumped_mass(nelements, nnodes, elems, xyz, w);
}
