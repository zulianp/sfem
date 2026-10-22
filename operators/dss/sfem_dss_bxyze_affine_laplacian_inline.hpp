#pragma once

#include "sfem_dss_bxyze_affine_laplacian.hpp"

#include <cassert>

// Inline all numerical stages so a specialized storage stride reaches their loops.

// Scratch scalars for one gradient batch, excluding the three output buffers.
template <int N, int Q, int B_EXEC>
static constexpr ptrdiff_t sfem_dss_bxyze_lagrange_gradient_workspace_n_values() {
    static_assert(N > 0 && Q > 0 && B_EXEC > 0, "Invalid contraction dimensions");
    return ((ptrdiff_t)2 * N * N * Q + (ptrdiff_t)3 * N * Q) * B_EXEC;
}

// The adjoint reuses the same two x buffers and three streamed qy planes.
template <int N, int Q, int B_EXEC>
static constexpr ptrdiff_t sfem_dss_bxyze_lagrange_gradient_transpose_workspace_n_values() {
    return sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, B_EXEC>();
}

/* Reference gradients at quadrature points for one full BXYZE batch:
 *   gx = (D_x S_y S_z) x, gy = (S_x D_y S_z) x, gz = (S_x S_y D_z) x.
 *
 * x_bxyze points to the first lane of B_EXEC valid elements; its nodal stride
 * is storage_bsize. The caller must provide a full batch.
 * Each output has Q * Q * Q * B_EXEC scalars, ordered [qz][qy][qx][b].
 * All B_EXEC lanes are written. No geometry or quadrature weights are applied.
 *
 * Workspace: two [iz][iy][qx][b] arrays, followed by three [iz][qx][b]
 * arrays. The latter hold one qy slice and are reused across qy iterations:
 *   sx = S_x x, dx = D_x x;
 *   sy_dx = S_y dx, dy_sx = D_y sx, sy_sx = S_y sx;
 *   gx = S_z sy_dx, gy = S_z dy_sx, gz = D_z sy_sx.
 * This shares S_x x and fuses the three y contractions and z contractions.
 * Scratch size is given by gradient_workspace_n_values<N, Q, B_EXEC>().
 * Outputs, input and workspace must not overlap. Thread parallelism is external;
 * concurrent calls need separate outputs and workspace. There are no allocations.
 */
template <int N, int Q, int B_EXEC>
static SFEM_FORCE_INLINE void sfem_dss_bxyze_lagrange_gradient(
        const int storage_bsize,
        const scalar_t *const SFEM_RESTRICT S,
        const scalar_t *const SFEM_RESTRICT D,
        const real_t *const SFEM_RESTRICT x_bxyze,
        scalar_t *const SFEM_RESTRICT gx,
        scalar_t *const SFEM_RESTRICT gy,
        scalar_t *const SFEM_RESTRICT gz,
        scalar_t *const SFEM_RESTRICT workspace) {
    static_assert(N > 0 && Q > 0 && B_EXEC > 0, "Invalid contraction dimensions");
    assert(storage_bsize >= B_EXEC);
    assert(S && D && x_bxyze && gx && gy && gz && workspace);

    constexpr ptrdiff_t x_size = (ptrdiff_t)N * N * Q * B_EXEC;
    constexpr ptrdiff_t y_size = (ptrdiff_t)N * Q * B_EXEC;
    scalar_t *const sx = workspace;
    scalar_t *const dx = sx + x_size;
    scalar_t *const sy_dx = dx + x_size;
    scalar_t *const dy_sx = sy_dx + y_size;
    scalar_t *const sy_sx = dy_sx + y_size;

    // Contract x once, sharing input reads between interpolation and derivative.
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            const ptrdiff_t line = (ptrdiff_t)iz * N + iy;
            const real_t *const input = &x_bxyze[line * N * storage_bsize];

            for (int qx = 0; qx < Q; ++qx) {
                scalar_t sum_s[B_EXEC];
                scalar_t sum_d[B_EXEC];
                const scalar_t *const sq = &S[(ptrdiff_t)qx * N];
                const scalar_t *const dq = &D[(ptrdiff_t)qx * N];
                const scalar_t s0 = sq[0];
                const scalar_t d0 = dq[0];

#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    const scalar_t u = (scalar_t)input[b];
                    sum_s[b] = s0 * u;
                    sum_d[b] = d0 * u;
                }

                for (int ix = 1; ix < N; ++ix) {
                    const scalar_t s = sq[ix];
                    const scalar_t d = dq[ix];
                    const real_t *const src = &input[(ptrdiff_t)ix * storage_bsize];

#pragma omp simd
                    for (int b = 0; b < B_EXEC; ++b) {
                        const scalar_t u = (scalar_t)src[b];
                        sum_s[b] += s * u;
                        sum_d[b] += d * u;
                    }
                }

                const ptrdiff_t out = (line * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sx[out + b] = sum_s[b];
                    dx[out + b] = sum_d[b];
                }
            }
        }
    }

    for (int qy = 0; qy < Q; ++qy) {
        const scalar_t *const sqy = &S[(ptrdiff_t)qy * N];
        const scalar_t *const dqy = &D[(ptrdiff_t)qy * N];

        // Contract y for one qy, keeping z nodal and x at quadrature points.
        for (int iz = 0; iz < N; ++iz) {
            for (int qx = 0; qx < Q; ++qx) {
                scalar_t sum_x[B_EXEC];
                scalar_t sum_y[B_EXEC];
                scalar_t sum_z[B_EXEC];
                const ptrdiff_t in = ((ptrdiff_t)iz * N * Q + qx) * B_EXEC;
                const scalar_t s0 = sqy[0];
                const scalar_t d0 = dqy[0];

#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    const scalar_t su = sx[in + b];
                    const scalar_t du = dx[in + b];
                    sum_x[b] = s0 * du;
                    sum_y[b] = d0 * su;
                    sum_z[b] = s0 * su;
                }

                for (int iy = 1; iy < N; ++iy) {
                    const scalar_t s = sqy[iy];
                    const scalar_t d = dqy[iy];
                    const ptrdiff_t off = in + (ptrdiff_t)iy * Q * B_EXEC;

#pragma omp simd
                    for (int b = 0; b < B_EXEC; ++b) {
                        const scalar_t su = sx[off + b];
                        const scalar_t du = dx[off + b];
                        sum_x[b] += s * du;
                        sum_y[b] += d * su;
                        sum_z[b] += s * su;
                    }
                }

                const ptrdiff_t out = ((ptrdiff_t)iz * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sy_dx[out + b] = sum_x[b];
                    dy_sx[out + b] = sum_y[b];
                    sy_sx[out + b] = sum_z[b];
                }
            }
        }

        // Contract z, writing the complete qy slice before reusing its scratch.
        for (int qz = 0; qz < Q; ++qz) {
            const scalar_t *const sqz = &S[(ptrdiff_t)qz * N];
            const scalar_t *const dqz = &D[(ptrdiff_t)qz * N];
            for (int qx = 0; qx < Q; ++qx) {
                scalar_t sum_x[B_EXEC];
                scalar_t sum_y[B_EXEC];
                scalar_t sum_z[B_EXEC];
                const ptrdiff_t in = (ptrdiff_t)qx * B_EXEC;
                const scalar_t s0 = sqz[0];
                const scalar_t d0 = dqz[0];

#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sum_x[b] = s0 * sy_dx[in + b];
                    sum_y[b] = s0 * dy_sx[in + b];
                    sum_z[b] = d0 * sy_sx[in + b];
                }

                for (int iz = 1; iz < N; ++iz) {
                    const scalar_t s = sqz[iz];
                    const scalar_t d = dqz[iz];
                    const ptrdiff_t off = in + (ptrdiff_t)iz * Q * B_EXEC;

#pragma omp simd
                    for (int b = 0; b < B_EXEC; ++b) {
                        sum_x[b] += s * sy_dx[off + b];
                        sum_y[b] += s * dy_sx[off + b];
                        sum_z[b] += d * sy_sx[off + b];
                    }
                }

                const ptrdiff_t out = (((ptrdiff_t)qz * Q + qy) * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    gx[out + b] = sum_x[b];
                    gy[out + b] = sum_y[b];
                    gz[out + b] = sum_z[b];
                }
            }
        }
    }
}

/* Apply G^T to one full batch of quadrature fields, overwriting nodal values.
 * Inputs fx/fy/fz are [qz][qy][qx][b], each Q^3 * B_EXEC scalars.
 * y_bxyze points to the first batch lane, with nodal stride storage_bsize.
 * S and D retain their forward [q][i] ordering; no weights are applied here.
 *
 * Eight contractions (the reverse of the gradient graph):
 *   A = S_z^T fx, B = S_z^T fy, C = D_z^T fz;
 *   dx = S_y^T A, sx = D_y^T B + S_y^T C;
 *   y = D_x^T dx + S_x^T sx.
 * Stream one qy plane through A/B/C, accumulating sx/dx across qy.
 * Scratch: two [iz][iy][qx][b] arrays, then three [iz][qx][b] arrays,
 * exactly gradient_transpose_workspace_n_values<N, Q, B_EXEC>() scalars.
 * All scratch entries are initialized before use; no clearing is required.
 * Inputs, output and scratch must be disjoint. Thread parallelism is external.
 */
template <int N, int Q, int B_EXEC>
static SFEM_FORCE_INLINE void sfem_dss_bxyze_lagrange_gradient_transpose(
        const int storage_bsize,
        const scalar_t *const SFEM_RESTRICT S,
        const scalar_t *const SFEM_RESTRICT D,
        const scalar_t *const SFEM_RESTRICT fx,
        const scalar_t *const SFEM_RESTRICT fy,
        const scalar_t *const SFEM_RESTRICT fz,
        real_t *const SFEM_RESTRICT y_bxyze,
        scalar_t *const SFEM_RESTRICT workspace) {
    static_assert(N > 0 && Q > 0 && B_EXEC > 0, "Invalid contraction dimensions");
    assert(storage_bsize >= B_EXEC);
    assert(S && D && fx && fy && fz && y_bxyze && workspace);
    constexpr ptrdiff_t x_size = (ptrdiff_t)N * N * Q * B_EXEC;
    constexpr ptrdiff_t plane_size = (ptrdiff_t)N * Q * B_EXEC;
    scalar_t *const sx = workspace;
    scalar_t *const dx = sx + x_size;
    scalar_t *const sz_fx = dx + x_size;
    scalar_t *const sz_fy = sz_fx + plane_size;
    scalar_t *const dz_fz = sz_fy + plane_size;

    for (int qy = 0; qy < Q; ++qy) {
        // Three z contractions for one qy plane, keeping x at quadrature points.
        for (int iz = 0; iz < N; ++iz) {
            const scalar_t s0 = S[iz], d0 = D[iz];
            for (int qx = 0; qx < Q; ++qx) {
                scalar_t sum_x[B_EXEC], sum_y[B_EXEC], sum_z[B_EXEC];
                const ptrdiff_t in = ((ptrdiff_t)qy * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sum_x[b] = s0 * fx[in + b];
                    sum_y[b] = s0 * fy[in + b];
                    sum_z[b] = d0 * fz[in + b];
                }
                for (int qz = 1; qz < Q; ++qz) {
                    const scalar_t s = S[(ptrdiff_t)qz * N + iz];
                    const scalar_t d = D[(ptrdiff_t)qz * N + iz];
                    const ptrdiff_t off = in + (ptrdiff_t)qz * Q * Q * B_EXEC;
#pragma omp simd
                    for (int b = 0; b < B_EXEC; ++b) {
                        sum_x[b] += s * fx[off + b];
                        sum_y[b] += s * fy[off + b];
                        sum_z[b] += d * fz[off + b];
                    }
                }
                const ptrdiff_t out = ((ptrdiff_t)iz * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sz_fx[out + b] = sum_x[b];
                    sz_fy[out + b] = sum_y[b];
                    dz_fz[out + b] = sum_z[b];
                }
            }
        }

        // Three y contractions merge into two buffers; seed them on first qy.
        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                const scalar_t s = S[(ptrdiff_t)qy * N + iy];
                const scalar_t d = D[(ptrdiff_t)qy * N + iy];
                for (int qx = 0; qx < Q; ++qx) {
                    const ptrdiff_t in = ((ptrdiff_t)iz * Q + qx) * B_EXEC;
                    const ptrdiff_t out = (((ptrdiff_t)iz * N + iy) * Q + qx) * B_EXEC;
                    if (qy == 0) {
#pragma omp simd
                        for (int b = 0; b < B_EXEC; ++b) {
                            dx[out + b] = s * sz_fx[in + b];
                            sx[out + b] = d * sz_fy[in + b] + s * dz_fz[in + b];
                        }
                    } else {
#pragma omp simd
                        for (int b = 0; b < B_EXEC; ++b) {
                            dx[out + b] += s * sz_fx[in + b];
                            sx[out + b] += d * sz_fy[in + b] + s * dz_fz[in + b];
                        }
                    }
                }
            }
        }
    }

    // Two x contractions, summed and stored once per node directly in BXYZE.
    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            const ptrdiff_t line = (ptrdiff_t)iz * N + iy;
            const ptrdiff_t in = line * Q * B_EXEC;
            for (int ix = 0; ix < N; ++ix) {
                scalar_t sum_s[B_EXEC], sum_d[B_EXEC];
                const scalar_t s0 = S[ix], d0 = D[ix];
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    sum_s[b] = s0 * sx[in + b];
                    sum_d[b] = d0 * dx[in + b];
                }
                for (int qx = 1; qx < Q; ++qx) {
                    const scalar_t s = S[(ptrdiff_t)qx * N + ix];
                    const scalar_t d = D[(ptrdiff_t)qx * N + ix];
                    const ptrdiff_t off = in + (ptrdiff_t)qx * B_EXEC;
#pragma omp simd
                    for (int b = 0; b < B_EXEC; ++b) {
                        sum_s[b] += s * sx[off + b];
                        sum_d[b] += d * dx[off + b];
                    }
                }
                const ptrdiff_t out = (line * N + ix) * storage_bsize;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) y_bxyze[out + b] = (real_t)(sum_s[b] + sum_d[b]);
            }
        }
    }
}

/* Replace reference gradients with weighted fluxes: f_q = w_q H_e grad(u)_q.
 * geometry points to the first batch lane, ordered [H00,H01,H02,H11,H12,H22]
 * with storage_bsize scalars between components. H includes det(J), not weights.
 * gx/gy/gz each contain Q^3 * B_EXEC scalars, ordered [qz][qy][qx][b].
 * Full batches only; all arrays must be disjoint. No additional scratch.
 */
template <int Q, int B_EXEC>
static SFEM_FORCE_INLINE void sfem_dss_bxyze_affine_metric_apply(
        const int storage_bsize,
        const scalar_t *const SFEM_RESTRICT qw,
        const scalar_t *const SFEM_RESTRICT geometry,
        scalar_t *const SFEM_RESTRICT gx,
        scalar_t *const SFEM_RESTRICT gy,
        scalar_t *const SFEM_RESTRICT gz) {
    static_assert(Q > 0 && B_EXEC > 0, "Invalid metric dimensions");
    assert(storage_bsize >= B_EXEC);
    assert(qw && geometry && gx && gy && gz);
    const scalar_t *const h00 = geometry;
    const scalar_t *const h01 = h00 + storage_bsize;
    const scalar_t *const h02 = h01 + storage_bsize;
    const scalar_t *const h11 = h02 + storage_bsize;
    const scalar_t *const h12 = h11 + storage_bsize;
    const scalar_t *const h22 = h12 + storage_bsize;

    for (int qz = 0; qz < Q; ++qz) {
        for (int qy = 0; qy < Q; ++qy) {
            const scalar_t wyz = qw[qz] * qw[qy];
            for (int qx = 0; qx < Q; ++qx) {
                const scalar_t w = wyz * qw[qx];
                const ptrdiff_t off = (((ptrdiff_t)qz * Q + qy) * Q + qx) * B_EXEC;
#pragma omp simd
                for (int b = 0; b < B_EXEC; ++b) {
                    const scalar_t ux = gx[off + b], uy = gy[off + b], uz = gz[off + b];
                    const scalar_t m00 = h00[b], m01 = h01[b], m02 = h02[b];
                    const scalar_t m11 = h11[b], m12 = h12[b], m22 = h22[b];
                    gx[off + b] = w * (m00 * ux + m01 * uy + m02 * uz);
                    gy[off + b] = w * (m01 * ux + m11 * uy + m12 * uz);
                    gz[off + b] = w * (m02 * ux + m12 * uy + m22 * uz);
                }
            }
        }
    }
}
