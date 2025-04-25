#ifndef CU_SPECTRAL_HEX_LAPLACIAN_INLINE_H
#define CU_SPECTRAL_HEX_LAPLACIAN_INLINE_H

#include "sfem_base.h"

template <int N, typename T>
inline __host__ __device__ void cu_spectral_hex_laplacian_apply(
        // Shape functions per quad point Q x S
        const T* const SFEM_RESTRICT D,
        // Metric 6
        const T* const SFEM_RESTRICT FFF,
        // Quad-weights  Q
        const T* const SFEM_RESTRICT qw,
        // Coefficients S x S x S
        const T* const SFEM_RESTRICT u,
        // Evaluation Q x Q x Q
        T* const SFEM_RESTRICT out) {
    static const int N2 = N * N;
    static const int N3 = N2 * N;

    T gx[N3];
    T gy[N3];
    T gz[N3];

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                const int idx = k * N2 + j * N + i;

                const T* const D0 = &D[N * i];
                const T* const D1 = &D[N * j];
                const T* const D2 = &D[N * k];

                const T* const u0 = &u[k * N2 + j * N];
                const T* const u1 = &u[k * N2 + i];
                const T* const u2 = &u[j * N + i];

                T acc[3] = {D0[0] * u0[0], D1[0] * u1[0], D2[0] * u2[0]};
                for (int n = 1; n < N; n++) {
                    acc[0] += D0[n] * u0[n];
                    acc[1] += D1[n] * u1[n * N];
                    acc[2] += D2[n] * u2[n * N2];
                }

                const T gxq = FFF[0] * acc[0] + FFF[1] * acc[1] + FFF[2] * acc[2];
                const T gyq = FFF[1] * acc[0] + FFF[3] * acc[1] + FFF[4] * acc[2];
                const T gzq = FFF[2] * acc[0] + FFF[4] * acc[1] + FFF[5] * acc[2];
                const T w   = qw[i] * qw[j] * qw[k];

                gx[idx] = gxq * w;
                gy[idx] = gyq * w;
                gz[idx] = gzq * w;
            }
        }
    }

    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            T* const out_kj = &out[k * N2 + j * N];

            for (int i = 0; i < N; i++) {
                const T* const D0 = &D[i];
                const T* const D1 = &D[j];
                const T* const D2 = &D[k];

                const T* const g0 = &gx[k * N2 + j * N];
                const T* const g1 = &gy[k * N2 + i];
                const T* const g2 = &gz[j * N + i];

                T acc[3] = {D0[0] * g0[0], D1[0] * g1[0], D2[0] * g2[0]};
                for (int n = 1; n < N; n++) {
                    const int nidx = n * N;
                    acc[0] += D0[nidx] * g0[n];
                    acc[1] += D1[nidx] * g1[nidx];
                    acc[2] += D2[nidx] * g2[n * N2];
                }

                out_kj[i] += acc[0] + acc[1] + acc[2];
            }
        }
    }
}

template <int N, typename T>
inline __device__ T cu_spectral_hex_laplacian_apply_tiled(const int i,
                                                          const int j,
                                                          const int k,
                                                          // --------------------------
                                                          // Registers
                                                          // --------------------------
                                                          // Shape functions per quad point Q x S
                                                          const T* const SFEM_RESTRICT D,
                                                          // Metric 6
                                                          const T* const SFEM_RESTRICT FFF,
                                                          // Quad-weights  Q
                                                          const T* const SFEM_RESTRICT qw,
                                                          // Coefficients S x S x S
                                                          const T* const SFEM_RESTRICT u,
                                                          // --------------------------
                                                          // Shared mem
                                                          // --------------------------
                                                          T* const SFEM_RESTRICT gx,
                                                          T* const SFEM_RESTRICT gy,
                                                          T* const SFEM_RESTRICT gz) {
    static const int N2 = N * N;

    const int idx = k * N2 + j * N + i;
    {
        const T* const D0 = &D[N * i];
        const T* const D1 = &D[N * j];
        const T* const D2 = &D[N * k];

        // Reading from shared mem
        const T* const u0 = &u[k * N2 + j * N];
        const T* const u1 = &u[k * N2 + i];
        const T* const u2 = &u[j * N + i];

        T acc[3] = {D0[0] * u0[0], D1[0] * u1[0], D2[0] * u2[0]};

#pragma unroll
        for (int n = 1; n < N; n++) {
            acc[0] += D0[n] * u0[n];
            acc[1] += D1[n] * u1[n * N];
            acc[2] += D2[n] * u2[n * N2];
        }

        const T w   = qw[i] * qw[j] * qw[k];
        const T gxq = (FFF[0] * acc[0] + FFF[1] * acc[1] + FFF[2] * acc[2]) * w;
        const T gyq = (FFF[1] * acc[0] + FFF[3] * acc[1] + FFF[4] * acc[2]) * w;
        const T gzq = (FFF[2] * acc[0] + FFF[4] * acc[1] + FFF[5] * acc[2]) * w;

        // Writing to shared mem
        gx[idx] = gxq;
        gy[idx] = gyq;
        gz[idx] = gzq;
    }

    {
        const T* const D0 = &D[i];
        const T* const D1 = &D[j];
        const T* const D2 = &D[k];

        // Reading from shared mem
        const T* const g0 = &gx[k * N2 + j * N];
        const T* const g1 = &gy[k * N2 + i];
        const T* const g2 = &gz[j * N + i];

        T acc[3] = {D0[0] * g0[0], D1[0] * g1[0], D2[0] * g2[0]};

#pragma unroll
        for (int n = 1; n < N; n++) {
            const int nidx = n * N;
            acc[0] += D0[nidx] * g0[n];
            acc[1] += D1[nidx] * g1[nidx];
            acc[2] += D2[nidx] * g2[n * N2];
        }

        // Return (i, j, k) value
        return acc[0] + acc[1] + acc[2];
    }
}


#endif  // CU_SPECTRAL_HEX_LAPLACIAN_INLINE_H