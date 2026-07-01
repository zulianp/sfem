#ifndef SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_HPP
#define SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_HPP

#include <stddef.h>

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif


#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT 
#endif

namespace sfem {
namespace codegen {

static constexpr int ipow(const int base, const int exponent) {
    return exponent == 0 ? 1 : base * ipow(base, exponent - 1);
}

static constexpr int integer_root_search(const int value, const int exponent, const int candidate) {
    return ipow(candidate, exponent) >= value ? candidate : integer_root_search(value, exponent, candidate + 1);
}

static constexpr int integer_root(const int value, const int exponent) {
    return integer_root_search(value, exponent, 1);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM>
struct TensorProductWeakOps;

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
struct TensorProductWeakOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2> {
    static SFEM_INLINE void gradient(
            const ptrdiff_t nelems,
            const scalar_t *const SFEM_RESTRICT shape_1d,
            const scalar_t *const SFEM_RESTRICT grad_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_SHAPE * 2],
            const int component,
            scalar_t *const SFEM_RESTRICT gradient) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t value_x[Q * S * VECTOR_SIZE];
        scalar_t grad_x[Q * S * VECTOR_SIZE];
        for (int qx = 0; qx < Q; ++qx) {
            for (int sy = 0; sy < S; ++sy) {
                #pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t v = scalar_t(0);
                    scalar_t gx = scalar_t(0);
                    for (int sx = 0; sx < S; ++sx) {
                        const int shape = sx + S * sy;
                        const scalar_t u = streams[shape * 2 + component][lane];
                        v += u * shape_1d[qx * S + sx];
                        gx += u * grad_1d[qx * S + sx];
                    }
                    const int i = (qx * S + sy) * VECTOR_SIZE + lane;
                    value_x[i] = v;
                    grad_x[i] = gx;
                }
            }
        }
        for (int qy = 0; qy < Q; ++qy) {
            for (int qx = 0; qx < Q; ++qx) {
                const int q = qx + Q * qy;
                #pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t gx = scalar_t(0);
                    scalar_t gy = scalar_t(0);
                    for (int sy = 0; sy < S; ++sy) {
                        const int i = (qx * S + sy) * VECTOR_SIZE + lane;
                        gx += grad_x[i] * shape_1d[qy * S + sy];
                        gy += value_x[i] * grad_1d[qy * S + sy];
                    }
                    gradient[(q * 2 + 0) * VECTOR_SIZE + lane] = gx;
                    gradient[(q * 2 + 1) * VECTOR_SIZE + lane] = gy;
                }
            }
        }
    }

    static SFEM_INLINE void test(
            const ptrdiff_t nelems,
            const scalar_t *const SFEM_RESTRICT shape_1d,
            const scalar_t *const SFEM_RESTRICT grad_1d,
            const scalar_t *const SFEM_RESTRICT flux,
            scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2],
            const int component) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t stage_x[Q * S * VECTOR_SIZE];
        scalar_t stage_y[Q * S * VECTOR_SIZE];
        for (int qx = 0; qx < Q; ++qx) {
            for (int sy = 0; sy < S; ++sy) {
                #pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t tx = scalar_t(0);
                    scalar_t ty = scalar_t(0);
                    for (int qy = 0; qy < Q; ++qy) {
                        const int q = qx + Q * qy;
                        tx += flux[(q * 2 + 0) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                        ty += flux[(q * 2 + 1) * VECTOR_SIZE + lane] * grad_1d[qy * S + sy];
                    }
                    const int i = (qx * S + sy) * VECTOR_SIZE + lane;
                    stage_x[i] = tx;
                    stage_y[i] = ty;
                }
            }
        }
        for (int sy = 0; sy < S; ++sy) {
            for (int sx = 0; sx < S; ++sx) {
                const int shape = sx + S * sy;
                #pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t value = scalar_t(0);
                    for (int qx = 0; qx < Q; ++qx) {
                        const int i = (qx * S + sy) * VECTOR_SIZE + lane;
                        value += stage_x[i] * grad_1d[qx * S + sx]
                               + stage_y[i] * shape_1d[qx * S + sx];
                    }
                    out_streams[shape * 2 + component][lane] += value;
                }
            }
        }
    }
};

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
struct TensorProductWeakOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3> {
    static SFEM_INLINE void gradient(
            const ptrdiff_t nelems,
            const scalar_t *const SFEM_RESTRICT shape_1d,
            const scalar_t *const SFEM_RESTRICT grad_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_SHAPE * 3],
            const int component,
            scalar_t *const SFEM_RESTRICT gradient) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t value_x[Q * S * S * VECTOR_SIZE];
        scalar_t grad_x[Q * S * S * VECTOR_SIZE];
        scalar_t value_xy[Q * Q * S * VECTOR_SIZE];
        scalar_t grad_x_xy[Q * Q * S * VECTOR_SIZE];
        scalar_t grad_y_xy[Q * Q * S * VECTOR_SIZE];
        for (int qx = 0; qx < Q; ++qx) {
            for (int sy = 0; sy < S; ++sy) {
                for (int sz = 0; sz < S; ++sz) {
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t v = scalar_t(0);
                        scalar_t gx = scalar_t(0);
                        for (int sx = 0; sx < S; ++sx) {
                            const int shape = sx + S * (sy + S * sz);
                            const scalar_t u = streams[shape * 3 + component][lane];
                            v += u * shape_1d[qx * S + sx];
                            gx += u * grad_1d[qx * S + sx];
                        }
                        const int i = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                        value_x[i] = v;
                        grad_x[i] = gx;
                    }
                }
            }
        }
        for (int qx = 0; qx < Q; ++qx) {
            for (int qy = 0; qy < Q; ++qy) {
                for (int sz = 0; sz < S; ++sz) {
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t v = scalar_t(0);
                        scalar_t gx = scalar_t(0);
                        scalar_t gy = scalar_t(0);
                        for (int sy = 0; sy < S; ++sy) {
                            const int i = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                            v += value_x[i] * shape_1d[qy * S + sy];
                            gx += grad_x[i] * shape_1d[qy * S + sy];
                            gy += value_x[i] * grad_1d[qy * S + sy];
                        }
                        const int j = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                        value_xy[j] = v;
                        grad_x_xy[j] = gx;
                        grad_y_xy[j] = gy;
                    }
                }
            }
        }
        for (int qz = 0; qz < Q; ++qz) {
            for (int qy = 0; qy < Q; ++qy) {
                for (int qx = 0; qx < Q; ++qx) {
                    const int q = qx + Q * (qy + Q * qz);
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t gx = scalar_t(0);
                        scalar_t gy = scalar_t(0);
                        scalar_t gz = scalar_t(0);
                        for (int sz = 0; sz < S; ++sz) {
                            const int j = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                            gx += grad_x_xy[j] * shape_1d[qz * S + sz];
                            gy += grad_y_xy[j] * shape_1d[qz * S + sz];
                            gz += value_xy[j] * grad_1d[qz * S + sz];
                        }
                        gradient[(q * 3 + 0) * VECTOR_SIZE + lane] = gx;
                        gradient[(q * 3 + 1) * VECTOR_SIZE + lane] = gy;
                        gradient[(q * 3 + 2) * VECTOR_SIZE + lane] = gz;
                    }
                }
            }
        }
    }

    static SFEM_INLINE void test(
            const ptrdiff_t nelems,
            const scalar_t *const SFEM_RESTRICT shape_1d,
            const scalar_t *const SFEM_RESTRICT grad_1d,
            const scalar_t *const SFEM_RESTRICT flux,
            scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3],
            const int component) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t stage_x[Q * Q * S * VECTOR_SIZE];
        scalar_t stage_y[Q * Q * S * VECTOR_SIZE];
        scalar_t stage_z[Q * Q * S * VECTOR_SIZE];
        scalar_t stage_xy_x[Q * S * S * VECTOR_SIZE];
        scalar_t stage_xy_y[Q * S * S * VECTOR_SIZE];
        scalar_t stage_xy_z[Q * S * S * VECTOR_SIZE];
        for (int qx = 0; qx < Q; ++qx) {
            for (int qy = 0; qy < Q; ++qy) {
                for (int sz = 0; sz < S; ++sz) {
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t tx = scalar_t(0);
                        scalar_t ty = scalar_t(0);
                        scalar_t tz = scalar_t(0);
                        for (int qz = 0; qz < Q; ++qz) {
                            const int q = qx + Q * (qy + Q * qz);
                            tx += flux[(q * 3 + 0) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                            ty += flux[(q * 3 + 1) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                            tz += flux[(q * 3 + 2) * VECTOR_SIZE + lane] * grad_1d[qz * S + sz];
                        }
                        const int i = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                        stage_x[i] = tx;
                        stage_y[i] = ty;
                        stage_z[i] = tz;
                    }
                }
            }
        }
        for (int qx = 0; qx < Q; ++qx) {
            for (int sy = 0; sy < S; ++sy) {
                for (int sz = 0; sz < S; ++sz) {
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t tx = scalar_t(0);
                        scalar_t ty = scalar_t(0);
                        scalar_t tz = scalar_t(0);
                        for (int qy = 0; qy < Q; ++qy) {
                            const int i = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                            tx += stage_x[i] * shape_1d[qy * S + sy];
                            ty += stage_y[i] * grad_1d[qy * S + sy];
                            tz += stage_z[i] * shape_1d[qy * S + sy];
                        }
                        const int j = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                        stage_xy_x[j] = tx;
                        stage_xy_y[j] = ty;
                        stage_xy_z[j] = tz;
                    }
                }
            }
        }
        for (int sz = 0; sz < S; ++sz) {
            for (int sy = 0; sy < S; ++sy) {
                for (int sx = 0; sx < S; ++sx) {
                    const int shape = sx + S * (sy + S * sz);
                    #pragma omp simd
                    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                        scalar_t value = scalar_t(0);
                        for (int qx = 0; qx < Q; ++qx) {
                            const int j = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                            value += stage_xy_x[j] * grad_1d[qx * S + sx]
                                   + (stage_xy_y[j] + stage_xy_z[j]) * shape_1d[qx * S + sx];
                        }
                        out_streams[shape * 3 + component][lane] += value;
                    }
                }
            }
        }
    }
};

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM>
static SFEM_INLINE void tensor_gradient(
        const ptrdiff_t nelems,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT streams[N_SHAPE * DIM],
        const int component,
        scalar_t *const SFEM_RESTRICT gradient) {
    TensorProductWeakOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::gradient(
            nelems, shape_1d, grad_1d, streams, component, gradient);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM>
static SFEM_INLINE void tensor_test(
        const ptrdiff_t nelems,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT flux,
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * DIM],
        const int component) {
    TensorProductWeakOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::test(
            nelems, shape_1d, grad_1d, flux, out_streams, component);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM>
struct TensorProductResidualOps;

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
struct TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2> {
    template <int N_FIELDS>
    static SFEM_INLINE void evaluate(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const grad_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
            scalar_t *const value,
            scalar_t *const gradient) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t vx[N_FIELDS * Q * S * VECTOR_SIZE];
        scalar_t gx[N_FIELDS * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                scalar_t g = scalar_t(0);
                for (int sx = 0; sx < S; ++sx) {
                    const int s = sx + S * sy;
                    const scalar_t u = streams[s * N_FIELDS + f][lane];
                    v += u * shape_1d[qx * S + sx];
                    g += u * grad_1d[qx * S + sx];
                }
                const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;
                vx[i] = v;
                gx[i] = g;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {
            const int q = qx + Q * qy;
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                scalar_t g0 = scalar_t(0);
                scalar_t g1 = scalar_t(0);
                for (int sy = 0; sy < S; ++sy) {
                    const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;
                    v += vx[i] * shape_1d[qy * S + sy];
                    g0 += gx[i] * shape_1d[qy * S + sy];
                    g1 += vx[i] * grad_1d[qy * S + sy];
                }
                value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;
                gradient[((f * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = g0;
                gradient[((f * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = g1;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void evaluate_value(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
            scalar_t *const value) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t vx[N_FIELDS * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int sx = 0; sx < S; ++sx) {
                    const int s = sx + S * sy;
                    v += streams[s * N_FIELDS + f][lane] * shape_1d[qx * S + sx];
                }
                vx[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] = v;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {
            const int q = qx + Q * qy;
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int sy = 0; sy < S; ++sy) {
                    v += vx[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                }
                value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void integrate(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const grad_1d,
            const scalar_t *const value_coeff,
            const scalar_t *const grad_coeff,
            scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t sv[N_FIELDS * Q * S * VECTOR_SIZE];
        scalar_t sg[N_FIELDS * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                scalar_t b = scalar_t(0);
                for (int qy = 0; qy < Q; ++qy) {
                    const int q = qx + Q * qy;
                    a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy]
                       + grad_coeff[((f * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] * grad_1d[qy * S + sy];
                    b += grad_coeff[((f * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                }
                const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;
                sv[i] = a;
                sg[i] = b;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {
            const int s = sx + S * sy;
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int qx = 0; qx < Q; ++qx) {
                    const int i = ((f * Q + qx) * S + sy) * VECTOR_SIZE + lane;
                    v += sv[i] * shape_1d[qx * S + sx] + sg[i] * grad_1d[qx * S + sx];
                }
                output[s * N_FIELDS + f][lane] += v;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void integrate_value(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const value_coeff,
            scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
        static constexpr int Q = integer_root(N_QP, 2);
        static constexpr int S = integer_root(N_SHAPE, 2);
        scalar_t sv[N_FIELDS * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                for (int qy = 0; qy < Q; ++qy) {
                    const int q = qx + Q * qy;
                    a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                }
                sv[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] = a;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {
            const int s = sx + S * sy;
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int qx = 0; qx < Q; ++qx) {
                    v += sv[((f * Q + qx) * S + sy) * VECTOR_SIZE + lane] * shape_1d[qx * S + sx];
                }
                output[s * N_FIELDS + f][lane] += v;
            }
        }
    }
};

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
struct TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3> {
    template <int N_FIELDS>
    static SFEM_INLINE void evaluate(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const grad_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
            scalar_t *const value,
            scalar_t *const gradient) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE];
        scalar_t gx[N_FIELDS * Q * S * S * VECTOR_SIZE];
        scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t g0xy[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t g1xy[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                scalar_t g = scalar_t(0);
                for (int sx = 0; sx < S; ++sx) {
                    const int s = sx + S * (sy + S * sz);
                    const scalar_t u = streams[s * N_FIELDS + f][lane];
                    v += u * shape_1d[qx * S + sx];
                    g += u * grad_1d[qx * S + sx];
                }
                const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                vx[i] = v;
                gx[i] = g;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                scalar_t g0 = scalar_t(0);
                scalar_t g1 = scalar_t(0);
                for (int sy = 0; sy < S; ++sy) {
                    const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                    v += vx[i] * shape_1d[qy * S + sy];
                    g0 += gx[i] * shape_1d[qy * S + sy];
                    g1 += vx[i] * grad_1d[qy * S + sy];
                }
                const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                vxy[j] = v;
                g0xy[j] = g0;
                g1xy[j] = g1;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qz = 0; qz < Q; ++qz) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {
            const int q = qx + Q * (qy + Q * qz);
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                scalar_t g0 = scalar_t(0);
                scalar_t g1 = scalar_t(0);
                scalar_t g2 = scalar_t(0);
                for (int sz = 0; sz < S; ++sz) {
                    const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                    v += vxy[j] * shape_1d[qz * S + sz];
                    g0 += g0xy[j] * shape_1d[qz * S + sz];
                    g1 += g1xy[j] * shape_1d[qz * S + sz];
                    g2 += vxy[j] * grad_1d[qz * S + sz];
                }
                value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;
                gradient[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = g0;
                gradient[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = g1;
                gradient[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = g2;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void evaluate_value(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
            scalar_t *const value) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE];
        scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int sx = 0; sx < S; ++sx) {
                    const int s = sx + S * (sy + S * sz);
                    v += streams[s * N_FIELDS + f][lane] * shape_1d[qx * S + sx];
                }
                vx[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] = v;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int sy = 0; sy < S; ++sy) {
                    v += vx[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                }
                vxy[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] = v;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qz = 0; qz < Q; ++qz) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {
            const int q = qx + Q * (qy + Q * qz);
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int sz = 0; sz < S; ++sz) {
                    v += vxy[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                }
                value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void integrate(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const grad_1d,
            const scalar_t *const value_coeff,
            const scalar_t *const grad_coeff,
            scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t z1[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t z2[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE];
        scalar_t yz1[N_FIELDS * Q * S * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                scalar_t b = scalar_t(0);
                scalar_t c = scalar_t(0);
                for (int qz = 0; qz < Q; ++qz) {
                    const int q = qx + Q * (qy + Q * qz);
                    a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz]
                       + grad_coeff[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] * grad_1d[qz * S + sz];
                    b += grad_coeff[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                    c += grad_coeff[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                }
                const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                z0[i] = a;
                z1[i] = b;
                z2[i] = c;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                scalar_t b = scalar_t(0);
                for (int qy = 0; qy < Q; ++qy) {
                    const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                    a += z0[i] * shape_1d[qy * S + sy] + z2[i] * grad_1d[qy * S + sy];
                    b += z1[i] * shape_1d[qy * S + sy];
                }
                const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                yz0[j] = a;
                yz1[j] = b;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {
            const int s = sx + S * (sy + S * sz);
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int qx = 0; qx < Q; ++qx) {
                    const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                    v += yz0[j] * shape_1d[qx * S + sx] + yz1[j] * grad_1d[qx * S + sx];
                }
                output[s * N_FIELDS + f][lane] += v;
            }
        }
    }

    template <int N_FIELDS>
    static SFEM_INLINE void integrate_value(
            const ptrdiff_t nelems,
            const scalar_t *const shape_1d,
            const scalar_t *const value_coeff,
            scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
        static constexpr int Q = integer_root(N_QP, 3);
        static constexpr int S = integer_root(N_SHAPE, 3);
        scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE];
        scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE];
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                for (int qz = 0; qz < Q; ++qz) {
                    const int q = qx + Q * (qy + Q * qz);
                    a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                }
                z0[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] = a;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t a = scalar_t(0);
                for (int qy = 0; qy < Q; ++qy) {
                    a += z0[(((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qy * S + sy];
                }
                yz0[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] = a;
            }
        }
        for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {
            const int s = sx + S * (sy + S * sz);
            #pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                scalar_t v = scalar_t(0);
                for (int qx = 0; qx < Q; ++qx) {
                    v += yz0[(((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane] * shape_1d[qx * S + sx];
                }
                output[s * N_FIELDS + f][lane] += v;
            }
        }
    }
};

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM, int N_FIELDS>
static SFEM_INLINE void tensor_evaluate(
        const ptrdiff_t nelems,
        const scalar_t *const shape_1d,
        const scalar_t *const grad_1d,
        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
        scalar_t *const value,
        scalar_t *const gradient) {
    TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::template evaluate<N_FIELDS>(
            nelems, shape_1d, grad_1d, streams, value, gradient);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM, int N_FIELDS>
static SFEM_INLINE void tensor_evaluate_value(
        const ptrdiff_t nelems,
        const scalar_t *const shape_1d,
        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE],
        scalar_t *const value) {
    TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::template evaluate_value<N_FIELDS>(
            nelems, shape_1d, streams, value);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM, int N_FIELDS>
static SFEM_INLINE void tensor_integrate(
        const ptrdiff_t nelems,
        const scalar_t *const shape_1d,
        const scalar_t *const grad_1d,
        const scalar_t *const value_coeff,
        const scalar_t *const grad_coeff,
        scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
    TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::template integrate<N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int DIM, int N_FIELDS>
static SFEM_INLINE void tensor_integrate_value(
        const ptrdiff_t nelems,
        const scalar_t *const shape_1d,
        const scalar_t *const value_coeff,
        scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
    TensorProductResidualOps<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM>::template integrate_value<N_FIELDS>(
            nelems, shape_1d, value_coeff, output);
}

} // namespace codegen
} // namespace sfem

#endif
