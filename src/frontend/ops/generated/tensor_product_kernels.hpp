#ifndef SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_HPP
#define SFEM_CODEGEN_TENSOR_PRODUCT_KERNELS_HPP

#include <stddef.h>

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif


#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
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

template <typename s_t, int NQ, int NS, int VS, int ND>
struct TensorProductWeakOps;

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductWeakOps<s_t, NQ, NS, VS, 2> {
  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t value_x[NQ1 * NS1 * VS];
    s_t grad_x[NQ1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t v = s_t(0);
          s_t gx = s_t(0);
          for (int sx = 0; sx < NS1; ++sx) {
            const int shape = sx + NS1 * sy;
            const s_t u = streams[shape * NC + component][lane];
            v += u * shape_1d[qx * NS1 + sx];
            gx += u * grad_1d[qx * NS1 + sx];
          }
          const int i = (qx * NS1 + sy) * VS + lane;
          value_x[i] = v;
          grad_x[i] = gx;
        }
      }
    }
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int qx = 0; qx < NQ1; ++qx) {
        const int q = qx + NQ1 * qy;
        s_t *const RSTR gradient_q0 = &gradient[(q * 2 + 0) * VS];
        s_t *const RSTR gradient_q1 = &gradient[(q * 2 + 1) * VS];
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy) * VS + lane;
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient_q0[lane] = gx;
          gradient_q1[lane] = gy;
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NC * NS][VS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t value_x[NQ1 * NS1 * VS];
    s_t grad_x[NQ1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t v = s_t(0);
          s_t gx = s_t(0);
          for (int sx = 0; sx < NS1; ++sx) {
            const int shape = sx + NS1 * sy;
            const s_t u = streams[shape * NC + component][lane];
            v += u * shape_1d[qx * NS1 + sx];
            gx += u * grad_1d[qx * NS1 + sx];
          }
          const int i = (qx * NS1 + sy) * VS + lane;
          value_x[i] = v;
          grad_x[i] = gx;
        }
      }
    }
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int qx = 0; qx < NQ1; ++qx) {
        const int q = qx + NQ1 * qy;
        s_t *const RSTR gradient_q0 = &gradient[(q * 2 + 0) * VS];
        s_t *const RSTR gradient_q1 = &gradient[(q * 2 + 1) * VS];
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy) * VS + lane;
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient_q0[lane] = gx;
          gradient_q1[lane] = gy;
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(ne, shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void gradient_contiguous(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC][VS],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(ne, shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void test(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t stage_x[NQ1 * NS1 * VS];
    s_t stage_y[NQ1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t tx = s_t(0);
          s_t ty = s_t(0);
          for (int qy = 0; qy < NQ1; ++qy) {
            const int q = qx + NQ1 * qy;
            tx += flux[(q * 2 + 0) * VS + lane] * shape_1d[qy * NS1 + sy];
            ty += flux[(q * 2 + 1) * VS + lane] * grad_1d[qy * NS1 + sy];
          }
          const int i = (qx * NS1 + sy) * VS + lane;
          stage_x[i] = tx;
          stage_y[i] = ty;
        }
      }
    }
    for (int sy = 0; sy < NS1; ++sy) {
      for (int sx = 0; sx < NS1; ++sx) {
        const int shape = sx + NS1 * sy;
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t value = s_t(0);
          for (int qx = 0; qx < NQ1; ++qx) {
            const int i = (qx * NS1 + sy) * VS + lane;
            value += stage_x[i] * grad_1d[qx * NS1 + sx]
                               + stage_y[i] * shape_1d[qx * NS1 + sx];
          }
          out_streams[shape * NC + component][lane] += value;
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductWeakOps<s_t, NQ, NS, VS, 3> {
  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t value_x[NQ1 * NS1 * NS1 * VS];
    s_t grad_x[NQ1 * NS1 * NS1 * VS];
    s_t value_xy[NQ1 * NQ1 * NS1 * VS];
    s_t grad_x_xy[NQ1 * NQ1 * NS1 * VS];
    s_t grad_y_xy[NQ1 * NQ1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            for (int sx = 0; sx < NS1; ++sx) {
              const int shape = sx + NS1 * (sy + NS1 * sz);
              const s_t u = streams[shape * NC + component][lane];
              v += u * shape_1d[qx * NS1 + sx];
              gx += u * grad_1d[qx * NS1 + sx];
            }
            const int i = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
            value_x[i] = v;
            grad_x[i] = gx;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            for (int sy = 0; sy < NS1; ++sy) {
              const int i = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
              v += value_x[i] * shape_1d[qy * NS1 + sy];
              gx += grad_x[i] * shape_1d[qy * NS1 + sy];
              gy += value_x[i] * grad_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
            value_xy[j] = v;
            grad_x_xy[j] = gx;
            grad_y_xy[j] = gy;
          }
        }
      }
    }
    for (int qz = 0; qz < NQ1; ++qz) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          s_t *const RSTR gradient_q0 = &gradient[(q * 3 + 0) * VS];
          s_t *const RSTR gradient_q1 = &gradient[(q * 3 + 1) * VS];
          s_t *const RSTR gradient_q2 = &gradient[(q * 3 + 2) * VS];
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            s_t gz = s_t(0);
            for (int sz = 0; sz < NS1; ++sz) {
              const int j = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
              gx += grad_x_xy[j] * shape_1d[qz * NS1 + sz];
              gy += grad_y_xy[j] * shape_1d[qz * NS1 + sz];
              gz += value_xy[j] * grad_1d[qz * NS1 + sz];
            }
            gradient_q0[lane] = gx;
            gradient_q1[lane] = gy;
            gradient_q2[lane] = gz;
          }
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NC * NS][VS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t value_x[NQ1 * NS1 * NS1 * VS];
    s_t grad_x[NQ1 * NS1 * NS1 * VS];
    s_t value_xy[NQ1 * NQ1 * NS1 * VS];
    s_t grad_x_xy[NQ1 * NQ1 * NS1 * VS];
    s_t grad_y_xy[NQ1 * NQ1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            for (int sx = 0; sx < NS1; ++sx) {
              const int shape = sx + NS1 * (sy + NS1 * sz);
              const s_t u = streams[shape * NC + component][lane];
              v += u * shape_1d[qx * NS1 + sx];
              gx += u * grad_1d[qx * NS1 + sx];
            }
            const int i = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
            value_x[i] = v;
            grad_x[i] = gx;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            for (int sy = 0; sy < NS1; ++sy) {
              const int i = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
              v += value_x[i] * shape_1d[qy * NS1 + sy];
              gx += grad_x[i] * shape_1d[qy * NS1 + sy];
              gy += value_x[i] * grad_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
            value_xy[j] = v;
            grad_x_xy[j] = gx;
            grad_y_xy[j] = gy;
          }
        }
      }
    }
    for (int qz = 0; qz < NQ1; ++qz) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          s_t *const RSTR gradient_q0 = &gradient[(q * 3 + 0) * VS];
          s_t *const RSTR gradient_q1 = &gradient[(q * 3 + 1) * VS];
          s_t *const RSTR gradient_q2 = &gradient[(q * 3 + 2) * VS];
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            s_t gz = s_t(0);
            for (int sz = 0; sz < NS1; ++sz) {
              const int j = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
              gx += grad_x_xy[j] * shape_1d[qz * NS1 + sz];
              gy += grad_y_xy[j] * shape_1d[qz * NS1 + sz];
              gz += value_xy[j] * grad_1d[qz * NS1 + sz];
            }
            gradient_q0[lane] = gx;
            gradient_q1[lane] = gy;
            gradient_q2[lane] = gz;
          }
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(ne, shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void gradient_contiguous(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC][VS],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(ne, shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void test(
      const int ne,
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t stage_x[NQ1 * NQ1 * NS1 * VS];
    s_t stage_y[NQ1 * NQ1 * NS1 * VS];
    s_t stage_z[NQ1 * NQ1 * NS1 * VS];
    s_t stage_xy_x[NQ1 * NS1 * NS1 * VS];
    s_t stage_xy_y[NQ1 * NS1 * NS1 * VS];
    s_t stage_xy_z[NQ1 * NS1 * NS1 * VS];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qz = 0; qz < NQ1; ++qz) {
              const int q = qx + NQ1 * (qy + NQ1 * qz);
              tx += flux[(q * 3 + 0) * VS + lane] * shape_1d[qz * NS1 + sz];
              ty += flux[(q * 3 + 1) * VS + lane] * shape_1d[qz * NS1 + sz];
              tz += flux[(q * 3 + 2) * VS + lane] * grad_1d[qz * NS1 + sz];
            }
            const int i = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
            stage_x[i] = tx;
            stage_y[i] = ty;
            stage_z[i] = tz;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qy = 0; qy < NQ1; ++qy) {
              const int i = ((qx * NQ1 + qy) * NS1 + sz) * VS + lane;
              tx += stage_x[i] * shape_1d[qy * NS1 + sy];
              ty += stage_y[i] * grad_1d[qy * NS1 + sy];
              tz += stage_z[i] * shape_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
            stage_xy_x[j] = tx;
            stage_xy_y[j] = ty;
            stage_xy_z[j] = tz;
          }
        }
      }
    }
    for (int sz = 0; sz < NS1; ++sz) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sx = 0; sx < NS1; ++sx) {
          const int shape = sx + NS1 * (sy + NS1 * sz);
                    #pragma omp simd
                    for (int lane = 0; lane < ne; ++lane) {
            s_t value = s_t(0);
            for (int qx = 0; qx < NQ1; ++qx) {
              const int j = ((qx * NS1 + sy) * NS1 + sz) * VS + lane;
              value += stage_xy_x[j] * grad_1d[qx * NS1 + sx]
                                   + (stage_xy_y[j] + stage_xy_z[j]) * shape_1d[qx * NS1 + sx];
            }
            out_streams[shape * NC + component][lane] += value;
          }
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static SFEM_INLINE void tensor_gradient(
    const int ne,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR streams[NS * NC],
    const int component,
    s_t *const RSTR gradient) {
  TensorProductWeakOps<s_t, NQ, NS, VS, ND>::template gradient<NC>(
      ne, shape_1d, grad_1d, streams, component, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static SFEM_INLINE void tensor_gradient_contiguous(
    const int ne,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t streams[NS * NC][VS],
    const int component,
    s_t *const RSTR gradient) {
  TensorProductWeakOps<s_t, NQ, NS, VS, ND>::template gradient_contiguous<NC>(
      ne, shape_1d, grad_1d, streams, component, gradient);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
template <typename s_t, int NQ, int NS, int VS, int ND, int NC = ND>
static SFEM_INLINE void tensor_test(
    const int ne,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR flux,
    s_t *const RSTR out_streams[NS * NC],
    const int component) {
  TensorProductWeakOps<s_t, NQ, NS, VS, ND>::template test<NC>(
      ne, shape_1d, grad_1d, flux, out_streams, component);
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <typename s_t, int NQ, int NS, int ND>
struct TensorProductWeakOpsScalar;

template <typename s_t, int NQ, int NS>
struct TensorProductWeakOpsScalar<s_t, NQ, NS, 2> {
  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t value_x[NQ1 * NS1];
    s_t grad_x[NQ1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                {
          s_t v = s_t(0);
          s_t gx = s_t(0);
          for (int sx = 0; sx < NS1; ++sx) {
            const int shape = sx + NS1 * sy;
            const s_t u = streams[shape * NC + component][0];
            v += u * shape_1d[qx * NS1 + sx];
            gx += u * grad_1d[qx * NS1 + sx];
          }
          const int i = (qx * NS1 + sy);
          value_x[i] = v;
          grad_x[i] = gx;
        }
      }
    }
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int qx = 0; qx < NQ1; ++qx) {
        const int q = qx + NQ1 * qy;
        s_t *const RSTR gradient_q0 = &gradient[(q * 2 + 0)];
        s_t *const RSTR gradient_q1 = &gradient[(q * 2 + 1)];
                {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy);
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient_q0[0] = gx;
          gradient_q1[0] = gy;
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t value_x[NQ1 * NS1];
    s_t grad_x[NQ1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                {
          s_t v = s_t(0);
          s_t gx = s_t(0);
          for (int sx = 0; sx < NS1; ++sx) {
            const int shape = sx + NS1 * sy;
            const s_t u = streams[shape * NC + component];
            v += u * shape_1d[qx * NS1 + sx];
            gx += u * grad_1d[qx * NS1 + sx];
          }
          const int i = (qx * NS1 + sy);
          value_x[i] = v;
          grad_x[i] = gx;
        }
      }
    }
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int qx = 0; qx < NQ1; ++qx) {
        const int q = qx + NQ1 * qy;
        s_t *const RSTR gradient_q0 = &gradient[(q * 2 + 0)];
        s_t *const RSTR gradient_q1 = &gradient[(q * 2 + 1)];
                {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy);
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient_q0[0] = gx;
          gradient_q1[0] = gy;
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void gradient_contiguous(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void test(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t stage_x[NQ1 * NS1];
    s_t stage_y[NQ1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
                {
          s_t tx = s_t(0);
          s_t ty = s_t(0);
          for (int qy = 0; qy < NQ1; ++qy) {
            const int q = qx + NQ1 * qy;
            tx += flux[(q * 2 + 0)] * shape_1d[qy * NS1 + sy];
            ty += flux[(q * 2 + 1)] * grad_1d[qy * NS1 + sy];
          }
          const int i = (qx * NS1 + sy);
          stage_x[i] = tx;
          stage_y[i] = ty;
        }
      }
    }
    for (int sy = 0; sy < NS1; ++sy) {
      for (int sx = 0; sx < NS1; ++sx) {
        const int shape = sx + NS1 * sy;
                {
          s_t value = s_t(0);
          for (int qx = 0; qx < NQ1; ++qx) {
            const int i = (qx * NS1 + sy);
            value += stage_x[i] * grad_1d[qx * NS1 + sx]
                               + stage_y[i] * shape_1d[qx * NS1 + sx];
          }
          out_streams[shape * NC + component][0] += value;
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS>
struct TensorProductWeakOpsScalar<s_t, NQ, NS, 3> {
  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t value_x[NQ1 * NS1 * NS1];
    s_t grad_x[NQ1 * NS1 * NS1];
    s_t value_xy[NQ1 * NQ1 * NS1];
    s_t grad_x_xy[NQ1 * NQ1 * NS1];
    s_t grad_y_xy[NQ1 * NQ1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            for (int sx = 0; sx < NS1; ++sx) {
              const int shape = sx + NS1 * (sy + NS1 * sz);
              const s_t u = streams[shape * NC + component][0];
              v += u * shape_1d[qx * NS1 + sx];
              gx += u * grad_1d[qx * NS1 + sx];
            }
            const int i = ((qx * NS1 + sy) * NS1 + sz);
            value_x[i] = v;
            grad_x[i] = gx;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            for (int sy = 0; sy < NS1; ++sy) {
              const int i = ((qx * NS1 + sy) * NS1 + sz);
              v += value_x[i] * shape_1d[qy * NS1 + sy];
              gx += grad_x[i] * shape_1d[qy * NS1 + sy];
              gy += value_x[i] * grad_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NQ1 + qy) * NS1 + sz);
            value_xy[j] = v;
            grad_x_xy[j] = gx;
            grad_y_xy[j] = gy;
          }
        }
      }
    }
    for (int qz = 0; qz < NQ1; ++qz) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          s_t *const RSTR gradient_q0 = &gradient[(q * 3 + 0)];
          s_t *const RSTR gradient_q1 = &gradient[(q * 3 + 1)];
          s_t *const RSTR gradient_q2 = &gradient[(q * 3 + 2)];
                    {
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            s_t gz = s_t(0);
            for (int sz = 0; sz < NS1; ++sz) {
              const int j = ((qx * NQ1 + qy) * NS1 + sz);
              gx += grad_x_xy[j] * shape_1d[qz * NS1 + sz];
              gy += grad_y_xy[j] * shape_1d[qz * NS1 + sz];
              gz += value_xy[j] * grad_1d[qz * NS1 + sz];
            }
            gradient_q0[0] = gx;
            gradient_q1[0] = gy;
            gradient_q2[0] = gz;
          }
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient_impl(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NC * NS],
      const int component,
      s_t *const RSTR gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t value_x[NQ1 * NS1 * NS1];
    s_t grad_x[NQ1 * NS1 * NS1];
    s_t value_xy[NQ1 * NQ1 * NS1];
    s_t grad_x_xy[NQ1 * NQ1 * NS1];
    s_t grad_y_xy[NQ1 * NQ1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            for (int sx = 0; sx < NS1; ++sx) {
              const int shape = sx + NS1 * (sy + NS1 * sz);
              const s_t u = streams[shape * NC + component];
              v += u * shape_1d[qx * NS1 + sx];
              gx += u * grad_1d[qx * NS1 + sx];
            }
            const int i = ((qx * NS1 + sy) * NS1 + sz);
            value_x[i] = v;
            grad_x[i] = gx;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t v = s_t(0);
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            for (int sy = 0; sy < NS1; ++sy) {
              const int i = ((qx * NS1 + sy) * NS1 + sz);
              v += value_x[i] * shape_1d[qy * NS1 + sy];
              gx += grad_x[i] * shape_1d[qy * NS1 + sy];
              gy += value_x[i] * grad_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NQ1 + qy) * NS1 + sz);
            value_xy[j] = v;
            grad_x_xy[j] = gx;
            grad_y_xy[j] = gy;
          }
        }
      }
    }
    for (int qz = 0; qz < NQ1; ++qz) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int qx = 0; qx < NQ1; ++qx) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          s_t *const RSTR gradient_q0 = &gradient[(q * 3 + 0)];
          s_t *const RSTR gradient_q1 = &gradient[(q * 3 + 1)];
          s_t *const RSTR gradient_q2 = &gradient[(q * 3 + 2)];
                    {
            s_t gx = s_t(0);
            s_t gy = s_t(0);
            s_t gz = s_t(0);
            for (int sz = 0; sz < NS1; ++sz) {
              const int j = ((qx * NQ1 + qy) * NS1 + sz);
              gx += grad_x_xy[j] * shape_1d[qz * NS1 + sz];
              gy += grad_y_xy[j] * shape_1d[qz * NS1 + sz];
              gz += value_xy[j] * grad_1d[qz * NS1 + sz];
            }
            gradient_q0[0] = gx;
            gradient_q1[0] = gy;
            gradient_q2[0] = gz;
          }
        }
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void gradient(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void gradient_contiguous(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t streams[NS * NC],
      const int component,
      s_t *const RSTR gradient) {
    gradient_impl<NC>(shape_1d, grad_1d, streams, component, gradient);
  }

  template <int NC>
  static SFEM_INLINE void test(
      const s_t *const RSTR shape_1d,
      const s_t *const RSTR grad_1d,
      const s_t *const RSTR flux,
      s_t *const RSTR out_streams[NS * NC],
      const int component) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t stage_x[NQ1 * NQ1 * NS1];
    s_t stage_y[NQ1 * NQ1 * NS1];
    s_t stage_z[NQ1 * NQ1 * NS1];
    s_t stage_xy_x[NQ1 * NS1 * NS1];
    s_t stage_xy_y[NQ1 * NS1 * NS1];
    s_t stage_xy_z[NQ1 * NS1 * NS1];
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int qy = 0; qy < NQ1; ++qy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qz = 0; qz < NQ1; ++qz) {
              const int q = qx + NQ1 * (qy + NQ1 * qz);
              tx += flux[(q * 3 + 0)] * shape_1d[qz * NS1 + sz];
              ty += flux[(q * 3 + 1)] * shape_1d[qz * NS1 + sz];
              tz += flux[(q * 3 + 2)] * grad_1d[qz * NS1 + sz];
            }
            const int i = ((qx * NQ1 + qy) * NS1 + sz);
            stage_x[i] = tx;
            stage_y[i] = ty;
            stage_z[i] = tz;
          }
        }
      }
    }
    for (int qx = 0; qx < NQ1; ++qx) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sz = 0; sz < NS1; ++sz) {
                    {
            s_t tx = s_t(0);
            s_t ty = s_t(0);
            s_t tz = s_t(0);
            for (int qy = 0; qy < NQ1; ++qy) {
              const int i = ((qx * NQ1 + qy) * NS1 + sz);
              tx += stage_x[i] * shape_1d[qy * NS1 + sy];
              ty += stage_y[i] * grad_1d[qy * NS1 + sy];
              tz += stage_z[i] * shape_1d[qy * NS1 + sy];
            }
            const int j = ((qx * NS1 + sy) * NS1 + sz);
            stage_xy_x[j] = tx;
            stage_xy_y[j] = ty;
            stage_xy_z[j] = tz;
          }
        }
      }
    }
    for (int sz = 0; sz < NS1; ++sz) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sx = 0; sx < NS1; ++sx) {
          const int shape = sx + NS1 * (sy + NS1 * sz);
                    {
            s_t value = s_t(0);
            for (int qx = 0; qx < NQ1; ++qx) {
              const int j = ((qx * NS1 + sy) * NS1 + sz);
              value += stage_xy_x[j] * grad_1d[qx * NS1 + sx]
                                   + (stage_xy_y[j] + stage_xy_z[j]) * shape_1d[qx * NS1 + sx];
            }
            out_streams[shape * NC + component][0] += value;
          }
        }
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int ND, int NC = ND>
static SFEM_INLINE void tensor_gradient_scalar(
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR streams[NS * NC],
    const int component,
    s_t *const RSTR gradient) {
  TensorProductWeakOpsScalar<s_t, NQ, NS, ND>::template gradient<NC>(
      shape_1d, grad_1d, streams, component, gradient);
}

template <typename s_t, int NQ, int NS, int ND, int NC = ND>
static SFEM_INLINE void tensor_gradient_contiguous_scalar(
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t streams[NS * NC],
    const int component,
    s_t *const RSTR gradient) {
  TensorProductWeakOpsScalar<s_t, NQ, NS, ND>::template gradient_contiguous<NC>(
      shape_1d, grad_1d, streams, component, gradient);
}

template <typename s_t, int NQ, int NS, int ND, int NC = ND>
static SFEM_INLINE void tensor_test_scalar(
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR flux,
    s_t *const RSTR out_streams[NS * NC],
    const int component) {
  TensorProductWeakOpsScalar<s_t, NQ, NS, ND>::template test<NC>(
      shape_1d, grad_1d, flux, out_streams, component);
}

template <typename s_t, int NQ, int NS, int VS, int ND>
struct TensorProductResidualOps;

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductResidualOps<s_t, NQ, NS, VS, 2> {
  template <int NC>
  static SFEM_INLINE void evaluate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          const s_t u = streams[s * NC + f][lane];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 2 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 2 + 1) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value_q[lane] = v;
        gradient_q0[lane] = g0;
        gradient_q1[lane] = g1;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t streams[NC * NS][VS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          const s_t u = streams[s * NC + f][lane];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 2 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 2 + 1) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value_q[lane] = v;
        gradient_q0[lane] = g0;
        gradient_q1[lane] = g1;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR vx_q = &vx[((f * NQ1 + qx) * NS1 + sy) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        value_q[lane] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t streams[NC * NS][VS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR vx_q = &vx[((f * NQ1 + qx) * NS1 + sy) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        value_q[lane] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    s_t sg[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1)  * VS + lane] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
        sv[i] = a;
        sg[i] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
          v += sv[i] * shape_1d[qx * NS1 + sx] + sg[i] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t output[NC * NS][VS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    s_t sg[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1)  * VS + lane] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
        sv[i] = a;
        sg[i] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int i = ((f * NQ1 + qx) * NS1 + sy)  * VS + lane;
          v += sv[i] * shape_1d[qx * NS1 + sx] + sg[i] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR sv_q = &sv[((f * NQ1 + qx) * NS1 + sy) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        sv_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy)  * VS + lane] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t output[NC * NS][VS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR sv_q = &sv[((f * NQ1 + qx) * NS1 + sy) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        sv_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy)  * VS + lane] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS>
struct TensorProductResidualOps<s_t, NQ, NS, VS, 3> {
  template <int NC>
  static SFEM_INLINE void evaluate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g0xy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g1xy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t u = streams[s * NC + f][lane];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 3 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 3 + 1) * VS];
      s_t *const RSTR gradient_q2 = &gradient[((f * NQ + q) * 3 + 2) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value_q[lane] = v;
        gradient_q0[lane] = g0;
        gradient_q1[lane] = g1;
        gradient_q2[lane] = g2;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t streams[NC * NS][VS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t gx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g0xy[NC * NQ1 * NQ1 * NS1 * VS];
    s_t g1xy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t u = streams[s * NC + f][lane];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 3 + 0) * VS];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 3 + 1) * VS];
      s_t *const RSTR gradient_q2 = &gradient[((f * NQ + q) * 3 + 2) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value_q[lane] = v;
        gradient_q0[lane] = g0;
        gradient_q1[lane] = g1;
        gradient_q2[lane] = g2;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vx_q = &vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vxy_q = &vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        vxy_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        value_q[lane] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t streams[NC * NS][VS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1 * VS];
    s_t vxy[NC * NQ1 * NQ1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vx_q = &vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vxy_q = &vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        vxy_q[lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        value_q[lane] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z1[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z2[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    s_t yz1[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        s_t c = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2)  * VS + lane] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0)  * VS + lane] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
        z0[i] = a;
        z1[i] = b;
        z2[i] = c;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
        yz0[j] = a;
        yz1[j] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
          v += yz0[j] * shape_1d[qx * NS1 + sx] + yz1[j] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t output[NC * NS][VS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z1[NC * NQ1 * NQ1 * NS1 * VS];
    s_t z2[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    s_t yz1[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        s_t c = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2)  * VS + lane] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0)  * VS + lane] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
        z0[i] = a;
        z1[i] = b;
        z2[i] = c;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
        yz0[j] = a;
        yz1[j] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane;
          v += yz0[j] * shape_1d[qx * NS1 + sx] + yz1[j] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR z0_q = &z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        z0_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR yz0_q = &yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        yz0_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value_contiguous(
      const int ne,
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t output[NC * NS][VS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1 * VS];
    s_t yz0[NC * NQ1 * NS1 * NS1 * VS];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR z0_q = &z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q)  * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        z0_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR yz0_q = &yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)  * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        yz0_q[lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)  * VS + lane] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][lane] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate<NC>(
      ne, shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t streams[NC * NS][VS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_contiguous<NC>(
      ne, shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_value(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_value<NC>(
      ne, shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_value_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t streams[NC * NS][VS],
    s_t *const value) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template evaluate_value_contiguous<NC>(
      ne, shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_integrate(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate<NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t output[NC * NS][VS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_contiguous<NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_value(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_value<NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_value_contiguous(
    const int ne,
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t output[NC * NS][VS]) {
  TensorProductResidualOps<s_t, NQ, NS, VS, ND>::template integrate_value_contiguous<NC>(
      ne, shape_1d, value_coeff, output);
}



template <typename s_t, int NQ, int NS, int ND>
struct TensorProductResidualOpsScalar;

template <typename s_t, int NQ, int NS>
struct TensorProductResidualOpsScalar<s_t, NQ, NS, 2> {
  template <int NC>
  static SFEM_INLINE void evaluate(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1];
    s_t gx[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          const s_t u = streams[s * NC + f][0];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) ;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q)];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 2 + 0)];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 2 + 1)];
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) ;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value_q[0] = v;
        gradient_q0[0] = g0;
        gradient_q1[0] = g1;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_contiguous(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1];
    s_t gx[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          const s_t u = streams[s * NC + f];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) ;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q)];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 2 + 0)];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 2 + 1)];
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) ;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value_q[0] = v;
        gradient_q0[0] = g0;
        gradient_q1[0] = g1;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value(
      const s_t *const shape_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR vx_q = &vx[((f * NQ1 + qx) * NS1 + sy)];
            {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][0] * shape_1d[qx * NS1 + sx];
        }
        vx_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q)];
            {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy) ] * shape_1d[qy * NS1 + sy];
        }
        value_q[0] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value_contiguous(
      const s_t *const shape_1d,
      const s_t streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t vx[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR vx_q = &vx[((f * NQ1 + qx) * NS1 + sy)];
            {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f] * shape_1d[qx * NS1 + sx];
        }
        vx_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
      s_t *const RSTR value_q = &value[(f * NQ + q)];
            {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy) ] * shape_1d[qy * NS1 + sy];
        }
        value_q[0] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1];
    s_t sg[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) ] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1) ] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0) ] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) ;
        sv[i] = a;
        sg[i] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) ;
          v += sv[i] * shape_1d[qx * NS1 + sx] + sg[i] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][0] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_contiguous(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1];
    s_t sg[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) ] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1) ] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0) ] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) ;
        sv[i] = a;
        sg[i] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) ;
          v += sv[i] * shape_1d[qx * NS1 + sx] + sg[i] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value(
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR sv_q = &sv[((f * NQ1 + qx) * NS1 + sy)];
            {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) ] * shape_1d[qy * NS1 + sy];
        }
        sv_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy) ] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][0] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value_contiguous(
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    s_t sv[NC * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) {
    s_t *const RSTR sv_q = &sv[((f * NQ1 + qx) * NS1 + sy)];
            {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) ] * shape_1d[qy * NS1 + sy];
        }
        sv_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy) ] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS>
struct TensorProductResidualOpsScalar<s_t, NQ, NS, 3> {
  template <int NC>
  static SFEM_INLINE void evaluate(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1];
    s_t gx[NC * NQ1 * NS1 * NS1];
    s_t vxy[NC * NQ1 * NQ1 * NS1];
    s_t g0xy[NC * NQ1 * NQ1 * NS1];
    s_t g1xy[NC * NQ1 * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t u = streams[s * NC + f][0];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q)];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 3 + 0)];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 3 + 1)];
      s_t *const RSTR gradient_q2 = &gradient[((f * NQ + q) * 3 + 2)];
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value_q[0] = v;
        gradient_q0[0] = g0;
        gradient_q1[0] = g1;
        gradient_q2[0] = g2;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_contiguous(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t streams[NC * NS],
      s_t *const value,
      s_t *const gradient) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1];
    s_t gx[NC * NQ1 * NS1 * NS1];
    s_t vxy[NC * NQ1 * NQ1 * NS1];
    s_t g0xy[NC * NQ1 * NQ1 * NS1];
    s_t g1xy[NC * NQ1 * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t v = s_t(0);
        s_t g = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t u = streams[s * NC + f];
          v += u * shape_1d[qx * NS1 + sx];
          g += u * grad_1d[qx * NS1 + sx];
        }
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q)];
      s_t *const RSTR gradient_q0 = &gradient[((f * NQ + q) * 3 + 0)];
      s_t *const RSTR gradient_q1 = &gradient[((f * NQ + q) * 3 + 1)];
      s_t *const RSTR gradient_q2 = &gradient[((f * NQ + q) * 3 + 2)];
            {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value_q[0] = v;
        gradient_q0[0] = g0;
        gradient_q1[0] = g1;
        gradient_q2[0] = g2;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value(
      const s_t *const shape_1d,
      const s_t *const RSTR streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1];
    s_t vxy[NC * NQ1 * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vx_q = &vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)];
            {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][0] * shape_1d[qx * NS1 + sx];
        }
        vx_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vxy_q = &vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)];
            {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ] * shape_1d[qy * NS1 + sy];
        }
        vxy_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q)];
            {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ] * shape_1d[qz * NS1 + sz];
        }
        value_q[0] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void evaluate_value_contiguous(
      const s_t *const shape_1d,
      const s_t streams[NC * NS],
      s_t *const value) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t vx[NC * NQ1 * NS1 * NS1];
    s_t vxy[NC * NQ1 * NQ1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vx_q = &vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)];
            {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f] * shape_1d[qx * NS1 + sx];
        }
        vx_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR vxy_q = &vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)];
            {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ] * shape_1d[qy * NS1 + sy];
        }
        vxy_q[0] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
      s_t *const RSTR value_q = &value[(f * NQ + q)];
            {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ] * shape_1d[qz * NS1 + sz];
        }
        value_q[0] = v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1];
    s_t z1[NC * NQ1 * NQ1 * NS1];
    s_t z2[NC * NQ1 * NQ1 * NS1];
    s_t yz0[NC * NQ1 * NS1 * NS1];
    s_t yz1[NC * NQ1 * NS1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        s_t c = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) ] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2) ] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0) ] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1) ] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
        z0[i] = a;
        z1[i] = b;
        z2[i] = c;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
        yz0[j] = a;
        yz1[j] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
          v += yz0[j] * shape_1d[qx * NS1 + sx] + yz1[j] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f][0] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_contiguous(
      const s_t *const shape_1d,
      const s_t *const grad_1d,
      const s_t *const value_coeff,
      const s_t *const grad_coeff,
      s_t output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1];
    s_t z1[NC * NQ1 * NQ1 * NS1];
    s_t z2[NC * NQ1 * NQ1 * NS1];
    s_t yz0[NC * NQ1 * NS1 * NS1];
    s_t yz1[NC * NQ1 * NS1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        s_t c = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) ] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2) ] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0) ] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1) ] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
        z0[i] = a;
        z1[i] = b;
        z2[i] = c;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            {
        s_t a = s_t(0);
        s_t b = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
        yz0[j] = a;
        yz1[j] = b;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ;
          v += yz0[j] * shape_1d[qx * NS1 + sx] + yz1[j] * grad_1d[qx * NS1 + sx];
        }
        output[s * NC + f] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value(
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t *const RSTR output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1];
    s_t yz0[NC * NQ1 * NS1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR z0_q = &z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)];
            {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) ] * shape_1d[qz * NS1 + sz];
        }
        z0_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR yz0_q = &yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)];
            {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ] * shape_1d[qy * NS1 + sy];
        }
        yz0_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f][0] += v;
      }
    }
  }

  template <int NC>
  static SFEM_INLINE void integrate_value_contiguous(
      const s_t *const shape_1d,
      const s_t *const value_coeff,
      s_t output[NC * NS]) {
    static constexpr int NQ1 = integer_root(NQ, 3);
    static constexpr int NS1 = integer_root(NS, 3);
    s_t z0[NC * NQ1 * NQ1 * NS1];
    s_t yz0[NC * NQ1 * NS1 * NS1];
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR z0_q = &z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz)];
            {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) ] * shape_1d[qz * NS1 + sz];
        }
        z0_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
    s_t *const RSTR yz0_q = &yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz)];
            {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) ] * shape_1d[qy * NS1 + sy];
        }
        yz0_q[0] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) ] * shape_1d[qx * NS1 + sx];
        }
        output[s * NC + f] += v;
      }
    }
  }
};

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_scalar(
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template evaluate<NC>(
      shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_contiguous_scalar(
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t streams[NC * NS],
    s_t *const value,
    s_t *const gradient) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template evaluate_contiguous<NC>(
      shape_1d, grad_1d, streams, value, gradient);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_value_scalar(
    const s_t *const shape_1d,
    const s_t *const RSTR streams[NC * NS],
    s_t *const value) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template evaluate_value<NC>(
      shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_evaluate_value_contiguous_scalar(
    const s_t *const shape_1d,
    const s_t streams[NC * NS],
    s_t *const value) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template evaluate_value_contiguous<NC>(
      shape_1d, streams, value);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_scalar(
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template integrate<NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_contiguous_scalar(
    const s_t *const shape_1d,
    const s_t *const grad_1d,
    const s_t *const value_coeff,
    const s_t *const grad_coeff,
    s_t output[NC * NS]) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template integrate_contiguous<NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff, output);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_value_scalar(
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t *const RSTR output[NC * NS]) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template integrate_value<NC>(
      shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int ND, int NC>
static SFEM_INLINE void tensor_integrate_value_contiguous_scalar(
    const s_t *const shape_1d,
    const s_t *const value_coeff,
    s_t output[NC * NS]) {
  TensorProductResidualOpsScalar<s_t, NQ, NS, ND>::template integrate_value_contiguous<NC>(
      shape_1d, value_coeff, output);
}



//! One point's contribution to a symmetric element matrix, upper triangle only:
//!   K[(i, s), (j, t)] += w sum_{d, e} g[d][s] D[(i, d), (j, e)] g[e][t],
//! with g the shape gradients pushed forward by the adjugate (so w = qw / det),
//! g_ref laid out [k][s] and D, the physical tangent, (NC * ND) x (NC * ND).
//! `mirror_upper_triangle_scalar` completes the matrix once the rule is done.
template <typename s_t, int NS, int ND, int NC>
static SFEM_INLINE void point_test_symmetric_scalar(
    const s_t *const RSTR g_ref,
    const s_t *const RSTR adj,
    const s_t w,
    const s_t *const RSTR D,
    s_t *const RSTR element_matrix) {
  static_assert(ND == 2 || ND == 3, "point_test_symmetric_scalar is spelled for 2 and 3 dimensions");
  static constexpr int NDOFS = NC * NS;
  static constexpr int NA = NC * ND;
  // Written out per dimension rather than looped over it: on Grace (gcc 13)
  // the loop form of these sums cost 11% of the whole HEX8 assembly.
  s_t g[ND][NS];
  for (int s = 0; s < NS; ++s) {
    if constexpr (ND == 3) {
      const s_t r0 = g_ref[0 * NS + s], r1 = g_ref[1 * NS + s], r2 = g_ref[2 * NS + s];
      g[0][s] = r0 * adj[0] + r1 * adj[3] + r2 * adj[6];
      g[1][s] = r0 * adj[1] + r1 * adj[4] + r2 * adj[7];
      g[2][s] = r0 * adj[2] + r1 * adj[5] + r2 * adj[8];
    } else {
      const s_t r0 = g_ref[0 * NS + s], r1 = g_ref[1 * NS + s];
      g[0][s] = r0 * adj[0] + r1 * adj[2];
      g[1][s] = r0 * adj[1] + r1 * adj[3];
    }
  }
  // Every trial column's weighted flux at the point: W[(i, d)][(j, t)] =
  // w sum_e D[(i, d), (j, e)] g[e][t].
  s_t W[NA][NDOFS];
  for (int a = 0; a < NA; ++a) {
    for (int j = 0; j < NC; ++j) {
      if constexpr (ND == 3) {
        const s_t a0 = w * D[a * NA + j * ND + 0];
        const s_t a1 = w * D[a * NA + j * ND + 1];
        const s_t a2 = w * D[a * NA + j * ND + 2];
        for (int t = 0; t < NS; ++t) W[a][j * NS + t] = a0 * g[0][t] + a1 * g[1][t] + a2 * g[2][t];
      } else {
        const s_t a0 = w * D[a * NA + j * ND + 0];
        const s_t a1 = w * D[a * NA + j * ND + 1];
        for (int t = 0; t < NS; ++t) W[a][j * NS + t] = a0 * g[0][t] + a1 * g[1][t];
      }
    }
  }
  for (int i = 0; i < NC; ++i) {
    for (int s = 0; s < NS; ++s) {
      const int r = i * NS + s;
      s_t *const RSTR row = &element_matrix[r * NDOFS];
      if constexpr (ND == 3) {
        const s_t g0 = g[0][s], g1 = g[1][s], g2 = g[2][s];
        const s_t *const RSTR w0 = W[i * ND + 0];
        const s_t *const RSTR w1 = W[i * ND + 1];
        const s_t *const RSTR w2 = W[i * ND + 2];
        for (int c = r; c < NDOFS; ++c) row[c] += g0 * w0[c] + g1 * w1[c] + g2 * w2[c];
      } else {
        const s_t g0 = g[0][s], g1 = g[1][s];
        const s_t *const RSTR w0 = W[i * ND + 0];
        const s_t *const RSTR w1 = W[i * ND + 1];
        for (int c = r; c < NDOFS; ++c) row[c] += g0 * w0[c] + g1 * w1[c];
      }
    }
  }
}

template <typename s_t, int NDOFS>
static SFEM_INLINE void mirror_upper_triangle_scalar(s_t *const RSTR element_matrix) {
  for (int r = 1; r < NDOFS; ++r) {
    for (int c = 0; c < r; ++c) element_matrix[r * NDOFS + c] = element_matrix[c * NDOFS + r];
  }
}

//! The reference gradients of every shape of a tensor-product basis at point q
//! of its rule, laid out [k][s], from the 1-D tables.
template <typename s_t, int NQ, int NS, int ND>
static SFEM_INLINE void tensor_point_reference_gradients_scalar(
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const int q,
    s_t *const RSTR g_ref) {
  static constexpr int NQ1 = integer_root(NQ, ND);
  static constexpr int NS1 = integer_root(NS, ND);
  const int qx = q % NQ1;
  if constexpr (ND == 2) {
    const int qy = q / NQ1;
    for (int sy = 0; sy < NS1; ++sy) {
      for (int sx = 0; sx < NS1; ++sx) {
        const int s = sx + NS1 * sy;
        const s_t bx = shape_1d[qx * NS1 + sx], gx = grad_1d[qx * NS1 + sx];
        const s_t by = shape_1d[qy * NS1 + sy], gy = grad_1d[qy * NS1 + sy];
        g_ref[0 * NS + s] = gx * by;
        g_ref[1 * NS + s] = bx * gy;
      }
    }
  } else {
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    for (int sz = 0; sz < NS1; ++sz) {
      for (int sy = 0; sy < NS1; ++sy) {
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          const s_t bx = shape_1d[qx * NS1 + sx], gx = grad_1d[qx * NS1 + sx];
          const s_t by = shape_1d[qy * NS1 + sy], gy = grad_1d[qy * NS1 + sy];
          const s_t bz = shape_1d[qz * NS1 + sz], gz = grad_1d[qz * NS1 + sz];
          g_ref[0 * NS + s] = gx * by * bz;
          g_ref[1 * NS + s] = bx * gy * bz;
          g_ref[2 * NS + s] = bx * by * gz;
        }
      }
    }
  }
}

//! One slab of a tensor-product rule -- every point whose last coordinate is
//! `q_last`, NQ1^(ND-1) of them -- tested against every shape of one field
//! component and accumulated into `out[s]`:
//!   out[s] += sum_{p in slab} sum_k G_k(s; p) flux[k][p],
//! factorised one dimension at a time, the slab's own factor applied per shape.
template <typename s_t, int NQ1, int NS1, int ND>
static SFEM_INLINE void tensor_slab_test_scalar(
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const int q_last,
    const s_t *const RSTR flux,
    s_t *const RSTR out) {
  if constexpr (ND == 2) {
    const int qy = q_last;
    for (int sx = 0; sx < NS1; ++sx) {
      s_t a0 = s_t(0), a1 = s_t(0);
      for (int qx = 0; qx < NQ1; ++qx) {
        a0 += grad_1d[qx * NS1 + sx] * flux[0 * NQ1 + qx];
        a1 += shape_1d[qx * NS1 + sx] * flux[1 * NQ1 + qx];
      }
      for (int sy = 0; sy < NS1; ++sy) {
        out[sx + NS1 * sy] += shape_1d[qy * NS1 + sy] * a0 + grad_1d[qy * NS1 + sy] * a1;
      }
    }
  } else {
    static constexpr int NP = NQ1 * NQ1;
    const int qz = q_last;
    s_t a[3 * NQ1 * NS1];
    for (int qy = 0; qy < NQ1; ++qy) {
      for (int sx = 0; sx < NS1; ++sx) {
        s_t a0 = s_t(0), a1 = s_t(0), a2 = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          const int p = qx + NQ1 * qy;
          const s_t bx = shape_1d[qx * NS1 + sx], gx = grad_1d[qx * NS1 + sx];
          a0 += gx * flux[0 * NP + p];
          a1 += bx * flux[1 * NP + p];
          a2 += bx * flux[2 * NP + p];
        }
        a[(0 * NQ1 + qy) * NS1 + sx] = a0;
        a[(1 * NQ1 + qy) * NS1 + sx] = a1;
        a[(2 * NQ1 + qy) * NS1 + sx] = a2;
      }
    }
    for (int sy = 0; sy < NS1; ++sy) {
      for (int sx = 0; sx < NS1; ++sx) {
        s_t cb = s_t(0), cg = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const s_t by = shape_1d[qy * NS1 + sy], gy = grad_1d[qy * NS1 + sy];
          cb += by * a[(0 * NQ1 + qy) * NS1 + sx] + gy * a[(1 * NQ1 + qy) * NS1 + sx];
          cg += by * a[(2 * NQ1 + qy) * NS1 + sx];
        }
        for (int sz = 0; sz < NS1; ++sz) {
          out[sx + NS1 * (sy + NS1 * sz)] += shape_1d[qz * NS1 + sz] * cb + grad_1d[qz * NS1 + sz] * cg;
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
