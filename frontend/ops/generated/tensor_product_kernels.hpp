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
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy) * VS + lane;
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient[(q * 2 + 0) * VS + lane] = gx;
          gradient[(q * 2 + 1) * VS + lane] = gy;
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
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
          s_t gx = s_t(0);
          s_t gy = s_t(0);
          for (int sy = 0; sy < NS1; ++sy) {
            const int i = (qx * NS1 + sy) * VS + lane;
            gx += grad_x[i] * shape_1d[qy * NS1 + sy];
            gy += value_x[i] * grad_1d[qy * NS1 + sy];
          }
          gradient[(q * 2 + 0) * VS + lane] = gx;
          gradient[(q * 2 + 1) * VS + lane] = gy;
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
            gradient[(q * 3 + 0) * VS + lane] = gx;
            gradient[(q * 3 + 1) * VS + lane] = gy;
            gradient[(q * 3 + 2) * VS + lane] = gz;
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
            gradient[(q * 3 + 0) * VS + lane] = gx;
            gradient[(q * 3 + 1) * VS + lane] = gy;
            gradient[(q * 3 + 2) * VS + lane] = gz;
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
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value[(f * NQ + q) * VS + lane] = v;
        gradient[((f * NQ + q) * 2 + 0) * VS + lane] = g0;
        gradient[((f * NQ + q) * 2 + 1) * VS + lane] = g1;
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
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
        vx[i] = v;
        gx[i] = g;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        value[(f * NQ + q) * VS + lane] = v;
        gradient[((f * NQ + q) * 2 + 0) * VS + lane] = g0;
        gradient[((f * NQ + q) * 2 + 1) * VS + lane] = g1;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx[((f * NQ1 + qx) * NS1 + sy) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        value[(f * NQ + q) * VS + lane] = v;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * sy;
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx[((f * NQ1 + qx) * NS1 + sy) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * qy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[((f * NQ1 + qx) * NS1 + sy) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        value[(f * NQ + q) * VS + lane] = v;
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
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1) * VS + lane] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
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
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
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
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qy * NS1 + sy]
                       + grad_coeff[((f * NQ + q) * 2 + 1) * VS + lane] * grad_1d[qy * NS1 + sy];
          b += grad_coeff[((f * NQ + q) * 2 + 0) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
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
          const int i = ((f * NQ1 + qx) * NS1 + sy) * VS + lane;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        sv[((f * NQ1 + qx) * NS1 + sy) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy) * VS + lane] * shape_1d[qx * NS1 + sx];
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          const int q = qx + NQ1 * qy;
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        sv[((f * NQ1 + qx) * NS1 + sy) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * sy;
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += sv[((f * NQ1 + qx) * NS1 + sy) * VS + lane] * shape_1d[qx * NS1 + sx];
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
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value[(f * NQ + q) * VS + lane] = v;
        gradient[((f * NQ + q) * 3 + 0) * VS + lane] = g0;
        gradient[((f * NQ + q) * 3 + 1) * VS + lane] = g1;
        gradient[((f * NQ + q) * 3 + 2) * VS + lane] = g2;
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
        const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
          const int i = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
          v += vx[i] * shape_1d[qy * NS1 + sy];
          g0 += gx[i] * shape_1d[qy * NS1 + sy];
          g1 += vx[i] * grad_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
        vxy[j] = v;
        g0xy[j] = g0;
        g1xy[j] = g1;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        s_t g0 = s_t(0);
        s_t g1 = s_t(0);
        s_t g2 = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          const int j = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
          v += vxy[j] * shape_1d[qz * NS1 + sz];
          g0 += g0xy[j] * shape_1d[qz * NS1 + sz];
          g1 += g1xy[j] * shape_1d[qz * NS1 + sz];
          g2 += vxy[j] * grad_1d[qz * NS1 + sz];
        }
        value[(f * NQ + q) * VS + lane] = v;
        gradient[((f * NQ + q) * 3 + 0) * VS + lane] = g0;
        gradient[((f * NQ + q) * 3 + 1) * VS + lane] = g1;
        gradient[((f * NQ + q) * 3 + 2) * VS + lane] = g2;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        value[(f * NQ + q) * VS + lane] = v;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sx = 0; sx < NS1; ++sx) {
          const int s = sx + NS1 * (sy + NS1 * sz);
          v += streams[s * NC + f][lane] * shape_1d[qx * NS1 + sx];
        }
        vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int qy = 0; qy < NQ1; ++qy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sy = 0; sy < NS1; ++sy) {
          v += vx[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] = v;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qz = 0; qz < NQ1; ++qz) for (int qy = 0; qy < NQ1; ++qy) for (int qx = 0; qx < NQ1; ++qx) {
      const int q = qx + NQ1 * (qy + NQ1 * qz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int sz = 0; sz < NS1; ++sz) {
          v += vxy[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        value[(f * NQ + q) * VS + lane] = v;
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
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2) * VS + lane] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0) * VS + lane] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
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
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qz * NS1 + sz]
                       + grad_coeff[((f * NQ + q) * 3 + 2) * VS + lane] * grad_1d[qz * NS1 + sz];
          b += grad_coeff[((f * NQ + q) * 3 + 0) * VS + lane] * shape_1d[qz * NS1 + sz];
          c += grad_coeff[((f * NQ + q) * 3 + 1) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
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
          const int i = (((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane;
          a += z0[i] * shape_1d[qy * NS1 + sy] + z2[i] * grad_1d[qy * NS1 + sy];
          b += z1[i] * shape_1d[qy * NS1 + sy];
        }
        const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
          const int j = (((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane;
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] * shape_1d[qx * NS1 + sx];
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
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qz = 0; qz < NQ1; ++qz) {
          const int q = qx + NQ1 * (qy + NQ1 * qz);
          a += value_coeff[(f * NQ + q) * VS + lane] * shape_1d[qz * NS1 + sz];
        }
        z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int qx = 0; qx < NQ1; ++qx) for (int sy = 0; sy < NS1; ++sy) for (int sz = 0; sz < NS1; ++sz) {
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t a = s_t(0);
        for (int qy = 0; qy < NQ1; ++qy) {
          a += z0[(((f * NQ1 + qx) * NQ1 + qy) * NS1 + sz) * VS + lane] * shape_1d[qy * NS1 + sy];
        }
        yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] = a;
      }
    }
    for (int f = 0; f < NC; ++f) for (int sz = 0; sz < NS1; ++sz) for (int sy = 0; sy < NS1; ++sy) for (int sx = 0; sx < NS1; ++sx) {
      const int s = sx + NS1 * (sy + NS1 * sz);
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
        s_t v = s_t(0);
        for (int qx = 0; qx < NQ1; ++qx) {
          v += yz0[(((f * NQ1 + qx) * NS1 + sy) * NS1 + sz) * VS + lane] * shape_1d[qx * NS1 + sx];
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

} // namespace codegen
} // namespace sfem

#endif
