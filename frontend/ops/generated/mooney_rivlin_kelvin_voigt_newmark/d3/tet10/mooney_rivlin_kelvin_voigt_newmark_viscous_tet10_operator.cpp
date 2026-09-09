#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "../mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#include "../../../packed_thread_scratch.hpp"
#if defined(__has_include)
#if __has_include("smesh_types.hpp")
#include "smesh_types.hpp"
#endif
#endif

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
    const int,
    const g_t *const RSTR source,
    s_t *const RSTR,
    std::true_type) {
  return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
    const int ne,
    const g_t *const RSTR source,
    s_t *const RSTR converted,
    std::false_type) {
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    converted[lane] = s_t(source[lane]);
  }
  return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data {
  static const s_t *shape() {
    static const s_t data[40] = {s_t(0.099999999999999936), s_t(-0.099999999999999992), s_t(-0.099999999999999992), s_t(-0.099999999999999992), s_t(0.32360679774997891), s_t(0.076393202250021025), s_t(0.32360679774997891), s_t(0.32360679774997891), s_t(0.076393202250021025), s_t(0.076393202250021025), s_t(-0.099999999999999978), s_t(0.10000000000000007), s_t(-0.099999999999999992), s_t(-0.099999999999999992), s_t(0.32360679774997886), s_t(0.32360679774997897), s_t(0.076393202250020983), s_t(0.076393202250020983), s_t(0.32360679774997897), s_t(0.076393202250021025), s_t(-0.099999999999999978), s_t(-0.099999999999999992), s_t(0.10000000000000007), s_t(-0.099999999999999992), s_t(0.076393202250020983), s_t(0.32360679774997897), s_t(0.32360679774997886), s_t(0.076393202250020983), s_t(0.076393202250021025), s_t(0.32360679774997897), s_t(-0.099999999999999964), s_t(-0.099999999999999992), s_t(-0.099999999999999992), s_t(0.10000000000000007), s_t(0.07639320225002097), s_t(0.076393202250021025), s_t(0.07639320225002097), s_t(0.3236067977499788), s_t(0.32360679774997897), s_t(0.32360679774997897)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[40] = {s_t(-1.3416407864998741), s_t(-0.44721359549995798), s_t(0), s_t(0), s_t(1.7888543819998319), s_t(0.55278640450004202), s_t(-0.55278640450004202), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(0), s_t(0.44721359549995832), s_t(1.3416407864998741), s_t(0), s_t(0), s_t(-1.7888543819998315), s_t(0.55278640450004202), s_t(-0.55278640450004202), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(0), s_t(0.44721359549995832), s_t(-0.44721359549995798), s_t(0), s_t(0), s_t(0), s_t(2.3416407864998741), s_t(-2.3416407864998741), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(0), s_t(0.44721359549995832), s_t(-0.44721359549995798), s_t(0), s_t(0), s_t(0), s_t(0.55278640450004202), s_t(-0.55278640450004202), s_t(-2.3416407864998741), s_t(2.3416407864998741), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[40] = {s_t(-1.3416407864998741), s_t(0), s_t(-0.44721359549995798), s_t(0), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(1.7888543819998319), s_t(-0.55278640450004202), s_t(0), s_t(0.55278640450004202), s_t(0.44721359549995832), s_t(0), s_t(-0.44721359549995798), s_t(0), s_t(-2.3416407864998741), s_t(2.3416407864998741), s_t(0), s_t(-0.55278640450004202), s_t(0), s_t(0.55278640450004202), s_t(0.44721359549995832), s_t(0), s_t(1.3416407864998741), s_t(0), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(-1.7888543819998315), s_t(-0.55278640450004202), s_t(0), s_t(0.55278640450004202), s_t(0.44721359549995832), s_t(0), s_t(-0.44721359549995798), s_t(0), s_t(-0.55278640450004202), s_t(0.55278640450004202), s_t(0), s_t(-2.3416407864998741), s_t(0), s_t(2.3416407864998741)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[40] = {s_t(-1.3416407864998741), s_t(0), s_t(0), s_t(-0.44721359549995798), s_t(-0.55278640450004202), s_t(0), s_t(-0.55278640450004202), s_t(1.7888543819998319), s_t(0.55278640450004202), s_t(0.55278640450004202), s_t(0.44721359549995832), s_t(0), s_t(0), s_t(-0.44721359549995798), s_t(-2.3416407864998741), s_t(0), s_t(-0.55278640450004202), s_t(0), s_t(2.3416407864998741), s_t(0.55278640450004202), s_t(0.44721359549995832), s_t(0), s_t(0), s_t(-0.44721359549995798), s_t(-0.55278640450004202), s_t(0), s_t(-2.3416407864998741), s_t(0), s_t(0.55278640450004202), s_t(2.3416407864998741), s_t(0.44721359549995832), s_t(0), s_t(0), s_t(1.3416407864998741), s_t(-0.55278640450004202), s_t(0), s_t(-0.55278640450004202), s_t(-1.7888543819998315), s_t(0.55278640450004202), s_t(0.55278640450004202)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[4] = {s_t(0.041666666666666664), s_t(0.041666666666666664), s_t(0.041666666666666664), s_t(0.041666666666666664)};
    return data;
  }
};

template <typename s_t>
struct mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data {
  static const s_t *shape() {
    static const s_t data[110] = {s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.44897959183673491), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.22448979591836737), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.020408163265306121), s_t(-0.061224489795918387), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836743), s_t(0.22448979591836732), s_t(0.020408163265306128), s_t(0.020408163265306128), s_t(0.22448979591836732), s_t(0.020408163265306121), s_t(-0.061224489795918401), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(0.020408163265306135), s_t(0.22448979591836732), s_t(0.22448979591836751), s_t(0.020408163265306135), s_t(0.020408163265306121), s_t(0.22448979591836732), s_t(-0.061224489795918421), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(0.020408163265306145), s_t(0.020408163265306121), s_t(0.020408163265306145), s_t(0.2244897959183676), s_t(0.22448979591836732), s_t(0.22448979591836732), s_t(-0.080357142857142821), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428564), s_t(0.63809286661931275), s_t(0.16071428571428564), s_t(0.040478561952115862), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.040478561952115875), s_t(0.1607142857142857), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(0.040478561952115875), s_t(0.16071428571428573), s_t(0.1607142857142857), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(-1), s_t(1), s_t(0), s_t(-2.1428571428571428), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(2.1428571428571428), s_t(0), s_t(0), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(3.1428571428571428), s_t(-3.1428571428571428), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571441), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(0), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719645), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(2.8571428571428572), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(2.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(-2.8571428571428568), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571441), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(3.1428571428571428), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(-1.1952286093343947), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(-1.1952286093343938), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(1.1952286093343933), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719645), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(1.1952286093343933), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(3.1428571428571428), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(0.2857142857142857), s_t(3.1428571428571428), s_t(0.71428571428571441), s_t(0), s_t(0), s_t(2.1428571428571428), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719645), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.40238569533280322)};
    return data;
  }
  static const s_t *q_weight() {
    static const s_t data[11] = {s_t(-0.013155555555555556), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887)};
    return data;
  }
};

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[30],
    const double *const RSTR previous[30],
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    double *const RSTR output[30]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<double, 11, 10, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[30],
    const float *const RSTR previous[30],
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    float *const RSTR output[30]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<float, 11, 10, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10][VS];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, eta_b, eta_s, newmark_velocity_alpha, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = bcoordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J01 = bcoordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J02 = bcoordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J10 = bcoordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J11 = bcoordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J12 = bcoordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J20 = bcoordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J21 = bcoordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J22 = bcoordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, eta_b, eta_s, newmark_velocity_alpha, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    const double *const RSTR current,
    const double *const RSTR previous,
    double *const RSTR output
) {
  return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    const float *const RSTR current,
    const float *const RSTR previous,
    float *const RSTR output
) {
  return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_i_msoa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_esoa(
    const int ne,
    const ptrdiff_t geometry_stride,
    const double *const RSTR determinant,
    const double *const RSTR adjugate[9],
    const double *const RSTR current[30],
    const double *const RSTR previous[30],
    const double *const RSTR direction[30],
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    double *const RSTR output[30]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<double, 11, 10, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_esoa_float(
    const int ne,
    const ptrdiff_t geometry_stride,
    const float *const RSTR determinant,
    const float *const RSTR adjugate[9],
    const float *const RSTR current[30],
    const float *const RSTR previous[30],
    const float *const RSTR direction[30],
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    float *const RSTR output[30]
) {
  sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<float, 11, 10, 16>(ne, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
  return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const g_t *const RSTR g_adj0,
    const g_t *const RSTR g_adj1,
    const g_t *const RSTR g_adj2,
    const g_t *const RSTR g_adj3,
    const g_t *const RSTR g_adj4,
    const g_t *const RSTR g_adj5,
    const g_t *const RSTR g_adj6,
    const g_t *const RSTR g_adj7,
    const g_t *const RSTR g_adj8,
    const g_t *const RSTR g_det0,
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 4;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const affine_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::shape();
  const s_t *const affine_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_x();
  const s_t *const affine_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_y();
  const s_t *const affine_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::grad_ref_z();
  const s_t *const affine_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_affine_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
          bdirection[stream][lane] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[10] = {g_adj0 + evb, g_adj1 + evb, g_adj2 + evb, g_adj3 + evb, g_adj4 + evb, g_adj5 + evb, g_adj6 + evb, g_adj7 + evb, g_adj8 + evb, g_det0 + evb};
    s_t baffine_geometry_data[10][VS];
    const s_t *bageom_streams[10];
    for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
      bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
          ne, affine_geometry_sources[geometry_stream], baffine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
    }
    const s_t *badjugate[9];
    for (int component = 0; component < 9; ++component) {
      badjugate[component] = bageom_streams[component];
    }

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, 0, bageom_streams[9], badjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, newmark_velocity_alpha, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl<double, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_adj0,
    const geom_t *const RSTR g_adj1,
    const geom_t *const RSTR g_adj2,
    const geom_t *const RSTR g_adj3,
    const geom_t *const RSTR g_adj4,
    const geom_t *const RSTR g_adj5,
    const geom_t *const RSTR g_adj6,
    const geom_t *const RSTR g_adj7,
    const geom_t *const RSTR g_adj8,
    const geom_t *const RSTR g_det0,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_a_msoa_impl<float, geom_t>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const s_t *const RSTR u0,
    const s_t *const RSTR u1,
    const s_t *const RSTR u2,
    const ptrdiff_t previous_stride,
    const s_t *const RSTR u0_old,
    const s_t *const RSTR u1_old,
    const s_t *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const s_t *const RSTR u0_direction,
    const s_t *const RSTR u1_direction,
    const s_t *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    s_t *const RSTR u0_out,
    s_t *const RSTR u1_out,
    s_t *const RSTR u2_out
) {
  static constexpr int ND = 3;
  static constexpr int NQ = 11;
  static constexpr int NS = 10;
  static constexpr int NC = 3;
  static constexpr int VS = 16;
  (void)nnodes;
  const s_t *const isoparametric_shape = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::shape();
  const s_t *const isoparametric_grad_ref_x = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_x();
  const s_t *const isoparametric_grad_ref_y = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_y();
  const s_t *const isoparametric_grad_ref_z = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::grad_ref_z();
  const s_t *const isoparametric_q_weight = sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<s_t>::q_weight();

#pragma omp parallel for schedule(static)
  for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
    const int ne = (int)MIN((ptrdiff_t)VS, nelements - evb);
    s_t bcoordinates[3 * NS][VS];
    s_t badjugate_data[9][NQ * VS];
    s_t bdeterminant[NQ * VS];
    s_t bcurrent[NC * NS][VS];
    s_t bprevious[NC * NS][VS];
    s_t bdirection[NC * NS][VS];
    s_t boutput[NC * NS][VS];

    const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int d = 0; d < ND; ++d) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcoordinates[shape * ND + d][lane] = coordinate_components[d][node];
        }
      }
    }
    const s_t *const current_components[NC] = {u0, u1, u2};
    const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
    const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const idx_t node = element_shape[evb + lane];
          bcurrent[stream][lane] = current_components[field][node * current_stride];
          bprevious[stream][lane] = previous_components[field][node * previous_stride];
          bdirection[stream][lane] = direction_components[field][node * direction_stride];
        }
      }
    }

    for (int stream = 0; stream < 30; ++stream) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        boutput[stream][lane] = s_t(0);
      }
    }

    s_t *badjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t J00 = bcoordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J01 = bcoordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J02 = bcoordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J10 = bcoordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J11 = bcoordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J12 = bcoordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
        const s_t J20 = bcoordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
        const s_t J21 = bcoordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
        const s_t J22 = bcoordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + bcoordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + bcoordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + bcoordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + bcoordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + bcoordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + bcoordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + bcoordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + bcoordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + bcoordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            badjugate_streams, bdeterminant, q * VS + lane);
      }
    }

    const s_t *const badjugate[9] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};

    mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(ne, VS, bdeterminant, badjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, bcurrent, bprevious, bdirection, eta_b, eta_s, newmark_velocity_alpha, boutput);

    s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
    for (int shape = 0; shape < NS; ++shape) {
      const idx_t *const RSTR element_shape = elements[shape];
      for (int field = 0; field < NC; ++field) {
        const int stream = shape * NC + field;
        s_t *const RSTR out = output_components[field];
        for (int scatter = 0; scatter < ne; ++scatter) {
          #pragma omp atomic update
          out[element_shape[evb + scatter] * out_stride] += boutput[stream][scatter];
        }
      }
    }
  }

  return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double eta_b,
    const double eta_s,
    const double newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const double *const RSTR u0,
    const double *const RSTR u1,
    const double *const RSTR u2,
    const ptrdiff_t previous_stride,
    const double *const RSTR u0_old,
    const double *const RSTR u1_old,
    const double *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const double *const RSTR u0_direction,
    const double *const RSTR u1_direction,
    const double *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    double *const RSTR u0_out,
    double *const RSTR u1_out,
    double *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float eta_b,
    const float eta_s,
    const float newmark_velocity_alpha,
    const ptrdiff_t current_stride,
    const float *const RSTR u0,
    const float *const RSTR u1,
    const float *const RSTR u2,
    const ptrdiff_t previous_stride,
    const float *const RSTR u0_old,
    const float *const RSTR u1_old,
    const float *const RSTR u2_old,
    const ptrdiff_t direction_stride,
    const float *const RSTR u0_direction,
    const float *const RSTR u1_direction,
    const float *const RSTR u2_direction,
    const ptrdiff_t out_stride,
    float *const RSTR u0_out,
    float *const RSTR u1_out,
    float *const RSTR u2_out
) {
  return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const double *const RSTR parameters,
    const double *const RSTR current,
    const double *const RSTR previous,
    const double *const RSTR direction,
    double *const RSTR output
) {
  return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_maos_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points,
    const float *const RSTR parameters,
    const float *const RSTR current,
    const float *const RSTR previous,
    const float *const RSTR direction,
    float *const RSTR output
) {
  return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_i_msoa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}
