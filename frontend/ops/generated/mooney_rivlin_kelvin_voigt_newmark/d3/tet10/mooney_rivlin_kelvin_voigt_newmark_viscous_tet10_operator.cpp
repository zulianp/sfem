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
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename s_t, typename g_t, int VS>
SFEM_INLINE const s_t *ageom_stream(
        const int nelems,
        const g_t *const SFEM_RESTRICT source,
        s_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
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

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[30],
        const double *const SFEM_RESTRICT previous[30],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[30]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<double, 11, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[30],
        const float *const SFEM_RESTRICT previous[30],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[30]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block<float, 11, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::q_weight(), current, previous, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const s_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const s_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out,
        s_t *const SFEM_RESTRICT u2_out
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
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_output[NC * NS][VS];
        const s_t *const current_components[NC] = {u0, u1, u2};
        const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 30; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_adjugate4 + evb, g_jacobian_adjugate5 + evb, g_jacobian_adjugate6 + evb, g_jacobian_adjugate7 + evb, g_jacobian_adjugate8 + evb, g_jacobian_determinant0 + evb};
        s_t block_affine_geometry_data[10][VS];
        const s_t *bageom_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = bageom_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[9], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, block_current, block_previous, eta_b, eta_s, newmark_velocity_alpha, block_output);

        s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const s_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const s_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out,
        s_t *const SFEM_RESTRICT u2_out
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
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_coordinates[3 * NS][VS];
        s_t block_adjugate_data[9][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_output[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_coordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {u0, u1, u2};
        const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                }
            }
        }

        for (int stream = 0; stream < 30; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t *block_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
                const s_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
                const s_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VS + lane);
            }
        }

        const s_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current, block_previous, eta_b, eta_s, newmark_velocity_alpha, block_output);

        s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        double *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        float *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_residual_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_element_soa(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const double *const SFEM_RESTRICT determinant,
        const double *const SFEM_RESTRICT adjugate[9],
        const double *const SFEM_RESTRICT current[30],
        const double *const SFEM_RESTRICT previous[30],
        const double *const SFEM_RESTRICT direction[30],
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        double *const SFEM_RESTRICT output[30]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<double, 11, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<double>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_element_soa_float(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const float *const SFEM_RESTRICT determinant,
        const float *const SFEM_RESTRICT adjugate[9],
        const float *const SFEM_RESTRICT current[30],
        const float *const SFEM_RESTRICT previous[30],
        const float *const SFEM_RESTRICT direction[30],
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        float *const SFEM_RESTRICT output[30]
) {
    sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block<float, 11, 10, 16>(nelems, geometry_stride, determinant, adjugate, sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::shape(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_x(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_y(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::grad_ref_z(), sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_isoparametric_reference_data<float>::q_weight(), current, previous, direction, eta_b, eta_s, newmark_velocity_alpha, output);
    return SFEM_SUCCESS;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const g_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const s_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const s_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u0_direction,
        const s_t *const SFEM_RESTRICT u1_direction,
        const s_t *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out,
        s_t *const SFEM_RESTRICT u2_out
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
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_direction[NC * NS][VS];
        s_t block_output[NC * NS][VS];
        const s_t *const current_components[NC] = {u0, u1, u2};
        const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
        const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 30; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        const g_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evb, g_jacobian_adjugate1 + evb, g_jacobian_adjugate2 + evb, g_jacobian_adjugate3 + evb, g_jacobian_adjugate4 + evb, g_jacobian_adjugate5 + evb, g_jacobian_adjugate6 + evb, g_jacobian_adjugate7 + evb, g_jacobian_adjugate8 + evb, g_jacobian_determinant0 + evb};
        s_t block_affine_geometry_data[10][VS];
        const s_t *bageom_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            bageom_streams[geometry_stream] = ageom_stream<s_t, g_t, VS>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<g_t, s_t>());
        }
        const s_t *block_adjugate[9];
        for (int component = 0; component < 9; ++component) {
            block_adjugate[component] = bageom_streams[component];
        }

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, 0, bageom_streams[9], block_adjugate, affine_shape, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, block_current, block_previous, block_direction, eta_b, eta_s, newmark_velocity_alpha, block_output);

        s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_affine_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_affine_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const geom_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const geom_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const s_t eta_b,
        const s_t eta_s,
        const s_t newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u0,
        const s_t *const SFEM_RESTRICT u1,
        const s_t *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const s_t *const SFEM_RESTRICT u0_old,
        const s_t *const SFEM_RESTRICT u1_old,
        const s_t *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const s_t *const SFEM_RESTRICT u0_direction,
        const s_t *const SFEM_RESTRICT u1_direction,
        const s_t *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u0_out,
        s_t *const SFEM_RESTRICT u1_out,
        s_t *const SFEM_RESTRICT u2_out
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
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_coordinates[3 * NS][VS];
        s_t block_adjugate_data[9][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[NC * NS][VS];
        s_t block_previous[NC * NS][VS];
        s_t block_direction[NC * NS][VS];
        s_t block_output[NC * NS][VS];

        const geom_t *const coordinate_components[ND] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < ND; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_coordinates[shape * ND + d][lane] = coordinate_components[d][node];
                }
            }
        }
        const s_t *const current_components[NC] = {u0, u1, u2};
        const s_t *const previous_components[NC] = {u0_old, u1_old, u2_old};
        const s_t *const direction_components[NC] = {u0_direction, u1_direction, u2_direction};

        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evb + lane];
                    block_current[stream][lane] = current_components[field][node * current_stride];
                    block_previous[stream][lane] = previous_components[field][node * previous_stride];
                    block_direction[stream][lane] = direction_components[field][node * direction_stride];
                }
            }
        }

        for (int stream = 0; stream < 30; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t *block_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = block_coordinates[0][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J01 = block_coordinates[0][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J02 = block_coordinates[0][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[3][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[6][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[9][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[12][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[15][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[18][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[21][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[24][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[27][lane] * isoparametric_grad_ref_z[q * NS + 9];
                const s_t J10 = block_coordinates[1][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J11 = block_coordinates[1][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J12 = block_coordinates[1][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[4][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[7][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[10][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[13][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[16][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[19][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[22][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[25][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[28][lane] * isoparametric_grad_ref_z[q * NS + 9];
                const s_t J20 = block_coordinates[2][lane] * isoparametric_grad_ref_x[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_x[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_x[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_x[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_x[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_x[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_x[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_x[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_x[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_x[q * NS + 9];
                const s_t J21 = block_coordinates[2][lane] * isoparametric_grad_ref_y[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_y[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_y[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_y[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_y[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_y[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_y[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_y[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_y[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_y[q * NS + 9];
                const s_t J22 = block_coordinates[2][lane] * isoparametric_grad_ref_z[q * NS + 0] + block_coordinates[5][lane] * isoparametric_grad_ref_z[q * NS + 1] + block_coordinates[8][lane] * isoparametric_grad_ref_z[q * NS + 2] + block_coordinates[11][lane] * isoparametric_grad_ref_z[q * NS + 3] + block_coordinates[14][lane] * isoparametric_grad_ref_z[q * NS + 4] + block_coordinates[17][lane] * isoparametric_grad_ref_z[q * NS + 5] + block_coordinates[20][lane] * isoparametric_grad_ref_z[q * NS + 6] + block_coordinates[23][lane] * isoparametric_grad_ref_z[q * NS + 7] + block_coordinates[26][lane] * isoparametric_grad_ref_z[q * NS + 8] + block_coordinates[29][lane] * isoparametric_grad_ref_z[q * NS + 9];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VS + lane);
            }
        }

        const s_t *const block_adjugate[9] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous<s_t, NQ, NS, VS>(nelems, VS, block_determinant, block_adjugate, isoparametric_shape, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, block_current, block_previous, block_direction, eta_b, eta_s, newmark_velocity_alpha, block_output);

        s_t *const output_components[NC] = {u0_out, u1_out, u2_out};
        for (int shape = 0; shape < NS; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int field = 0; field < NC; ++field) {
                const int stream = shape * NC + field;
                s_t *const SFEM_RESTRICT out = output_components[field];
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double eta_b,
        const double eta_s,
        const double newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u0,
        const double *const SFEM_RESTRICT u1,
        const double *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u0_old,
        const double *const SFEM_RESTRICT u1_old,
        const double *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u0_direction,
        const double *const SFEM_RESTRICT u1_direction,
        const double *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u0_out,
        double *const SFEM_RESTRICT u1_out,
        double *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float eta_b,
        const float eta_s,
        const float newmark_velocity_alpha,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u0,
        const float *const SFEM_RESTRICT u1,
        const float *const SFEM_RESTRICT u2,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u0_old,
        const float *const SFEM_RESTRICT u1_old,
        const float *const SFEM_RESTRICT u2_old,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u0_direction,
        const float *const SFEM_RESTRICT u1_direction,
        const float *const SFEM_RESTRICT u2_direction,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u0_out,
        float *const SFEM_RESTRICT u1_out,
        float *const SFEM_RESTRICT u2_out
) {
    return sfem::codegen::mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, eta_b, eta_s, newmark_velocity_alpha, current_stride, u0, u1, u2, previous_stride, u0_old, u1_old, u2_old, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double *const SFEM_RESTRICT parameters,
        const double *const SFEM_RESTRICT current,
        const double *const SFEM_RESTRICT previous,
        const double *const SFEM_RESTRICT direction,
        double *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}

extern "C" int mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_aos_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float *const SFEM_RESTRICT parameters,
        const float *const SFEM_RESTRICT current,
        const float *const SFEM_RESTRICT previous,
        const float *const SFEM_RESTRICT direction,
        float *const SFEM_RESTRICT output
) {
    return mooney_rivlin_kelvin_voigt_newmark_viscous_tet10_jacobian_action_isoparametric_mesh_soa_float(nelements, nnodes, elements, points, parameters[0], parameters[1], parameters[2], 3, current + 0, current + 1, current + 2, 3, previous + 0, previous + 1, previous + 2, 3, direction + 0, direction + 1, direction + 2, 3, output + 0, output + 1, output + 2);
}
