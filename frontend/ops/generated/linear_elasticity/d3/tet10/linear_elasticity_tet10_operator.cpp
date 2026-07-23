#include <cstdio>
#include <type_traits>
#include "../linear_elasticity_d3_simplex_local.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <cstdlib>
#include "../../../packed_thread_scratch.hpp"
#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT,
        std::true_type) {
    return source;
}

template <typename scalar_t, typename jacobian_t, int VECTOR_SIZE>
SFEM_INLINE const scalar_t *affine_geometry_stream(
        const int nelems,
        const jacobian_t *const SFEM_RESTRICT source,
        scalar_t *const SFEM_RESTRICT converted,
        std::false_type) {
    #pragma omp simd
    for (int lane = 0; lane < nelems; ++lane) {
        converted[lane] = scalar_t(source[lane]);
    }
    return converted;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {


template <typename scalar_t>
struct linear_elasticity_tet10_affine_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[40] = {scalar_t(0.099999999999999936), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.32360679774997891), scalar_t(0.32360679774997891), scalar_t(0.076393202250021025), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.32360679774997886), scalar_t(0.32360679774997897), scalar_t(0.076393202250020983), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.076393202250021025), scalar_t(-0.099999999999999978), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(-0.099999999999999992), scalar_t(0.076393202250020983), scalar_t(0.32360679774997897), scalar_t(0.32360679774997886), scalar_t(0.076393202250020983), scalar_t(0.076393202250021025), scalar_t(0.32360679774997897), scalar_t(-0.099999999999999964), scalar_t(-0.099999999999999992), scalar_t(-0.099999999999999992), scalar_t(0.10000000000000007), scalar_t(0.07639320225002097), scalar_t(0.076393202250021025), scalar_t(0.07639320225002097), scalar_t(0.3236067977499788), scalar_t(0.32360679774997897), scalar_t(0.32360679774997897)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(-2.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(0.44721359549995832), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(-0.55278640450004202), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(2.3416407864998741)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[40] = {scalar_t(-1.3416407864998741), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(1.7888543819998319), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(2.3416407864998741), scalar_t(0.55278640450004202), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(-0.44721359549995798), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-2.3416407864998741), scalar_t(0), scalar_t(0.55278640450004202), scalar_t(2.3416407864998741), scalar_t(0.44721359549995832), scalar_t(0), scalar_t(0), scalar_t(1.3416407864998741), scalar_t(-0.55278640450004202), scalar_t(0), scalar_t(-0.55278640450004202), scalar_t(-1.7888543819998315), scalar_t(0.55278640450004202), scalar_t(0.55278640450004202)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[4] = {scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664), scalar_t(0.041666666666666664)};
        return data;
    }
};

template <typename scalar_t>
struct linear_elasticity_tet10_isoparametric_reference_data {
    static const scalar_t *shape() {
        static const scalar_t data[110] = {scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.44897959183673491), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.22448979591836737), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918387), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836743), scalar_t(0.22448979591836732), scalar_t(0.020408163265306128), scalar_t(0.020408163265306128), scalar_t(0.22448979591836732), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918401), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(0.020408163265306135), scalar_t(0.22448979591836732), scalar_t(0.22448979591836751), scalar_t(0.020408163265306135), scalar_t(0.020408163265306121), scalar_t(0.22448979591836732), scalar_t(-0.061224489795918421), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(0.020408163265306145), scalar_t(0.020408163265306121), scalar_t(0.020408163265306145), scalar_t(0.2244897959183676), scalar_t(0.22448979591836732), scalar_t(0.22448979591836732), scalar_t(-0.080357142857142821), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428564), scalar_t(0.63809286661931275), scalar_t(0.16071428571428564), scalar_t(0.040478561952115862), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.040478561952115875), scalar_t(0.1607142857142857), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(0.040478561952115875), scalar_t(0.16071428571428573), scalar_t(0.1607142857142857), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573)};
        return data;
    }
    static const scalar_t *grad_ref_x() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-2.1428571428571428), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(-3.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571441), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719645), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0)};
        return data;
    }
    static const scalar_t *grad_ref_y() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(-1.1952286093343947), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(-1.1952286093343938), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969)};
        return data;
    }
    static const scalar_t *grad_ref_z() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(3.1428571428571428), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.40238569533280322)};
        return data;
    }
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tet10_tet10_objective_soa_diagnostics_data = {
    "linear_elasticity_tet10_tet10_objective_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    11,
    6,
    0,
    0,
    7,
    0,
    0,
    0,
    6,
    1,
    24,
    0,
    0,
    0,
    11,
    10,
    330,
    11,
    2,
    30,
    0,
    1,
    1,
    1,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_tet10_objective_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data;
}

extern "C" double linear_elasticity_tet10_tet10_objective_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_tet10_tet10_objective_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_objective_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_objective_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_objective_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_objective_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_objective_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_objective_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_objective_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_objective_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_objective_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_objective_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int linear_elasticity_tet10_tet10_objective_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_steps_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_objective_steps_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_objective_steps_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_value[lane] = scalar_t(0);
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_value);

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            value[evbegin + lane] += block_value[lane];
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}

extern "C" int linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, value);
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const int nsteps,
        const scalar_t *const SFEM_RESTRICT steps,
        scalar_t *const SFEM_RESTRICT value
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_value[VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }

        const scalar_t *const u_components[DIM] = {ux, uy, uz};
        const scalar_t *const h_components[DIM] = {hx, hy, hz};
        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_base_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }

        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        for (int step = 0; step < nsteps; ++step) {
            const scalar_t alpha = steps[step];
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                for (int d = 0; d < DIM; ++d) {
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                    }
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_value[lane] = scalar_t(0);
            }

            linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_value);

            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

extern "C" int linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_objective_steps_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_objective_steps_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const int nsteps,
        const double *const SFEM_RESTRICT steps,
        double *const SFEM_RESTRICT value
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_objective_steps_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const int nsteps,
        const float *const SFEM_RESTRICT steps,
        float *const SFEM_RESTRICT value
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    (void)n_shared_nodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u_base = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_base_component = pack_u_base + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[k] = u_component[node * u_stride];
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_base_component[n_contiguous + k] = u_component[node * u_stride];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_u_base_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_value[VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};

                const scalar_t *block_u_streams[N_SHAPE * DIM] = {block_u_data[0], block_u_data[1], block_u_data[2], block_u_data[3], block_u_data[4], block_u_data[5], block_u_data[6], block_u_data[7], block_u_data[8], block_u_data[9], block_u_data[10], block_u_data[11], block_u_data[12], block_u_data[13], block_u_data[14], block_u_data[15], block_u_data[16], block_u_data[17], block_u_data[18], block_u_data[19], block_u_data[20], block_u_data[21], block_u_data[22], block_u_data[23], block_u_data[24], block_u_data[25], block_u_data[26], block_u_data[27], block_u_data[28], block_u_data[29]};

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_base_data[shape * DIM + d][lane] = pack_u_base[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                for (int step = 0; step < nsteps; ++step) {
                    const scalar_t alpha = steps[step];
                    for (int shape = 0; shape < N_SHAPE; ++shape) {
                        for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                            for (int lane = 0; lane < nelems; ++lane) {
                                block_u_data[shape * DIM + d][lane] = block_u_base_data[shape * DIM + d][lane] + alpha * block_h_data[shape * DIM + d][lane];
                            }
                        }
                    }
#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        block_value[lane] = scalar_t(0);
                    }

                    linear_elasticity_d3_simplex_objective_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_value);

#pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        value[(ptrdiff_t)step * nelements + evbegin + lane] = block_value[lane];
                    }
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data = {
    "linear_elasticity_tet10_tet10_gradient_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    8,
    8,
    0,
    0,
    0,
    0,
    0,
    0,
    6,
    14,
    16,
    0,
    0,
    5,
    8,
    10,
    330,
    11,
    2,
    30,
    0,
    30,
    30,
    30,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_tet10_gradient_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data;
}

extern "C" double linear_elasticity_tet10_tet10_gradient_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_tet10_tet10_gradient_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_gradient_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_gradient_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_gradient_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_gradient_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_gradient_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_gradient_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_gradient_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_gradient_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_gradient_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t u_stride,
        const scalar_t *const SFEM_RESTRICT ux,
        const scalar_t *const SFEM_RESTRICT uy,
        const scalar_t *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const u_components[DIM] = {ux, uy, uz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_u_data[shape * DIM + d][lane] = u_components[d][node * u_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_u_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_u_streams[stream] = block_u_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_gradient_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_gradient_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t u_stride,
        const double *const SFEM_RESTRICT ux,
        const double *const SFEM_RESTRICT uy,
        const double *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_gradient_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t u_stride,
        const float *const SFEM_RESTRICT ux,
        const float *const SFEM_RESTRICT uy,
        const float *const SFEM_RESTRICT uz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_u = sfem::codegen::thread_scratch<scalar_t>(1, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const u_components[DIM] = {ux, uy, uz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_u_component = pack_u + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT u_component = u_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_u_component[k] = u_component[node * u_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_u_component[n_contiguous + k] = u_component[node * u_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_u_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_u_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_u_streams[stream] = block_u_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_u_data[shape * DIM + d][lane] = pack_u[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                linear_elasticity_d3_simplex_gradient_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_u_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

static const KernelDiagnostics linear_elasticity_tet10_tet10_apply_soa_diagnostics_data = {
    "linear_elasticity_tet10_tet10_apply_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    8,
    8,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    14,
    16,
    0,
    0,
    5,
    8,
    10,
    330,
    11,
    2,
    0,
    30,
    30,
    30,
    30,
    1.0,
    1.0,
    8.0,
    12.0,
    16.0,
    20.0,
    20.0,
    24.0,
    1.0,
    1.0
};

} // namespace codegen
} // namespace sfem

extern "C" const sfem::codegen::KernelDiagnostics *linear_elasticity_tet10_tet10_apply_soa_diagnostics(void) {
    return &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data;
}

extern "C" double linear_elasticity_tet10_tet10_apply_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(&sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data, nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void linear_elasticity_tet10_tet10_apply_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_apply_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_apply_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "linear_elasticity_tet10_tet10_apply_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_apply_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_apply_affine_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_apply_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "linear_elasticity_tet10_tet10_apply_affine_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_float",
            &sfem::codegen::linear_elasticity_tet10_tet10_apply_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}


namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_apply_affine_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate0,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate1,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate2,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate3,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate4,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate5,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate6,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate7,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate8,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }
        scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
        scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
        const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

        linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_apply_affine_mesh_soa(
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
        const double lmbda,
        const double mu,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_apply_affine_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_tet10_apply_affine_mesh_soa_float(
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
        const float lmbda,
        const float mu,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_apply_affine_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, lmbda, mu, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_apply_packed_affine_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const double lmbda,
        const double mu,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_apply_packed_affine_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
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
        const float lmbda,
        const float mu,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 4;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const scalar_t *const affine_grad_ref_x = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const affine_grad_ref_y = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const affine_grad_ref_z = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const affine_q_weight = sfem::codegen::linear_elasticity_tet10_affine_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }

                scalar_t block_jacobian_adjugate0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate0 + evbegin, block_jacobian_adjugate0_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate1_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate1 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate1 + evbegin, block_jacobian_adjugate1_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate2_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate2 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate2 + evbegin, block_jacobian_adjugate2_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate3_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate3 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate3 + evbegin, block_jacobian_adjugate3_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate4_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate4 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate4 + evbegin, block_jacobian_adjugate4_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate5_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate5 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate5 + evbegin, block_jacobian_adjugate5_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate6_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate6 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate6 + evbegin, block_jacobian_adjugate6_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate7_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate7 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate7 + evbegin, block_jacobian_adjugate7_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_adjugate8_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_adjugate8 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_adjugate8 + evbegin, block_jacobian_adjugate8_data, std::is_same<jacobian_t, scalar_t>());
                scalar_t block_jacobian_determinant0_data[VECTOR_SIZE];
                const scalar_t *const block_jacobian_determinant0 = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                        nelems, g_jacobian_determinant0 + evbegin, block_jacobian_determinant0_data, std::is_same<jacobian_t, scalar_t>());

                linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, affine_grad_ref_x, affine_grad_ref_y, affine_grad_ref_z, affine_q_weight, lmbda, mu, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem


namespace sfem {
namespace codegen {

template <typename scalar_t, typename geometry_t>
static SFEM_INLINE int linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geometry_t *const *const SFEM_RESTRICT points,
        const scalar_t lmbda,
        const scalar_t mu,
        const ptrdiff_t h_stride,
        const scalar_t *const SFEM_RESTRICT hx,
        const scalar_t *const SFEM_RESTRICT hy,
        const scalar_t *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT outx,
        scalar_t *const SFEM_RESTRICT outy,
        scalar_t *const SFEM_RESTRICT outz
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const geometry_t *const SFEM_RESTRICT x = points[0];
    const geometry_t *const SFEM_RESTRICT y = points[1];
    const geometry_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * N_SHAPE];
        scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
        scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
        scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];

        for (int element_node = 0; element_node < N_SHAPE; ++element_node) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[element_node];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                ev[element_node * VECTOR_SIZE + lane] = element_shape[evbegin + lane];
            }
        }
        const geometry_t *const coordinate_components[DIM] = {x, y, z};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    block_coordinate_data[shape * DIM + d][lane] = coordinate_components[d][ev[shape * VECTOR_SIZE + lane]];
                }
            }
        }
        const scalar_t *const h_components[DIM] = {hx, hy, hz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = ev[shape * VECTOR_SIZE + lane];
                    block_h_data[shape * DIM + d][lane] = h_components[d][node * h_stride];
                }
            }
        }
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_out_data[stream][lane] = scalar_t(0);
            }
        }

        const scalar_t *block_h_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_h_streams[stream] = block_h_data[stream];
        }
        scalar_t *block_out_streams[N_SHAPE * DIM];
        for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
            block_out_streams[stream] = block_out_data[stream];
        }

        for (int q = 0; q < N_QP; ++q) {
            scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
            scalar_t J00_values[VECTOR_SIZE];
            scalar_t J01_values[VECTOR_SIZE];
            scalar_t J02_values[VECTOR_SIZE];
            scalar_t J10_values[VECTOR_SIZE];
            scalar_t J11_values[VECTOR_SIZE];
            scalar_t J12_values[VECTOR_SIZE];
            scalar_t J20_values[VECTOR_SIZE];
            scalar_t J21_values[VECTOR_SIZE];
            scalar_t J22_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J00_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J01_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J02_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J10_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J11_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J12_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J20_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J21_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                J22_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = J00_values[lane];
                const scalar_t J01 = J01_values[lane];
                const scalar_t J02 = J02_values[lane];
                const scalar_t J10 = J10_values[lane];
                const scalar_t J11 = J11_values[lane];
                const scalar_t J12 = J12_values[lane];
                const scalar_t J20 = J20_values[lane];
                const scalar_t J21 = J21_values[lane];
                const scalar_t J22 = J22_values[lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
            }
        }

        linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_h_streams, block_out_streams);

        scalar_t *const out_components[DIM] = {outx, outy, outz};

        for (int shape = 0; shape < N_SHAPE; ++shape) {
            for (int d = 0; d < DIM; ++d) {
                {
                    for (int scatter = 0; scatter < nelems; ++scatter) {
                        #pragma omp atomic update
                        out_components[d][ev[shape * VECTOR_SIZE + scatter] * out_stride] += block_out_data[shape * DIM + d][scatter];
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_impl<double, geom_t>(nelements, nnodes, elements, points, lmbda, mu, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

extern "C" int linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    return sfem::codegen::linear_elasticity_tet10_tet10_apply_isoparametric_mesh_soa_impl<float, geom_t>(nelements, nnodes, elements, points, lmbda, mu, h_stride, hx, hy, hz, out_stride, outx, outy, outz);
}

namespace sfem {
namespace codegen {

extern "C" int linear_elasticity_tet10_tet10_apply_packed_isoparametric_mesh_soa(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const double lmbda,
        const double mu,
        const ptrdiff_t h_stride,
        const double *const SFEM_RESTRICT hx,
        const double *const SFEM_RESTRICT hy,
        const double *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT outx,
        double *const SFEM_RESTRICT outy,
        double *const SFEM_RESTRICT outz
) {
    using scalar_t = double;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

extern "C" int linear_elasticity_tet10_tet10_apply_packed_isoparametric_mesh_soa_float(
        const ptrdiff_t n_packs,
        const ptrdiff_t n_elements_per_pack,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        const ptrdiff_t max_nodes_per_pack,
        uint16_t **const SFEM_RESTRICT elements,
        const ptrdiff_t *const SFEM_RESTRICT owned_nodes_ptr,
        const ptrdiff_t *const SFEM_RESTRICT n_shared_nodes,
        const ptrdiff_t *const SFEM_RESTRICT ghost_ptr,
        const idx_t *const SFEM_RESTRICT ghost_idx,
        const geom_t *const *const SFEM_RESTRICT points,
        const float lmbda,
        const float mu,
        const ptrdiff_t h_stride,
        const float *const SFEM_RESTRICT hx,
        const float *const SFEM_RESTRICT hy,
        const float *const SFEM_RESTRICT hz,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT outx,
        float *const SFEM_RESTRICT outy,
        float *const SFEM_RESTRICT outz
) {
    using scalar_t = float;
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int N_SHAPE = 10;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];
    const scalar_t *const isoparametric_grad_ref_x = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_x();
    const scalar_t *const isoparametric_grad_ref_y = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_y();
    const scalar_t *const isoparametric_grad_ref_z = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::grad_ref_z();
    const scalar_t *const isoparametric_q_weight = sfem::codegen::linear_elasticity_tet10_isoparametric_reference_data<scalar_t>::q_weight();

#pragma omp parallel
    {
        scalar_t *const SFEM_RESTRICT pack_coordinates = sfem::codegen::thread_scratch<scalar_t>(0, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_h = sfem::codegen::thread_scratch<scalar_t>(2, (size_t)DIM * (size_t)max_nodes_per_pack);
        scalar_t *const SFEM_RESTRICT pack_out = sfem::codegen::thread_scratch<scalar_t>(3, (size_t)DIM * (size_t)max_nodes_per_pack);

#pragma omp for schedule(static)
        for (ptrdiff_t pack = 0; pack < n_packs; ++pack) {
            const ptrdiff_t e_start = pack * n_elements_per_pack;
            const ptrdiff_t e_end = MIN(nelements, (pack + 1) * n_elements_per_pack);
            const ptrdiff_t n_contiguous = owned_nodes_ptr[pack + 1] - owned_nodes_ptr[pack];
            const ptrdiff_t n_shared = n_shared_nodes[pack];
            const ptrdiff_t n_not_shared = n_contiguous - n_shared;
            const ptrdiff_t n_ghost = ghost_ptr[pack + 1] - ghost_ptr[pack];
            const ptrdiff_t n_pack_nodes = n_contiguous + n_ghost;
            const idx_t *const SFEM_RESTRICT ghosts = &ghost_idx[ghost_ptr[pack]];
            const geom_t *const coordinate_components[DIM] = {x, y, z};
            const scalar_t *const h_components[DIM] = {hx, hy, hz};
            scalar_t *const out_components[DIM] = {outx, outy, outz};
            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_coordinate = pack_coordinates + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT pack_h_component = pack_h + d * max_nodes_per_pack;
                const geom_t *const SFEM_RESTRICT coordinate_component = coordinate_components[d];
                const scalar_t *const SFEM_RESTRICT h_component = h_components[d];
                for (ptrdiff_t k = 0; k < n_pack_nodes; ++k) {
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_contiguous; ++k) {
                    const idx_t node = owned_nodes_ptr[pack] + k;
                    pack_coordinate[k] = scalar_t(coordinate_component[node]);
                    pack_h_component[k] = h_component[node * h_stride];
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
                    const idx_t node = ghosts[k];
                    pack_coordinate[n_contiguous + k] = scalar_t(coordinate_component[node]);
                    pack_h_component[n_contiguous + k] = h_component[node * h_stride];
                }
            }

            for (ptrdiff_t evbegin = e_start; evbegin < e_end; evbegin += VECTOR_SIZE) {
                const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, e_end - evbegin);
                scalar_t block_h_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_out_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_coordinate_data[N_SHAPE * DIM][VECTOR_SIZE];
                scalar_t block_jacobian_adjugate0[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate1[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate2[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate3[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate4[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate5[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate6[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate7[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_adjugate8[N_QP * VECTOR_SIZE];
                scalar_t block_jacobian_determinant0[N_QP * VECTOR_SIZE];
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                const scalar_t *block_h_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_h_streams[stream] = block_h_data[stream];
                }
                scalar_t *block_out_streams[N_SHAPE * DIM];
                for (int stream = 0; stream < N_SHAPE * DIM; ++stream) {
                    block_out_streams[stream] = block_out_data[stream];
                }

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
#pragma omp simd
                        for (int lane = 0; lane < nelems; ++lane) {
                            const uint16_t packed_node = element_shape[evbegin + lane];
                            block_coordinate_data[shape * DIM + d][lane] = pack_coordinates[d * max_nodes_per_pack + packed_node];
                            block_h_data[shape * DIM + d][lane] = pack_h[d * max_nodes_per_pack + packed_node];
                            block_out_data[shape * DIM + d][lane] = scalar_t(0);
                        }
                    }
                }


                for (int q = 0; q < N_QP; ++q) {
                scalar_t *block_jacobian_adjugate_streams[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
                scalar_t J00_values[VECTOR_SIZE];
                scalar_t J01_values[VECTOR_SIZE];
                scalar_t J02_values[VECTOR_SIZE];
                scalar_t J10_values[VECTOR_SIZE];
                scalar_t J11_values[VECTOR_SIZE];
                scalar_t J12_values[VECTOR_SIZE];
                scalar_t J20_values[VECTOR_SIZE];
                scalar_t J21_values[VECTOR_SIZE];
                scalar_t J22_values[VECTOR_SIZE];
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J00_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J01_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J02_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J10_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J11_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J12_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J20_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J21_values[lane] = scalar_t(0);
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    J22_values[lane] = scalar_t(0);
                }
                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const scalar_t g0 = isoparametric_grad_ref_x[q * N_SHAPE + shape];
                    const scalar_t g1 = isoparametric_grad_ref_y[q * N_SHAPE + shape];
                    const scalar_t g2 = isoparametric_grad_ref_z[q * N_SHAPE + shape];
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J00_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J01_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J02_values[lane] += block_coordinate_data[shape * 3 + 0][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J10_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J11_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J12_values[lane] += block_coordinate_data[shape * 3 + 1][lane] * g2;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J20_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g0;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J21_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g1;
                    }
                    #pragma omp simd
                    for (int lane = 0; lane < nelems; ++lane) {
                        J22_values[lane] += block_coordinate_data[shape * 3 + 2][lane] * g2;
                    }
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const scalar_t J00 = J00_values[lane];
                    const scalar_t J01 = J01_values[lane];
                    const scalar_t J02 = J02_values[lane];
                    const scalar_t J10 = J10_values[lane];
                    const scalar_t J11 = J11_values[lane];
                    const scalar_t J12 = J12_values[lane];
                    const scalar_t J20 = J20_values[lane];
                    const scalar_t J21 = J21_values[lane];
                    const scalar_t J22 = J22_values[lane];
                    geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                            J00, J01, J02, J10, J11, J12, J20, J21, J22,
                            block_jacobian_adjugate_streams, block_jacobian_determinant0, q * VECTOR_SIZE + lane);
                }
                }

                linear_elasticity_d3_simplex_apply_block<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8, block_jacobian_determinant0, isoparametric_grad_ref_x, isoparametric_grad_ref_y, isoparametric_grad_ref_z, isoparametric_q_weight, lmbda, mu, block_h_streams, block_out_streams);

                for (int shape = 0; shape < N_SHAPE; ++shape) {
                    const uint16_t *const SFEM_RESTRICT element_shape = elements[shape];
                    for (int d = 0; d < DIM; ++d) {
                        scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                        for (int lane = 0; lane < nelems; ++lane) {
                            pack_component_out[element_shape[evbegin + lane]] += block_out_data[shape * DIM + d][lane];
                        }
                    }
                }
            }

            for (int d = 0; d < DIM; ++d) {
                scalar_t *const SFEM_RESTRICT pack_component_out = pack_out + d * max_nodes_per_pack;
                scalar_t *const SFEM_RESTRICT global_out = out_components[d];
                for (ptrdiff_t k = 0; k < n_not_shared; ++k) {
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = n_not_shared; k < n_contiguous; ++k) {
#pragma omp atomic update
                    global_out[(owned_nodes_ptr[pack] + k) * out_stride] += pack_component_out[k];
                    pack_component_out[k] = scalar_t(0);
                }
                for (ptrdiff_t k = 0; k < n_ghost; ++k) {
#pragma omp atomic update
                    global_out[ghosts[k] * out_stride] += pack_component_out[n_contiguous + k];
                    pack_component_out[n_contiguous + k] = scalar_t(0);
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
