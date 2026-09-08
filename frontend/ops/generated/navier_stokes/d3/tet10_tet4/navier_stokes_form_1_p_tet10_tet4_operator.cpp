#include <type_traits>
#include "../navier_stokes_form_1_p_d3_simplex_mixed_local.hpp"
#include "../../../kernel_math.hpp"
#include "../../../geometry_kernels.hpp"
#include "../../../kernel_diagnostics.hpp"

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

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
struct navier_stokes_form_1_p_affine_reference_data {
    static const s_t *q_weight() {
        static const s_t data[11] = {s_t(-0.013155555555555556), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887)};
        return data;
    }
    static const s_t *tet10_shape() {
        static const s_t data[110] = {s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.44897959183673491), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.22448979591836737), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.020408163265306121), s_t(-0.061224489795918387), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836743), s_t(0.22448979591836732), s_t(0.020408163265306128), s_t(0.020408163265306128), s_t(0.22448979591836732), s_t(0.020408163265306121), s_t(-0.061224489795918401), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(0.020408163265306135), s_t(0.22448979591836732), s_t(0.22448979591836751), s_t(0.020408163265306135), s_t(0.020408163265306121), s_t(0.22448979591836732), s_t(-0.061224489795918421), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(0.020408163265306145), s_t(0.020408163265306121), s_t(0.020408163265306145), s_t(0.2244897959183676), s_t(0.22448979591836732), s_t(0.22448979591836732), s_t(-0.080357142857142821), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428564), s_t(0.63809286661931275), s_t(0.16071428571428564), s_t(0.040478561952115862), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.040478561952115875), s_t(0.1607142857142857), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(0.040478561952115875), s_t(0.16071428571428573), s_t(0.1607142857142857), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573)};
        return data;
    }
    static const s_t *tet10_grad_ref_x() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(-1), s_t(1), s_t(0), s_t(-2.1428571428571428), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(2.1428571428571428), s_t(0), s_t(0), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(3.1428571428571428), s_t(-3.1428571428571428), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571441), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(0), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719645), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0)};
        return data;
    }
    static const s_t *tet10_grad_ref_y() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(2.8571428571428572), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(2.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(-2.8571428571428568), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571441), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(3.1428571428571428), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(-1.1952286093343947), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(-1.1952286093343938), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(1.1952286093343933), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719645), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(1.1952286093343933), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969)};
        return data;
    }
    static const s_t *tet10_grad_ref_z() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(3.1428571428571428), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(0.2857142857142857), s_t(3.1428571428571428), s_t(0.71428571428571441), s_t(0), s_t(0), s_t(2.1428571428571428), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719645), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.40238569533280322)};
        return data;
    }
    static const s_t *tet4_shape() {
        static const s_t data[44] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.78571428571428581), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.071428571428571452), s_t(0.7857142857142857), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.07142857142857148), s_t(0.071428571428571425), s_t(0.7857142857142857), s_t(0.071428571428571425), s_t(0.071428571428571508), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.7857142857142857), s_t(0.10059642383320075), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.10059642383320078), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.10059642383320078), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.1005964238332008), s_t(0.39940357616679922)};
        return data;
    }
    static const s_t *tet4_grad_ref_x() {
        static const s_t data[44] = {s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0)};
        return data;
    }
    static const s_t *tet4_grad_ref_y() {
        static const s_t data[44] = {s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *tet4_grad_ref_z() {
        static const s_t data[44] = {s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1)};
        return data;
    }
};

template <typename s_t>
struct navier_stokes_form_1_p_isoparametric_reference_data {
    static const s_t *q_weight() {
        static const s_t data[11] = {s_t(-0.013155555555555556), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.0076222222222222221), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887), s_t(0.024888888888888887)};
        return data;
    }
    static const s_t *tet10_shape() {
        static const s_t data[110] = {s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(-0.125), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.44897959183673491), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.22448979591836737), s_t(0.22448979591836737), s_t(0.020408163265306121), s_t(0.020408163265306121), s_t(-0.061224489795918387), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.22448979591836743), s_t(0.22448979591836732), s_t(0.020408163265306128), s_t(0.020408163265306128), s_t(0.22448979591836732), s_t(0.020408163265306121), s_t(-0.061224489795918401), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(-0.061224489795918366), s_t(0.020408163265306135), s_t(0.22448979591836732), s_t(0.22448979591836751), s_t(0.020408163265306135), s_t(0.020408163265306121), s_t(0.22448979591836732), s_t(-0.061224489795918421), s_t(-0.061224489795918366), s_t(-0.061224489795918366), s_t(0.44897959183673464), s_t(0.020408163265306145), s_t(0.020408163265306121), s_t(0.020408163265306145), s_t(0.2244897959183676), s_t(0.22448979591836732), s_t(0.22448979591836732), s_t(-0.080357142857142821), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428564), s_t(0.63809286661931275), s_t(0.16071428571428564), s_t(0.040478561952115862), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.040478561952115875), s_t(0.1607142857142857), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(0.040478561952115875), s_t(0.16071428571428573), s_t(0.1607142857142857), s_t(0.1607142857142857), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(-0.080357142857142849), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(0.16071428571428573), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(-0.080357142857142849), s_t(-0.080357142857142863), s_t(-0.080357142857142863), s_t(-0.080357142857142849), s_t(0.16071428571428573), s_t(0.040478561952115882), s_t(0.16071428571428573), s_t(0.63809286661931275), s_t(0.16071428571428573), s_t(0.16071428571428573)};
        return data;
    }
    static const s_t *tet10_grad_ref_x() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(-1), s_t(1), s_t(0), s_t(-2.1428571428571428), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(2.1428571428571428), s_t(0), s_t(0), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571397), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(3.1428571428571428), s_t(-3.1428571428571428), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(0.71428571428571441), s_t(-0.7142857142857143), s_t(0), s_t(0), s_t(0), s_t(0.2857142857142857), s_t(-0.2857142857142857), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(0), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-0.59761430466719689), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719689), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(-1.5976143046671969), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.59761430466719645), s_t(-0.59761430466719678), s_t(0), s_t(0), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(-0.40238569533280322), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0)};
        return data;
    }
    static const s_t *tet10_grad_ref_y() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(2.8571428571428572), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-3.1428571428571428), s_t(3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(2.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(-2.8571428571428568), s_t(-0.2857142857142857), s_t(0), s_t(0.2857142857142857), s_t(0.71428571428571441), s_t(0), s_t(-0.7142857142857143), s_t(0), s_t(-0.2857142857142857), s_t(0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(3.1428571428571428), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(-1.1952286093343947), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(-1.1952286093343938), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-1.5976143046671969), s_t(1.5976143046671969), s_t(1.1952286093343933), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0.59761430466719689), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(-0.59761430466719645), s_t(0), s_t(-0.59761430466719678), s_t(0), s_t(-0.40238569533280322), s_t(0.40238569533280322), s_t(1.1952286093343933), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969)};
        return data;
    }
    static const s_t *tet10_grad_ref_z() {
        static const s_t data[110] = {s_t(0), s_t(0), s_t(0), s_t(0), s_t(-1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(1), s_t(-2.1428571428571428), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(2.8571428571428572), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-3.1428571428571428), s_t(0), s_t(-0.2857142857142857), s_t(0), s_t(3.1428571428571428), s_t(0.2857142857142857), s_t(0.71428571428571397), s_t(0), s_t(0), s_t(-0.7142857142857143), s_t(-0.2857142857142857), s_t(0), s_t(-3.1428571428571428), s_t(0), s_t(0.2857142857142857), s_t(3.1428571428571428), s_t(0.71428571428571441), s_t(0), s_t(0), s_t(2.1428571428571428), s_t(-0.2857142857142857), s_t(0), s_t(-0.2857142857142857), s_t(-2.8571428571428568), s_t(0.2857142857142857), s_t(0.2857142857142857), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-1.5976143046671969), s_t(0), s_t(1.5976143046671969), s_t(1.5976143046671969), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(-1.1952286093343947), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(0.59761430466719689), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(-1.1952286093343938), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-1.5976143046671969), s_t(0), s_t(-0.40238569533280322), s_t(1.1952286093343933), s_t(1.5976143046671969), s_t(0.40238569533280322), s_t(-0.59761430466719689), s_t(0), s_t(0), s_t(-0.59761430466719678), s_t(-0.40238569533280322), s_t(0), s_t(-1.5976143046671969), s_t(1.1952286093343933), s_t(0.40238569533280322), s_t(1.5976143046671969), s_t(-0.59761430466719645), s_t(0), s_t(0), s_t(0.59761430466719689), s_t(-0.40238569533280322), s_t(0), s_t(-0.40238569533280322), s_t(0), s_t(0.40238569533280322), s_t(0.40238569533280322)};
        return data;
    }
    static const s_t *tet4_shape() {
        static const s_t data[44] = {s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.25), s_t(0.78571428571428581), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.071428571428571452), s_t(0.7857142857142857), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.07142857142857148), s_t(0.071428571428571425), s_t(0.7857142857142857), s_t(0.071428571428571425), s_t(0.071428571428571508), s_t(0.071428571428571425), s_t(0.071428571428571425), s_t(0.7857142857142857), s_t(0.10059642383320075), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.10059642383320078), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.10059642383320078), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.39940357616679922), s_t(0.1005964238332008), s_t(0.1005964238332008), s_t(0.39940357616679922)};
        return data;
    }
    static const s_t *tet4_grad_ref_x() {
        static const s_t data[44] = {s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0)};
        return data;
    }
    static const s_t *tet4_grad_ref_y() {
        static const s_t data[44] = {s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0)};
        return data;
    }
    static const s_t *tet4_grad_ref_z() {
        static const s_t data[44] = {s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_1_p_tet10_tet4_residual_element_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    32,
    52,
    1,
    0,
    0,
    0,
    0,
    0,
    36,
    11,
    92,
    0,
    0,
    7,
    21,
    10,
    616,
    11,
    0,
    34,
    0,
    34,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_p_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_p_tet10_tet4_residual_element_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_p_tet10_tet4_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    29,
    60,
    1,
    0,
    0,
    0,
    0,
    0,
    33,
    21,
    97,
    0,
    0,
    17,
    26,
    10,
    616,
    11,
    0,
    0,
    0,
    34,
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_1_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_mixed_impl(
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
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u_data[3],
        const s_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 11;
    static constexpr int CELL_NS = 10;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const field_shape[NC] = {sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet10_shape(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet4_shape()};
    const s_t *const fgref[NC * ND] = {sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet10_grad_ref_x(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet10_grad_ref_y(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet10_grad_ref_z(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet4_grad_ref_x(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet4_grad_ref_y(), sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::tet4_grad_ref_z()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_current[N_FIELD_STREAMS][VS];
        s_t block_output[N_FIELD_STREAMS][VS];

        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[0][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 10 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[1][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 20 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[2][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 4; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 30 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = p_data[node * current_stride];
            }
        }

        for (int stream = 0; stream < 34; ++stream) {
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
        const s_t *block_adjugate[ND * ND];
        for (int component = 0; component < ND * ND; ++component) {
            block_adjugate[component] = bageom_streams[component];
        }

        navier_stokes_form_1_p_d3_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, 0, bageom_streams[9], block_adjugate, field_shape, fgref, sfem::codegen::navier_stokes_form_1_p_affine_reference_data<s_t>::q_weight(), block_current, block_output);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 10 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 20 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 4; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 30 + local_shape;
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

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa(
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
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_soa_float(
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
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const s_t *const SFEM_RESTRICT u_data[3],
        const s_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 11;
    static constexpr int CELL_NS = 10;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VS = 16;
    (void)nnodes;
    const s_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_x();
    const s_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_y();
    const s_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_z();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evb = 0; evb < nelements; evb += VS) {
        const int nelems = (int)MIN((ptrdiff_t)VS, nelements - evb);
        s_t block_coordinates[ND * CELL_NS][VS];
        s_t block_adjugate_data[ND * ND][NQ * VS];
        s_t block_determinant[NQ * VS];
        s_t block_current[N_FIELD_STREAMS][VS];
        s_t block_output[N_FIELD_STREAMS][VS];

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

        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[0][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 10 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[1][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 20 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = u_data[2][node * current_stride];
            }
        }
        for (int local_shape = 0; local_shape < 4; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 30 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evb + lane];
                block_current[stream][lane] = p_data[node * current_stride];
            }
        }

        for (int stream = 0; stream < 34; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = s_t(0);
            }
        }

        s_t *block_adjugate_streams[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < NQ; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
                const s_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
                const s_t J02 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
                const s_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
                const s_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
                const s_t J12 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
                const s_t J20 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_0[q * CELL_NS + 9];
                const s_t J21 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_1[q * CELL_NS + 9];
                const s_t J22 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_2[q * CELL_NS + 9];
                geometry_jacobian_adjugate_and_determinant_3<s_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VS + lane);
            }
        }

        const s_t *const field_shape[NC] = {sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_shape(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet4_shape()};
        const s_t *const fgref[NC * ND] = {sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_x(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_y(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet10_grad_ref_z(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet4_grad_ref_x(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet4_grad_ref_y(), sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::tet4_grad_ref_z()};
        const s_t *const block_adjugate[ND * ND] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        navier_stokes_form_1_p_d3_simplex_mixed_residual_block_contiguous<s_t, NQ, CELL_NS, VS>(nelems, VS, block_determinant, block_adjugate, field_shape, fgref, sfem::codegen::navier_stokes_form_1_p_isoparametric_reference_data<s_t>::q_weight(), block_current, block_output);

        {
            s_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 10 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 20 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evb + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            s_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 4; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 30 + local_shape;
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

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_1_p_tet10_tet4_residual_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, current_stride, u_data, p_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
static SFEM_INLINE int navier_stokes_form_1_p_tet10_tet4_jacobian_action_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const g_t *const SFEM_RESTRICT g_jacobian_determinant0,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 11;
    static constexpr int CELL_NS = 10;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename s_t>
static SFEM_INLINE int navier_stokes_form_1_p_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        s_t *const SFEM_RESTRICT u_out[3],
        s_t *const SFEM_RESTRICT p_out
) {
    static constexpr int ND = 3;
    static constexpr int NQ = 11;
    static constexpr int CELL_NS = 10;
    static constexpr int NS = CELL_NS;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VS = 16;
    (void)nnodes;
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem
