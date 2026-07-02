#include <type_traits>
#include "../poro_hyperelasticity_poro_d3_simplex_mixed_local.hpp"
#include "../../kernel_math.hpp"
#include "../../geometry_kernels.hpp"
#include "../../kernel_diagnostics.hpp"

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
typedef double geom_t;
#endif
#ifdef _OPENMP
#include <omp.h>
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
struct poro_hyperelasticity_poro_affine_reference_data {
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
    static const scalar_t *tet10_shape() {
        static const scalar_t data[110] = {scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.44897959183673491), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.22448979591836737), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918387), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836743), scalar_t(0.22448979591836732), scalar_t(0.020408163265306128), scalar_t(0.020408163265306128), scalar_t(0.22448979591836732), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918401), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(0.020408163265306135), scalar_t(0.22448979591836732), scalar_t(0.22448979591836751), scalar_t(0.020408163265306135), scalar_t(0.020408163265306121), scalar_t(0.22448979591836732), scalar_t(-0.061224489795918421), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(0.020408163265306145), scalar_t(0.020408163265306121), scalar_t(0.020408163265306145), scalar_t(0.2244897959183676), scalar_t(0.22448979591836732), scalar_t(0.22448979591836732), scalar_t(-0.080357142857142821), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428564), scalar_t(0.63809286661931275), scalar_t(0.16071428571428564), scalar_t(0.040478561952115862), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.040478561952115875), scalar_t(0.1607142857142857), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(0.040478561952115875), scalar_t(0.16071428571428573), scalar_t(0.1607142857142857), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_x() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-2.1428571428571428), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(-3.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571441), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719645), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_y() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(-1.1952286093343947), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(-1.1952286093343938), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_z() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(3.1428571428571428), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.40238569533280322)};
        return data;
    }
    static const scalar_t *tet4_shape() {
        static const scalar_t data[44] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.78571428571428581), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571452), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.07142857142857148), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571508), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.10059642383320075), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.10059642383320078), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.10059642383320078), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_x() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_y() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_z() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
};

template <typename scalar_t>
struct poro_hyperelasticity_poro_isoparametric_reference_data {
    static const scalar_t *q_weight() {
        static const scalar_t data[11] = {scalar_t(-0.013155555555555556), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.0076222222222222221), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887), scalar_t(0.024888888888888887)};
        return data;
    }
    static const scalar_t *tet10_shape() {
        static const scalar_t data[110] = {scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(-0.125), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.44897959183673491), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.22448979591836737), scalar_t(0.22448979591836737), scalar_t(0.020408163265306121), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918387), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.22448979591836743), scalar_t(0.22448979591836732), scalar_t(0.020408163265306128), scalar_t(0.020408163265306128), scalar_t(0.22448979591836732), scalar_t(0.020408163265306121), scalar_t(-0.061224489795918401), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(-0.061224489795918366), scalar_t(0.020408163265306135), scalar_t(0.22448979591836732), scalar_t(0.22448979591836751), scalar_t(0.020408163265306135), scalar_t(0.020408163265306121), scalar_t(0.22448979591836732), scalar_t(-0.061224489795918421), scalar_t(-0.061224489795918366), scalar_t(-0.061224489795918366), scalar_t(0.44897959183673464), scalar_t(0.020408163265306145), scalar_t(0.020408163265306121), scalar_t(0.020408163265306145), scalar_t(0.2244897959183676), scalar_t(0.22448979591836732), scalar_t(0.22448979591836732), scalar_t(-0.080357142857142821), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428564), scalar_t(0.63809286661931275), scalar_t(0.16071428571428564), scalar_t(0.040478561952115862), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.040478561952115875), scalar_t(0.1607142857142857), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(0.040478561952115875), scalar_t(0.16071428571428573), scalar_t(0.1607142857142857), scalar_t(0.1607142857142857), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(-0.080357142857142849), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142863), scalar_t(-0.080357142857142849), scalar_t(0.16071428571428573), scalar_t(0.040478561952115882), scalar_t(0.16071428571428573), scalar_t(0.63809286661931275), scalar_t(0.16071428571428573), scalar_t(0.16071428571428573)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_x() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-2.1428571428571428), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571397), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(-3.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(0.71428571428571441), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(-0.2857142857142857), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719689), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(-1.5976143046671969), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.59761430466719645), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(0), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(-0.40238569533280322), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_y() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(-1.1952286093343947), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(-1.1952286093343938), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969)};
        return data;
    }
    static const scalar_t *tet10_grad_ref_z() {
        static const scalar_t data[110] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(1), scalar_t(-2.1428571428571428), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(2.8571428571428572), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(3.1428571428571428), scalar_t(0.2857142857142857), scalar_t(0.71428571428571397), scalar_t(0), scalar_t(0), scalar_t(-0.7142857142857143), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-3.1428571428571428), scalar_t(0), scalar_t(0.2857142857142857), scalar_t(3.1428571428571428), scalar_t(0.71428571428571441), scalar_t(0), scalar_t(0), scalar_t(2.1428571428571428), scalar_t(-0.2857142857142857), scalar_t(0), scalar_t(-0.2857142857142857), scalar_t(-2.8571428571428568), scalar_t(0.2857142857142857), scalar_t(0.2857142857142857), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(1.5976143046671969), scalar_t(1.5976143046671969), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(-1.1952286093343947), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(-1.1952286093343938), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-1.5976143046671969), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(1.1952286093343933), scalar_t(1.5976143046671969), scalar_t(0.40238569533280322), scalar_t(-0.59761430466719689), scalar_t(0), scalar_t(0), scalar_t(-0.59761430466719678), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-1.5976143046671969), scalar_t(1.1952286093343933), scalar_t(0.40238569533280322), scalar_t(1.5976143046671969), scalar_t(-0.59761430466719645), scalar_t(0), scalar_t(0), scalar_t(0.59761430466719689), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(-0.40238569533280322), scalar_t(0), scalar_t(0.40238569533280322), scalar_t(0.40238569533280322)};
        return data;
    }
    static const scalar_t *tet4_shape() {
        static const scalar_t data[44] = {scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.25), scalar_t(0.78571428571428581), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.071428571428571452), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.07142857142857148), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.071428571428571425), scalar_t(0.071428571428571508), scalar_t(0.071428571428571425), scalar_t(0.071428571428571425), scalar_t(0.7857142857142857), scalar_t(0.10059642383320075), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.10059642383320078), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.10059642383320078), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922), scalar_t(0.1005964238332008), scalar_t(0.1005964238332008), scalar_t(0.39940357616679922)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_x() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0), scalar_t(-1), scalar_t(1), scalar_t(0), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_y() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0), scalar_t(-1), scalar_t(0), scalar_t(1), scalar_t(0)};
        return data;
    }
    static const scalar_t *tet4_grad_ref_z() {
        static const scalar_t data[44] = {scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1), scalar_t(-1), scalar_t(0), scalar_t(0), scalar_t(1)};
        return data;
    }
};

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_tet10_tet4_residual_element_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    10,
    20,
    1,
    0,
    0,
    0,
    0,
    0,
    22,
    5,
    38,
    0,
    0,
    1,
    19,
    10,
    616,
    11,
    4,
    68,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_tet10_tet4_residual_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data = {
    "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa",
    "TET10",
    3,
    11,
    10,
    16,
    4,
    6,
    20,
    1,
    0,
    0,
    0,
    0,
    0,
    18,
    7,
    34,
    0,
    0,
    3,
    14,
    10,
    616,
    11,
    4,
    0,
    34,
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

extern "C" const sfem::codegen::KernelDiagnostics *poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs,
        const int repeat) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs, repeat,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_mixed_impl(
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
        const scalar_t alpha,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u_data[3],
        const scalar_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[3],
        const scalar_t *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int CELL_N_SHAPE = 10;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_shape(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_z()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 9] = elements[9][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_current[0][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[0][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[1][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[1][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[2][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[2][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[3][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[3][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[4][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[4][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[5][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[5][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[6][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[6][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[7][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[7][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[8][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[8][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[9][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[9][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[10][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[10][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[11][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[11][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[12][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[12][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[13][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[13][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[14][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[14][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[15][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[15][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[16][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[16][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[17][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[17][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[18][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[18][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[19][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[19][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[20][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[20][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[21][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[21][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[22][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[22][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[23][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[23][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[24][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[24][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[25][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[25][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[26][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[26][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[27][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[27][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[28][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[28][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[29][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[29][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[30][lane] = p_data[ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[30][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[31][lane] = p_data[ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[31][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[32][lane] = p_data[ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[32][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[33][lane] = p_data[ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[33][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 3] * previous_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
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
        const scalar_t *const block_adjugate[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        const scalar_t *const block_current_streams[N_FIELD_STREAMS] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33]};
        const scalar_t *const block_previous_streams[N_FIELD_STREAMS] = {block_previous[0], block_previous[1], block_previous[2], block_previous[3], block_previous[4], block_previous[5], block_previous[6], block_previous[7], block_previous[8], block_previous[9], block_previous[10], block_previous[11], block_previous[12], block_previous[13], block_previous[14], block_previous[15], block_previous[16], block_previous[17], block_previous[18], block_previous[19], block_previous[20], block_previous[21], block_previous[22], block_previous[23], block_previous[24], block_previous[25], block_previous[26], block_previous[27], block_previous[28], block_previous[29], block_previous[30], block_previous[31], block_previous[32], block_previous[33]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33]};

        poro_hyperelasticity_poro_d3_simplex_mixed_residual_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::q_weight(), block_current_streams, block_previous_streams, alpha, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[33][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa(
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
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, dt, hydraulic_conductivity, storage, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_soa_float(
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
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, dt, hydraulic_conductivity, storage, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t alpha,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t current_stride,
        const scalar_t *const SFEM_RESTRICT u_data[3],
        const scalar_t *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const scalar_t *const SFEM_RESTRICT u_old_data[3],
        const scalar_t *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int CELL_N_SHAPE = 10;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y();
    const scalar_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_current[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_previous[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 9] = elements[9][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[2][lane] = points[2][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[3][lane] = points[0][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[4][lane] = points[1][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[5][lane] = points[2][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[6][lane] = points[0][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[7][lane] = points[1][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[8][lane] = points[2][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[9][lane] = points[0][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[10][lane] = points[1][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[11][lane] = points[2][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[12][lane] = points[0][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[13][lane] = points[1][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[14][lane] = points[2][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[15][lane] = points[0][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[16][lane] = points[1][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[17][lane] = points[2][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[18][lane] = points[0][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[19][lane] = points[1][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[20][lane] = points[2][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[21][lane] = points[0][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[22][lane] = points[1][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[23][lane] = points[2][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[24][lane] = points[0][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[25][lane] = points[1][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[26][lane] = points[2][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[27][lane] = points[0][ev[lane * CELL_N_SHAPE + 9]];
            block_coordinates[28][lane] = points[1][ev[lane * CELL_N_SHAPE + 9]];
            block_coordinates[29][lane] = points[2][ev[lane * CELL_N_SHAPE + 9]];
            block_current[0][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[0][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[1][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[1][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[2][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[2][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[3][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[3][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[4][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[4][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[5][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[5][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[6][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[6][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[7][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[7][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[8][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[8][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[9][lane] = u_data[0][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[9][lane] = u_old_data[0][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[10][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[10][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[11][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[11][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[12][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[12][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[13][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[13][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[14][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[14][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[15][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[15][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[16][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[16][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[17][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[17][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[18][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[18][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[19][lane] = u_data[1][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[19][lane] = u_old_data[1][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[20][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[20][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[21][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[21][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[22][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[22][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[23][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[23][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 3] * previous_stride];
            block_current[24][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 4] * current_stride];
            block_previous[24][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 4] * previous_stride];
            block_current[25][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 5] * current_stride];
            block_previous[25][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 5] * previous_stride];
            block_current[26][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 6] * current_stride];
            block_previous[26][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 6] * previous_stride];
            block_current[27][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 7] * current_stride];
            block_previous[27][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 7] * previous_stride];
            block_current[28][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 8] * current_stride];
            block_previous[28][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 8] * previous_stride];
            block_current[29][lane] = u_data[2][ev[lane * CELL_N_SHAPE + 9] * current_stride];
            block_previous[29][lane] = u_old_data[2][ev[lane * CELL_N_SHAPE + 9] * previous_stride];
            block_current[30][lane] = p_data[ev[lane * CELL_N_SHAPE + 0] * current_stride];
            block_previous[30][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 0] * previous_stride];
            block_current[31][lane] = p_data[ev[lane * CELL_N_SHAPE + 1] * current_stride];
            block_previous[31][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 1] * previous_stride];
            block_current[32][lane] = p_data[ev[lane * CELL_N_SHAPE + 2] * current_stride];
            block_previous[32][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 2] * previous_stride];
            block_current[33][lane] = p_data[ev[lane * CELL_N_SHAPE + 3] * current_stride];
            block_previous[33][lane] = p_old_data[ev[lane * CELL_N_SHAPE + 3] * previous_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_shape(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_z()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_current_streams[N_FIELD_STREAMS] = {block_current[0], block_current[1], block_current[2], block_current[3], block_current[4], block_current[5], block_current[6], block_current[7], block_current[8], block_current[9], block_current[10], block_current[11], block_current[12], block_current[13], block_current[14], block_current[15], block_current[16], block_current[17], block_current[18], block_current[19], block_current[20], block_current[21], block_current[22], block_current[23], block_current[24], block_current[25], block_current[26], block_current[27], block_current[28], block_current[29], block_current[30], block_current[31], block_current[32], block_current[33]};
        const scalar_t *const block_previous_streams[N_FIELD_STREAMS] = {block_previous[0], block_previous[1], block_previous[2], block_previous[3], block_previous[4], block_previous[5], block_previous[6], block_previous[7], block_previous[8], block_previous[9], block_previous[10], block_previous[11], block_previous[12], block_previous[13], block_previous[14], block_previous[15], block_previous[16], block_previous[17], block_previous[18], block_previous[19], block_previous[20], block_previous[21], block_previous[22], block_previous[23], block_previous[24], block_previous[25], block_previous[26], block_previous[27], block_previous[28], block_previous[29], block_previous[30], block_previous[31], block_previous[32], block_previous[33]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33]};

        poro_hyperelasticity_poro_d3_simplex_mixed_residual_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::q_weight(), block_current_streams, block_previous_streams, alpha, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[33][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t current_stride,
        const double *const SFEM_RESTRICT u_data[3],
        const double *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const double *const SFEM_RESTRICT u_old_data[3],
        const double *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t current_stride,
        const float *const SFEM_RESTRICT u_data[3],
        const float *const SFEM_RESTRICT p_data,
        const ptrdiff_t previous_stride,
        const float *const SFEM_RESTRICT u_old_data[3],
        const float *const SFEM_RESTRICT p_old_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_residual_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, current_stride, u_data, p_data, previous_stride, u_old_data, p_old_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_mixed_impl(
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
        const scalar_t alpha,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int CELL_N_SHAPE = 10;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_shape(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::tet4_grad_ref_z()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 9] = elements[9][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_direction[0][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[3][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[4][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[5][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[6][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[7][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[8][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[9][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[10][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[11][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[12][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[13][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[14][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[15][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[16][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[17][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[18][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[19][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[20][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[21][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[22][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[23][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[24][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[25][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[26][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[27][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[28][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[29][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[30][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[31][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[32][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[33][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 3] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
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
        const scalar_t *const block_adjugate[DIM * DIM] = {block_jacobian_adjugate0, block_jacobian_adjugate1, block_jacobian_adjugate2, block_jacobian_adjugate3, block_jacobian_adjugate4, block_jacobian_adjugate5, block_jacobian_adjugate6, block_jacobian_adjugate7, block_jacobian_adjugate8};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33]};

        poro_hyperelasticity_poro_d3_simplex_mixed_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_jacobian_determinant0, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_affine_reference_data<scalar_t>::q_weight(), block_direction_streams, alpha, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[33][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa(
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
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, dt, hydraulic_conductivity, storage, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_soa_float(
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
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, alpha, dt, hydraulic_conductivity, storage, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const scalar_t alpha,
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        const ptrdiff_t direction_stride,
        const scalar_t *const SFEM_RESTRICT u_direction_data[3],
        const scalar_t *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        scalar_t *const SFEM_RESTRICT u_out[3],
        scalar_t *const SFEM_RESTRICT p_out
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int CELL_N_SHAPE = 10;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VECTOR_SIZE = 16;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y();
    const scalar_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        idx_t ev[VECTOR_SIZE * CELL_N_SHAPE];
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            ev[lane * CELL_N_SHAPE + 0] = elements[0][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 1] = elements[1][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 2] = elements[2][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 3] = elements[3][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 4] = elements[4][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 5] = elements[5][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 6] = elements[6][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 7] = elements[7][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 8] = elements[8][evbegin + lane];
            ev[lane * CELL_N_SHAPE + 9] = elements[9][evbegin + lane];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_coordinates[0][lane] = points[0][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[1][lane] = points[1][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[2][lane] = points[2][ev[lane * CELL_N_SHAPE + 0]];
            block_coordinates[3][lane] = points[0][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[4][lane] = points[1][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[5][lane] = points[2][ev[lane * CELL_N_SHAPE + 1]];
            block_coordinates[6][lane] = points[0][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[7][lane] = points[1][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[8][lane] = points[2][ev[lane * CELL_N_SHAPE + 2]];
            block_coordinates[9][lane] = points[0][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[10][lane] = points[1][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[11][lane] = points[2][ev[lane * CELL_N_SHAPE + 3]];
            block_coordinates[12][lane] = points[0][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[13][lane] = points[1][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[14][lane] = points[2][ev[lane * CELL_N_SHAPE + 4]];
            block_coordinates[15][lane] = points[0][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[16][lane] = points[1][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[17][lane] = points[2][ev[lane * CELL_N_SHAPE + 5]];
            block_coordinates[18][lane] = points[0][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[19][lane] = points[1][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[20][lane] = points[2][ev[lane * CELL_N_SHAPE + 6]];
            block_coordinates[21][lane] = points[0][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[22][lane] = points[1][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[23][lane] = points[2][ev[lane * CELL_N_SHAPE + 7]];
            block_coordinates[24][lane] = points[0][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[25][lane] = points[1][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[26][lane] = points[2][ev[lane * CELL_N_SHAPE + 8]];
            block_coordinates[27][lane] = points[0][ev[lane * CELL_N_SHAPE + 9]];
            block_coordinates[28][lane] = points[1][ev[lane * CELL_N_SHAPE + 9]];
            block_coordinates[29][lane] = points[2][ev[lane * CELL_N_SHAPE + 9]];
            block_direction[0][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[1][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[2][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[3][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[4][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[5][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[6][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[7][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[8][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[9][lane] = u_direction_data[0][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[10][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[11][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[12][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[13][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[14][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[15][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[16][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[17][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[18][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[19][lane] = u_direction_data[1][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[20][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[21][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[22][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[23][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 3] * direction_stride];
            block_direction[24][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 4] * direction_stride];
            block_direction[25][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 5] * direction_stride];
            block_direction[26][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 6] * direction_stride];
            block_direction[27][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 7] * direction_stride];
            block_direction[28][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 8] * direction_stride];
            block_direction[29][lane] = u_direction_data[2][ev[lane * CELL_N_SHAPE + 9] * direction_stride];
            block_direction[30][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 0] * direction_stride];
            block_direction[31][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 1] * direction_stride];
            block_direction[32][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 2] * direction_stride];
            block_direction[33][lane] = p_direction_data[ev[lane * CELL_N_SHAPE + 3] * direction_stride];
        }

        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            block_output[0][lane] = scalar_t(0);
            block_output[1][lane] = scalar_t(0);
            block_output[2][lane] = scalar_t(0);
            block_output[3][lane] = scalar_t(0);
            block_output[4][lane] = scalar_t(0);
            block_output[5][lane] = scalar_t(0);
            block_output[6][lane] = scalar_t(0);
            block_output[7][lane] = scalar_t(0);
            block_output[8][lane] = scalar_t(0);
            block_output[9][lane] = scalar_t(0);
            block_output[10][lane] = scalar_t(0);
            block_output[11][lane] = scalar_t(0);
            block_output[12][lane] = scalar_t(0);
            block_output[13][lane] = scalar_t(0);
            block_output[14][lane] = scalar_t(0);
            block_output[15][lane] = scalar_t(0);
            block_output[16][lane] = scalar_t(0);
            block_output[17][lane] = scalar_t(0);
            block_output[18][lane] = scalar_t(0);
            block_output[19][lane] = scalar_t(0);
            block_output[20][lane] = scalar_t(0);
            block_output[21][lane] = scalar_t(0);
            block_output[22][lane] = scalar_t(0);
            block_output[23][lane] = scalar_t(0);
            block_output[24][lane] = scalar_t(0);
            block_output[25][lane] = scalar_t(0);
            block_output[26][lane] = scalar_t(0);
            block_output[27][lane] = scalar_t(0);
            block_output[28][lane] = scalar_t(0);
            block_output[29][lane] = scalar_t(0);
            block_output[30][lane] = scalar_t(0);
            block_output[31][lane] = scalar_t(0);
            block_output[32][lane] = scalar_t(0);
            block_output[33][lane] = scalar_t(0);
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t J00 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J01 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J02 = block_coordinates[0][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[3][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[6][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[9][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[12][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[15][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[18][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[21][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[24][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[27][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                const scalar_t J10 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J11 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J12 = block_coordinates[1][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[4][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[7][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[10][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[13][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[16][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[19][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[22][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[25][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[28][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                const scalar_t J20 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_0[q * CELL_N_SHAPE + 9];
                const scalar_t J21 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_1[q * CELL_N_SHAPE + 9];
                const scalar_t J22 = block_coordinates[2][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 0] + block_coordinates[5][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 1] + block_coordinates[8][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 2] + block_coordinates[11][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 3] + block_coordinates[14][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 4] + block_coordinates[17][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 5] + block_coordinates[20][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 6] + block_coordinates[23][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 7] + block_coordinates[26][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 8] + block_coordinates[29][lane] * isoparametric_cell_grad_ref_2[q * CELL_N_SHAPE + 9];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        block_adjugate_streams, block_determinant, q * VECTOR_SIZE + lane);
            }
        }

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_shape(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::tet4_grad_ref_z()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        const scalar_t *const block_direction_streams[N_FIELD_STREAMS] = {block_direction[0], block_direction[1], block_direction[2], block_direction[3], block_direction[4], block_direction[5], block_direction[6], block_direction[7], block_direction[8], block_direction[9], block_direction[10], block_direction[11], block_direction[12], block_direction[13], block_direction[14], block_direction[15], block_direction[16], block_direction[17], block_direction[18], block_direction[19], block_direction[20], block_direction[21], block_direction[22], block_direction[23], block_direction[24], block_direction[25], block_direction[26], block_direction[27], block_direction[28], block_direction[29], block_direction[30], block_direction[31], block_direction[32], block_direction[33]};
        scalar_t *const block_output_streams[N_FIELD_STREAMS] = {block_output[0], block_output[1], block_output[2], block_output[3], block_output[4], block_output[5], block_output[6], block_output[7], block_output[8], block_output[9], block_output[10], block_output[11], block_output[12], block_output[13], block_output[14], block_output[15], block_output[16], block_output[17], block_output[18], block_output[19], block_output[20], block_output[21], block_output[22], block_output[23], block_output[24], block_output[25], block_output[26], block_output[27], block_output[28], block_output[29], block_output[30], block_output[31], block_output[32], block_output[33]};

        poro_hyperelasticity_poro_d3_simplex_mixed_jacobian_action_block<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::poro_hyperelasticity_poro_isoparametric_reference_data<scalar_t>::q_weight(), block_direction_streams, alpha, dt, hydraulic_conductivity, storage, block_output_streams);

        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[0][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[1][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[2][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[3][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[4][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[5][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[6][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[7][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[8][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[0][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[9][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[10][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[11][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[12][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[13][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[14][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[15][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[16][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[17][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[18][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[1][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[19][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[20][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[21][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[22][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[23][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 4] * out_stride] += block_output[24][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 5] * out_stride] += block_output[25][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 6] * out_stride] += block_output[26][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 7] * out_stride] += block_output[27][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 8] * out_stride] += block_output[28][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                u_out[2][ev[scatter * CELL_N_SHAPE + 9] * out_stride] += block_output[29][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 0] * out_stride] += block_output[30][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 1] * out_stride] += block_output[31][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 2] * out_stride] += block_output[32][scatter];
            }
        }
        {
            for (int scatter = 0; scatter < nelems; ++scatter) {
                #pragma omp atomic update
                p_out[ev[scatter * CELL_N_SHAPE + 3] * out_stride] += block_output[33][scatter];
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const double alpha,
        const double dt,
        const double hydraulic_conductivity,
        const double storage,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const float alpha,
        const float dt,
        const float hydraulic_conductivity,
        const float storage,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::poro_hyperelasticity_poro_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, alpha, dt, hydraulic_conductivity, storage, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}
