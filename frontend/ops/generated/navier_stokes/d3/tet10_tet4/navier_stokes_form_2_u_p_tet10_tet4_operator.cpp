#include <type_traits>
#include "../navier_stokes_form_2_u_p_d3_simplex_mixed_local.hpp"
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
struct navier_stokes_form_2_u_p_affine_reference_data {
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
struct navier_stokes_form_2_u_p_isoparametric_reference_data {
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

static const KernelDiagnostics navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_residual_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data = {
    "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa",
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

extern "C" const sfem::codegen::KernelDiagnostics *navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics(void) {
    return &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data;
}

extern "C" double navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_arithmetic_intensity(
        const ptrdiff_t nelements,
        const size_t scalar_bytes,
        const size_t real_bytes,
        const size_t accumulator_bytes) {
    return sfem::codegen::KernelDiagnostics_arithmetic_intensity(
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            nelements, scalar_bytes, real_bytes, accumulator_bytes);
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_affine_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(double), sizeof(double), sizeof(double));
}

extern "C" void navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float_print_rate(
        const double elapsed,
        const ptrdiff_t nelements,
        const ptrdiff_t ndofs) {
    sfem::codegen::KernelDiagnostics_print_rate_isoparametric_mesh(
            "navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float",
            &sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_element_soa_diagnostics_data,
            elapsed, nelements, ndofs,
            sizeof(float), sizeof(float), sizeof(float));
}

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_tet10_tet4_residual_affine_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant0,
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
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_tet10_tet4_residual_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

namespace sfem {
namespace codegen {

template <typename scalar_t, typename jacobian_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_mixed_impl(
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
    const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet10_shape(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet4_shape()};
    const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::tet4_grad_ref_z()};

#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 10 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 20 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 4; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 30 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 34; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
        }
        const jacobian_t *const affine_geometry_sources[10] = {g_jacobian_adjugate0 + evbegin, g_jacobian_adjugate1 + evbegin, g_jacobian_adjugate2 + evbegin, g_jacobian_adjugate3 + evbegin, g_jacobian_adjugate4 + evbegin, g_jacobian_adjugate5 + evbegin, g_jacobian_adjugate6 + evbegin, g_jacobian_adjugate7 + evbegin, g_jacobian_adjugate8 + evbegin, g_jacobian_determinant0 + evbegin};
        scalar_t block_affine_geometry_data[10][VECTOR_SIZE];
        const scalar_t *block_affine_geometry_streams[10];
        for (int geometry_stream = 0; geometry_stream < 10; ++geometry_stream) {
            block_affine_geometry_streams[geometry_stream] = affine_geometry_stream<scalar_t, jacobian_t, VECTOR_SIZE>(
                    nelems, affine_geometry_sources[geometry_stream], block_affine_geometry_data[geometry_stream], std::is_same<jacobian_t, scalar_t>());
        }
        const scalar_t *block_adjugate[DIM * DIM];
        for (int component = 0; component < DIM * DIM; ++component) {
            block_adjugate[component] = block_affine_geometry_streams[component];
        }

        navier_stokes_form_2_u_p_d3_simplex_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_affine_geometry_streams[9], block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_p_affine_reference_data<scalar_t>::q_weight(), block_direction, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 10 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 20 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 4; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 30 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa(
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
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_mixed_impl<double, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_soa_float(
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
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_affine_mesh_mixed_impl<float, geom_t>(nelements, nnodes, elements, g_jacobian_adjugate0, g_jacobian_adjugate1, g_jacobian_adjugate2, g_jacobian_adjugate3, g_jacobian_adjugate4, g_jacobian_adjugate5, g_jacobian_adjugate6, g_jacobian_adjugate7, g_jacobian_adjugate8, g_jacobian_determinant0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
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
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y();
    const scalar_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t evbegin = 0; evbegin < nelements; evbegin += VECTOR_SIZE) {
        const int nelems = (int)MIN((ptrdiff_t)VECTOR_SIZE, nelements - evbegin);
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }

        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 0 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[0][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 10 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[1][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 10; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 20 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = u_direction_data[2][node * direction_stride];
            }
        }
        for (int local_shape = 0; local_shape < 4; ++local_shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
            const int stream = 30 + local_shape;
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const idx_t node = element_shape[evbegin + lane];
                block_direction[stream][lane] = p_direction_data[node * direction_stride];
            }
        }

        for (int stream = 0; stream < 34; ++stream) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                block_output[stream][lane] = scalar_t(0);
            }
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

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_shape(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_z()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        navier_stokes_form_2_u_p_d3_simplex_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, VECTOR_SIZE, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::q_weight(), block_direction, block_output);

        {
            scalar_t *const SFEM_RESTRICT out = u_out[0];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 0 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[1];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 10 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = u_out[2];
            for (int local_shape = 0; local_shape < 10; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 20 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
        {
            scalar_t *const SFEM_RESTRICT out = p_out;
            for (int local_shape = 0; local_shape < 4; ++local_shape) {
                const idx_t *const SFEM_RESTRICT element_shape = elements[local_shape];
                const int stream = 30 + local_shape;
                for (int scatter = 0; scatter < nelems; ++scatter) {
                    #pragma omp atomic update
                    out[element_shape[evbegin + scatter] * out_stride] += block_output[stream][scatter];
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const double *const SFEM_RESTRICT u_direction_data[3],
        const double *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        double *const SFEM_RESTRICT u_out[3],
        double *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl<double>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t direction_stride,
        const float *const SFEM_RESTRICT u_direction_data[3],
        const float *const SFEM_RESTRICT p_direction_data,
        const ptrdiff_t out_stride,
        float *const SFEM_RESTRICT u_out[3],
        float *const SFEM_RESTRICT p_out
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_jacobian_action_isoparametric_mesh_mixed_impl<float>(nelements, nnodes, elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out);
}

namespace sfem {
namespace codegen {

template <typename scalar_t>
static SFEM_INLINE void navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(
        idx_t **const SFEM_RESTRICT elements,
        const ptrdiff_t element,
        const ptrdiff_t out_stride,
        const scalar_t *const SFEM_RESTRICT element_matrix,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values) {
    static constexpr int N_ROW_STREAMS = 30;
    static constexpr int N_COL_STREAMS = 4;
    static constexpr int ROW_COMPONENT[N_ROW_STREAMS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    static constexpr int ROW_SHAPE[N_ROW_STREAMS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    static constexpr int COL_COMPONENT[N_COL_STREAMS] = {3, 3, 3, 3};
    static constexpr int COL_SHAPE[N_COL_STREAMS] = {0, 1, 2, 3};
    const ptrdiff_t element_offset = element * N_ROW_STREAMS * N_COL_STREAMS;
    for (int row_stream = 0; row_stream < N_ROW_STREAMS; ++row_stream) {
        const idx_t row_node = elements[ROW_SHAPE[row_stream]][element];
        const idx_t global_row = row_node * out_stride + ROW_COMPONENT[row_stream];
        for (int col_stream = 0; col_stream < N_COL_STREAMS; ++col_stream) {
            const idx_t col_node = elements[COL_SHAPE[col_stream]][element];
            const ptrdiff_t entry = element_offset + row_stream * N_COL_STREAMS + col_stream;
            rows[entry] = global_row;
            cols[entry] = col_node * out_stride + COL_COMPONENT[col_stream];
            values[entry] = element_matrix[row_stream * N_COL_STREAMS + col_stream];
        }
    }
}

template <typename scalar_t>
static SFEM_INLINE int navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 3;
    static constexpr int N_QP = 11;
    static constexpr int CELL_N_SHAPE = 10;
    static constexpr int N_SHAPE = CELL_N_SHAPE;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    static constexpr int VECTOR_SIZE = 1;
    (void)nnodes;
    const scalar_t *const isoparametric_cell_grad_ref_0 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x();
    const scalar_t *const isoparametric_cell_grad_ref_1 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y();
    const scalar_t *const isoparametric_cell_grad_ref_2 = sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t element = 0; element < nelements; ++element) {
        const ptrdiff_t evbegin = element;
        const int nelems = 1;
        scalar_t block_coordinates[DIM * CELL_N_SHAPE][VECTOR_SIZE];
        scalar_t block_adjugate_data[DIM * DIM][N_QP * VECTOR_SIZE];
        scalar_t block_determinant[N_QP * VECTOR_SIZE];
        scalar_t block_direction[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t block_output[N_FIELD_STREAMS][VECTOR_SIZE];
        scalar_t element_matrix[120];

        const geom_t *const coordinate_components[DIM] = {points[0], points[1], points[2]};
        for (int shape = 0; shape < N_SHAPE; ++shape) {
            const idx_t *const SFEM_RESTRICT element_shape = elements[shape];
            for (int d = 0; d < DIM; ++d) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    const idx_t node = element_shape[evbegin + lane];
                    block_coordinates[shape * DIM + d][lane] = coordinate_components[d][node];
                }
            }
        }

        scalar_t *block_adjugate_streams[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};
        for (int q = 0; q < N_QP; ++q) {
            const int lane = 0;
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

        const scalar_t *const field_shape[N_FIELDS] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_shape(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_shape()};
        const scalar_t *const field_grad_ref[N_FIELDS * DIM] = {sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet10_grad_ref_z(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_x(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_y(), sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::tet4_grad_ref_z()};
        const scalar_t *const block_adjugate[DIM * DIM] = {block_adjugate_data[0], block_adjugate_data[1], block_adjugate_data[2], block_adjugate_data[3], block_adjugate_data[4], block_adjugate_data[5], block_adjugate_data[6], block_adjugate_data[7], block_adjugate_data[8]};

        for (int entry = 0; entry < 120; ++entry) {
            element_matrix[entry] = scalar_t(0);
        }
        for (int trial_local = 0; trial_local < 4; ++trial_local) {
            const int trial = (trial_local + 30);
            for (int stream = 0; stream < N_FIELD_STREAMS; ++stream) {
                block_direction[stream][0] = scalar_t(0);
                block_output[stream][0] = scalar_t(0);
            }
            block_direction[trial][0] = scalar_t(1);
            navier_stokes_form_2_u_p_d3_simplex_mixed_jacobian_action_block_contiguous<scalar_t, N_QP, CELL_N_SHAPE, VECTOR_SIZE>(nelems, 0, block_determinant, block_adjugate, field_shape, field_grad_ref, sfem::codegen::navier_stokes_form_2_u_p_isoparametric_reference_data<scalar_t>::q_weight(), block_direction, block_output);
            for (int test_local = 0; test_local < 30; ++test_local) {
                const int test = test_local;
                element_matrix[test_local * 4 + trial_local] = block_output[test][0];
            }
        }

        navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_scatter_coo_triplets(elements, element, out_stride, element_matrix, rows, cols, values);
    }

    return SFEM_SUCCESS;
}

} // namespace codegen
} // namespace sfem

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        double *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_impl<double>(nelements, nnodes, elements, points, out_stride, rows, cols, values);
}

extern "C" int navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_float(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        const geom_t *const *const SFEM_RESTRICT points,
        const ptrdiff_t out_stride,
        idx_t *const SFEM_RESTRICT rows,
        idx_t *const SFEM_RESTRICT cols,
        float *const SFEM_RESTRICT values
) {
    return sfem::codegen::navier_stokes_form_2_u_p_tet10_tet4_hessian_coo_triplet_isoparametric_mesh_soa_impl<float>(nelements, nnodes, elements, points, out_stride, rows, cols, values);
}
