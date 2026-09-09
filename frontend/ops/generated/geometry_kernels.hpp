#ifndef SFEM_CODEGEN_GEOMETRY_KERNELS_HPP
#define SFEM_CODEGEN_GEOMETRY_KERNELS_HPP

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

template <typename s_t, int ND, int NQ, int VS>
struct GeometryJacobianAdjugateDeterminant;

template <typename s_t>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_2(
    const s_t J00,
    const s_t J01,
    const s_t J10,
    const s_t J11,
    s_t *const *const RSTR adjugate,
    s_t *const RSTR determinant,
    const ptrdiff_t offset) {
  adjugate[0][offset] = J11;
  adjugate[1][offset] = -J01;
  adjugate[2][offset] = -J10;
  adjugate[3][offset] = J00;
  determinant[offset] = J00 * J11 - J01 * J10;
}

template <typename s_t>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_3(
    const s_t J00,
    const s_t J01,
    const s_t J02,
    const s_t J10,
    const s_t J11,
    const s_t J12,
    const s_t J20,
    const s_t J21,
    const s_t J22,
    s_t *const *const RSTR adjugate,
    s_t *const RSTR determinant,
    const ptrdiff_t offset) {
  adjugate[0][offset] = J11 * J22 - J12 * J21;
  adjugate[1][offset] = J02 * J21 - J01 * J22;
  adjugate[2][offset] = J01 * J12 - J02 * J11;
  adjugate[3][offset] = J12 * J20 - J10 * J22;
  adjugate[4][offset] = J00 * J22 - J02 * J20;
  adjugate[5][offset] = J02 * J10 - J00 * J12;
  adjugate[6][offset] = J10 * J21 - J11 * J20;
  adjugate[7][offset] = J01 * J20 - J00 * J21;
  adjugate[8][offset] = J00 * J11 - J01 * J10;
  determinant[offset] = J00 * (J11 * J22 - J12 * J21)
      - J01 * (J10 * J22 - J12 * J20)
      + J02 * (J10 * J21 - J11 * J20);
}

template <typename s_t, int NQ, int VS>
struct GeometryJacobianAdjugateDeterminant<s_t, 2, NQ, VS> {
  static SFEM_INLINE void eval(
      const int ne,
      const s_t *const RSTR coordinate_grad_ref,
      s_t *const *const RSTR adjugate,
      s_t *const RSTR determinant) {
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t offset = q * VS + lane;
        const s_t J00 = coordinate_grad_ref[((0 * NQ + q) * 2 + 0) * VS + lane];
        const s_t J01 = coordinate_grad_ref[((0 * NQ + q) * 2 + 1) * VS + lane];
        const s_t J10 = coordinate_grad_ref[((1 * NQ + q) * 2 + 0) * VS + lane];
        const s_t J11 = coordinate_grad_ref[((1 * NQ + q) * 2 + 1) * VS + lane];
        geometry_jacobian_adjugate_and_determinant_2<s_t>(
            J00, J01, J10, J11, adjugate, determinant, offset);
      }
    }
  }
};

template <typename s_t, int NQ, int VS>
struct GeometryJacobianAdjugateDeterminant<s_t, 3, NQ, VS> {
  static SFEM_INLINE void eval(
      const int ne,
      const s_t *const RSTR coordinate_grad_ref,
      s_t *const *const RSTR adjugate,
      s_t *const RSTR determinant) {
    for (int q = 0; q < NQ; ++q) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t offset = q * VS + lane;
        const s_t J00 = coordinate_grad_ref[((0 * NQ + q) * 3 + 0) * VS + lane];
        const s_t J01 = coordinate_grad_ref[((0 * NQ + q) * 3 + 1) * VS + lane];
        const s_t J02 = coordinate_grad_ref[((0 * NQ + q) * 3 + 2) * VS + lane];
        const s_t J10 = coordinate_grad_ref[((1 * NQ + q) * 3 + 0) * VS + lane];
        const s_t J11 = coordinate_grad_ref[((1 * NQ + q) * 3 + 1) * VS + lane];
        const s_t J12 = coordinate_grad_ref[((1 * NQ + q) * 3 + 2) * VS + lane];
        const s_t J20 = coordinate_grad_ref[((2 * NQ + q) * 3 + 0) * VS + lane];
        const s_t J21 = coordinate_grad_ref[((2 * NQ + q) * 3 + 1) * VS + lane];
        const s_t J22 = coordinate_grad_ref[((2 * NQ + q) * 3 + 2) * VS + lane];
        geometry_jacobian_adjugate_and_determinant_3<s_t>(
            J00, J01, J02, J10, J11, J12, J20, J21, J22,
            adjugate, determinant, offset);
      }
    }
  }
};

template <typename s_t, int ND, int NQ, int VS>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant(
    const int ne,
    const s_t *const RSTR coordinate_grad_ref,
    s_t *const *const RSTR adjugate,
    s_t *const RSTR determinant) {
  GeometryJacobianAdjugateDeterminant<s_t, ND, NQ, VS>::eval(
      ne, coordinate_grad_ref, adjugate, determinant);
}

} // namespace codegen
} // namespace sfem

#endif
