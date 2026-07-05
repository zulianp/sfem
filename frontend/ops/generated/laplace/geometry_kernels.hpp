#ifndef SFEM_CODEGEN_GEOMETRY_KERNELS_HPP
#define SFEM_CODEGEN_GEOMETRY_KERNELS_HPP

#include <stddef.h>

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>
struct GeometryJacobianAdjugateDeterminant;

template <typename scalar_t>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_2(
        const scalar_t J00,
        const scalar_t J01,
        const scalar_t J10,
        const scalar_t J11,
        scalar_t *const *const SFEM_RESTRICT adjugate,
        scalar_t *const SFEM_RESTRICT determinant,
        const ptrdiff_t offset) {
    adjugate[0][offset] = J11;
    adjugate[1][offset] = -J01;
    adjugate[2][offset] = -J10;
    adjugate[3][offset] = J00;
    determinant[offset] = J00 * J11 - J01 * J10;
}

template <typename scalar_t>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant_3(
        const scalar_t J00,
        const scalar_t J01,
        const scalar_t J02,
        const scalar_t J10,
        const scalar_t J11,
        const scalar_t J12,
        const scalar_t J20,
        const scalar_t J21,
        const scalar_t J22,
        scalar_t *const *const SFEM_RESTRICT adjugate,
        scalar_t *const SFEM_RESTRICT determinant,
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

template <typename scalar_t, int N_QP, int VECTOR_SIZE>
struct GeometryJacobianAdjugateDeterminant<scalar_t, 2, N_QP, VECTOR_SIZE> {
    static SFEM_INLINE void eval(
            const int nelems,
            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,
            scalar_t *const *const SFEM_RESTRICT adjugate,
            scalar_t *const SFEM_RESTRICT determinant) {
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t offset = q * VECTOR_SIZE + lane;
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
                geometry_jacobian_adjugate_and_determinant_2<scalar_t>(
                        J00, J01, J10, J11, adjugate, determinant, offset);
            }
        }
    }
};

template <typename scalar_t, int N_QP, int VECTOR_SIZE>
struct GeometryJacobianAdjugateDeterminant<scalar_t, 3, N_QP, VECTOR_SIZE> {
    static SFEM_INLINE void eval(
            const int nelems,
            const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,
            scalar_t *const *const SFEM_RESTRICT adjugate,
            scalar_t *const SFEM_RESTRICT determinant) {
        for (int q = 0; q < N_QP; ++q) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t offset = q * VECTOR_SIZE + lane;
                const scalar_t J00 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
                const scalar_t J01 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
                const scalar_t J02 = coordinate_grad_ref[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
                const scalar_t J10 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
                const scalar_t J11 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
                const scalar_t J12 = coordinate_grad_ref[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
                const scalar_t J20 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
                const scalar_t J21 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
                const scalar_t J22 = coordinate_grad_ref[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
                geometry_jacobian_adjugate_and_determinant_3<scalar_t>(
                        J00, J01, J02, J10, J11, J12, J20, J21, J22,
                        adjugate, determinant, offset);
            }
        }
    }
};

template <typename scalar_t, int DIM, int N_QP, int VECTOR_SIZE>
static SFEM_INLINE void geometry_jacobian_adjugate_and_determinant(
        const int nelems,
        const scalar_t *const SFEM_RESTRICT coordinate_grad_ref,
        scalar_t *const *const SFEM_RESTRICT adjugate,
        scalar_t *const SFEM_RESTRICT determinant) {
    GeometryJacobianAdjugateDeterminant<scalar_t, DIM, N_QP, VECTOR_SIZE>::eval(
            nelems, coordinate_grad_ref, adjugate, determinant);
}

} // namespace codegen
} // namespace sfem

#endif
