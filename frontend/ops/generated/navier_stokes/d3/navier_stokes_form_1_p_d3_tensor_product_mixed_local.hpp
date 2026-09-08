#ifndef NAVIER_STOKES_FORM_1_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_1_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../kernel_math.hpp"
#include "../../tensor_product_kernels.hpp"

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_p_d3_tensor_product_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR field_shape_1d[2],
        const s_t *const RSTR field_grad_1d[2],
        const s_t *const RSTR q_weight_1d,
        const s_t *const RSTR current[89],
        s_t *const RSTR output[89]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 27;
    static constexpr int P_NS = 8;
    static constexpr int NQ1 = integer_root(NQ, ND);
    static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
    static constexpr int U_NS1 = integer_root(U_NS, ND);
    static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
    static constexpr int P_NS1 = integer_root(P_NS, ND);
    static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
    s_t current_u0_value[NQ * VS];
    s_t current_u0_grad_ref[NQ * ND * VS];
    const s_t *const current_u0_streams[U_NS] = {current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7], current[8], current[9], current[10], current[11], current[12], current[13], current[14], current[15], current[16], current[17], current[18], current[19], current[20], current[21], current[22], current[23], current[24], current[25], current[26]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current_u0_streams, current_u0_value, current_u0_grad_ref);
    s_t current_u1_value[NQ * VS];
    s_t current_u1_grad_ref[NQ * ND * VS];
    const s_t *const current_u1_streams[U_NS] = {current[27], current[28], current[29], current[30], current[31], current[32], current[33], current[34], current[35], current[36], current[37], current[38], current[39], current[40], current[41], current[42], current[43], current[44], current[45], current[46], current[47], current[48], current[49], current[50], current[51], current[52], current[53]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current_u1_streams, current_u1_value, current_u1_grad_ref);
    s_t current_u2_value[NQ * VS];
    s_t current_u2_grad_ref[NQ * ND * VS];
    const s_t *const current_u2_streams[U_NS] = {current[54], current[55], current[56], current[57], current[58], current[59], current[60], current[61], current[62], current[63], current[64], current[65], current[66], current[67], current[68], current[69], current[70], current[71], current[72], current[73], current[74], current[75], current[76], current[77], current[78], current[79], current[80]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current_u2_streams, current_u2_value, current_u2_grad_ref);
    s_t current_p_value[NQ * VS];
    s_t current_p_grad_ref[NQ * ND * VS];
    const s_t *const current_p_streams[P_NS] = {current[81], current[82], current[83], current[84], current[85], current[86], current[87], current[88]};
    tensor_evaluate<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], current_p_streams, current_p_value, current_p_grad_ref);
    s_t u0_value_coeff[NQ * VS];
    s_t u1_value_coeff[NQ * VS];
    s_t u2_value_coeff[NQ * VS];
    s_t p_value_coeff[NQ * VS];
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = (q / NQ1) % NQ1;
        const int qz = q / (NQ1 * NQ1);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t adj4 = adjugate[4][goff];
            const s_t adj5 = adjugate[5][goff];
            const s_t adj6 = adjugate[6][goff];
            const s_t adj7 = adjugate[7][goff];
            const s_t adj8 = adjugate[8][goff];
            const s_t u0_grad_0_ref = current_u0_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u0_grad_1_ref = current_u0_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u0_grad_2_ref = current_u0_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const s_t u1_grad_0_ref = current_u1_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u1_grad_1_ref = current_u1_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u1_grad_2_ref = current_u1_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const s_t u2_grad_0_ref = current_u2_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u2_grad_1_ref = current_u2_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u2_grad_2_ref = current_u2_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const s_t p_grad_0_ref = current_p_grad_ref[(q * ND + 0) * VS + lane];
            const s_t p_grad_1_ref = current_p_grad_ref[(q * ND + 1) * VS + lane];
            const s_t p_grad_2_ref = current_p_grad_ref[(q * ND + 2) * VS + lane];
            const s_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj3 + p_grad_2_ref * adj6) / det;
            const s_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj4 + p_grad_2_ref * adj7) / det;
            const s_t p_grad_2 = (p_grad_0_ref * adj2 + p_grad_1_ref * adj5 + p_grad_2_ref * adj8) / det;
            const s_t value_coeff3 = u0_grad_0 + u1_grad_1 + u2_grad_2;
            u0_value_coeff[q * VS + lane] = s_t(0);
            u1_value_coeff[q * VS + lane] = s_t(0);
            u2_value_coeff[q * VS + lane] = s_t(0);
            p_value_coeff[q * VS + lane] = qw * det * value_coeff3;
        }
    }
    s_t *const u0_output_streams[U_NS] = {output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9], output[10], output[11], output[12], output[13], output[14], output[15], output[16], output[17], output[18], output[19], output[20], output[21], output[22], output[23], output[24], output[25], output[26]};
    tensor_integrate_value<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u0_value_coeff, u0_output_streams);
    s_t *const u1_output_streams[U_NS] = {output[27], output[28], output[29], output[30], output[31], output[32], output[33], output[34], output[35], output[36], output[37], output[38], output[39], output[40], output[41], output[42], output[43], output[44], output[45], output[46], output[47], output[48], output[49], output[50], output[51], output[52], output[53]};
    tensor_integrate_value<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u1_value_coeff, u1_output_streams);
    s_t *const u2_output_streams[U_NS] = {output[54], output[55], output[56], output[57], output[58], output[59], output[60], output[61], output[62], output[63], output[64], output[65], output[66], output[67], output[68], output[69], output[70], output[71], output[72], output[73], output[74], output[75], output[76], output[77], output[78], output[79], output[80]};
    tensor_integrate_value<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u2_value_coeff, u2_output_streams);
    s_t *const p_output_streams[P_NS] = {output[81], output[82], output[83], output[84], output[85], output[86], output[87], output[88]};
    tensor_integrate_value<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], p_value_coeff, p_output_streams);
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_p_d3_tensor_product_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR field_shape_1d[2],
        const s_t *const RSTR field_grad_1d[2],
        const s_t *const RSTR q_weight_1d,
        const s_t current[89][VS],
        s_t output[89][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 27;
    static constexpr int P_NS = 8;
    static constexpr int NQ1 = integer_root(NQ, ND);
    static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
    static constexpr int U_NS1 = integer_root(U_NS, ND);
    static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
    static constexpr int P_NS1 = integer_root(P_NS, ND);
    static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
    s_t current_u0_value[NQ * VS];
    s_t current_u0_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current + 0, current_u0_value, current_u0_grad_ref);
    s_t current_u1_value[NQ * VS];
    s_t current_u1_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current + 27, current_u1_value, current_u1_grad_ref);
    s_t current_u2_value[NQ * VS];
    s_t current_u2_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], current + 54, current_u2_value, current_u2_grad_ref);
    s_t current_p_value[NQ * VS];
    s_t current_p_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], current + 81, current_p_value, current_p_grad_ref);
    s_t u0_value_coeff[NQ * VS];
    s_t u1_value_coeff[NQ * VS];
    s_t u2_value_coeff[NQ * VS];
    s_t p_value_coeff[NQ * VS];
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = (q / NQ1) % NQ1;
        const int qz = q / (NQ1 * NQ1);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t adj4 = adjugate[4][goff];
            const s_t adj5 = adjugate[5][goff];
            const s_t adj6 = adjugate[6][goff];
            const s_t adj7 = adjugate[7][goff];
            const s_t adj8 = adjugate[8][goff];
            const s_t u0_grad_0_ref = current_u0_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u0_grad_1_ref = current_u0_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u0_grad_2_ref = current_u0_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const s_t u1_grad_0_ref = current_u1_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u1_grad_1_ref = current_u1_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u1_grad_2_ref = current_u1_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const s_t u2_grad_0_ref = current_u2_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u2_grad_1_ref = current_u2_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u2_grad_2_ref = current_u2_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const s_t p_grad_0_ref = current_p_grad_ref[(q * ND + 0) * VS + lane];
            const s_t p_grad_1_ref = current_p_grad_ref[(q * ND + 1) * VS + lane];
            const s_t p_grad_2_ref = current_p_grad_ref[(q * ND + 2) * VS + lane];
            const s_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj3 + p_grad_2_ref * adj6) / det;
            const s_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj4 + p_grad_2_ref * adj7) / det;
            const s_t p_grad_2 = (p_grad_0_ref * adj2 + p_grad_1_ref * adj5 + p_grad_2_ref * adj8) / det;
            const s_t value_coeff3 = u0_grad_0 + u1_grad_1 + u2_grad_2;
            u0_value_coeff[q * VS + lane] = s_t(0);
            u1_value_coeff[q * VS + lane] = s_t(0);
            u2_value_coeff[q * VS + lane] = s_t(0);
            p_value_coeff[q * VS + lane] = qw * det * value_coeff3;
        }
    }
    tensor_integrate_value_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u0_value_coeff, output + 0);
    tensor_integrate_value_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u1_value_coeff, output + 27);
    tensor_integrate_value_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], u2_value_coeff, output + 54);
    tensor_integrate_value_contiguous<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], p_value_coeff, output + 81);
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_p_d3_tensor_product_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR field_shape_1d[2],
        const s_t *const RSTR q_weight_1d,
        s_t *const RSTR output[89]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 27;
    static constexpr int P_NS = 8;
    static constexpr int NQ1 = integer_root(NQ, ND);
    static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
    static constexpr int U_NS1 = integer_root(U_NS, ND);
    static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
    static constexpr int P_NS1 = integer_root(P_NS, ND);
    static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_p_d3_tensor_product_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR field_shape_1d[2],
        const s_t *const RSTR q_weight_1d,
        s_t output[89][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 27;
    static constexpr int P_NS = 8;
    static constexpr int NQ1 = integer_root(NQ, ND);
    static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
    static constexpr int U_NS1 = integer_root(U_NS, ND);
    static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
    static constexpr int P_NS1 = integer_root(P_NS, ND);
    static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
}

} // namespace codegen
} // namespace sfem

#endif
