#ifndef NAVIER_STOKES_FORM_2_P_U_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_P_U_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_tensor_product_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT field_shape_1d[2],
        const s_t *const SFEM_RESTRICT q_weight_1d,
        s_t *const SFEM_RESTRICT output[89]
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
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_tensor_product_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT field_shape_1d[2],
        const s_t *const SFEM_RESTRICT q_weight_1d,
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

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_tensor_product_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT adjugate[9],
        const s_t *const SFEM_RESTRICT field_shape_1d[2],
        const s_t *const SFEM_RESTRICT field_grad_1d[2],
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t *const SFEM_RESTRICT direction[89],
        s_t *const SFEM_RESTRICT output[89]
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
    s_t direction_u0_value[NQ * VS];
    s_t direction_u0_grad_ref[NQ * ND * VS];
    const s_t *const direction_u0_streams[U_NS] = {direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7], direction[8], direction[9], direction[10], direction[11], direction[12], direction[13], direction[14], direction[15], direction[16], direction[17], direction[18], direction[19], direction[20], direction[21], direction[22], direction[23], direction[24], direction[25], direction[26]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u0_streams, direction_u0_value, direction_u0_grad_ref);
    s_t direction_u1_value[NQ * VS];
    s_t direction_u1_grad_ref[NQ * ND * VS];
    const s_t *const direction_u1_streams[U_NS] = {direction[27], direction[28], direction[29], direction[30], direction[31], direction[32], direction[33], direction[34], direction[35], direction[36], direction[37], direction[38], direction[39], direction[40], direction[41], direction[42], direction[43], direction[44], direction[45], direction[46], direction[47], direction[48], direction[49], direction[50], direction[51], direction[52], direction[53]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u1_streams, direction_u1_value, direction_u1_grad_ref);
    s_t direction_u2_value[NQ * VS];
    s_t direction_u2_grad_ref[NQ * ND * VS];
    const s_t *const direction_u2_streams[U_NS] = {direction[54], direction[55], direction[56], direction[57], direction[58], direction[59], direction[60], direction[61], direction[62], direction[63], direction[64], direction[65], direction[66], direction[67], direction[68], direction[69], direction[70], direction[71], direction[72], direction[73], direction[74], direction[75], direction[76], direction[77], direction[78], direction[79], direction[80]};
    tensor_evaluate<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u2_streams, direction_u2_value, direction_u2_grad_ref);
    s_t direction_p_value[NQ * VS];
    s_t direction_p_grad_ref[NQ * ND * VS];
    const s_t *const direction_p_streams[P_NS] = {direction[81], direction[82], direction[83], direction[84], direction[85], direction[86], direction[87], direction[88]};
    tensor_evaluate<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], direction_p_streams, direction_p_value, direction_p_grad_ref);
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
            const s_t u0_direction_grad_0_ref = direction_u0_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u0_direction_grad_1_ref = direction_u0_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u0_direction_grad_2_ref = direction_u0_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const s_t u1_direction_grad_0_ref = direction_u1_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u1_direction_grad_1_ref = direction_u1_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u1_direction_grad_2_ref = direction_u1_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const s_t u2_direction_grad_0_ref = direction_u2_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u2_direction_grad_1_ref = direction_u2_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u2_direction_grad_2_ref = direction_u2_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const s_t p_direction_grad_0_ref = direction_p_grad_ref[(q * ND + 0) * VS + lane];
            const s_t p_direction_grad_1_ref = direction_p_grad_ref[(q * ND + 1) * VS + lane];
            const s_t p_direction_grad_2_ref = direction_p_grad_ref[(q * ND + 2) * VS + lane];
            const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
            const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
            const s_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
            const s_t value_coeff3 = u0_direction_grad_0 + u1_direction_grad_1 + u2_direction_grad_2;
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
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_tensor_product_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT adjugate[9],
        const s_t *const SFEM_RESTRICT field_shape_1d[2],
        const s_t *const SFEM_RESTRICT field_grad_1d[2],
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t direction[89][VS],
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
    s_t direction_u0_value[NQ * VS];
    s_t direction_u0_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 0, direction_u0_value, direction_u0_grad_ref);
    s_t direction_u1_value[NQ * VS];
    s_t direction_u1_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 27, direction_u1_value, direction_u1_grad_ref);
    s_t direction_u2_value[NQ * VS];
    s_t direction_u2_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 54, direction_u2_value, direction_u2_grad_ref);
    s_t direction_p_value[NQ * VS];
    s_t direction_p_grad_ref[NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, P_NS, VS, ND, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], direction + 81, direction_p_value, direction_p_grad_ref);
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
            const s_t u0_direction_grad_0_ref = direction_u0_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u0_direction_grad_1_ref = direction_u0_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u0_direction_grad_2_ref = direction_u0_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const s_t u1_direction_grad_0_ref = direction_u1_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u1_direction_grad_1_ref = direction_u1_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u1_direction_grad_2_ref = direction_u1_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const s_t u2_direction_grad_0_ref = direction_u2_grad_ref[(q * ND + 0) * VS + lane];
            const s_t u2_direction_grad_1_ref = direction_u2_grad_ref[(q * ND + 1) * VS + lane];
            const s_t u2_direction_grad_2_ref = direction_u2_grad_ref[(q * ND + 2) * VS + lane];
            const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const s_t p_direction_grad_0_ref = direction_p_grad_ref[(q * ND + 0) * VS + lane];
            const s_t p_direction_grad_1_ref = direction_p_grad_ref[(q * ND + 1) * VS + lane];
            const s_t p_direction_grad_2_ref = direction_p_grad_ref[(q * ND + 2) * VS + lane];
            const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
            const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
            const s_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
            const s_t value_coeff3 = u0_direction_grad_0 + u1_direction_grad_1 + u2_direction_grad_2;
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

} // namespace codegen
} // namespace sfem

#endif
