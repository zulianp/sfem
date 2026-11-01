#ifndef SFEM_CODEGEN_INEXACT_TET4_Q1_CUH
#define SFEM_CODEGEN_INEXACT_TET4_Q1_CUH

namespace sfem {
namespace codegen {

// The stored tangent's element half for TET4 on its 1-point rule (plans/inexact_element.py).
struct inexact_tet4_q1 {
  static constexpr const char *name = "tet4_q1";
  static constexpr int DIM = 3;
  static constexpr int N = 4;   // nodes
  static constexpr int R = 1;   // gradient vectors the store contracts with
  static constexpr int NV = 1;  // stored tensors per element
  static constexpr int NQ = 1;  // quadrature points of the set-up
  static constexpr bool moments = false;
  static constexpr bool foldable = false;
  static constexpr bool reordered = false;
  static constexpr double KVOL = 0.16666666666666666;
  // DT[r][m][j] = dphi_j/dxi_m at gradient point r.
  static constexpr double DT[R][3][N] = {{{-1.0, 1.0, 0.0, 0.0}, {-1.0, 0.0, 1.0, 0.0}, {-1.0, 0.0, 0.0, 1.0}}};
  // Each stored tensor's reference weight (what the element mean contributes to it).
  static constexpr double X[NV] = {0.16666666666666666};
  static constexpr double QW[NQ] = {0.16666666666666666};

  // P[r][k*3+m] = sum_j dphi_j/dxi_m (at r) h[k][j]: in the vector's precision, then
  // rounded to the arithmetic type, so differences of nodal values are formed first.
  template <typename h_t, typename t_out>
  static __host__ __device__ __forceinline__ void stage_in(const h_t h[3][4], t_out P[1][9]) {
    using t = h_t;
    {
      const t h0_0 = h[0][0];
      const t h0_1 = h[0][1];
      const t h0_2 = h[0][2];
      const t h0_3 = h[0][3];
      const t h1_0 = h[1][0];
      const t h1_1 = h[1][1];
      const t h1_2 = h[1][2];
      const t h1_3 = h[1][3];
      const t h2_0 = h[2][0];
      const t h2_1 = h[2][1];
      const t h2_2 = h[2][2];
      const t h2_3 = h[2][3];
      P[0][0] = t_out(-h0_0 + h0_1);
      P[0][1] = t_out(-h0_0 + h0_2);
      P[0][2] = t_out(-h0_0 + h0_3);
      P[0][3] = t_out(-h1_0 + h1_1);
      P[0][4] = t_out(-h1_0 + h1_2);
      P[0][5] = t_out(-h1_0 + h1_3);
      P[0][6] = t_out(-h2_0 + h2_1);
      P[0][7] = t_out(-h2_0 + h2_2);
      P[0][8] = t_out(-h2_0 + h2_3);
    }
  }

  // out[i][p] = sum_r sum_n dphi_p/dxi_n (at r) Y[r][i*3+n], accumulated in out_t.
  template <typename y_t, typename out_t>
  static __host__ __device__ __forceinline__ void stage_out(const y_t Y[1][9], out_t out[3][4]) {
    using t = out_t;
    {
      const t y0_0 = t(Y[0][0]);
      const t y0_1 = t(Y[0][1]);
      const t y0_2 = t(Y[0][2]);
      const t y0_3 = t(Y[0][3]);
      const t y0_4 = t(Y[0][4]);
      const t y0_5 = t(Y[0][5]);
      const t y0_6 = t(Y[0][6]);
      const t y0_7 = t(Y[0][7]);
      const t y0_8 = t(Y[0][8]);
      out[0][0] = -y0_0 - y0_1 - y0_2;
      out[0][1] = y0_0;
      out[0][2] = y0_1;
      out[0][3] = y0_2;
      out[1][0] = -y0_3 - y0_4 - y0_5;
      out[1][1] = y0_3;
      out[1][2] = y0_4;
      out[1][3] = y0_5;
      out[2][0] = -y0_6 - y0_7 - y0_8;
      out[2][1] = y0_6;
      out[2][2] = y0_7;
      out[2][3] = y0_8;
    }
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
