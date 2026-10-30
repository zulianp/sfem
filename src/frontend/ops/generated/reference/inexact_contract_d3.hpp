#ifndef SFEM_CODEGEN_INEXACT_CONTRACT_D3_HPP
#define SFEM_CODEGEN_INEXACT_CONTRACT_D3_HPP

namespace sfem {
namespace codegen {

template <bool SYM>
struct inexact_contract_d3;

template <>
struct inexact_contract_d3<true> {
  static constexpr int NC = 45;
  // Row r, column c of the packed tangent.
  static constexpr int SLOT[9][9] = {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {1, 9, 10, 11, 12, 13, 14, 15, 16}, {2, 10, 17, 18, 19, 20, 21, 22, 23}, {3, 11, 18, 24, 25, 26, 27, 28, 29}, {4, 12, 19, 25, 30, 31, 32, 33, 34}, {5, 13, 20, 26, 31, 35, 36, 37, 38}, {6, 14, 21, 27, 32, 36, 39, 40, 41}, {7, 15, 22, 28, 33, 37, 40, 42, 43}, {8, 16, 23, 29, 34, 38, 41, 43, 44}};
  static constexpr int slot(const int r, const int c) { return SLOT[r][c]; }

  template <typename t>
  static SFEM_INLINE void contract(const t *const S, const t *const g, t *const y) {
    y[0] += S[0]*g[0] + S[1]*g[1] + S[2]*g[2] + S[3]*g[3] + S[4]*g[4] + S[5]*g[5] + S[6]*g[6] + S[7]*g[7] + S[8]*g[8];
    y[1] += S[1]*g[0] + S[9]*g[1] + S[10]*g[2] + S[11]*g[3] + S[12]*g[4] + S[13]*g[5] + S[14]*g[6] + S[15]*g[7] + S[16]*g[8];
    y[2] += S[2]*g[0] + S[10]*g[1] + S[17]*g[2] + S[18]*g[3] + S[19]*g[4] + S[20]*g[5] + S[21]*g[6] + S[22]*g[7] + S[23]*g[8];
    y[3] += S[3]*g[0] + S[11]*g[1] + S[18]*g[2] + S[24]*g[3] + S[25]*g[4] + S[26]*g[5] + S[27]*g[6] + S[28]*g[7] + S[29]*g[8];
    y[4] += S[4]*g[0] + S[12]*g[1] + S[19]*g[2] + S[25]*g[3] + S[30]*g[4] + S[31]*g[5] + S[32]*g[6] + S[33]*g[7] + S[34]*g[8];
    y[5] += S[5]*g[0] + S[13]*g[1] + S[20]*g[2] + S[26]*g[3] + S[31]*g[4] + S[35]*g[5] + S[36]*g[6] + S[37]*g[7] + S[38]*g[8];
    y[6] += S[6]*g[0] + S[14]*g[1] + S[21]*g[2] + S[27]*g[3] + S[32]*g[4] + S[36]*g[5] + S[39]*g[6] + S[40]*g[7] + S[41]*g[8];
    y[7] += S[7]*g[0] + S[15]*g[1] + S[22]*g[2] + S[28]*g[3] + S[33]*g[4] + S[37]*g[5] + S[40]*g[6] + S[42]*g[7] + S[43]*g[8];
    y[8] += S[8]*g[0] + S[16]*g[1] + S[23]*g[2] + S[29]*g[3] + S[34]*g[4] + S[38]*g[5] + S[41]*g[6] + S[43]*g[7] + S[44]*g[8];
  }

  // The same for a block of lanes, a loop over them per row.
  template <typename t, int VS>
  static SFEM_INLINE void contract_lanes(const int ne, const t S[45][VS], const t G[9][VS], t Y[9][VS]) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[1][lane] += S[1][lane]*G[0][lane] + S[9][lane]*G[1][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[12][lane]*G[4][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[15][lane]*G[7][lane] + S[16][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[17][lane]*G[2][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[20][lane]*G[5][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane] + S[23][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[3][lane] += S[3][lane]*G[0][lane] + S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[24][lane]*G[3][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[27][lane]*G[6][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[4][lane] += S[4][lane]*G[0][lane] + S[12][lane]*G[1][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[30][lane]*G[4][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[33][lane]*G[7][lane] + S[34][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[35][lane]*G[5][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane] + S[38][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[6][lane] += S[6][lane]*G[0][lane] + S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[27][lane]*G[3][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[39][lane]*G[6][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[7][lane] += S[7][lane]*G[0][lane] + S[15][lane]*G[1][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[33][lane]*G[4][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[42][lane]*G[7][lane] + S[43][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[23][lane]*G[2][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[38][lane]*G[5][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
    }
  }

  // The folded store's contraction: bit d of MASK drops the m = n = d terms.
  template <int MASK, typename t, int VS>
  static SFEM_INLINE void contract_lanes_masked(const int ne, const t S[45][VS], const t G[9][VS], t Y[9][VS]) {
    if constexpr (MASK == 0) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[9][lane]*G[1][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[12][lane]*G[4][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[15][lane]*G[7][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[17][lane]*G[2][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[20][lane]*G[5][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane] + S[23][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[3][lane]*G[0][lane] + S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[24][lane]*G[3][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[27][lane]*G[6][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[12][lane]*G[1][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[30][lane]*G[4][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[33][lane]*G[7][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[35][lane]*G[5][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane] + S[38][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[6][lane]*G[0][lane] + S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[27][lane]*G[3][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[39][lane]*G[6][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[15][lane]*G[1][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[33][lane]*G[4][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[42][lane]*G[7][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[23][lane]*G[2][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[38][lane]*G[5][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 1) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[9][lane]*G[1][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[12][lane]*G[4][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[15][lane]*G[7][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[17][lane]*G[2][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[20][lane]*G[5][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane] + S[23][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[12][lane]*G[1][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[30][lane]*G[4][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[33][lane]*G[7][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[35][lane]*G[5][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane] + S[38][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[15][lane]*G[1][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[33][lane]*G[4][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[42][lane]*G[7][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[23][lane]*G[2][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[38][lane]*G[5][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 2) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[17][lane]*G[2][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[20][lane]*G[5][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane] + S[23][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[3][lane]*G[0][lane] + S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[24][lane]*G[3][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[27][lane]*G[6][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[35][lane]*G[5][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane] + S[38][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[6][lane]*G[0][lane] + S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[27][lane]*G[3][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[39][lane]*G[6][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[23][lane]*G[2][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[38][lane]*G[5][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 3) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[17][lane]*G[2][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[20][lane]*G[5][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane] + S[23][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[35][lane]*G[5][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane] + S[38][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[23][lane]*G[2][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[38][lane]*G[5][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 4) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[9][lane]*G[1][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[12][lane]*G[4][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[15][lane]*G[7][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[3][lane]*G[0][lane] + S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[24][lane]*G[3][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[27][lane]*G[6][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[12][lane]*G[1][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[30][lane]*G[4][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[33][lane]*G[7][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[6][lane]*G[0][lane] + S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[27][lane]*G[3][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[39][lane]*G[6][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[15][lane]*G[1][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[33][lane]*G[4][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[42][lane]*G[7][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 5) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[9][lane]*G[1][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[12][lane]*G[4][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[15][lane]*G[7][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[12][lane]*G[1][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[30][lane]*G[4][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[33][lane]*G[7][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[15][lane]*G[1][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[33][lane]*G[4][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[42][lane]*G[7][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 6) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[3][lane]*G[0][lane] + S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[24][lane]*G[3][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[27][lane]*G[6][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[6][lane]*G[0][lane] + S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[27][lane]*G[3][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[39][lane]*G[6][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 7) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[1][lane]*G[0][lane] + S[10][lane]*G[2][lane] + S[11][lane]*G[3][lane] + S[13][lane]*G[5][lane] + S[14][lane]*G[6][lane] + S[16][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[2][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[18][lane]*G[3][lane] + S[19][lane]*G[4][lane] + S[21][lane]*G[6][lane] + S[22][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[11][lane]*G[1][lane] + S[18][lane]*G[2][lane] + S[25][lane]*G[4][lane] + S[26][lane]*G[5][lane] + S[28][lane]*G[7][lane] + S[29][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[4][lane]*G[0][lane] + S[19][lane]*G[2][lane] + S[25][lane]*G[3][lane] + S[31][lane]*G[5][lane] + S[32][lane]*G[6][lane] + S[34][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[5][lane]*G[0][lane] + S[13][lane]*G[1][lane] + S[26][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[36][lane]*G[6][lane] + S[37][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[14][lane]*G[1][lane] + S[21][lane]*G[2][lane] + S[32][lane]*G[4][lane] + S[36][lane]*G[5][lane] + S[40][lane]*G[7][lane] + S[41][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[7][lane]*G[0][lane] + S[22][lane]*G[2][lane] + S[28][lane]*G[3][lane] + S[37][lane]*G[5][lane] + S[40][lane]*G[6][lane] + S[43][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[8][lane]*G[0][lane] + S[16][lane]*G[1][lane] + S[29][lane]*G[3][lane] + S[34][lane]*G[4][lane] + S[41][lane]*G[6][lane] + S[43][lane]*G[7][lane];
      }
    }
  }

  template <typename t, int VS>
  static SFEM_INLINE void contract_lanes_masked(const int mask, const int ne, const t S[45][VS], const t G[9][VS], t Y[9][VS]) {
    switch (mask) {
      case 0: contract_lanes_masked<0, t, VS>(ne, S, G, Y); break;
      case 1: contract_lanes_masked<1, t, VS>(ne, S, G, Y); break;
      case 2: contract_lanes_masked<2, t, VS>(ne, S, G, Y); break;
      case 3: contract_lanes_masked<3, t, VS>(ne, S, G, Y); break;
      case 4: contract_lanes_masked<4, t, VS>(ne, S, G, Y); break;
      case 5: contract_lanes_masked<5, t, VS>(ne, S, G, Y); break;
      case 6: contract_lanes_masked<6, t, VS>(ne, S, G, Y); break;
      case 7: contract_lanes_masked<7, t, VS>(ne, S, G, Y); break;
    }
  }
};

template <>
struct inexact_contract_d3<false> {
  static constexpr int NC = 81;
  // Row r, column c of the packed tangent.
  static constexpr int slot(const int r, const int c) { return r * 9 + c; }

  template <typename t>
  static SFEM_INLINE void contract(const t *const S, const t *const g, t *const y) {
    y[0] += S[0]*g[0] + S[1]*g[1] + S[2]*g[2] + S[3]*g[3] + S[4]*g[4] + S[5]*g[5] + S[6]*g[6] + S[7]*g[7] + S[8]*g[8];
    y[1] += S[9]*g[0] + S[10]*g[1] + S[11]*g[2] + S[12]*g[3] + S[13]*g[4] + S[14]*g[5] + S[15]*g[6] + S[16]*g[7] + S[17]*g[8];
    y[2] += S[18]*g[0] + S[19]*g[1] + S[20]*g[2] + S[21]*g[3] + S[22]*g[4] + S[23]*g[5] + S[24]*g[6] + S[25]*g[7] + S[26]*g[8];
    y[3] += S[27]*g[0] + S[28]*g[1] + S[29]*g[2] + S[30]*g[3] + S[31]*g[4] + S[32]*g[5] + S[33]*g[6] + S[34]*g[7] + S[35]*g[8];
    y[4] += S[36]*g[0] + S[37]*g[1] + S[38]*g[2] + S[39]*g[3] + S[40]*g[4] + S[41]*g[5] + S[42]*g[6] + S[43]*g[7] + S[44]*g[8];
    y[5] += S[45]*g[0] + S[46]*g[1] + S[47]*g[2] + S[48]*g[3] + S[49]*g[4] + S[50]*g[5] + S[51]*g[6] + S[52]*g[7] + S[53]*g[8];
    y[6] += S[54]*g[0] + S[55]*g[1] + S[56]*g[2] + S[57]*g[3] + S[58]*g[4] + S[59]*g[5] + S[60]*g[6] + S[61]*g[7] + S[62]*g[8];
    y[7] += S[63]*g[0] + S[64]*g[1] + S[65]*g[2] + S[66]*g[3] + S[67]*g[4] + S[68]*g[5] + S[69]*g[6] + S[70]*g[7] + S[71]*g[8];
    y[8] += S[72]*g[0] + S[73]*g[1] + S[74]*g[2] + S[75]*g[3] + S[76]*g[4] + S[77]*g[5] + S[78]*g[6] + S[79]*g[7] + S[80]*g[8];
  }

  // The same for a block of lanes, a loop over them per row.
  template <typename t, int VS>
  static SFEM_INLINE void contract_lanes(const int ne, const t S[81][VS], const t G[9][VS], t Y[9][VS]) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[1][lane] += S[9][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[13][lane]*G[4][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[16][lane]*G[7][lane] + S[17][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[23][lane]*G[5][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane] + S[26][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[3][lane] += S[27][lane]*G[0][lane] + S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[30][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[33][lane]*G[6][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[4][lane] += S[36][lane]*G[0][lane] + S[37][lane]*G[1][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[40][lane]*G[4][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[47][lane]*G[2][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[50][lane]*G[5][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane] + S[53][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[6][lane] += S[54][lane]*G[0][lane] + S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[57][lane]*G[3][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[60][lane]*G[6][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[7][lane] += S[63][lane]*G[0][lane] + S[64][lane]*G[1][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[67][lane]*G[4][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[70][lane]*G[7][lane] + S[71][lane]*G[8][lane];
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[74][lane]*G[2][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[77][lane]*G[5][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane] + S[80][lane]*G[8][lane];
    }
  }

  // The folded store's contraction: bit d of MASK drops the m = n = d terms.
  template <int MASK, typename t, int VS>
  static SFEM_INLINE void contract_lanes_masked(const int ne, const t S[81][VS], const t G[9][VS], t Y[9][VS]) {
    if constexpr (MASK == 0) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[13][lane]*G[4][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[16][lane]*G[7][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[23][lane]*G[5][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane] + S[26][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[27][lane]*G[0][lane] + S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[30][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[33][lane]*G[6][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[37][lane]*G[1][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[40][lane]*G[4][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[47][lane]*G[2][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[50][lane]*G[5][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane] + S[53][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[54][lane]*G[0][lane] + S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[57][lane]*G[3][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[60][lane]*G[6][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[64][lane]*G[1][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[67][lane]*G[4][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[70][lane]*G[7][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[74][lane]*G[2][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[77][lane]*G[5][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane] + S[80][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 1) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[13][lane]*G[4][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[16][lane]*G[7][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[23][lane]*G[5][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane] + S[26][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[37][lane]*G[1][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[40][lane]*G[4][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[47][lane]*G[2][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[50][lane]*G[5][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane] + S[53][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[64][lane]*G[1][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[67][lane]*G[4][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[70][lane]*G[7][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[74][lane]*G[2][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[77][lane]*G[5][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane] + S[80][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 2) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[23][lane]*G[5][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane] + S[26][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[27][lane]*G[0][lane] + S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[30][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[33][lane]*G[6][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[47][lane]*G[2][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[50][lane]*G[5][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane] + S[53][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[54][lane]*G[0][lane] + S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[57][lane]*G[3][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[60][lane]*G[6][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[74][lane]*G[2][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[77][lane]*G[5][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane] + S[80][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 3) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[20][lane]*G[2][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[23][lane]*G[5][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane] + S[26][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[47][lane]*G[2][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[50][lane]*G[5][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane] + S[53][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[74][lane]*G[2][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[77][lane]*G[5][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane] + S[80][lane]*G[8][lane];
      }
    } else if constexpr (MASK == 4) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[13][lane]*G[4][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[16][lane]*G[7][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[27][lane]*G[0][lane] + S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[30][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[33][lane]*G[6][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[37][lane]*G[1][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[40][lane]*G[4][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[54][lane]*G[0][lane] + S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[57][lane]*G[3][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[60][lane]*G[6][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[64][lane]*G[1][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[67][lane]*G[4][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[70][lane]*G[7][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 5) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[10][lane]*G[1][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[13][lane]*G[4][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[16][lane]*G[7][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[37][lane]*G[1][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[40][lane]*G[4][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[43][lane]*G[7][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[64][lane]*G[1][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[67][lane]*G[4][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[70][lane]*G[7][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 6) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[0][lane]*G[0][lane] + S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[3][lane]*G[3][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[6][lane]*G[6][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[27][lane]*G[0][lane] + S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[30][lane]*G[3][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[33][lane]*G[6][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[54][lane]*G[0][lane] + S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[57][lane]*G[3][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[60][lane]*G[6][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane];
      }
    } else if constexpr (MASK == 7) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[0][lane] += S[1][lane]*G[1][lane] + S[2][lane]*G[2][lane] + S[4][lane]*G[4][lane] + S[5][lane]*G[5][lane] + S[7][lane]*G[7][lane] + S[8][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[1][lane] += S[9][lane]*G[0][lane] + S[11][lane]*G[2][lane] + S[12][lane]*G[3][lane] + S[14][lane]*G[5][lane] + S[15][lane]*G[6][lane] + S[17][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[2][lane] += S[18][lane]*G[0][lane] + S[19][lane]*G[1][lane] + S[21][lane]*G[3][lane] + S[22][lane]*G[4][lane] + S[24][lane]*G[6][lane] + S[25][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[3][lane] += S[28][lane]*G[1][lane] + S[29][lane]*G[2][lane] + S[31][lane]*G[4][lane] + S[32][lane]*G[5][lane] + S[34][lane]*G[7][lane] + S[35][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[4][lane] += S[36][lane]*G[0][lane] + S[38][lane]*G[2][lane] + S[39][lane]*G[3][lane] + S[41][lane]*G[5][lane] + S[42][lane]*G[6][lane] + S[44][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[5][lane] += S[45][lane]*G[0][lane] + S[46][lane]*G[1][lane] + S[48][lane]*G[3][lane] + S[49][lane]*G[4][lane] + S[51][lane]*G[6][lane] + S[52][lane]*G[7][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[6][lane] += S[55][lane]*G[1][lane] + S[56][lane]*G[2][lane] + S[58][lane]*G[4][lane] + S[59][lane]*G[5][lane] + S[61][lane]*G[7][lane] + S[62][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[7][lane] += S[63][lane]*G[0][lane] + S[65][lane]*G[2][lane] + S[66][lane]*G[3][lane] + S[68][lane]*G[5][lane] + S[69][lane]*G[6][lane] + S[71][lane]*G[8][lane];
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        Y[8][lane] += S[72][lane]*G[0][lane] + S[73][lane]*G[1][lane] + S[75][lane]*G[3][lane] + S[76][lane]*G[4][lane] + S[78][lane]*G[6][lane] + S[79][lane]*G[7][lane];
      }
    }
  }

  template <typename t, int VS>
  static SFEM_INLINE void contract_lanes_masked(const int mask, const int ne, const t S[81][VS], const t G[9][VS], t Y[9][VS]) {
    switch (mask) {
      case 0: contract_lanes_masked<0, t, VS>(ne, S, G, Y); break;
      case 1: contract_lanes_masked<1, t, VS>(ne, S, G, Y); break;
      case 2: contract_lanes_masked<2, t, VS>(ne, S, G, Y); break;
      case 3: contract_lanes_masked<3, t, VS>(ne, S, G, Y); break;
      case 4: contract_lanes_masked<4, t, VS>(ne, S, G, Y); break;
      case 5: contract_lanes_masked<5, t, VS>(ne, S, G, Y); break;
      case 6: contract_lanes_masked<6, t, VS>(ne, S, G, Y); break;
      case 7: contract_lanes_masked<7, t, VS>(ne, S, G, Y); break;
    }
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
