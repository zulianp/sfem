#ifndef SFEM_LAGRANGE_HPP
#define SFEM_LAGRANGE_HPP

#include "sfem_base.h"

template <typename scalar_t>
int lagrange_eval(const int order, const int Q,
                  const scalar_t *const SFEM_RESTRICT qx,
                  scalar_t *const SFEM_RESTRICT S) {

  const int N = order + 1;
  switch (order) {
  case 1: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Sq = &S[q * N];
      Sq[0] = 1 - x;
      Sq[1] = x;
    }

    break;
  }
  case 2: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Sq = &S[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x - 1;
      Sq[0] = x0 * x1;
      Sq[1] = -4 * x * x0;
      Sq[2] = x * x1;
    }

    break;
  }
  case 4: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Sq = &S[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x - 1;
      const scalar_t x2 = 4 * x;
      const scalar_t x3 = x2 - 3;
      const scalar_t x4 = x0 * x1 * x3;
      const scalar_t x5 = x2 - 1;
      const scalar_t x6 = (1.0 / 3.0) * x5;
      const scalar_t x7 = (16.0 / 3.0) * x;
      const scalar_t x8 = x0 * x5;
      Sq[0] = x4 * x6;
      Sq[1] = -x4 * x7;
      Sq[2] = x2 * x3 * x8;
      Sq[3] = -x1 * x7 * x8;
      Sq[4] = x * x1 * x3 * x6;
    }

    break;
  }
  case 8: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Sq = &S[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x - 1;
      const scalar_t x2 = 4 * x;
      const scalar_t x3 = x2 - 1;
      const scalar_t x4 = 8 * x;
      const scalar_t x5 = x4 - 7;
      const scalar_t x6 = x4 - 5;
      const scalar_t x7 = x2 - 3;
      const scalar_t x8 = x4 - 3;
      const scalar_t x9 = x0 * x1 * x3 * x5 * x6 * x7 * x8;
      const scalar_t x10 = x4 - 1;
      const scalar_t x11 = (1.0 / 315.0) * x10;
      const scalar_t x12 = (64.0 / 315.0) * x;
      const scalar_t x13 = x0 * x1 * x10 * x6 * x7 * x8;
      const scalar_t x14 = x * x5;
      const scalar_t x15 = (16.0 / 45.0) * x14;
      const scalar_t x16 = x0 * x1 * x10 * x3 * x6;
      const scalar_t x17 = x14 * x7;
      const scalar_t x18 = (64.0 / 45.0) * x17;
      const scalar_t x19 = x0 * x10 * x3 * x8;
      const scalar_t x20 = x17 * x6;
      Sq[0] = x11 * x9;
      Sq[1] = -x12 * x9;
      Sq[2] = x13 * x15;
      Sq[3] = -x16 * x18;
      Sq[4] = (4.0 / 9.0) * x19 * x20;
      Sq[5] = -x1 * x18 * x19;
      Sq[6] = x15 * x16 * x8;
      Sq[7] = -x12 * x13 * x3;
      Sq[8] = x1 * x11 * x20 * x3 * x8;
    }

    break;
  }
  case 16: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Sq = &S[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x - 1;
      const scalar_t x2 = 4 * x;
      const scalar_t x3 = x2 - 1;
      const scalar_t x4 = 8 * x;
      const scalar_t x5 = x4 - 1;
      const scalar_t x6 = 16 * x;
      const scalar_t x7 = x6 - 15;
      const scalar_t x8 = x6 - 13;
      const scalar_t x9 = x6 - 11;
      const scalar_t x10 = x6 - 9;
      const scalar_t x11 = x4 - 7;
      const scalar_t x12 = x6 - 7;
      const scalar_t x13 = x4 - 5;
      const scalar_t x14 = x6 - 5;
      const scalar_t x15 = x2 - 3;
      const scalar_t x16 = x4 - 3;
      const scalar_t x17 = x6 - 3;
      const scalar_t x18 = x0 * x1 * x10 * x11 * x12 * x13 * x14 * x15 * x16 *
                           x17 * x3 * x5 * x7 * x8 * x9;
      const scalar_t x19 = x6 - 1;
      const scalar_t x20 = (1.0 / 638512875.0) * x19;
      const scalar_t x21 = (256.0 / 638512875.0) * x;
      const scalar_t x22 = x0 * x1 * x10 * x11 * x12 * x13 * x14 * x15 * x16 *
                           x17 * x19 * x3 * x8 * x9;
      const scalar_t x23 = x * x7;
      const scalar_t x24 = (64.0 / 42567525.0) * x23;
      const scalar_t x25 =
          x0 * x1 * x10 * x12 * x13 * x14 * x15 * x16 * x19 * x3 * x5 * x8 * x9;
      const scalar_t x26 = x11 * x23;
      const scalar_t x27 = (256.0 / 18243225.0) * x26;
      const scalar_t x28 =
          x0 * x1 * x10 * x12 * x13 * x14 * x15 * x16 * x17 * x19 * x5 * x9;
      const scalar_t x29 = x26 * x8;
      const scalar_t x30 = (16.0 / 1403325.0) * x29;
      const scalar_t x31 =
          x0 * x1 * x10 * x12 * x13 * x16 * x17 * x19 * x3 * x5 * x9;
      const scalar_t x32 = x15 * x29;
      const scalar_t x33 = (256.0 / 2338875.0) * x32;
      const scalar_t x34 =
          x0 * x1 * x10 * x12 * x13 * x14 * x17 * x19 * x3 * x5;
      const scalar_t x35 = x32 * x9;
      const scalar_t x36 = (64.0 / 637875.0) * x35;
      const scalar_t x37 = x0 * x1 * x10 * x14 * x16 * x17 * x19 * x3 * x5;
      const scalar_t x38 = x13 * x35;
      const scalar_t x39 = (256.0 / 893025.0) * x38;
      const scalar_t x40 = x0 * x12 * x14 * x16 * x17 * x19 * x3 * x5;
      const scalar_t x41 = x10 * x38;
      Sq[0] = x18 * x20;
      Sq[1] = -x18 * x21;
      Sq[2] = x22 * x24;
      Sq[3] = -x25 * x27;
      Sq[4] = x28 * x30;
      Sq[5] = -x31 * x33;
      Sq[6] = x34 * x36;
      Sq[7] = -x37 * x39;
      Sq[8] = (4.0 / 99225.0) * x40 * x41;
      Sq[9] = -x1 * x39 * x40;
      Sq[10] = x12 * x36 * x37;
      Sq[11] = -x16 * x33 * x34;
      Sq[12] = x14 * x30 * x31;
      Sq[13] = -x27 * x28 * x3;
      Sq[14] = x17 * x24 * x25;
      Sq[15] = -x21 * x22 * x5;
      Sq[16] = x1 * x12 * x14 * x16 * x17 * x20 * x3 * x41 * x5;
    }

    break;
  }
  default:
    return 1;
  }

  return 0;
}
template <typename scalar_t>
int lagrange_diff_eval(const int order, const int Q,
                       const scalar_t *const SFEM_RESTRICT qx,
                       scalar_t *const SFEM_RESTRICT D) {

  const int N = order + 1;
  switch (order) {
  case 1: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Dq = &D[q * N];
      Dq[0] = -1;
      Dq[1] = 1;
    }

    break;
  }
  case 2: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Dq = &D[q * N];
      const scalar_t x0 = 4 * x;
      Dq[0] = x0 - 3;
      Dq[1] = 4 * (1 - 2 * x);
      Dq[2] = x0 - 1;
    }

    break;
  }
  case 4: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Dq = &D[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x;
      const scalar_t x2 = x1 - 1;
      const scalar_t x3 = 4 * x;
      const scalar_t x4 = x3 - 3;
      const scalar_t x5 = x2 * x4;
      const scalar_t x6 = x0 * x5;
      const scalar_t x7 = x3 - 1;
      const scalar_t x8 = x2 * x7;
      const scalar_t x9 = x0 * x8;
      const scalar_t x10 = x4 * x7;
      const scalar_t x11 = x0 * x10;
      const scalar_t x12 = x5 * x7;
      const scalar_t x13 = x0 * x3;
      const scalar_t x14 = x13 * x2;
      const scalar_t x15 = x0 * x4;
      Dq[0] = (2.0 / 3.0) * x11 + (1.0 / 3.0) * x12 + (4.0 / 3.0) * x6 +
              (4.0 / 3.0) * x9;
      Dq[1] = -16.0 / 3.0 * x * x5 - 16.0 / 3.0 * x1 * x15 - 16.0 / 3.0 * x14 -
              16.0 / 3.0 * x6;
      Dq[2] = 4 * x * x10 + 4 * x11 + 4 * x13 * x7 + 4 * x15 * x3;
      Dq[3] = -16.0 / 3.0 * x * x8 - 16.0 / 3.0 * x0 * x1 * x7 -
              16.0 / 3.0 * x14 - 16.0 / 3.0 * x9;
      Dq[4] = (1.0 / 3.0) * x1 * x10 + (1.0 / 3.0) * x12 +
              (1.0 / 3.0) * x3 * x5 + (1.0 / 3.0) * x3 * x8;
    }

    break;
  }
  case 8: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Dq = &D[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x;
      const scalar_t x2 = x1 - 1;
      const scalar_t x3 = 4 * x;
      const scalar_t x4 = x3 - 1;
      const scalar_t x5 = 8 * x;
      const scalar_t x6 = x5 - 7;
      const scalar_t x7 = x5 - 5;
      const scalar_t x8 = x3 - 3;
      const scalar_t x9 = x5 - 3;
      const scalar_t x10 = x4 * x6 * x7 * x8 * x9;
      const scalar_t x11 = x10 * x2;
      const scalar_t x12 = x0 * x11;
      const scalar_t x13 = x5 - 1;
      const scalar_t x14 = x13 * x2;
      const scalar_t x15 = x14 * x8;
      const scalar_t x16 = x15 * x6;
      const scalar_t x17 = x16 * x7;
      const scalar_t x18 = x0 * x4;
      const scalar_t x19 = x17 * x18;
      const scalar_t x20 = x16 * x9;
      const scalar_t x21 = x18 * x20;
      const scalar_t x22 = x7 * x9;
      const scalar_t x23 = x15 * x22;
      const scalar_t x24 = x18 * x23;
      const scalar_t x25 = x6 * x7 * x9;
      const scalar_t x26 = x15 * x25;
      const scalar_t x27 = x0 * x26;
      const scalar_t x28 = x14 * x25;
      const scalar_t x29 = x18 * x28;
      const scalar_t x30 = x10 * x13;
      const scalar_t x31 = x0 * x30;
      const scalar_t x32 = x10 * x14;
      const scalar_t x33 = x6 * x7;
      const scalar_t x34 = x18 * x2;
      const scalar_t x35 = x5 * x8;
      const scalar_t x36 = x34 * x35;
      const scalar_t x37 = x33 * x36;
      const scalar_t x38 = x6 * x9;
      const scalar_t x39 = x36 * x38;
      const scalar_t x40 = x22 * x36;
      const scalar_t x41 = x25 * x3;
      const scalar_t x42 = x0 * x2 * x8;
      const scalar_t x43 = x0 * x1;
      const scalar_t x44 = x25 * x5;
      const scalar_t x45 = x0 * x5;
      const scalar_t x46 = x0 * x3;
      const scalar_t x47 = x28 * x46;
      const scalar_t x48 = x13 * x8;
      const scalar_t x49 = x18 * x5;
      const scalar_t x50 = x16 * x49;
      const scalar_t x51 = x15 * x49;
      const scalar_t x52 = x51 * x7;
      const scalar_t x53 = x18 * x33;
      const scalar_t x54 = x14 * x3;
      const scalar_t x55 = x1 * x48;
      const scalar_t x56 = x * x4;
      const scalar_t x57 = x13 * x18;
      const scalar_t x58 = x35 * x57;
      const scalar_t x59 = x51 * x9;
      const scalar_t x60 = x18 * x38;
      const scalar_t x61 = x14 * x49;
      const scalar_t x62 = x18 * x22;
      const scalar_t x63 = x4 * x5;
      Dq[0] = (8.0 / 315.0) * x12 + (8.0 / 315.0) * x19 + (8.0 / 315.0) * x21 +
              (8.0 / 315.0) * x24 + (4.0 / 315.0) * x27 + (4.0 / 315.0) * x29 +
              (2.0 / 315.0) * x31 + (1.0 / 315.0) * x32;
      Dq[1] = -64.0 / 315.0 * x * x11 - 64.0 / 315.0 * x10 * x43 -
              64.0 / 315.0 * x12 - 64.0 / 315.0 * x34 * x41 -
              64.0 / 315.0 * x37 - 64.0 / 315.0 * x39 - 64.0 / 315.0 * x40 -
              64.0 / 315.0 * x41 * x42;
      Dq[2] = (16.0 / 45.0) * x * x26 + (16.0 / 45.0) * x17 * x45 +
              (16.0 / 45.0) * x20 * x45 + (16.0 / 45.0) * x23 * x45 +
              (16.0 / 45.0) * x25 * x43 * x48 + (16.0 / 45.0) * x27 +
              (16.0 / 45.0) * x42 * x44 + (16.0 / 45.0) * x47;
      Dq[3] = -64.0 / 45.0 * x17 * x46 - 64.0 / 45.0 * x17 * x56 -
              64.0 / 45.0 * x19 - 64.0 / 45.0 * x37 - 64.0 / 45.0 * x50 -
              64.0 / 45.0 * x52 - 64.0 / 45.0 * x53 * x54 -
              64.0 / 45.0 * x53 * x55;
      Dq[4] = (4.0 / 9.0) * x * x30 + (4.0 / 9.0) * x0 * x41 * x48 +
              (4.0 / 9.0) * x10 * x45 + (4.0 / 9.0) * x13 * x35 * x53 +
              (4.0 / 9.0) * x22 * x58 + (4.0 / 9.0) * x31 +
              (4.0 / 9.0) * x38 * x58 + (4.0 / 9.0) * x41 * x57;
      Dq[5] = -64.0 / 45.0 * x20 * x46 - 64.0 / 45.0 * x20 * x56 -
              64.0 / 45.0 * x21 - 64.0 / 45.0 * x39 - 64.0 / 45.0 * x50 -
              64.0 / 45.0 * x54 * x60 - 64.0 / 45.0 * x55 * x60 -
              64.0 / 45.0 * x59;
      Dq[6] = (16.0 / 45.0) * x1 * x25 * x57 + (16.0 / 45.0) * x14 * x5 * x53 +
              (16.0 / 45.0) * x22 * x61 + (16.0 / 45.0) * x28 * x56 +
              (16.0 / 45.0) * x29 + (16.0 / 45.0) * x34 * x44 +
              (16.0 / 45.0) * x38 * x61 + (16.0 / 45.0) * x47;
      Dq[7] = -64.0 / 315.0 * x23 * x46 - 64.0 / 315.0 * x23 * x56 -
              64.0 / 315.0 * x24 - 64.0 / 315.0 * x40 - 64.0 / 315.0 * x52 -
              64.0 / 315.0 * x54 * x62 - 64.0 / 315.0 * x55 * x62 -
              64.0 / 315.0 * x59;
      Dq[8] = (1.0 / 315.0) * x1 * x30 + (1.0 / 315.0) * x11 * x5 +
              (1.0 / 315.0) * x17 * x63 + (1.0 / 315.0) * x20 * x63 +
              (1.0 / 315.0) * x23 * x63 + (1.0 / 315.0) * x26 * x3 +
              (1.0 / 315.0) * x28 * x3 * x4 + (1.0 / 315.0) * x32;
    }

    break;
  }
  case 16: {
    for (int q = 0; q < Q; q++) {
      const scalar_t x = qx[q];
      scalar_t *const Dq = &D[q * N];
      const scalar_t x0 = x - 1;
      const scalar_t x1 = 2 * x;
      const scalar_t x2 = x1 - 1;
      const scalar_t x3 = 4 * x;
      const scalar_t x4 = x3 - 1;
      const scalar_t x5 = 8 * x;
      const scalar_t x6 = x5 - 1;
      const scalar_t x7 = 16 * x;
      const scalar_t x8 = x7 - 15;
      const scalar_t x9 = x7 - 13;
      const scalar_t x10 = x7 - 11;
      const scalar_t x11 = x7 - 9;
      const scalar_t x12 = x5 - 7;
      const scalar_t x13 = x7 - 7;
      const scalar_t x14 = x5 - 5;
      const scalar_t x15 = x7 - 5;
      const scalar_t x16 = x3 - 3;
      const scalar_t x17 = x5 - 3;
      const scalar_t x18 = x7 - 3;
      const scalar_t x19 = x10 * x11 * x12 * x13 * x14 * x15 * x16 * x17 * x18 *
                           x4 * x6 * x8 * x9;
      const scalar_t x20 = x19 * x2;
      const scalar_t x21 = x0 * x20;
      const scalar_t x22 = x7 - 1;
      const scalar_t x23 = x2 * x22;
      const scalar_t x24 = x16 * x23;
      const scalar_t x25 = x10 * x11 * x12 * x13 * x14 * x15 * x17 * x8 * x9;
      const scalar_t x26 = x24 * x25;
      const scalar_t x27 = x26 * x4;
      const scalar_t x28 = x0 * x6;
      const scalar_t x29 = x27 * x28;
      const scalar_t x30 = x10 * x11 * x12 * x13 * x14 * x18 * x4 * x8 * x9;
      const scalar_t x31 = x17 * x24;
      const scalar_t x32 = x30 * x31;
      const scalar_t x33 = x28 * x32;
      const scalar_t x34 = x15 * x28;
      const scalar_t x35 = x10 * x11 * x12 * x18 * x4 * x8 * x9;
      const scalar_t x36 = x14 * x31;
      const scalar_t x37 = x35 * x36;
      const scalar_t x38 = x34 * x37;
      const scalar_t x39 = x10 * x18 * x4 * x8 * x9;
      const scalar_t x40 = x12 * x36;
      const scalar_t x41 = x39 * x40;
      const scalar_t x42 = x13 * x34;
      const scalar_t x43 = x41 * x42;
      const scalar_t x44 = x18 * x4 * x8;
      const scalar_t x45 = x40 * x9;
      const scalar_t x46 = x44 * x45;
      const scalar_t x47 = x11 * x42;
      const scalar_t x48 = x46 * x47;
      const scalar_t x49 = x40 * x44;
      const scalar_t x50 = x10 * x47;
      const scalar_t x51 = x49 * x50;
      const scalar_t x52 = x18 * x4;
      const scalar_t x53 = x45 * x52;
      const scalar_t x54 = x50 * x53;
      const scalar_t x55 =
          x10 * x11 * x12 * x13 * x14 * x15 * x17 * x18 * x4 * x8 * x9;
      const scalar_t x56 = x24 * x55;
      const scalar_t x57 = x0 * x56;
      const scalar_t x58 = x24 * x30;
      const scalar_t x59 = x34 * x58;
      const scalar_t x60 = x31 * x35;
      const scalar_t x61 = x42 * x60;
      const scalar_t x62 = x36 * x39;
      const scalar_t x63 = x47 * x62;
      const scalar_t x64 = x18 * x26;
      const scalar_t x65 = x28 * x64;
      const scalar_t x66 = x23 * x55;
      const scalar_t x67 = x28 * x66;
      const scalar_t x68 = x19 * x22;
      const scalar_t x69 = x0 * x68;
      const scalar_t x70 = x19 * x23;
      const scalar_t x71 = x16 * x2;
      const scalar_t x72 = x7 * x71;
      const scalar_t x73 = x25 * x28;
      const scalar_t x74 = x4 * x73;
      const scalar_t x75 = x72 * x74;
      const scalar_t x76 = x17 * x72;
      const scalar_t x77 = x28 * x30;
      const scalar_t x78 = x76 * x77;
      const scalar_t x79 = x14 * x76;
      const scalar_t x80 = x34 * x35;
      const scalar_t x81 = x79 * x80;
      const scalar_t x82 = x14 * x39;
      const scalar_t x83 = x12 * x42;
      const scalar_t x84 = x82 * x83;
      const scalar_t x85 = x76 * x84;
      const scalar_t x86 = x44 * x79;
      const scalar_t x87 = x12 * x9;
      const scalar_t x88 = x47 * x87;
      const scalar_t x89 = x86 * x88;
      const scalar_t x90 = x12 * x50;
      const scalar_t x91 = x86 * x90;
      const scalar_t x92 = x50 * x87;
      const scalar_t x93 = x52 * x92;
      const scalar_t x94 = x79 * x93;
      const scalar_t x95 = x0 * x5;
      const scalar_t x96 = x2 * x55;
      const scalar_t x97 = x16 * x96;
      const scalar_t x98 = x5 * x71;
      const scalar_t x99 = x30 * x34;
      const scalar_t x100 = x17 * x98;
      const scalar_t x101 = x35 * x42;
      const scalar_t x102 = x47 * x82;
      const scalar_t x103 = x28 * x3;
      const scalar_t x104 = x103 * x25;
      const scalar_t x105 = x104 * x18;
      const scalar_t x106 = x0 * x1;
      const scalar_t x107 = x0 * x7;
      const scalar_t x108 = x107 * x15;
      const scalar_t x109 = x108 * x13;
      const scalar_t x110 = x109 * x11;
      const scalar_t x111 = x10 * x110;
      const scalar_t x112 = x15 * x95;
      const scalar_t x113 = x112 * x58;
      const scalar_t x114 = x112 * x13;
      const scalar_t x115 = x114 * x60;
      const scalar_t x116 = x11 * x62;
      const scalar_t x117 = x114 * x116;
      const scalar_t x118 = x0 * x3;
      const scalar_t x119 = x16 * x22;
      const scalar_t x120 = x119 * x55;
      const scalar_t x121 = x13 * x28;
      const scalar_t x122 = x10 * x11;
      const scalar_t x123 = x4 * x8;
      const scalar_t x124 = x45 * x7;
      const scalar_t x125 = x123 * x124;
      const scalar_t x126 = x122 * x125;
      const scalar_t x127 = x121 * x126;
      const scalar_t x128 = x126 * x34;
      const scalar_t x129 = x10 * x42;
      const scalar_t x130 = x125 * x129;
      const scalar_t x131 = x125 * x47;
      const scalar_t x132 = x123 * x50;
      const scalar_t x133 = x40 * x7;
      const scalar_t x134 = x132 * x133;
      const scalar_t x135 = x124 * x50;
      const scalar_t x136 = x135 * x4;
      const scalar_t x137 = x24 * x5;
      const scalar_t x138 = x137 * x14;
      const scalar_t x139 = x123 * x92;
      const scalar_t x140 = x31 * x5;
      const scalar_t x141 = x36 * x9;
      const scalar_t x142 = x141 * x5;
      const scalar_t x143 = x1 * x119;
      const scalar_t x144 = x * x6;
      const scalar_t x145 = x18 * x73;
      const scalar_t x146 = x28 * x7;
      const scalar_t x147 = x18 * x8;
      const scalar_t x148 = x124 * x147;
      const scalar_t x149 = x121 * x122;
      const scalar_t x150 = x122 * x34;
      const scalar_t x151 = x147 * x50;
      const scalar_t x152 = x147 * x92;
      const scalar_t x153 = x105 * x23;
      const scalar_t x154 = x146 * x37;
      const scalar_t x155 = x121 * x7;
      const scalar_t x156 = x155 * x41;
      const scalar_t x157 = x11 * x46;
      const scalar_t x158 = x155 * x157;
      const scalar_t x159 = x149 * x7;
      const scalar_t x160 = x159 * x49;
      const scalar_t x161 = x159 * x53;
      const scalar_t x162 = x5 * x58;
      const scalar_t x163 = x121 * x5;
      const scalar_t x164 = x147 * x45;
      const scalar_t x165 = x122 * x13;
      const scalar_t x166 = x17 * x23;
      const scalar_t x167 = x143 * x17;
      const scalar_t x168 = x14 * x24;
      const scalar_t x169 = x168 * x7;
      const scalar_t x170 = x7 * x84;
      const scalar_t x171 = x169 * x44;
      const scalar_t x172 = x101 * x137;
      const scalar_t x173 = x102 * x137;
      const scalar_t x174 = x152 * x3;
      const scalar_t x175 = x23 * x99;
      const scalar_t x176 = x144 * x15;
      const scalar_t x177 = x34 * x7;
      const scalar_t x178 = x177 * x41;
      const scalar_t x179 = x157 * x177;
      const scalar_t x180 = x150 * x7;
      const scalar_t x181 = x180 * x49;
      const scalar_t x182 = x180 * x53;
      const scalar_t x183 = x34 * x5;
      const scalar_t x184 = x164 * x3;
      const scalar_t x185 = x14 * x80;
      const scalar_t x186 = x166 * x3;
      const scalar_t x187 = x119 * x7;
      const scalar_t x188 = x17 * x187;
      const scalar_t x189 = x14 * x188;
      const scalar_t x190 = x44 * x88;
      const scalar_t x191 = x44 * x90;
      const scalar_t x192 = x119 * x5;
      const scalar_t x193 = x17 * x192;
      const scalar_t x194 = x22 * x55;
      const scalar_t x195 = x42 * x46 * x7;
      const scalar_t x196 = x129 * x7;
      const scalar_t x197 = x196 * x49;
      const scalar_t x198 = x196 * x53;
      const scalar_t x199 = x140 * x39;
      const scalar_t x200 = x42 * x62;
      const scalar_t x201 = x13 * x176;
      const scalar_t x202 = x31 * x7;
      const scalar_t x203 = x199 * x47;
      const scalar_t x204 = x47 * x7;
      const scalar_t x205 = x204 * x49;
      const scalar_t x206 = x204 * x53;
      const scalar_t x207 = x44 * x47;
      const scalar_t x208 = x14 * x190;
      const scalar_t x209 = x11 * x201;
      const scalar_t x210 = x166 * x7;
      const scalar_t x211 = x14 * x210;
      const scalar_t x212 = x166 * x5;
      const scalar_t x213 = x50 * x52;
      const scalar_t x214 = x133 * x213;
      const scalar_t x215 = x114 * x122;
      const scalar_t x216 = x36 * x44 * x50;
      const scalar_t x217 = x151 * x3;
      const scalar_t x218 = x14 * x191;
      const scalar_t x219 = x10 * x209;
      const scalar_t x220 = x141 * x7;
      const scalar_t x221 = x14 * x93;
      const scalar_t x222 = x6 * x7;
      const scalar_t x223 = x15 * x6;
      const scalar_t x224 = x223 * x7;
      const scalar_t x225 = x13 * x223;
      const scalar_t x226 = x225 * x7;
      const scalar_t x227 = x165 * x224;
      const scalar_t x228 = x225 * x5;
      const scalar_t x229 = x3 * x6;
      Dq[0] = (16.0 / 638512875.0) * x21 + (16.0 / 638512875.0) * x29 +
              (16.0 / 638512875.0) * x33 + (16.0 / 638512875.0) * x38 +
              (16.0 / 638512875.0) * x43 + (16.0 / 638512875.0) * x48 +
              (16.0 / 638512875.0) * x51 + (16.0 / 638512875.0) * x54 +
              (8.0 / 638512875.0) * x57 + (8.0 / 638512875.0) * x59 +
              (8.0 / 638512875.0) * x61 + (8.0 / 638512875.0) * x63 +
              (4.0 / 638512875.0) * x65 + (4.0 / 638512875.0) * x67 +
              (2.0 / 638512875.0) * x69 + (1.0 / 638512875.0) * x70;
      Dq[1] =
          -256.0 / 638512875.0 * x * x20 - 256.0 / 638512875.0 * x100 * x101 -
          256.0 / 638512875.0 * x100 * x102 - 256.0 / 638512875.0 * x103 * x96 -
          256.0 / 638512875.0 * x105 * x71 - 256.0 / 638512875.0 * x106 * x19 -
          256.0 / 638512875.0 * x21 - 256.0 / 638512875.0 * x75 -
          256.0 / 638512875.0 * x78 - 256.0 / 638512875.0 * x81 -
          256.0 / 638512875.0 * x85 - 256.0 / 638512875.0 * x89 -
          256.0 / 638512875.0 * x91 - 256.0 / 638512875.0 * x94 -
          256.0 / 638512875.0 * x95 * x97 - 256.0 / 638512875.0 * x98 * x99;
      Dq[2] =
          (64.0 / 42567525.0) * x * x56 + (64.0 / 42567525.0) * x106 * x120 +
          (64.0 / 42567525.0) * x107 * x27 + (64.0 / 42567525.0) * x107 * x32 +
          (64.0 / 42567525.0) * x107 * x97 + (64.0 / 42567525.0) * x108 * x37 +
          (64.0 / 42567525.0) * x109 * x41 + (64.0 / 42567525.0) * x110 * x46 +
          (64.0 / 42567525.0) * x111 * x49 + (64.0 / 42567525.0) * x111 * x53 +
          (64.0 / 42567525.0) * x113 + (64.0 / 42567525.0) * x115 +
          (64.0 / 42567525.0) * x117 + (64.0 / 42567525.0) * x118 * x64 +
          (64.0 / 42567525.0) * x118 * x66 + (64.0 / 42567525.0) * x57;
      Dq[3] = -256.0 / 18243225.0 * x103 * x26 -
              256.0 / 18243225.0 * x104 * x23 * x4 - 256.0 / 18243225.0 * x127 -
              256.0 / 18243225.0 * x128 - 256.0 / 18243225.0 * x130 -
              256.0 / 18243225.0 * x131 - 256.0 / 18243225.0 * x132 * x142 -
              256.0 / 18243225.0 * x134 - 256.0 / 18243225.0 * x136 -
              256.0 / 18243225.0 * x138 * x139 -
              256.0 / 18243225.0 * x139 * x140 -
              256.0 / 18243225.0 * x143 * x74 -
              256.0 / 18243225.0 * x144 * x27 - 256.0 / 18243225.0 * x27 * x95 -
              256.0 / 18243225.0 * x29 - 256.0 / 18243225.0 * x75;
      Dq[4] =
          (16.0 / 1403325.0) * x129 * x148 + (16.0 / 1403325.0) * x133 * x151 +
          (16.0 / 1403325.0) * x135 * x18 + (16.0 / 1403325.0) * x138 * x152 +
          (16.0 / 1403325.0) * x140 * x152 + (16.0 / 1403325.0) * x142 * x151 +
          (16.0 / 1403325.0) * x143 * x145 + (16.0 / 1403325.0) * x144 * x64 +
          (16.0 / 1403325.0) * x145 * x72 + (16.0 / 1403325.0) * x146 * x26 +
          (16.0 / 1403325.0) * x148 * x149 + (16.0 / 1403325.0) * x148 * x150 +
          (16.0 / 1403325.0) * x148 * x47 + (16.0 / 1403325.0) * x153 +
          (16.0 / 1403325.0) * x64 * x95 + (16.0 / 1403325.0) * x65;
      Dq[5] = -256.0 / 2338875.0 * x103 * x164 * x165 -
              256.0 / 2338875.0 * x103 * x166 * x30 -
              256.0 / 2338875.0 * x116 * x163 - 256.0 / 2338875.0 * x127 -
              256.0 / 2338875.0 * x144 * x32 - 256.0 / 2338875.0 * x154 -
              256.0 / 2338875.0 * x156 - 256.0 / 2338875.0 * x158 -
              256.0 / 2338875.0 * x160 - 256.0 / 2338875.0 * x161 -
              256.0 / 2338875.0 * x162 * x28 - 256.0 / 2338875.0 * x163 * x60 -
              256.0 / 2338875.0 * x167 * x77 - 256.0 / 2338875.0 * x32 * x95 -
              256.0 / 2338875.0 * x33 - 256.0 / 2338875.0 * x78;
      Dq[6] = (64.0 / 637875.0) * x113 + (64.0 / 637875.0) * x139 * x169 +
              (64.0 / 637875.0) * x143 * x99 + (64.0 / 637875.0) * x146 * x58 +
              (64.0 / 637875.0) * x168 * x174 + (64.0 / 637875.0) * x169 * x80 +
              (64.0 / 637875.0) * x169 * x93 + (64.0 / 637875.0) * x170 * x24 +
              (64.0 / 637875.0) * x171 * x88 + (64.0 / 637875.0) * x171 * x90 +
              (64.0 / 637875.0) * x172 + (64.0 / 637875.0) * x173 +
              (64.0 / 637875.0) * x175 * x3 + (64.0 / 637875.0) * x176 * x58 +
              (64.0 / 637875.0) * x59 + (64.0 / 637875.0) * x72 * x99;
      Dq[7] = -256.0 / 893025.0 * x112 * x37 - 256.0 / 893025.0 * x116 * x183 -
              256.0 / 893025.0 * x128 - 256.0 / 893025.0 * x138 * x80 -
              256.0 / 893025.0 * x150 * x184 - 256.0 / 893025.0 * x154 -
              256.0 / 893025.0 * x167 * x185 - 256.0 / 893025.0 * x176 * x37 -
              256.0 / 893025.0 * x178 - 256.0 / 893025.0 * x179 -
              256.0 / 893025.0 * x181 - 256.0 / 893025.0 * x182 -
              256.0 / 893025.0 * x183 * x60 - 256.0 / 893025.0 * x185 * x186 -
              256.0 / 893025.0 * x38 - 256.0 / 893025.0 * x81;
      Dq[8] = (4.0 / 99225.0) * x * x68 + (4.0 / 99225.0) * x101 * x193 +
              (4.0 / 99225.0) * x102 * x193 + (4.0 / 99225.0) * x103 * x194 +
              (4.0 / 99225.0) * x105 * x119 + (4.0 / 99225.0) * x107 * x19 +
              (4.0 / 99225.0) * x119 * x17 * x170 +
              (4.0 / 99225.0) * x120 * x95 + (4.0 / 99225.0) * x185 * x188 +
              (4.0 / 99225.0) * x187 * x74 + (4.0 / 99225.0) * x188 * x77 +
              (4.0 / 99225.0) * x189 * x190 + (4.0 / 99225.0) * x189 * x191 +
              (4.0 / 99225.0) * x189 * x93 + (4.0 / 99225.0) * x192 * x99 +
              (4.0 / 99225.0) * x69;
      Dq[9] = -256.0 / 893025.0 * x114 * x41 - 256.0 / 893025.0 * x129 * x184 -
              256.0 / 893025.0 * x130 - 256.0 / 893025.0 * x137 * x84 -
              256.0 / 893025.0 * x156 - 256.0 / 893025.0 * x167 * x84 -
              256.0 / 893025.0 * x178 - 256.0 / 893025.0 * x186 * x84 -
              256.0 / 893025.0 * x195 - 256.0 / 893025.0 * x197 -
              256.0 / 893025.0 * x198 - 256.0 / 893025.0 * x199 * x83 -
              256.0 / 893025.0 * x200 * x5 - 256.0 / 893025.0 * x201 * x41 -
              256.0 / 893025.0 * x43 - 256.0 / 893025.0 * x85;
      Dq[10] =
          (64.0 / 637875.0) * x101 * x167 + (64.0 / 637875.0) * x101 * x186 +
          (64.0 / 637875.0) * x101 * x76 + (64.0 / 637875.0) * x115 +
          (64.0 / 637875.0) * x139 * x202 + (64.0 / 637875.0) * x155 * x60 +
          (64.0 / 637875.0) * x172 + (64.0 / 637875.0) * x174 * x31 +
          (64.0 / 637875.0) * x177 * x60 + (64.0 / 637875.0) * x190 * x202 +
          (64.0 / 637875.0) * x191 * x202 + (64.0 / 637875.0) * x201 * x60 +
          (64.0 / 637875.0) * x202 * x39 * x83 +
          (64.0 / 637875.0) * x202 * x93 + (64.0 / 637875.0) * x203 +
          (64.0 / 637875.0) * x61;
      Dq[11] =
          -256.0 / 2338875.0 * x114 * x157 - 256.0 / 2338875.0 * x131 -
          256.0 / 2338875.0 * x138 * x190 - 256.0 / 2338875.0 * x140 * x190 -
          256.0 / 2338875.0 * x142 * x207 - 256.0 / 2338875.0 * x158 -
          256.0 / 2338875.0 * x167 * x208 - 256.0 / 2338875.0 * x179 -
          256.0 / 2338875.0 * x184 * x47 - 256.0 / 2338875.0 * x186 * x208 -
          256.0 / 2338875.0 * x195 - 256.0 / 2338875.0 * x205 -
          256.0 / 2338875.0 * x206 - 256.0 / 2338875.0 * x209 * x46 -
          256.0 / 2338875.0 * x48 - 256.0 / 2338875.0 * x89;
      Dq[12] =
          (16.0 / 1403325.0) * x1 * x194 * x28 +
          (16.0 / 1403325.0) * x101 * x212 + (16.0 / 1403325.0) * x102 * x212 +
          (16.0 / 1403325.0) * x144 * x66 + (16.0 / 1403325.0) * x146 * x96 +
          (16.0 / 1403325.0) * x153 + (16.0 / 1403325.0) * x166 * x170 +
          (16.0 / 1403325.0) * x175 * x5 + (16.0 / 1403325.0) * x185 * x210 +
          (16.0 / 1403325.0) * x191 * x211 + (16.0 / 1403325.0) * x208 * x210 +
          (16.0 / 1403325.0) * x210 * x77 + (16.0 / 1403325.0) * x211 * x93 +
          (16.0 / 1403325.0) * x23 * x7 * x74 + (16.0 / 1403325.0) * x66 * x95 +
          (16.0 / 1403325.0) * x67;
      Dq[13] =
          -256.0 / 18243225.0 * x134 - 256.0 / 18243225.0 * x138 * x191 -
          256.0 / 18243225.0 * x140 * x191 - 256.0 / 18243225.0 * x160 -
          256.0 / 18243225.0 * x167 * x218 - 256.0 / 18243225.0 * x181 -
          256.0 / 18243225.0 * x186 * x218 - 256.0 / 18243225.0 * x197 -
          256.0 / 18243225.0 * x205 - 256.0 / 18243225.0 * x214 -
          256.0 / 18243225.0 * x215 * x49 - 256.0 / 18243225.0 * x216 * x5 -
          256.0 / 18243225.0 * x217 * x40 - 256.0 / 18243225.0 * x219 * x49 -
          256.0 / 18243225.0 * x51 - 256.0 / 18243225.0 * x91;
      Dq[14] =
          (64.0 / 42567525.0) * x102 * x167 +
          (64.0 / 42567525.0) * x102 * x186 + (64.0 / 42567525.0) * x102 * x76 +
          (64.0 / 42567525.0) * x116 * x155 +
          (64.0 / 42567525.0) * x116 * x177 + (64.0 / 42567525.0) * x117 +
          (64.0 / 42567525.0) * x132 * x220 +
          (64.0 / 42567525.0) * x141 * x217 + (64.0 / 42567525.0) * x173 +
          (64.0 / 42567525.0) * x200 * x7 + (64.0 / 42567525.0) * x203 +
          (64.0 / 42567525.0) * x207 * x220 + (64.0 / 42567525.0) * x209 * x62 +
          (64.0 / 42567525.0) * x213 * x220 + (64.0 / 42567525.0) * x216 * x7 +
          (64.0 / 42567525.0) * x63;
      Dq[15] = -256.0 / 638512875.0 * x136 - 256.0 / 638512875.0 * x138 * x93 -
               256.0 / 638512875.0 * x140 * x93 -
               256.0 / 638512875.0 * x142 * x213 - 256.0 / 638512875.0 * x161 -
               256.0 / 638512875.0 * x167 * x221 -
               256.0 / 638512875.0 * x18 * x3 * x45 * x50 -
               256.0 / 638512875.0 * x182 - 256.0 / 638512875.0 * x186 * x221 -
               256.0 / 638512875.0 * x198 - 256.0 / 638512875.0 * x206 -
               256.0 / 638512875.0 * x214 - 256.0 / 638512875.0 * x215 * x53 -
               256.0 / 638512875.0 * x219 * x53 - 256.0 / 638512875.0 * x54 -
               256.0 / 638512875.0 * x94;
      Dq[16] =
          (1.0 / 638512875.0) * x1 * x68 + (1.0 / 638512875.0) * x116 * x228 +
          (1.0 / 638512875.0) * x157 * x226 +
          (1.0 / 638512875.0) * x162 * x223 + (1.0 / 638512875.0) * x20 * x7 +
          (1.0 / 638512875.0) * x222 * x27 + (1.0 / 638512875.0) * x222 * x32 +
          (1.0 / 638512875.0) * x224 * x37 + (1.0 / 638512875.0) * x226 * x41 +
          (1.0 / 638512875.0) * x227 * x49 + (1.0 / 638512875.0) * x227 * x53 +
          (1.0 / 638512875.0) * x228 * x60 + (1.0 / 638512875.0) * x229 * x64 +
          (1.0 / 638512875.0) * x229 * x66 + (1.0 / 638512875.0) * x5 * x56 +
          (1.0 / 638512875.0) * x70;
    }

    break;
  }
  default:
    return 1;
  }

  return 0;
}

#endif  // SFEM_LAGRANGE_HPP
