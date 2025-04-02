#ifndef SFEM_LAGRANGE_LEGENDRE_GAUSS_LOBATTO_HPP
#define SFEM_LAGRANGE_LEGENDRE_GAUSS_LOBATTO_HPP

#include "sfem_base.h"

template <typename scalar_t>
int lagrange_GLL_eval(const int order, const int Q, const scalar_t *const SFEM_RESTRICT qx, scalar_t *const SFEM_RESTRICT S) {
    const int N = order + 1;
    switch (order) {
        case 1: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x  = qx[q];
                scalar_t *const Sq = &S[q * N];
                Sq[0]              = 1 - x;
                Sq[1]              = x;
            }

            break;
        }
        case 2: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x  = qx[q];
                scalar_t *const Sq = &S[q * N];
                const scalar_t  x0 = 2.0 * x - 1.0;
                Sq[0]              = 1.0 * x0 * (x - 1.0);
                Sq[1]              = -4.0 * x * (x - 1);
                Sq[2]              = x * x0;
            }

            break;
        }
        case 4: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x  = qx[q];
                scalar_t *const Sq = &S[q * N];
                const scalar_t  x0 = 2 * x - 1;
                const scalar_t  x1 = x - 1.0;
                const scalar_t  x2 = x - 0.82732683535141405;
                const scalar_t  x3 = x1 * x2;
                const scalar_t  x4 = x0 * x3;
                const scalar_t  x5 = x * (x - 0.17267316464494797);
                const scalar_t  x6 = x0 * x5;
                Sq[0]              = 1.2087121525255498 * x4 * (5.7912878475035541 * x - 1.0);
                Sq[1]              = -16.333333333519061 * x * x4;
                Sq[2]              = 37.333333333575915 * x3 * x5;
                Sq[3]              = -16.333333333402361 * x1 * x6;
                Sq[4]              = 6.999999999886632 * x2 * x6;
            }

            break;
        }
        case 8: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x   = qx[q];
                scalar_t *const Sq  = &S[q * N];
                const scalar_t  x0  = x - 0.16140686024300521;
                const scalar_t  x1  = 2 * x - 1;
                const scalar_t  x2  = x - 1.0;
                const scalar_t  x3  = x - 0.31844126808573492;
                const scalar_t  x4  = x - 0.68155873191426508;
                const scalar_t  x5  = x - 0.94987899770785589;
                const scalar_t  x6  = x - 0.83859313975699479;
                const scalar_t  x7  = x1 * x2 * x3 * x4 * x5 * x6;
                const scalar_t  x8  = x0 * x7;
                const scalar_t  x9  = x * (x - 0.050121002295782091);
                const scalar_t  x10 = x0 * x2 * x4 * x5 * x6 * x9;
                const scalar_t  x11 = x0 * x1 * x2 * x3 * x5 * x9;
                const scalar_t  x12 = x0 * x1 * x3 * x4 * x6 * x9;
                Sq[0]               = 35.836516639813759 * x8 * (19.951715931296349 * x - 1.0);
                Sq[1]               = -1745.2200960701314 * x * x8;
                Sq[2]               = 2247.8073580213991 * x7 * x9;
                Sq[3]               = -2525.0158333200243 * x1 * x10;
                Sq[4]               = 5229.7142854659533 * x10 * x3;
                Sq[5]               = -2525.0158332805872 * x11 * x6;
                Sq[6]               = 2247.8073579598263 * x11 * x4;
                Sq[7]               = -1745.2200960521598 * x12 * x2;
                Sq[8]               = 715.00000003868695 * x12 * x5;
            }

            break;
        }
        case 16: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x   = qx[q];
                scalar_t *const Sq  = &S[q * N];
                const scalar_t  x0  = x - 0.044560002042999258;
                const scalar_t  x1  = 2 * x - 1;
                const scalar_t  x2  = x - 1.0;
                const scalar_t  x3  = x - 0.95543999795336276;
                const scalar_t  x4  = x - 0.15448550968721975;
                const scalar_t  x5  = x - 0.90784812561469153;
                const scalar_t  x6  = x - 0.092151874388946453;
                const scalar_t  x7  = x - 0.31391278321825666;
                const scalar_t  x8  = x - 0.77069269966159482;
                const scalar_t  x9  = x - 0.84551449031278025;
                const scalar_t  x10 = x - 0.22930730033476721;
                const scalar_t  x11 = x - 0.5947559867563541;
                const scalar_t  x12 = x - 0.68608721678174334;
                const scalar_t  x13 = x - 0.40524401324000792;
                const scalar_t  x14 = x - 0.98656608831515769;
                const scalar_t  x15 = x1 * x10 * x11 * x12 * x13 * x14 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x16 = x0 * x15;
                const scalar_t  x17 = x * (x - 0.013433911684842315);
                const scalar_t  x18 = x0 * x1 * x10 * x11 * x12 * x13 * x14 * x17 * x2 * x3 * x5 * x7 * x8 * x9;
                const scalar_t  x19 = x0 * x1 * x11 * x12 * x13 * x14 * x17 * x2 * x3 * x4 * x5 * x6 * x8 * x9;
                const scalar_t  x20 = x0 * x10 * x11 * x12 * x14 * x17 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x21 = x0 * x1 * x10 * x13 * x14 * x17 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x22 = x0 * x1 * x10 * x11 * x12 * x13 * x14 * x17 * x2 * x3 * x4 * x5 * x6 * x7;
                const scalar_t  x23 = x0 * x1 * x10 * x11 * x12 * x13 * x14 * x17 * x2 * x4 * x6 * x7 * x8 * x9;
                const scalar_t  x24 = x0 * x1 * x10 * x11 * x12 * x13 * x17 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                Sq[0]               = 237495.90806882046 * x16 * (74.438482510857284 * x - 1.0);
                Sq[1]               = -43697073.381567873 * x * x16;
                Sq[2]               = 58020430.326339468 * x15 * x17;
                Sq[3]               = -68562514.624906674 * x18 * x4;
                Sq[4]               = 76584936.622250274 * x18 * x6;
                Sq[5]               = -82569195.289751619 * x19 * x7;
                Sq[6]               = 86739107.286408588 * x10 * x19;
                Sq[7]               = -89206186.022886083 * x1 * x20;
                Sq[8]               = 180046640.34226343 * x13 * x20;
                Sq[9]               = -89206186.021877989 * x12 * x21;
                Sq[10]              = 86739107.284455061 * x11 * x21;
                Sq[11]              = -82569195.284388319 * x22 * x9;
                Sq[12]              = 76584936.614642844 * x22 * x8;
                Sq[13]              = -68562514.626742154 * x23 * x3;
                Sq[14]              = 58020430.328498803 * x23 * x5;
                Sq[15]              = -43697073.376259953 * x2 * x24;
                Sq[16]              = 17678834.997722976 * x14 * x24;
            }

            break;
        }
        default:
            return 1;
    }

    return 0;
}
template <typename scalar_t>
int lagrange_GLL_diff_eval(const int                           order,
                           const int                           Q,
                           const scalar_t *const SFEM_RESTRICT qx,
                           scalar_t *const SFEM_RESTRICT       D) {
    const int N = order + 1;
    switch (order) {
        case 1: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x  = qx[q];
                scalar_t *const Dq = &D[q * N];
                Dq[0]              = -1;
                Dq[1]              = 1;
            }

            break;
        }
        case 2: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x  = qx[q];
                scalar_t *const Dq = &D[q * N];
                const scalar_t  x0 = 4.0 * x;
                Dq[0]              = x0 - 3.0;
                Dq[1]              = 4.0 - 8.0 * x;
                Dq[2]              = x0 - 1.0;
            }

            break;
        }
        case 4: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x   = qx[q];
                scalar_t *const Dq  = &D[q * N];
                const scalar_t  x0  = x - 0.82732683535141405;
                const scalar_t  x1  = x - 1.0;
                const scalar_t  x2  = 5.7912878475035541 * x - 1.0;
                const scalar_t  x3  = x1 * x2;
                const scalar_t  x4  = 2 * x - 1;
                const scalar_t  x5  = x0 * x4;
                const scalar_t  x6  = x1 * x5;
                const scalar_t  x7  = 1.2087121525255498 * x4;
                const scalar_t  x8  = x * x1;
                const scalar_t  x9  = x0 * x8;
                const scalar_t  x10 = x4 * x8;
                const scalar_t  x11 = x * x5;
                const scalar_t  x12 = x - 0.17267316464494797;
                const scalar_t  x13 = x12 * x8;
                const scalar_t  x14 = x0 * x12;
                const scalar_t  x15 = x * x14;
                const scalar_t  x16 = 16.333333333402361 * x12 * x4;
                const scalar_t  x17 = 6.999999999886632 * x4;
                Dq[0]               = x0 * x2 * x7 + 2.4174243050510995 * x0 * x3 + x3 * x7 + 7.0000000000510783 * x6;
                Dq[1] = -16.333333333519061 * x10 - 16.333333333519061 * x11 - 16.333333333519061 * x6 - 32.666666667038122 * x9;
                Dq[2] = 37.333333333575915 * x1 * x14 + 37.333333333575915 * x13 + 37.333333333575915 * x15 +
                        37.333333333575915 * x9;
                Dq[3] = -x * x16 - x1 * x16 - 16.333333333402361 * x10 - 32.666666666804723 * x13;
                Dq[4] = x * x12 * x17 + 6.999999999886632 * x11 + x14 * x17 + 13.999999999773264 * x15;
            }

            break;
        }
        case 8: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x   = qx[q];
                scalar_t *const Dq  = &D[q * N];
                const scalar_t  x0  = 19.951715931296349 * x - 1.0;
                const scalar_t  x1  = x - 0.83859313975699479;
                const scalar_t  x2  = x - 0.94987899770785589;
                const scalar_t  x3  = x - 1.0;
                const scalar_t  x4  = x - 0.31844126808573492;
                const scalar_t  x5  = x - 0.16140686024300521;
                const scalar_t  x6  = x - 0.68155873191426508;
                const scalar_t  x7  = x3 * x4 * x5 * x6;
                const scalar_t  x8  = x2 * x7;
                const scalar_t  x9  = x1 * x8;
                const scalar_t  x10 = 2 * x - 1;
                const scalar_t  x11 = x10 * x9;
                const scalar_t  x12 = x2 * x3 * x4;
                const scalar_t  x13 = x12 * x6;
                const scalar_t  x14 = 35.836516639813759 * x0;
                const scalar_t  x15 = x1 * x10;
                const scalar_t  x16 = x14 * x15;
                const scalar_t  x17 = x16 * x5;
                const scalar_t  x18 = x2 * x6;
                const scalar_t  x19 = x18 * x3;
                const scalar_t  x20 = x10 * x8;
                const scalar_t  x21 = x18 * x4;
                const scalar_t  x22 = x * x9;
                const scalar_t  x23 = 1745.2200960701314 * x;
                const scalar_t  x24 = x15 * x23;
                const scalar_t  x25 = x24 * x5;
                const scalar_t  x26 = x - 0.050121002295782091;
                const scalar_t  x27 = x * x26;
                const scalar_t  x28 = x1 * x27;
                const scalar_t  x29 = x13 * x28;
                const scalar_t  x30 = x * x15;
                const scalar_t  x31 = 2247.8073580213991 * x13;
                const scalar_t  x32 = 2247.8073580213991 * x10;
                const scalar_t  x33 = x28 * x32;
                const scalar_t  x34 = x13 * x27;
                const scalar_t  x35 = x4 * x6;
                const scalar_t  x36 = x3 * x35;
                const scalar_t  x37 = x15 * x26;
                const scalar_t  x38 = x28 * x5;
                const scalar_t  x39 = x3 * x38;
                const scalar_t  x40 = x18 * x39;
                const scalar_t  x41 = x30 * x5;
                const scalar_t  x42 = 2525.0158333200243 * x19;
                const scalar_t  x43 = 2525.0158333200243 * x10;
                const scalar_t  x44 = x19 * x43;
                const scalar_t  x45 = x39 * x43;
                const scalar_t  x46 = x27 * x5;
                const scalar_t  x47 = x18 * x38;
                const scalar_t  x48 = x37 * x5;
                const scalar_t  x49 = x12 * x38;
                const scalar_t  x50 = x27 * x8;
                const scalar_t  x51 = x28 * x7;
                const scalar_t  x52 = x21 * x38;
                const scalar_t  x53 = 2525.0158332805872 * x12;
                const scalar_t  x54 = 2525.0158332805872 * x10;
                const scalar_t  x55 = x12 * x54;
                const scalar_t  x56 = x39 * x54;
                const scalar_t  x57 = x2 * x38 * x4;
                const scalar_t  x58 = 2247.8073579598263 * x20;
                const scalar_t  x59 = 2247.8073579598263 * x10;
                const scalar_t  x60 = x46 * x59;
                const scalar_t  x61 = x27 * x7;
                const scalar_t  x62 = 1745.2200960521598 * x7;
                const scalar_t  x63 = 1745.2200960521598 * x10;
                const scalar_t  x64 = x39 * x63;
                const scalar_t  x65 = x35 * x38;
                const scalar_t  x66 = 715.00000003868695 * x21;
                const scalar_t  x67 = 715.00000003868695 * x10;
                const scalar_t  x68 = x21 * x67;
                Dq[0] = 71.673033279627518 * x0 * x9 + 714.99999996473889 * x11 + x12 * x17 + x13 * x16 + x14 * x20 + x16 * x7 +
                        x17 * x19 + x17 * x21;
                Dq[1] = -1745.2200960701314 * x11 - x12 * x25 - x13 * x24 - x19 * x25 - x20 * x23 - x21 * x25 -
                        3490.4401921402628 * x22 - x24 * x7;
                Dq[2] = x12 * x33 + x19 * x33 + x21 * x33 + 4495.6147160427981 * x29 + x30 * x31 + x31 * x37 + x32 * x34 +
                        x33 * x36;
                Dq[3] = -x2 * x45 - x28 * x44 - 5050.0316666400486 * x40 - x41 * x42 - x42 * x48 - x43 * x47 - x44 * x46 -
                        x45 * x6;
                Dq[4] = 5229.7142854659533 * x22 + 5229.7142854659533 * x26 * x9 + 5229.7142854659533 * x29 +
                        5229.7142854659533 * x40 + 5229.7142854659533 * x49 + 5229.7142854659533 * x50 +
                        5229.7142854659533 * x51 + 5229.7142854659533 * x52;
                Dq[5] = -x2 * x56 - x28 * x55 - x4 * x56 - x41 * x53 - x46 * x55 - x48 * x53 - 5050.0316665611745 * x49 -
                        x54 * x57;
                Dq[6] = x * x58 + x12 * x60 + x19 * x60 + x21 * x60 + x26 * x58 + x34 * x59 + 4495.6147159196526 * x50 +
                        x59 * x61;
                Dq[7] = -x28 * x36 * x63 - x30 * x62 - x37 * x62 - x4 * x64 - 3490.4401921043195 * x51 - x6 * x64 - x61 * x63 -
                        x63 * x65;
                Dq[8] = x28 * x68 + x41 * x66 + x46 * x68 + x47 * x67 + x48 * x66 + 1430.0000000773739 * x52 + x57 * x67 +
                        x65 * x67;
            }

            break;
        }
        case 16: {
            for (int q = 0; q < Q; q++) {
                const scalar_t  x    = qx[q];
                scalar_t *const Dq   = &D[q * N];
                const scalar_t  x0   = 74.438482510857284 * x - 1.0;
                const scalar_t  x1   = x - 0.98656608831515769;
                const scalar_t  x2   = x - 0.40524401324000792;
                const scalar_t  x3   = x - 1.0;
                const scalar_t  x4   = x - 0.95543999795336276;
                const scalar_t  x5   = x - 0.15448550968721975;
                const scalar_t  x6   = x - 0.044560002042999258;
                const scalar_t  x7   = x - 0.90784812561469153;
                const scalar_t  x8   = x - 0.092151874388946453;
                const scalar_t  x9   = x - 0.31391278321825666;
                const scalar_t  x10  = x - 0.77069269966159482;
                const scalar_t  x11  = x - 0.84551449031278025;
                const scalar_t  x12  = x - 0.22930730033476721;
                const scalar_t  x13  = x - 0.5947559867563541;
                const scalar_t  x14  = x - 0.68608721678174334;
                const scalar_t  x15  = x10 * x11 * x12 * x13 * x14 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x16  = x15 * x2;
                const scalar_t  x17  = x1 * x16;
                const scalar_t  x18  = 2 * x - 1;
                const scalar_t  x19  = x17 * x18;
                const scalar_t  x20  = x10 * x12 * x14 * x2 * x3 * x4 * x8;
                const scalar_t  x21  = 237495.90806882046 * x0;
                const scalar_t  x22  = x1 * x18;
                const scalar_t  x23  = x21 * x22;
                const scalar_t  x24  = x13 * x23;
                const scalar_t  x25  = x11 * x24;
                const scalar_t  x26  = x25 * x9;
                const scalar_t  x27  = x26 * x7;
                const scalar_t  x28  = x27 * x5;
                const scalar_t  x29  = x10 * x12 * x14 * x2 * x3 * x4 * x5 * x6;
                const scalar_t  x30  = x20 * x6;
                const scalar_t  x31  = x10 * x14 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x32  = x12 * x14 * x2 * x3 * x4 * x5 * x6 * x7 * x8;
                const scalar_t  x33  = x10 * x32;
                const scalar_t  x34  = x10 * x11 * x12 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9;
                const scalar_t  x35  = x14 * x34;
                const scalar_t  x36  = x12 * x31;
                const scalar_t  x37  = x29 * x8;
                const scalar_t  x38  = x10 * x12 * x14 * x2 * x6 * x8;
                const scalar_t  x39  = x3 * x38;
                const scalar_t  x40  = x16 * x18;
                const scalar_t  x41  = x38 * x4;
                const scalar_t  x42  = x * x17;
                const scalar_t  x43  = 43697073.381567873 * x;
                const scalar_t  x44  = x22 * x43;
                const scalar_t  x45  = x13 * x44;
                const scalar_t  x46  = x11 * x45;
                const scalar_t  x47  = x46 * x9;
                const scalar_t  x48  = x47 * x7;
                const scalar_t  x49  = x48 * x5;
                const scalar_t  x50  = x - 0.013433911684842315;
                const scalar_t  x51  = x * x50;
                const scalar_t  x52  = x1 * x51;
                const scalar_t  x53  = x13 * x52;
                const scalar_t  x54  = x11 * x53;
                const scalar_t  x55  = x54 * x9;
                const scalar_t  x56  = x55 * x7;
                const scalar_t  x57  = x5 * x56;
                const scalar_t  x58  = x20 * x57;
                const scalar_t  x59  = x11 * x13;
                const scalar_t  x60  = x7 * x9;
                const scalar_t  x61  = x59 * x60;
                const scalar_t  x62  = x22 * x61;
                const scalar_t  x63  = x * x62;
                const scalar_t  x64  = x20 * x5;
                const scalar_t  x65  = 58020430.326339468 * x64;
                const scalar_t  x66  = 58020430.326339468 * x18;
                const scalar_t  x67  = x3 * x57;
                const scalar_t  x68  = x10 * x12 * x4 * x67;
                const scalar_t  x69  = x66 * x68;
                const scalar_t  x70  = x2 * x69;
                const scalar_t  x71  = x20 * x66;
                const scalar_t  x72  = x14 * x8;
                const scalar_t  x73  = x10 * x72;
                const scalar_t  x74  = x2 * x4 * x67;
                const scalar_t  x75  = x66 * x74;
                const scalar_t  x76  = x5 * x71;
                const scalar_t  x77  = x54 * x7;
                const scalar_t  x78  = x60 * x76;
                const scalar_t  x79  = x11 * x52;
                const scalar_t  x80  = x12 * x72;
                const scalar_t  x81  = x67 * x73;
                const scalar_t  x82  = x12 * x2;
                const scalar_t  x83  = x66 * x82;
                const scalar_t  x84  = x51 * x61;
                const scalar_t  x85  = x4 * x57;
                const scalar_t  x86  = x73 * x85;
                const scalar_t  x87  = x50 * x62;
                const scalar_t  x88  = x29 * x56;
                const scalar_t  x89  = 68562514.624906674 * x29;
                const scalar_t  x90  = 68562514.624906674 * x18;
                const scalar_t  x91  = x68 * x90;
                const scalar_t  x92  = x2 * x91;
                const scalar_t  x93  = x14 * x6;
                const scalar_t  x94  = x10 * x93;
                const scalar_t  x95  = x82 * x94;
                const scalar_t  x96  = x90 * x95;
                const scalar_t  x97  = x3 * x4;
                const scalar_t  x98  = x56 * x97;
                const scalar_t  x99  = x74 * x90;
                const scalar_t  x100 = x29 * x90;
                const scalar_t  x101 = x100 * x60;
                const scalar_t  x102 = x12 * x93;
                const scalar_t  x103 = x30 * x56;
                const scalar_t  x104 = 76584936.622250274 * x30;
                const scalar_t  x105 = 76584936.622250274 * x18;
                const scalar_t  x106 = x105 * x56;
                const scalar_t  x107 = x105 * x98;
                const scalar_t  x108 = x6 * x73;
                const scalar_t  x109 = x108 * x2;
                const scalar_t  x110 = x105 * x30;
                const scalar_t  x111 = x108 * x12;
                const scalar_t  x112 = x110 * x60;
                const scalar_t  x113 = x107 * x6;
                const scalar_t  x114 = x10 * x8;
                const scalar_t  x115 = x114 * x82;
                const scalar_t  x116 = x2 * x80;
                const scalar_t  x117 = x31 * x54;
                const scalar_t  x118 = x22 * x59;
                const scalar_t  x119 = x * x118;
                const scalar_t  x120 = 82569195.289751619 * x31;
                const scalar_t  x121 = 82569195.289751619 * x18;
                const scalar_t  x122 = x121 * x74;
                const scalar_t  x123 = x109 * x121;
                const scalar_t  x124 = x5 * x77;
                const scalar_t  x125 = x124 * x97;
                const scalar_t  x126 = x121 * x6;
                const scalar_t  x127 = x126 * x81;
                const scalar_t  x128 = x121 * x31;
                const scalar_t  x129 = x126 * x74;
                const scalar_t  x130 = x5 * x97;
                const scalar_t  x131 = x130 * x55;
                const scalar_t  x132 = x51 * x59;
                const scalar_t  x133 = x2 * x86;
                const scalar_t  x134 = x118 * x50;
                const scalar_t  x135 = x33 * x54;
                const scalar_t  x136 = 86739107.286408588 * x33;
                const scalar_t  x137 = 86739107.286408588 * x18;
                const scalar_t  x138 = x137 * x77;
                const scalar_t  x139 = x125 * x137;
                const scalar_t  x140 = x137 * x33;
                const scalar_t  x141 = x115 * x6;
                const scalar_t  x142 = x137 * x54;
                const scalar_t  x143 = x124 * x137;
                const scalar_t  x144 = x15 * x52;
                const scalar_t  x145 = x * x22;
                const scalar_t  x146 = 89206186.022886083 * x15;
                const scalar_t  x147 = 89206186.022886083 * x18;
                const scalar_t  x148 = x147 * x68;
                const scalar_t  x149 = x111 * x147;
                const scalar_t  x150 = x147 * x6;
                const scalar_t  x151 = x150 * x4;
                const scalar_t  x152 = x60 * x79;
                const scalar_t  x153 = x130 * x149;
                const scalar_t  x154 = x68 * x8;
                const scalar_t  x155 = x67 * x80;
                const scalar_t  x156 = x53 * x60;
                const scalar_t  x157 = x12 * x150;
                const scalar_t  x158 = x15 * x51;
                const scalar_t  x159 = x22 * x50;
                const scalar_t  x160 = x35 * x52;
                const scalar_t  x161 = x34 * x53;
                const scalar_t  x162 = x32 * x55;
                const scalar_t  x163 = x36 * x53;
                const scalar_t  x164 = x37 * x55;
                const scalar_t  x165 = x38 * x67;
                const scalar_t  x166 = x16 * x51;
                const scalar_t  x167 = x41 * x57;
                const scalar_t  x168 = 89206186.021877989 * x35;
                const scalar_t  x169 = 89206186.021877989 * x18;
                const scalar_t  x170 = x152 * x169;
                const scalar_t  x171 = x169 * x79;
                const scalar_t  x172 = x111 * x130;
                const scalar_t  x173 = x169 * x52;
                const scalar_t  x174 = x171 * x9;
                const scalar_t  x175 = x170 * x5;
                const scalar_t  x176 = x35 * x51;
                const scalar_t  x177 = x13 * x145;
                const scalar_t  x178 = 86739107.284455061 * x34;
                const scalar_t  x179 = 86739107.284455061 * x18;
                const scalar_t  x180 = x179 * x6;
                const scalar_t  x181 = x141 * x179;
                const scalar_t  x182 = x180 * x74;
                const scalar_t  x183 = x179 * x34;
                const scalar_t  x184 = x12 * x8;
                const scalar_t  x185 = x130 * x156;
                const scalar_t  x186 = x13 * x51;
                const scalar_t  x187 = x13 * x159;
                const scalar_t  x188 = x32 * x9;
                const scalar_t  x189 = 82569195.284388319 * x188;
                const scalar_t  x190 = 82569195.284388319 * x18;
                const scalar_t  x191 = x190 * x74;
                const scalar_t  x192 = x190 * x6;
                const scalar_t  x193 = x116 * x192;
                const scalar_t  x194 = x192 * x74;
                const scalar_t  x195 = x190 * x32;
                const scalar_t  x196 = x195 * x9;
                const scalar_t  x197 = x132 * x9;
                const scalar_t  x198 = 76584936.614642844 * x36;
                const scalar_t  x199 = 76584936.614642844 * x18;
                const scalar_t  x200 = x156 * x199;
                const scalar_t  x201 = x199 * x53;
                const scalar_t  x202 = x199 * x36;
                const scalar_t  x203 = x37 * x9;
                const scalar_t  x204 = x200 * x5;
                const scalar_t  x205 = 68562514.626742154 * x203;
                const scalar_t  x206 = 68562514.626742154 * x18;
                const scalar_t  x207 = x206 * x55;
                const scalar_t  x208 = x131 * x206;
                const scalar_t  x209 = x206 * x37;
                const scalar_t  x210 = x203 * x206;
                const scalar_t  x211 = x116 * x6;
                const scalar_t  x212 = x207 * x5;
                const scalar_t  x213 = x5 * x63;
                const scalar_t  x214 = 58020430.328498803 * x39;
                const scalar_t  x215 = 58020430.328498803 * x18;
                const scalar_t  x216 = x215 * x81;
                const scalar_t  x217 = x215 * x67;
                const scalar_t  x218 = x215 * x39;
                const scalar_t  x219 = x12 * x6;
                const scalar_t  x220 = x218 * x5;
                const scalar_t  x221 = x5 * x84;
                const scalar_t  x222 = x38 * x57;
                const scalar_t  x223 = x5 * x87;
                const scalar_t  x224 = 43697073.376259953 * x40;
                const scalar_t  x225 = 43697073.376259953 * x18;
                const scalar_t  x226 = x225 * x84;
                const scalar_t  x227 = x132 * x225;
                const scalar_t  x228 = x186 * x225;
                const scalar_t  x229 = x197 * x225;
                const scalar_t  x230 = x221 * x225;
                const scalar_t  x231 = 17678834.997722976 * x41;
                const scalar_t  x232 = 17678834.997722976 * x18;
                const scalar_t  x233 = x232 * x86;
                const scalar_t  x234 = x232 * x85;
                const scalar_t  x235 = x232 * x41;
                const scalar_t  x236 = x235 * x5;
                Dq[0] = 474991.81613764091 * x0 * x17 + x15 * x23 + 17678834.999181062 * x19 + x20 * x28 + x21 * x40 + x23 * x35 +
                        x24 * x34 + x24 * x36 + x25 * x31 + x25 * x33 + x26 * x32 + x26 * x37 + x27 * x29 + x27 * x30 +
                        x28 * x39 + x28 * x41;
                Dq[1] = -x15 * x44 - 43697073.381567873 * x19 - x20 * x49 - x29 * x48 - x30 * x48 - x31 * x46 - x32 * x47 -
                        x33 * x46 - x34 * x45 - x35 * x44 - x36 * x45 - x37 * x47 - x39 * x49 - x40 * x43 - x41 * x49 -
                        87394146.763135746 * x42;
                Dq[2] = x14 * x70 + x53 * x78 + x55 * x76 + x56 * x71 + 116040860.65267894 * x58 + x63 * x65 + x65 * x87 +
                        x69 * x72 + x70 * x8 + x73 * x75 + x75 * x80 + x76 * x77 + x76 * x84 + x78 * x79 + x81 * x83 + x83 * x86;
                Dq[3] = -x100 * x55 - x100 * x77 - x100 * x84 - x101 * x53 - x101 * x79 - x102 * x99 - x14 * x92 - x6 * x92 -
                        x63 * x89 - x67 * x96 - x85 * x96 - x87 * x89 - 137125029.24981335 * x88 - x91 * x93 - x94 * x99 -
                        x96 * x98;
                Dq[4] = 153169873.24450055 * x103 + x104 * x63 + x104 * x87 + x106 * x20 + x106 * x39 + x106 * x41 + x107 * x109 +
                        x107 * x111 + x107 * x95 + x110 * x55 + x110 * x77 + x110 * x84 + x112 * x53 + x112 * x79 + x113 * x115 +
                        x113 * x116;
                Dq[5] = -x114 * x129 - 165138390.57950324 * x117 - x119 * x120 - x120 * x134 - x122 * x73 - x122 * x94 -
                        x123 * x125 - x123 * x131 - x123 * x98 - x126 * x133 - x127 * x2 - x127 * x4 - x128 * x132 - x128 * x53 -
                        x128 * x79 - x129 * x72;
                Dq[6] = x109 * x139 + x111 * x139 + x119 * x136 + x132 * x140 + x134 * x136 + 173478214.57281718 * x135 +
                        x138 * x29 + x138 * x30 + x138 * x64 + x139 * x141 + x140 * x53 + x140 * x79 + x142 * x32 + x142 * x37 +
                        x143 * x39 + x143 * x41;
                Dq[7] = -x125 * x149 - x131 * x149 - 178412372.04577217 * x144 - x145 * x146 - x146 * x159 - x147 * x158 -
                        x148 * x72 - x148 * x93 - x149 * x98 - x150 * x154 - x151 * x155 - x151 * x81 - x152 * x153 -
                        x153 * x156 - x157 * x81 - x157 * x86;
                Dq[8] = 180046640.34226343 * x103 + 180046640.34226343 * x117 + 180046640.34226343 * x135 +
                        180046640.34226343 * x144 + 180046640.34226343 * x160 + 180046640.34226343 * x161 +
                        180046640.34226343 * x162 + 180046640.34226343 * x163 + 180046640.34226343 * x164 +
                        180046640.34226343 * x165 + 180046640.34226343 * x166 + 180046640.34226343 * x167 +
                        180046640.34226343 * x17 * x50 + 180046640.34226343 * x42 + 180046640.34226343 * x58 +
                        180046640.34226343 * x88;
                Dq[9] = -x145 * x168 - x159 * x168 - 178412372.04375598 * x160 - x169 * x176 - x170 * x172 - x170 * x29 -
                        x170 * x30 - x170 * x64 - x171 * x31 - x171 * x33 - x173 * x34 - x173 * x36 - x174 * x32 - x174 * x37 -
                        x175 * x39 - x175 * x41;
                Dq[10] = x114 * x182 + x125 * x181 + x131 * x181 + x154 * x179 * x2 + x154 * x180 + 173478214.56891012 * x161 +
                         x177 * x178 + x178 * x187 + x180 * x2 * x68 + x181 * x185 + x181 * x67 + x181 * x85 + x181 * x98 +
                         x182 * x184 + x183 * x186 + x183 * x52;
                Dq[11] = -x102 * x191 - x119 * x189 - x131 * x193 - x134 * x189 - x155 * x192 * x4 - 165138390.56877664 * x162 -
                         x184 * x194 - x191 * x80 - x193 * x67 - x193 * x85 - x193 * x98 - x194 * x72 - x195 * x197 - x195 * x54 -
                         x196 * x53 - x196 * x79;
                Dq[12] = x141 * x185 * x199 + 153169873.22928569 * x163 + x172 * x200 + x177 * x198 + x186 * x202 + x187 * x198 +
                         x188 * x201 + x200 * x29 + x200 * x30 + x200 * x64 + x201 * x203 + x201 * x31 + x201 * x33 + x202 * x52 +
                         x204 * x39 + x204 * x41;
                Dq[13] = -x109 * x208 - x111 * x208 - x119 * x205 - x134 * x205 - x141 * x208 - 137125029.25348431 * x164 -
                         x197 * x209 - x207 * x29 - x207 * x30 - x207 * x64 - x208 * x211 - x209 * x54 - x210 * x53 - x210 * x79 -
                         x212 * x39 - x212 * x41;
                Dq[14] = x124 * x218 + x141 * x217 + x152 * x220 + x156 * x220 + 116040860.65699761 * x165 + x2 * x216 * x6 +
                         x211 * x217 + x213 * x214 + x214 * x223 + x215 * x222 + x216 * x219 + x216 * x82 + x217 * x95 +
                         x218 * x221 + x218 * x56 + x220 * x55;
                Dq[15] = -x * x224 - x158 * x225 - 87394146.752519906 * x166 - x176 * x225 - x224 * x50 - x226 * x29 -
                         x226 * x30 - x226 * x64 - x227 * x31 - x227 * x33 - x228 * x34 - x228 * x36 - x229 * x32 - x229 * x37 -
                         x230 * x39 - x230 * x41;
                Dq[16] = x124 * x235 + x133 * x232 * x6 + x141 * x234 + x152 * x236 + x156 * x236 + 35357669.995445952 * x167 +
                         x211 * x234 + x213 * x231 + x219 * x233 + x221 * x235 + x222 * x232 + x223 * x231 + x233 * x82 +
                         x234 * x95 + x235 * x56 + x236 * x55;
            }

            break;
        }
        default:
            return 1;
    }

    return 0;
}

#endif  // SFEM_LAGRANGE_LEGENDRE_GAUSS_LOBATTO_HPP
