#include "sfem_base.h"

#include <cassert>
#include <cmath>
#include <complex>

using complex_t = std::complex<real_t>;

inline static complex_t operator*(const int l, const complex_t &r) { return real_t(l) * r; }

inline static complex_t operator+(const complex_t &l, const int r) { return l + real_t(r); }

static SFEM_INLINE void neohookean_principal_stresses_kernel(const real_t mu,
                                                             const real_t lambda,
                                                             const real_t px0,
                                                             const real_t px1,
                                                             const real_t px2,
                                                             const real_t px3,
                                                             const real_t py0,
                                                             const real_t py1,
                                                             const real_t py2,
                                                             const real_t py3,
                                                             const real_t pz0,
                                                             const real_t pz1,
                                                             const real_t pz2,
                                                             const real_t pz3,
                                                             const real_t *const SFEM_RESTRICT u,
                                                             real_t *const SFEM_RESTRICT
                                                                 element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 3*ADD + 3*ASSIGNMENT + 6*MUL + 2*POW
    //       - Subexpressions: 79*ADD + 16*DIV + LOG + 219*MUL + 9*NEG + 9*POW + 68*SUB

    const complex_t x0 = py0 - py1;
    const complex_t x1 = pz0 - pz2;
    const complex_t x2 = py0 - py2;
    const complex_t x3 = pz0 - pz1;
    const complex_t x4 = x0 * x1 - x2 * x3;
    const complex_t x5 = -x4;
    const complex_t x6 = pz0 - pz3;
    const complex_t x7 = px0 - px1;
    const complex_t x8 = x2 * x7;
    const complex_t x9 = px0 - px2;
    const complex_t x10 = py0 - py3;
    const complex_t x11 = x10 * x9;
    const complex_t x12 = px0 - px3;
    const complex_t x13 = x0 * x12;
    const complex_t x14 = x10 * x7;
    const complex_t x15 = x0 * x9;
    const complex_t x16 = x12 * x2;
    const complex_t x17 = 1.0 / (x1 * x13 - x1 * x14 + x11 * x3 - x15 * x6 - x16 * x3 + x6 * x8);
    const complex_t x18 = u[10] * x17;
    const complex_t x19 = -x1 * x10 + x2 * x6;
    const complex_t x20 = -x19;
    const complex_t x21 = u[4] * x17;
    const complex_t x22 = x0 * x6;
    const complex_t x23 = x10 * x3;
    const complex_t x24 = x22 - x23;
    const complex_t x25 = u[7] * x17;
    const complex_t x26 = x19 - x22 + x23 + x4;
    const complex_t x27 = u[1] * x17;
    const complex_t x28 = x18 * x5 + x20 * x21 + x24 * x25 + x26 * x27;
    const complex_t x29 = x11 - x16;
    const complex_t x30 = -x29;
    const complex_t x31 = u[3] * x17;
    const complex_t x32 = -x13 + x14;
    const complex_t x33 = u[6] * x17;
    const complex_t x34 = -x15 + x8;
    const complex_t x35 = -x34;
    const complex_t x36 = u[9] * x17;
    const complex_t x37 = x13 - x14 + x29 + x34;
    const complex_t x38 = u[0] * x17;
    const complex_t x39 = x30 * x31 + x32 * x33 + x35 * x36 + x37 * x38;
    const complex_t x40 = x1 * x7 - x3 * x9;
    const complex_t x41 = u[11] * x17;
    const complex_t x42 = -x1 * x12 + x6 * x9;
    const complex_t x43 = u[5] * x17;
    const complex_t x44 = x6 * x7;
    const complex_t x45 = x12 * x3;
    const complex_t x46 = -x44 + x45;
    const complex_t x47 = u[8] * x17;
    const complex_t x48 = -x40 - x42 + x44 - x45;
    const complex_t x49 = u[2] * x17;
    const complex_t x50 = x40 * x41 + x42 * x43 + x46 * x47 + x48 * x49;
    const complex_t x51 = x31 * x42 + x33 * x46 + x36 * x40 + x38 * x48;
    const complex_t x52 = x30 * x43 + x32 * x47 + x35 * x41 + x37 * x49 + 1;
    const complex_t x53 = x39 * x50 - x51 * x52;
    const complex_t x54 = x28 * x50;
    const complex_t x55 = x18 * x35 + x21 * x30 + x25 * x32 + x27 * x37;
    const complex_t x56 = x20 * x43 + x24 * x47 + x26 * x49 + x41 * x5;
    const complex_t x57 = x55 * x56;
    const complex_t x58 = x18 * x40 + x21 * x42 + x25 * x46 + x27 * x48 + 1;
    const complex_t x59 = x56 * x58;
    const complex_t x60 = x28 * x52;
    const complex_t x61 = x20 * x31 + x24 * x33 + x26 * x38 + x36 * x5 + 1;
    const complex_t x62 = x50 * x55;
    const complex_t x63 =
        x39 * x54 - x39 * x59 + x51 * x57 - x51 * x60 + x52 * x58 * x61 - x61 * x62;
    const complex_t x64 = 1.0 / x63;
    const complex_t x65 = mu * x64;
    const complex_t x66 = lambda * x64 * log(x63);
    const complex_t x67 = mu * x28 - x53 * x65 + x53 * x66;
    const complex_t x68 = -x50 * x61 + x51 * x56;
    const complex_t x69 = mu * x55 - x65 * x68 + x66 * x68;
    const complex_t x70 = -x39 * x56 + x52 * x61;
    const complex_t x71 = mu * x58 - x65 * x70 + x66 * x70;
    const complex_t x72 = x28 * x67 + x55 * x69 + x58 * x71;
    const complex_t x73 = x57 - x60;
    const complex_t x74 = mu * x51 - x65 * x73 + x66 * x73;
    const complex_t x75 = x54 - x59;
    const complex_t x76 = mu * x39 - x65 * x75 + x66 * x75;
    const complex_t x77 = x52 * x58 - x62;
    const complex_t x78 = mu * x61 - x65 * x77 + x66 * x77;
    const complex_t x79 = x50 * x74 + x52 * x76 + x56 * x78;
    const complex_t x80 = x28 * x39 - x55 * x61;
    const complex_t x81 = mu * x50 - x65 * x80 + x66 * x80;
    const complex_t x82 = -x39 * x58 + x51 * x55;
    const complex_t x83 = mu * x56 - x65 * x82 + x66 * x82;
    const complex_t x84 = -x28 * x51 + x58 * x61;
    const complex_t x85 = mu * x52 - x65 * x84 + x66 * x84;
    const complex_t x86 = x39 * x85 + x51 * x81 + x61 * x83;
    const complex_t x87 = x79 * x86;
    const complex_t x88 = pow(x63, -3);
    const complex_t x89 = (27.0 / 2.0) * x88;
    const complex_t x90 = x28 * x78 + x55 * x76 + x58 * x74;
    const complex_t x91 = x50 * x71 + x52 * x69 + x56 * x67;
    const complex_t x92 = x86 * x90 * x91;
    const complex_t x93 = x50 * x81 + x52 * x85 + x56 * x83;
    const complex_t x94 = x72 * x93;
    const complex_t x95 = x39 * x76 + x51 * x74 + x61 * x78;
    const complex_t x96 = x89 * x95;
    const complex_t x97 = x39 * x69 + x51 * x71 + x61 * x67;
    const complex_t x98 = x90 * x97;
    const complex_t x99 = x28 * x83 + x55 * x85 + x58 * x81;
    const complex_t x100 = x91 * x99;
    const complex_t x101 = x79 * x97 * x99;
    const complex_t x102 = x64 * x93;
    const complex_t x103 = x64 * x72;
    const complex_t x104 = x64 * x95;
    const complex_t x105 = -x102 - x103 - x104;
    const complex_t x106 = pow(x105, 3);
    const complex_t x107 = pow(x63, -2);
    const complex_t x108 = x107 * x87;
    const complex_t x109 = x100 * x107;
    const complex_t x110 = x107 * x98;
    const complex_t x111 =
        (-9 * x102 - 9 * x103 - 9 * x104) *
        (x107 * x72 * x93 + x107 * x72 * x95 + x107 * x93 * x95 - x108 - x109 - x110);
    const complex_t x112 = x107 * x95;
    const complex_t x113 = pow(x105, 2) - 3 * x107 * x94 + 3 * x108 + 3 * x109 + 3 * x110 -
                           3 * x112 * x72 - 3 * x112 * x93;
    const complex_t x114 = 27 * x88;
    // const complex_t x115 =
    //     std::cbrt(x100 * x96 - x101 * x89 + x106 - 1.0 / 2.0 * x111 + x72 * x87 * x89 - x89 * x92
    //     +
    //          x89 * x93 * x98 - x94 * x96 +
    //          (1.0 / 2.0) * sqrt(-4 * pow(x113, 3) +
    //                             pow(-x101 * x114 + 2 * x106 - x111 - x114 * x92 - x114 * x94 *
    //                             x95 +
    //                                     27 * x72 * x79 * x86 * x88 + 27 * x88 * x90 * x93 * x97 +
    //                                     27 * x88 * x91 * x95 * x99,
    //                                 2)));
    const complex_t x115 =
        pow(x100 * x96 - x101 * x89 + x106 - 1.0 / 2.0 * x111 + x72 * x87 * x89 - x89 * x92 +
                x89 * x93 * x98 - x94 * x96 +
                (1.0 / 2.0) * sqrt(-4 * pow(x113, 3) +
                                   pow(-x101 * x114 + 2 * x106 - x111 - x114 * x92 -
                                           x114 * x94 * x95 + 27 * x72 * x79 * x86 * x88 +
                                           27 * x88 * x90 * x93 * x97 + 27 * x88 * x91 * x95 * x99,
                                       2)),
            1. / 3);

    const complex_t x116 = (1.0 / 3.0) * x115;
    const complex_t x117 = (1.0 / 3.0) * x113 / x115;
    const complex_t x118 = (1.0 / 3.0) * x102 + (1.0 / 3.0) * x103 + (1.0 / 3.0) * x104;
    const complex_t x119 = (1.0 / 2.0) * sqrt(3) * complex_t(0, 1);
    const complex_t x120 = x119 - 1.0 / 2.0;
    const complex_t x121 = -x119 - 1.0 / 2.0;

    element_vector[0] = std::real(-x116 - x117 + x118);
    element_vector[1] = std::real(-x116 * x120 - x117 / x120 + x118);
    element_vector[2] = std::real(-x116 * x121 - x117 / x121 + x118);

    std::sort(element_vector, element_vector + 3);

    assert(std::imag(element_vector[0]) < 1e-10);
    assert(std::imag(element_vector[1]) < 1e-10);
    assert(std::imag(element_vector[2]) < 1e-10);
}

extern "C" void neohookean_principal_stresses_aos(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  real_t *const SFEM_RESTRICT displacement,
                                                  real_t **const SFEM_RESTRICT stress) {
    SFEM_UNUSED(nnodes);

    static const int block_size = 3;
    idx_t ev[4];
    real_t element_displacement[4 * 3];
    real_t element_vector[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int enode = 0; enode < 4; ++enode) {
            idx_t edof = enode * block_size;
            idx_t dof = ev[enode] * block_size;

            for (int b = 0; b < block_size; ++b) {
                element_displacement[edof + b] = displacement[dof + b];
            }
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        neohookean_principal_stresses_kernel(mu,
                                             lambda,
                                             // X-coordinates
                                             xyz[0][i0],
                                             xyz[0][i1],
                                             xyz[0][i2],
                                             xyz[0][i3],
                                             // Y-coordinates
                                             xyz[1][i0],
                                             xyz[1][i1],
                                             xyz[1][i2],
                                             xyz[1][i3],
                                             // Z-coordinates
                                             xyz[2][i0],
                                             xyz[2][i1],
                                             xyz[2][i2],
                                             xyz[2][i3],
                                             // Data
                                             element_displacement,
                                             // Output
                                             element_vector);

        stress[0][i] = element_vector[0];
        stress[1][i] = element_vector[1];
        stress[2][i] = element_vector[2];
    }
}

extern "C" void neohookean_principal_stresses_soa(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t mu,
                                       const real_t lambda,
                                       real_t **const SFEM_RESTRICT displacement,
                                       real_t **const SFEM_RESTRICT stress)
{
        SFEM_UNUSED(nnodes);

        static const int block_size = 3;
        idx_t ev[4];
        real_t element_displacement[4 * 3];
        real_t element_vector[3];

        for (ptrdiff_t i = 0; i < nelements; ++i) {
    #pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

           for (int enode = 0; enode < 4; ++enode) {
               idx_t edof = enode * block_size;
               idx_t dof = ev[enode];

               element_displacement[edof + 0] = displacement[0][dof];
               element_displacement[edof + 1] = displacement[1][dof];
               element_displacement[edof + 2] = displacement[2][dof];
           }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            neohookean_principal_stresses_kernel(mu,
                                                 lambda,
                                                 // X-coordinates
                                                 xyz[0][i0],
                                                 xyz[0][i1],
                                                 xyz[0][i2],
                                                 xyz[0][i3],
                                                 // Y-coordinates
                                                 xyz[1][i0],
                                                 xyz[1][i1],
                                                 xyz[1][i2],
                                                 xyz[1][i3],
                                                 // Z-coordinates
                                                 xyz[2][i0],
                                                 xyz[2][i1],
                                                 xyz[2][i2],
                                                 xyz[2][i3],
                                                 // Data
                                                 element_displacement,
                                                 // Output
                                                 element_vector);

            stress[0][i] = element_vector[0];
            stress[1][i] = element_vector[1];
            stress[2][i] = element_vector[2];
        }
}

