#include "tet10_grad.h"

#include <mpi.h>
#include <stdio.h>

// Microkernel
static SFEM_INLINE void tet10_grad_kernel(const real_t px0,
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
                                          // Data
                                          const real_t *SFEM_RESTRICT f,
                                          // Output
                                          real_t *SFEM_RESTRICT dfdx,
                                          real_t *SFEM_RESTRICT dfdy,
                                          real_t *SFEM_RESTRICT dfdz) {
    // // FLOATING POINT OPS!
    // //       - Result: 12*ADD + 12*ASSIGNMENT + 36*MUL
    // //       - Subexpressions: 20*ADD + DIV + 47*MUL + 9*NEG + 27*SUB
    // const real_t x0 = -px0 + px2;
    // const real_t x1 = -py0 + py3;
    // const real_t x2 = x0 * x1;
    // const real_t x3 = -px0 + px3;
    // const real_t x4 = -py0 + py2;
    // const real_t x5 = x3 * x4;
    // const real_t x6 = x2 - x5;
    // const real_t x7 = -px0 + px1;
    // const real_t x8 = -pz0 + pz3;
    // const real_t x9 = x4 * x8;
    // const real_t x10 = -pz0 + pz1;
    // const real_t x11 = -py0 + py1;
    // const real_t x12 = -pz0 + pz2;
    // const real_t x13 = x1 * x12;
    // const real_t x14 = x0 * x8;
    // const real_t x15 = 1.0 / (x10 * x2 - x10 * x5 + x11 * x12 * x3 - x11 * x14 - x13 * x7 + x7 * x9);
    // const real_t x16 = 3 * f[0];
    // const real_t x17 = -4 * f[7];
    // const real_t x18 = x15 * (-f[3] - x16 - x17);
    // const real_t x19 = x12 * x3 - x14;
    // const real_t x20 = -4 * f[6];
    // const real_t x21 = x15 * (-f[2] - x16 - x20);
    // const real_t x22 = -x13 + x9;
    // const real_t x23 = -4 * f[4];
    // const real_t x24 = x15 * (-f[1] - x16 - x23);
    // const real_t x25 = -x1 * x7 + x11 * x3;
    // const real_t x26 = -x10 * x3 + x7 * x8;
    // const real_t x27 = x1 * x10 - x11 * x8;
    // const real_t x28 = -x0 * x11 + x4 * x7;
    // const real_t x29 = x0 * x10 - x12 * x7;
    // const real_t x30 = -x10 * x4 + x11 * x12;
    // const real_t x31 = f[0] + x23;
    // const real_t x32 = x15 * (3 * f[1] + x31);
    // const real_t x33 = -f[3];
    // const real_t x34 = 4 * f[8];
    // const real_t x35 = x15 * (x31 + x33 + x34);
    // const real_t x36 = -f[2];
    // const real_t x37 = 4 * f[5];
    // const real_t x38 = x15 * (x31 + x36 + x37);
    // const real_t x39 = f[0] + x20;
    // const real_t x40 = x15 * (3 * f[2] + x39);
    // const real_t x41 = 4 * f[9];
    // const real_t x42 = x15 * (x33 + x39 + x41);
    // const real_t x43 = -f[1];
    // const real_t x44 = x15 * (x37 + x39 + x43);
    // const real_t x45 = f[0] + x17;
    // const real_t x46 = x15 * (3 * f[3] + x45);
    // const real_t x47 = x15 * (x36 + x41 + x45);
    // const real_t x48 = x15 * (x34 + x43 + x45);
    // dfdx[0] = x18 * x6 + x19 * x21 + x22 * x24;
    // dfdy[0] = x18 * x25 + x21 * x26 + x24 * x27;
    // dfdz[0] = x18 * x28 + x21 * x29 + x24 * x30;
    // dfdx[1] = x19 * x38 + x22 * x32 + x35 * x6;
    // dfdy[1] = x25 * x35 + x26 * x38 + x27 * x32;
    // dfdz[1] = x28 * x35 + x29 * x38 + x30 * x32;
    // dfdx[2] = x19 * x40 + x22 * x44 + x42 * x6;
    // dfdy[2] = x25 * x42 + x26 * x40 + x27 * x44;
    // dfdz[2] = x28 * x42 + x29 * x40 + x30 * x44;
    // dfdx[3] = x19 * x47 + x22 * x48 + x46 * x6;
    // dfdy[3] = x25 * x46 + x26 * x47 + x27 * x48;
    // dfdz[3] = x28 * x46 + x29 * x47 + x30 * x48;
    //FLOATING POINT OPS!
    //      - Result: 12*ADD + 12*ASSIGNMENT + 36*MUL
    //      - Subexpressions: 20*ADD + DIV + 47*MUL + 9*NEG + 27*SUB
    const real_t x0 = -py0 + py1;
    const real_t x1 = -pz0 + pz2;
    const real_t x2 = x0*x1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = x3*x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = -px0 + px1;
    const real_t x8 = -pz0 + pz3;
    const real_t x9 = x3*x8;
    const real_t x10 = -px0 + px2;
    const real_t x11 = -py0 + py3;
    const real_t x12 = -px0 + px3;
    const real_t x13 = x1*x11;
    const real_t x14 = x0*x8;
    const real_t x15 = 1.0/(x10*x11*x4 - x10*x14 + x12*x2 - x12*x5 - x13*x7 + x7*x9);
    const real_t x16 = 3*f[0];
    const real_t x17 = -4*f[7];
    const real_t x18 = x15*(-f[3] - x16 - x17);
    const real_t x19 = x11*x4 - x14;
    const real_t x20 = -4*f[6];
    const real_t x21 = x15*(-f[2] - x16 - x20);
    const real_t x22 = -x13 + x9;
    const real_t x23 = -4*f[4];
    const real_t x24 = x15*(-f[1] - x16 - x23);
    const real_t x25 = -x1*x7 + x10*x4;
    const real_t x26 = -x12*x4 + x7*x8;
    const real_t x27 = x1*x12 - x10*x8;
    const real_t x28 = -x0*x10 + x3*x7;
    const real_t x29 = x0*x12 - x11*x7;
    const real_t x30 = x10*x11 - x12*x3;
    const real_t x31 = f[0] + x23;
    const real_t x32 = x15*(3*f[1] + x31);
    const real_t x33 = -f[3];
    const real_t x34 = 4*f[8];
    const real_t x35 = x15*(x31 + x33 + x34);
    const real_t x36 = -f[2];
    const real_t x37 = 4*f[5];
    const real_t x38 = x15*(x31 + x36 + x37);
    const real_t x39 = f[0] + x20;
    const real_t x40 = x15*(3*f[2] + x39);
    const real_t x41 = 4*f[9];
    const real_t x42 = x15*(x33 + x39 + x41);
    const real_t x43 = -f[1];
    const real_t x44 = x15*(x37 + x39 + x43);
    const real_t x45 = f[0] + x17;
    const real_t x46 = x15*(3*f[3] + x45);
    const real_t x47 = x15*(x36 + x41 + x45);
    const real_t x48 = x15*(x34 + x43 + x45);
    dfdx[0] = x18*x6 + x19*x21 + x22*x24;
    dfdy[0] = x18*x25 + x21*x26 + x24*x27;
    dfdz[0] = x18*x28 + x21*x29 + x24*x30;
    dfdx[1] = x19*x38 + x22*x32 + x35*x6;
    dfdy[1] = x25*x35 + x26*x38 + x27*x32;
    dfdz[1] = x28*x35 + x29*x38 + x30*x32;
    dfdx[2] = x19*x40 + x22*x44 + x42*x6;
    dfdy[2] = x25*x42 + x26*x40 + x27*x44;
    dfdz[2] = x28*x42 + x29*x40 + x30*x44;
    dfdx[3] = x19*x47 + x22*x48 + x46*x6;
    dfdy[3] = x25*x46 + x26*x47 + x27*x48;
    dfdz[3] = x28*x46 + x29*x47 + x30*x48;
}

void tet10_grad(const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **SFEM_RESTRICT xyz,
                const real_t *const SFEM_RESTRICT f,
                real_t *const SFEM_RESTRICT dfdx,
                real_t *const SFEM_RESTRICT dfdy,
                real_t *const SFEM_RESTRICT dfdz) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_f[10];
    real_t element_dfdx[4];
    real_t element_dfdy[4];
    real_t element_dfdz[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_f[v] = f[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_grad_kernel(
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
            element_f,
            // Output
            element_dfdx,
            element_dfdy,
            element_dfdz);

        // Write cell data
        for (int v = 0; v < 4; ++v) {
            dfdx[i * 4 + v] = element_dfdx[v];
        }

        for (int v = 0; v < 4; ++v) {
            dfdy[i * 4 + v] = element_dfdy[v];
        }

        for (int v = 0; v < 4; ++v) {
            dfdz[i * 4 + v] = element_dfdz[v];
        }
    }

    double tock = MPI_Wtime();
    printf("cgrad.c: cgrad3\t%g seconds\n", tock - tick);
}
