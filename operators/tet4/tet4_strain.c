
#include <stddef.h>
#include <math.h>

#include "sfem_base.h"

static SFEM_INLINE void strain_kernel(const real_t px0,
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
                                     const real_t *const SFEM_RESTRICT ux,
                                     const real_t *const SFEM_RESTRICT uy,
                                     const real_t *const SFEM_RESTRICT uz,
                                     // Output
                                     real_t *const SFEM_RESTRICT strain) {
    //FLOATING POINT OPS!
    //      - Result: 6*ADD + 6*ASSIGNMENT + 18*MUL + 9*POW
    //      - Subexpressions: 32*ADD + 4*DIV + 70*MUL + 3*NEG + 27*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0*x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3*x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = -px0 + px2;
    const real_t x8 = -pz0 + pz1;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10*x4;
    const real_t x12 = x1*x10;
    const real_t x13 = x0*x8;
    const real_t x14 = 1.0/(x11*x9 - x12*x7 - x13*x9 + x2*x6 + x3*x7*x8 - x5*x6);
    const real_t x15 = x14*(x2 - x5);
    const real_t x16 = x14*(-x12 + x3*x8);
    const real_t x17 = x14*(x11 - x13);
    const real_t x18 = -x15 - x16 - x17;
    const real_t x19 = uy[0]*x18 + uy[1]*x15 + uy[2]*x16 + uy[3]*x17;
    const real_t x20 = uz[0]*x18 + uz[1]*x15 + uz[2]*x16 + uz[3]*x17;
    const real_t x21 = ux[0]*x18 + ux[1]*x15 + ux[2]*x16 + ux[3]*x17 + 1;
    const real_t x22 = x14*(-x1*x7 + x4*x9);
    const real_t x23 = x14*(x1*x6 - x8*x9);
    const real_t x24 = x14*(-x4*x6 + x7*x8);
    const real_t x25 = -x22 - x23 - x24;
    const real_t x26 = uz[0]*x25 + uz[1]*x22 + uz[2]*x23 + uz[3]*x24;
    const real_t x27 = (1.0/2.0)*x20;
    const real_t x28 = ux[0]*x25 + ux[1]*x22 + ux[2]*x23 + ux[3]*x24;
    const real_t x29 = (1.0/2.0)*x21;
    const real_t x30 = uy[0]*x25 + uy[1]*x22 + uy[2]*x23 + uy[3]*x24 + 1;
    const real_t x31 = (1.0/2.0)*x19;
    const real_t x32 = x14*(-x0*x9 + x3*x7);
    const real_t x33 = x14*(x10*x9 - x3*x6);
    const real_t x34 = x14*(x0*x6 - x10*x7);
    const real_t x35 = -x32 - x33 - x34;
    const real_t x36 = uy[0]*x35 + uy[1]*x32 + uy[2]*x33 + uy[3]*x34;
    const real_t x37 = ux[0]*x35 + ux[1]*x32 + ux[2]*x33 + ux[3]*x34;
    const real_t x38 = uz[0]*x35 + uz[1]*x32 + uz[2]*x33 + uz[3]*x34 + 1;
    strain[0] = (1.0/2.0)*pow(x19, 2) + (1.0/2.0)*pow(x20, 2) + (1.0/2.0)*pow(x21, 2) - 1.0/2.0;
    strain[1] = x26*x27 + x28*x29 + x30*x31;
    strain[2] = x27*x38 + x29*x37 + x31*x36;
    strain[3] = (1.0/2.0)*pow(x26, 2) + (1.0/2.0)*pow(x28, 2) + (1.0/2.0)*pow(x30, 2) - 1.0/2.0;
    strain[4] = (1.0/2.0)*x26*x38 + (1.0/2.0)*x28*x37 + (1.0/2.0)*x30*x36;
    strain[5] = (1.0/2.0)*pow(x36, 2) + (1.0/2.0)*pow(x37, 2) + (1.0/2.0)*pow(x38, 2) - 1.0/2.0;
}

void strain(const ptrdiff_t nelements,
           const ptrdiff_t nnodes,
           idx_t **const SFEM_RESTRICT elems,
           geom_t **const SFEM_RESTRICT xyz,
           const real_t *const SFEM_RESTRICT ux,
           const real_t *const SFEM_RESTRICT uy,
           const real_t *const SFEM_RESTRICT uz,
           real_t *const SFEM_RESTRICT strain_xx,
           real_t *const SFEM_RESTRICT strain_xy,
           real_t *const SFEM_RESTRICT strain_xz,
           real_t *const SFEM_RESTRICT strain_yy,
           real_t *const SFEM_RESTRICT strain_yz,
           real_t *const SFEM_RESTRICT strain_zz) {
    SFEM_UNUSED(nnodes);

    idx_t ev[4];
    real_t element_vector[4];

    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];
    real_t element_strain[6];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uz[v] = uz[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        strain_kernel(
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
            element_ux,
            element_uy,
            element_uz,
            // Output
            element_strain);

        strain_xx[i] = element_strain[0];
        strain_xy[i] = element_strain[1];
        strain_xz[i] = element_strain[2];
        strain_yy[i] = element_strain[3];
        strain_yz[i] = element_strain[4];
        strain_zz[i] = element_strain[5];
    }
}
