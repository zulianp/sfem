
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void div_gradient(const real_t px0,
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
                                     const real_t *ux,
                                     const real_t *uy,
                                     const real_t *uz,
                                     real_t *element_vector) {
   //FLOATING POINT OPS!
   //      - Result: 4*ASSIGNMENT
   //      - Subexpressions: 13*ADD + 2*DIV + 47*MUL + 3*NEG + 27*SUB
   const real_t x0 = -px0 + px1;
   const real_t x1 = -py0 + py2;
   const real_t x2 = -pz0 + pz3;
   const real_t x3 = x1*x2;
   const real_t x4 = -px0 + px2;
   const real_t x5 = -py0 + py3;
   const real_t x6 = -pz0 + pz1;
   const real_t x7 = -px0 + px3;
   const real_t x8 = -py0 + py1;
   const real_t x9 = -pz0 + pz2;
   const real_t x10 = x8*x9;
   const real_t x11 = x5*x9;
   const real_t x12 = x2*x8;
   const real_t x13 = x1*x6;
   const real_t x14 = -x0*x11 + x0*x3 + x10*x7 - x12*x4 - x13*x7 + x4*x5*x6;
   const real_t x15 = 1.0/x14;
   const real_t x16 = x15*(-x11 + x3);
   const real_t x17 = x15*(-x12 + x5*x6);
   const real_t x18 = x15*(x10 - x13);
   const real_t x19 = x15*(-x2*x4 + x7*x9);
   const real_t x20 = x15*(x0*x2 - x6*x7);
   const real_t x21 = x15*(-x0*x9 + x4*x6);
   const real_t x22 = x15*(-x1*x7 + x4*x5);
   const real_t x23 = x15*(-x0*x5 + x7*x8);
   const real_t x24 = x15*(x0*x1 - x4*x8);
   const real_t x25 = (1.0/24.0)*x14*(ux[0]*(-x16 - x17 - x18) + ux[1]*x16 + ux[2]*x17 + ux[3]*x18 + uy[0]*(-x19 - x20 - x21) + uy[1]*x19 + uy[2]*x20 + uy[3]*x21 + uz[0]*(-x22 - x23 - x24) + uz[1]*x22 + uz[2]*x23 + 
   uz[3]*x24);
   element_vector[0] = x25;
   element_vector[1] = x25;
   element_vector[2] = x25;
   element_vector[3] = x25;
}

void div_apply(const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_vector[4];
    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];

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

        div_gradient(
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
            element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("div.c: div_apply\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void ediv(const real_t px0,
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
                             const real_t *ux,
                             const real_t *uy,
                             const real_t *uz,
                             real_t *element_value) {
    // FL//FLOATING POINT OPS!
    // FLOATING POINT OPS!
    //      - Result: ADD + ASSIGNMENT + 72*MUL
    //      - Subexpressions: 8*DIV
    const real_t x0 = (1.0 / 6.0) * px0;
    const real_t x1 = (1.0 / 6.0) * px1;
    const real_t x2 = (1.0 / 6.0) * px2;
    const real_t x3 = (1.0 / 6.0) * px3;
    const real_t x4 = (1.0 / 6.0) * py0;
    const real_t x5 = (1.0 / 6.0) * py1;
    const real_t x6 = (1.0 / 6.0) * py2;
    const real_t x7 = (1.0 / 6.0) * py3;
    element_value[0] =
        (1.0 / 6.0) * px0 * py1 * uz[3] + (1.0 / 6.0) * px0 * py2 * uz[1] + (1.0 / 6.0) * px0 * py3 * uz[2] +
        (1.0 / 6.0) * px0 * pz1 * uy[2] + (1.0 / 6.0) * px0 * pz2 * uy[3] + (1.0 / 6.0) * px0 * pz3 * uy[1] +
        (1.0 / 6.0) * px1 * py0 * uz[2] + (1.0 / 6.0) * px1 * py2 * uz[3] + (1.0 / 6.0) * px1 * py3 * uz[0] +
        (1.0 / 6.0) * px1 * pz0 * uy[3] + (1.0 / 6.0) * px1 * pz2 * uy[0] + (1.0 / 6.0) * px1 * pz3 * uy[2] +
        (1.0 / 6.0) * px2 * py0 * uz[3] + (1.0 / 6.0) * px2 * py1 * uz[0] + (1.0 / 6.0) * px2 * py3 * uz[1] +
        (1.0 / 6.0) * px2 * pz0 * uy[1] + (1.0 / 6.0) * px2 * pz1 * uy[3] + (1.0 / 6.0) * px2 * pz3 * uy[0] +
        (1.0 / 6.0) * px3 * py0 * uz[1] + (1.0 / 6.0) * px3 * py1 * uz[2] + (1.0 / 6.0) * px3 * py2 * uz[0] +
        (1.0 / 6.0) * px3 * pz0 * uy[2] + (1.0 / 6.0) * px3 * pz1 * uy[0] + (1.0 / 6.0) * px3 * pz2 * uy[1] +
        (1.0 / 6.0) * py0 * pz1 * ux[3] + (1.0 / 6.0) * py0 * pz2 * ux[1] + (1.0 / 6.0) * py0 * pz3 * ux[2] -
        py0 * uz[1] * x2 - py0 * uz[2] * x3 - py0 * uz[3] * x1 + (1.0 / 6.0) * py1 * pz0 * ux[2] +
        (1.0 / 6.0) * py1 * pz2 * ux[3] + (1.0 / 6.0) * py1 * pz3 * ux[0] - py1 * uz[0] * x3 - py1 * uz[2] * x0 -
        py1 * uz[3] * x2 + (1.0 / 6.0) * py2 * pz0 * ux[3] + (1.0 / 6.0) * py2 * pz1 * ux[0] +
        (1.0 / 6.0) * py2 * pz3 * ux[1] - py2 * uz[0] * x1 - py2 * uz[1] * x3 - py2 * uz[3] * x0 +
        (1.0 / 6.0) * py3 * pz0 * ux[1] + (1.0 / 6.0) * py3 * pz1 * ux[2] + (1.0 / 6.0) * py3 * pz2 * ux[0] -
        py3 * uz[0] * x2 - py3 * uz[1] * x0 - py3 * uz[2] * x1 - pz0 * ux[1] * x6 - pz0 * ux[2] * x7 -
        pz0 * ux[3] * x5 - pz0 * uy[1] * x3 - pz0 * uy[2] * x1 - pz0 * uy[3] * x2 - pz1 * ux[0] * x7 -
        pz1 * ux[2] * x4 - pz1 * ux[3] * x6 - pz1 * uy[0] * x2 - pz1 * uy[2] * x3 - pz1 * uy[3] * x0 -
        pz2 * ux[0] * x5 - pz2 * ux[1] * x7 - pz2 * ux[3] * x4 - pz2 * uy[0] * x3 - pz2 * uy[1] * x0 -
        pz2 * uy[3] * x1 - pz3 * ux[0] * x6 - pz3 * ux[1] * x4 - pz3 * ux[2] * x5 - pz3 * uy[0] * x1 -
        pz3 * uy[1] * x2 - pz3 * uy[2] * x0;
}

void integrate_div(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   const real_t *const uz,
                   real_t *const value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];

    *value = 0.;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];

            assert(ev[v] >= 0);
            assert(ev[v] < nnodes);
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

        real_t element_scalar = 0;

        ediv(
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
            &element_scalar);

        *value += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("div.c: integrate_div\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void cdiv_kernel(const real_t px0,
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
                                    const real_t *SFEM_RESTRICT ux,
                                    const real_t *SFEM_RESTRICT uy,
                                    const real_t *SFEM_RESTRICT uz,
                                    real_t *SFEM_RESTRICT element_value) {
    // FLOATING POINT OPS!
    //       - Result: 4*ADD + ASSIGNMENT + 21*MUL
    //       - Subexpressions: 2*ADD + DIV + 34*MUL + 21*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = -px0 + px2;
    const real_t x8 = -pz0 + pz1;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x4;
    const real_t x12 = x1 * x10;
    const real_t x13 = x0 * x8;
    const real_t x14 = 1.0 / (x11 * x9 - x12 * x7 - x13 * x9 + x2 * x6 + x3 * x7 * x8 - x5 * x6);
    const real_t x15 = x14 * (x2 - x5);
    const real_t x16 = x14 * (-x12 + x3 * x8);
    const real_t x17 = x14 * (x11 - x13);
    const real_t x18 = x14 * (-x1 * x7 + x4 * x9);
    const real_t x19 = x14 * (x1 * x6 - x8 * x9);
    const real_t x20 = x14 * (-x4 * x6 + x7 * x8);
    const real_t x21 = x14 * (-x0 * x9 + x3 * x7);
    const real_t x22 = x14 * (x10 * x9 - x3 * x6);
    const real_t x23 = x14 * (x0 * x6 - x10 * x7);
    element_value[0] = ux[0] * (-x15 - x16 - x17) + ux[1] * x15 + ux[2] * x16 + ux[3] * x17 +
                       uy[0] * (-x18 - x19 - x20) + uy[1] * x18 + uy[2] * x19 + uy[3] * x20 +
                       uz[0] * (-x21 - x22 - x23) + uz[1] * x21 + uz[2] * x22 + uz[3] * x23;
}

void cdiv(const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT div) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];

            assert(ev[v] >= 0);
            assert(ev[v] < nnodes);
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

        real_t element_scalar = 0;

        cdiv_kernel(
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
            &element_scalar);

        div[i] = element_scalar;
    }

    double tock = MPI_Wtime();
    printf("div.c: cdiv\t%g seconds\n", tock - tick);
}
