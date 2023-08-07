#include "neumann.h"

#include "sfem_defs.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] -
           mat[0] * mat[5] * mat[7] - mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

static SFEM_INLINE real_t area3(const real_t left[3], const real_t right[3]) {
    real_t a = (left[1] * right[2]) - (right[1] * left[2]);
    real_t b = (left[2] * right[0]) - (right[2] * left[0]);
    real_t c = (left[0] * right[1]) - (right[0] * left[1]);
    return sqrt(a * a + b * b + c * c);
}

static SFEM_INLINE void tri_shell_3_integrate(const real_t px0,
                                              const real_t px1,
                                              const real_t px2,
                                              const real_t py0,
                                              const real_t py1,
                                              const real_t py2,
                                              const real_t pz0,
                                              const real_t pz1,
                                              const real_t pz2,
                                              const real_t *const SFEM_RESTRICT u,
                                              real_t *const SFEM_RESTRICT element_vector) {
    static const int stride = 1;
    // FLOATING POINT OPS!
    //       - Result: 3*ASSIGNMENT
    //       - Subexpressions: 35*ADD + 2*DIV + 85*MUL + 10*POW + 27*SUB
    const real_t x0 = 2 * px0;
    const real_t x1 = px1 * x0;
    const real_t x2 = py0 * x1;
    const real_t x3 = py1 * py2;
    const real_t x4 = pz0 * pz1;
    const real_t x5 = pz0 * pz2;
    const real_t x6 = pz1 * pz2;
    const real_t x7 = px2 * x0;
    const real_t x8 = py0 * x7;
    const real_t x9 = px1 * px2;
    const real_t x10 = 2 * py0;
    const real_t x11 = py1 * x10;
    const real_t x12 = py2 * x10;
    const real_t x13 = 2 * x9;
    const real_t x14 = 2 * pz0;
    const real_t x15 = pz1 * x14;
    const real_t x16 = pz2 * x14;
    const real_t x17 = POW2(px0);
    const real_t x18 = POW2(py1);
    const real_t x19 = POW2(py2);
    const real_t x20 = POW2(pz1);
    const real_t x21 = POW2(pz2);
    const real_t x22 = POW2(px1);
    const real_t x23 = POW2(py0);
    const real_t x24 = POW2(pz0);
    const real_t x25 = POW2(px2);
    const real_t x26 = 2 * x17;
    const real_t x27 = 2 * x23;
    const real_t x28 = 2 * x24;
    const real_t x29 =
        (1.0 / 6.0) * u[0] *
        sqrt(-py1 * x2 + py1 * x8 + py2 * x2 - py2 * x8 - x1 * x19 - x1 * x21 + x1 * x3 - x1 * x4 +
             x1 * x5 + x1 * x6 - x11 * x21 - x11 * x25 - x11 * x4 + x11 * x5 + x11 * x6 + x11 * x9 -
             x12 * x20 - x12 * x22 + x12 * x4 - x12 * x5 + x12 * x6 + x12 * x9 - x13 * x3 -
             x13 * x6 - x15 * x19 - x15 * x25 + x15 * x3 + x15 * x9 - x16 * x18 - x16 * x22 +
             x16 * x3 + x16 * x9 + x17 * x18 + x17 * x19 + x17 * x20 + x17 * x21 + x18 * x21 +
             x18 * x24 + x18 * x25 - x18 * x7 + x19 * x20 + x19 * x22 + x19 * x24 + x20 * x23 +
             x20 * x25 - x20 * x7 + x21 * x22 + x21 * x23 + x22 * x23 + x22 * x24 + x23 * x25 +
             x24 * x25 - x26 * x3 - x26 * x6 - x27 * x6 - x27 * x9 - x28 * x3 - x28 * x9 -
             2 * x3 * x6 + x3 * x7 + x4 * x7 - x5 * x7 + x6 * x7);
    element_vector[0 * stride] = x29;
    element_vector[1 * stride] = x29;
    element_vector[2 * stride] = x29;
}

static void tri_shell_3_surface_forcing_function(const ptrdiff_t nfaces,
                                          const idx_t *SFEM_RESTRICT faces_neumann,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t value,
                                          const int stride,
                                          real_t *SFEM_RESTRICT output) {  // Neumann
    double tick = MPI_Wtime();

    real_t element_vector[3];
    for (idx_t f = 0; f < nfaces; ++f) {
        idx_t i0 = faces_neumann[f * 3];
        idx_t i1 = faces_neumann[f * 3 + 1];
        idx_t i2 = faces_neumann[f * 3 + 2];

        tri_shell_3_integrate(xyz[0][i0],
                              xyz[0][i1],
                              xyz[0][i2],
                              xyz[1][i0],
                              xyz[1][i1],
                              xyz[1][i2],
                              xyz[2][i0],
                              xyz[2][i1],
                              xyz[2][i2],
                              &value,
                              element_vector);

        output[i0 * stride] += element_vector[0];
        output[i1 * stride] += element_vector[1];
        output[i2 * stride] += element_vector[2];
    }

    double tock = MPI_Wtime();
    printf("neumann.c: tri_shell_3_surface_forcing_function\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void tri_shell_6_integrate(const real_t px0,
                                              const real_t px1,
                                              const real_t px2,
                                              const real_t py0,
                                              const real_t py1,
                                              const real_t py2,
                                              const real_t pz0,
                                              const real_t pz1,
                                              const real_t pz2,
                                              const real_t *SFEM_RESTRICT u,
                                              real_t *SFEM_RESTRICT element_vector) {
    static const int stride = 1;
    // FLOATING POINT OPS!
    //       - Result: 6*ASSIGNMENT
    //       - Subexpressions: 35*ADD + 2*DIV + 85*MUL + 10*POW + 27*SUB
    const real_t x0 = 2 * px0;
    const real_t x1 = px1 * x0;
    const real_t x2 = py0 * x1;
    const real_t x3 = py1 * py2;
    const real_t x4 = pz0 * pz1;
    const real_t x5 = pz0 * pz2;
    const real_t x6 = pz1 * pz2;
    const real_t x7 = px2 * x0;
    const real_t x8 = py0 * x7;
    const real_t x9 = px1 * px2;
    const real_t x10 = 2 * py0;
    const real_t x11 = py1 * x10;
    const real_t x12 = py2 * x10;
    const real_t x13 = 2 * x9;
    const real_t x14 = 2 * pz0;
    const real_t x15 = pz1 * x14;
    const real_t x16 = pz2 * x14;
    const real_t x17 = POW2(px0);
    const real_t x18 = POW2(py1);
    const real_t x19 = POW2(py2);
    const real_t x20 = POW2(pz1);
    const real_t x21 = POW2(pz2);
    const real_t x22 = POW2(px1);
    const real_t x23 = POW2(py0);
    const real_t x24 = POW2(pz0);
    const real_t x25 = POW2(px2);
    const real_t x26 = 2 * x17;
    const real_t x27 = 2 * x23;
    const real_t x28 = 2 * x24;
    const real_t x29 =
        (1.0 / 6.0) * u[0] *
        sqrt(-py1 * x2 + py1 * x8 + py2 * x2 - py2 * x8 - x1 * x19 - x1 * x21 + x1 * x3 - x1 * x4 +
             x1 * x5 + x1 * x6 - x11 * x21 - x11 * x25 - x11 * x4 + x11 * x5 + x11 * x6 + x11 * x9 -
             x12 * x20 - x12 * x22 + x12 * x4 - x12 * x5 + x12 * x6 + x12 * x9 - x13 * x3 -
             x13 * x6 - x15 * x19 - x15 * x25 + x15 * x3 + x15 * x9 - x16 * x18 - x16 * x22 +
             x16 * x3 + x16 * x9 + x17 * x18 + x17 * x19 + x17 * x20 + x17 * x21 + x18 * x21 +
             x18 * x24 + x18 * x25 - x18 * x7 + x19 * x20 + x19 * x22 + x19 * x24 + x20 * x23 +
             x20 * x25 - x20 * x7 + x21 * x22 + x21 * x23 + x22 * x23 + x22 * x24 + x23 * x25 +
             x24 * x25 - x26 * x3 - x26 * x6 - x27 * x6 - x27 * x9 - x28 * x3 - x28 * x9 -
             2 * x3 * x6 + x3 * x7 + x4 * x7 - x5 * x7 + x6 * x7);
    element_vector[0 * stride] = x29;
    element_vector[1 * stride] = x29;
    element_vector[2 * stride] = x29;
}

static void tri_shell_6_surface_forcing_function(const ptrdiff_t nfaces,
                                          const idx_t *SFEM_RESTRICT faces_neumann,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t value,
                                          const int stride,
                                          real_t *SFEM_RESTRICT output) {
    double tick = MPI_Wtime();

    real_t element_vector[3];

    for (idx_t f = 0; f < nfaces; ++f) {
        idx_t i0 = faces_neumann[f * 6];
        idx_t i1 = faces_neumann[f * 6 + 1];
        idx_t i2 = faces_neumann[f * 6 + 2];
        idx_t i3 = faces_neumann[f * 6 + 3];
        idx_t i4 = faces_neumann[f * 6 + 4];
        idx_t i5 = faces_neumann[f * 6 + 5];

        tri_shell_6_integrate(xyz[0][i0],
                              xyz[0][i1],
                              xyz[0][i2],
                              xyz[1][i0],
                              xyz[1][i1],
                              xyz[1][i2],
                              xyz[2][i0],
                              xyz[2][i1],
                              xyz[2][i2],
                              &value,
                              element_vector);

        // Only edge dofs
        output[i3 * stride] += element_vector[0];
        output[i4 * stride] += element_vector[1];
        output[i5 * stride] += element_vector[2];
    }

    double tock = MPI_Wtime();
    printf("neumann.c: tri_shell_6_surface_forcing_function\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void edge_shell_2_surface_forcing_function_kernel(
    const real_t px0,
    const real_t px1,
    const real_t py0,
    const real_t py1,
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_vector) {
    static const int stride = 1;
    // FLOATING POINT OPS!
    //       - Result: 2*ASSIGNMENT
    //       - Subexpressions: 3*ADD + 2*DIV + 5*MUL + 5*POW + 2*SUB
    const real_t x0 =
        (1.0 / 2.0) * u[0] *
        sqrt(pow(px0, 2) - 2 * px0 * px1 + pow(px1, 2) + pow(py0, 2) - 2 * py0 * py1 + pow(py1, 2));
    element_vector[0 * stride] = x0;
    element_vector[1 * stride] = x0;
}

static void edge_shell_2_surface_forcing_function(const ptrdiff_t nfaces,
                                           const idx_t *SFEM_RESTRICT faces_neumann,
                                           geom_t **const SFEM_RESTRICT xyz,
                                           const real_t value,
                                           const int stride,
                                           real_t *SFEM_RESTRICT output) {
    double tick = MPI_Wtime();

    real_t element_vector[2];

    for (idx_t f = 0; f < nfaces; ++f) {
        idx_t i0 = faces_neumann[f * 2];
        idx_t i1 = faces_neumann[f * 2 + 1];

        edge_shell_2_surface_forcing_function_kernel(
            xyz[0][i0], xyz[0][i1], xyz[1][i0], xyz[1][i1], &value, element_vector);

        // Only edge dofs
        output[i0 * stride] += element_vector[0];
        output[i1 * stride] += element_vector[1];
    }

    double tock = MPI_Wtime();
    printf("neumann.c: edge_shell_3_surface_forcing_function\t%g seconds\n", tock - tick);
}

void surface_forcing_function(const int element_type,
                              const ptrdiff_t nfaces,
                              const idx_t *SFEM_RESTRICT faces_neumann,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t value,
                              real_t *SFEM_RESTRICT output) {
    switch (element_type) {
        case EDGE2: {
            edge_shell_2_surface_forcing_function(nfaces, faces_neumann, xyz, value, 1, output);
            break;
        }
        case TRI3: {
            tri_shell_3_surface_forcing_function(nfaces, faces_neumann, xyz, value, 1, output);
            break;
        }
        case TRI6: {
            tri_shell_6_surface_forcing_function(nfaces, faces_neumann, xyz, value, 1, output);
            break;
        }
        default: {
            assert(0 && "Implement me!");
            break;
        }
    }
}

void surface_forcing_function_vec(const int element_type,
                                  const ptrdiff_t nfaces,
                                  const idx_t *faces_neumann,
                                  geom_t **const xyz,
                                  const real_t value,
                                  const int block_size,
                                  const int component,
                                  real_t *output) {
    switch (element_type) {
        case EDGE2: {
            edge_shell_2_surface_forcing_function(
                nfaces, faces_neumann, xyz, value, block_size, &output[component]);
            break;
        }
        case TRI3: {
            tri_shell_3_surface_forcing_function(
                nfaces, faces_neumann, xyz, value, block_size, &output[component]);
            break;
        }
        case TRI6: {
            tri_shell_6_surface_forcing_function(
                nfaces, faces_neumann, xyz, value, block_size, &output[component]);
            break;
        }
        default: {
            assert(0 && "Implement me!");
            break;
        }
    }
}
