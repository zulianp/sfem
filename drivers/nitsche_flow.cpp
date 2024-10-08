#include "sfem_Function.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"
#include "sfem_defs.h"

#include "matrixio_array.h"

#include <mpi.h>
#include <stdio.h>
#include <sys/stat.h>

#define POW2(x) ((x) * (x))

static SFEM_INLINE void quad4_fff_s(const scalar_t px0,
                                    const scalar_t px2,
                                    const scalar_t py0,
                                    const scalar_t py2,
                                    scalar_t *const fff) {
    const scalar_t x0 = -py0 + py2;
    const scalar_t x1 = -px0 + px2;
    assert(x0 != 0);
    assert(x1 != 0);
    fff[0] = x0 / x1;
    fff[1] = 0;
    fff[2] = x1 / x0;
}

static SFEM_INLINE scalar_t quad4_det_fff(const scalar_t *const fff) {
    return fff[0] * fff[2] - POW2(fff[1]);
}

static SFEM_INLINE void quad4_laplacian_apply_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                  const scalar_t *const SFEM_RESTRICT u,
                                                  accumulator_t *const SFEM_RESTRICT
                                                          element_vector) {
    const scalar_t x0 = (1.0 / 3.0) * fff[2];
    const scalar_t x1 = u[3] * x0;
    const scalar_t x2 = (1.0 / 6.0) * u[2];
    const scalar_t x3 = fff[2] * x2;
    const scalar_t x4 = u[0] * x0;
    const scalar_t x5 = (1.0 / 6.0) * fff[2];
    const scalar_t x6 = u[1] * x5;
    const scalar_t x7 = (1.0 / 2.0) * fff[1];
    const scalar_t x8 = u[0] * x7 - u[2] * x7;
    const scalar_t x9 = (1.0 / 3.0) * fff[0];
    const scalar_t x10 = (1.0 / 6.0) * u[3];
    const scalar_t x11 = fff[0] * x10 - fff[0] * x2 + u[0] * x9 - u[1] * x9;
    const scalar_t x12 = u[1] * x0;
    const scalar_t x13 = u[0] * x5;
    const scalar_t x14 = u[2] * x0;
    const scalar_t x15 = fff[2] * x10;
    const scalar_t x16 = u[1] * x7 - u[3] * x7;
    const scalar_t x17 = (1.0 / 6.0) * fff[0];
    const scalar_t x18 = u[0] * x17 - u[1] * x17 - u[2] * x9 + u[3] * x9;
    element_vector[0] = -x1 + x11 - x3 + x4 + x6 + x8;
    element_vector[1] = -x11 + x12 + x13 - x14 - x15 - x16;
    element_vector[2] = -x12 - x13 + x14 + x15 - x18 - x8;
    element_vector[3] = x1 + x16 + x18 + x3 - x4 - x6;
}

int aa_quad4_laplacian_nitsche_BC_apply(const ptrdiff_t nx,
                                        const ptrdiff_t ny,
                                        const ptrdiff_t *const strides,
                                        const geom_t ox,
                                        const geom_t oy,
                                        const geom_t dx,
                                        const geom_t dy,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny; ++j) {
        for (ptrdiff_t i = 0; i < nx; ++i) {
            idx_t ev[4];
            accumulator_t element_vector[4];
            scalar_t element_u[4];
            scalar_t fff[3];

            const geom_t x0 = ox + i * dx;
            const geom_t x2 = ox + (i + 1) * dx;

            const geom_t y0 = oy + j * dy;
            const geom_t y2 = oy + (j + 1) * dy;

            ev[0] = i * strides[0] + j * strides[1];
            ev[1] = (i + 1) * strides[0] + j * strides[1];
            ev[2] = (i + 1) * strides[0] + (j + 1) * strides[1];
            ev[3] = i * strides[0] + (j + 1) * strides[1];

            for (int v = 0; v < 4; ++v) {
                element_u[v] = u[ev[v]];
            }

            quad4_fff_s(x0, x2, y0, y2, fff);
            quad4_laplacian_apply_fff(fff, element_u, element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }

    return SFEM_SUCCESS;
}

int copy_at_BC(const ptrdiff_t nx,
               const ptrdiff_t ny,
               const ptrdiff_t *const SFEM_RESTRICT strides,
               const real_t *const SFEM_RESTRICT in,
               real_t *const SFEM_RESTRICT out) {
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny + 1; j++) {
        out[0 * strides[0] + j * strides[1]] = in[0 * strides[0] + j * strides[1]];
        out[nx * strides[0] + j * strides[1]] = in[nx * strides[0] + j * strides[1]];
    }

    return SFEM_SUCCESS;
}

int set_BC(const ptrdiff_t nx,
           const ptrdiff_t ny,
           const ptrdiff_t *const strides,
           real_t *const SFEM_RESTRICT x) {
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny + 1; j++) {
        x[0 * strides[0] + j * strides[1]] = -1;
        x[nx * strides[0] + j * strides[1]] = 1;
    }
    return SFEM_SUCCESS;
}

static SFEM_INLINE void edgeshell2_jacobian_adjugate_and_determinant_s(
        const geom_t px0,
        const geom_t px1,
        const geom_t py0,
        const geom_t py1,
        scalar_t *const SFEM_RESTRICT jacobian_adjugate,
        scalar_t *const SFEM_RESTRICT jacobian_determinant) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py1;
    const scalar_t x2 = sqrt(POW2(x0) + POW2(x1));
    const scalar_t x3 = 1.0 / x2;
    jacobian_adjugate[0] = x0 * x3;
    jacobian_adjugate[1] = x1 * x3;
    *jacobian_determinant = x2;
}

static SFEM_INLINE void nitsche_apply_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                          const scalar_t jacobian_determinant,
                                          const scalar_t gamma,
                                          const scalar_t eps,
                                          const scalar_t hE,
                                          const scalar_t *const SFEM_RESTRICT u,
                                          accumulator_t *const SFEM_RESTRICT element_vector) {
    const scalar_t x0 = gamma * hE;
    const scalar_t x1 = 1.0 / (eps + x0);
    const scalar_t x2 = 1.0 / jacobian_determinant;
    const scalar_t x3 = POW2(adjugate[0]);
    const scalar_t x4 = POW2(adjugate[1]);
    const scalar_t x5 = sqrt(x3 + x4);
    const scalar_t x6 = u[0] * x2 * x5;
    const scalar_t x7 = x0 * x1;
    const scalar_t x8 = x2 / x5;
    const scalar_t x9 = x3 * x8;
    const scalar_t x10 = x4 * x8;
    const scalar_t x11 = x10 + x9;
    const scalar_t x12 = -u[0] + u[1];
    const scalar_t x13 = eps * x7 * (x10 * x12 + x12 * x9);
    element_vector[0] = jacobian_determinant *
                        (x1 * ((1.0 / 3.0) * u[0] + (1.0 / 6.0) * u[1]) + x11 * x13 + x6 * x7);
    element_vector[1] =
            jacobian_determinant * (x1 * ((1.0 / 6.0) * u[0] + (1.0 / 3.0) * u[1]) - x11 * x13 -
                                    x7 * (-u[0] * x10 - u[0] * x9 + u[1] * x10 + u[1] * x9 + x6));
}

static SFEM_INLINE void nitsche_gradient_adj(const scalar_t *const SFEM_RESTRICT adjugate,
                                             const scalar_t jacobian_determinant,
                                             const scalar_t gamma,
                                             const scalar_t eps,
                                             const scalar_t hE,
                                             const scalar_t u0,
                                             const scalar_t g0,
                                             accumulator_t *const SFEM_RESTRICT element_vector) {
    const scalar_t x0 = POW2(adjugate[0]);
    const scalar_t x1 = POW2(adjugate[1]);
    const scalar_t x2 = 1 / (jacobian_determinant * sqrt(x0 + x1));
    const scalar_t x3 = x0 * x2 + x1 * x2;
    const scalar_t x4 = -x3;
    const scalar_t x5 = gamma * hE;
    const scalar_t x6 = 1.0 / (eps + x5);
    const scalar_t x7 = x5 * x6;
    const scalar_t x8 = u0 * x7;
    const scalar_t x9 = eps * g0 * x7;
    const scalar_t x10 = -1.0 / 2.0 * eps * g0 * x6 - 1.0 / 2.0 * u0 * x6;
    element_vector[0] = jacobian_determinant * (-x10 - x4 * x8 - x4 * x9);
    element_vector[1] = jacobian_determinant * (-x10 - x3 * x8 - x3 * x9);
}

static const scalar_t EPS_LEFT = 0.1;
static const scalar_t EPS_RIGHT = 1000;
static const scalar_t GAMMA = 0.001;

int aa_quad4_laplacian_nitsche_BC_rhs(const ptrdiff_t nx,
                                      const ptrdiff_t ny,
                                      const ptrdiff_t *const strides,
                                      const geom_t ox,
                                      const geom_t oy,
                                      const geom_t dx,
                                      const geom_t dy,
                                      real_t *const SFEM_RESTRICT values) {
    scalar_t u0_left = -1;
    scalar_t u0_right = 1;
    scalar_t g0_left = -2;
    scalar_t g0_right = 2;
    const scalar_t hE = dy;

// Left
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny; j++) {
        scalar_t jacobian_adjugate[2];
        scalar_t jacobian_determinant;
        accumulator_t element_vector[2];
        idx_t ev[2];

        ev[0] = (j + 1) * strides[1];
        ev[1] = j * strides[1];

        const geom_t x0 = ox;
        const geom_t x1 = ox;

        const geom_t y0 = oy + (j + 1) * dy;
        const geom_t y1 = oy + j * dy;

        edgeshell2_jacobian_adjugate_and_determinant_s(
                x0, x1, y0, y1, jacobian_adjugate, &jacobian_determinant);

        nitsche_gradient_adj(jacobian_adjugate,
                             jacobian_determinant,
                             GAMMA,
                             EPS_LEFT,
                             hE,
                             u0_left,
                             g0_left,
                             element_vector);

        for (int d = 0; d < 2; d++) {
#pragma omp atomic update
            values[ev[d]] += element_vector[d];
        }
    }

// Right
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny; j++) {
        scalar_t jacobian_adjugate[2];
        scalar_t jacobian_determinant;
        accumulator_t element_vector[2];
        idx_t ev[2];
        ev[0] = nx * strides[0] + j * strides[1];
        ev[1] = nx * strides[0] + (j + 1) * strides[1];

        const geom_t x0 = ox + nx * dx;
        const geom_t x1 = ox + nx * dx;

        const geom_t y0 = oy + j * dy;
        const geom_t y1 = oy + (j + 1) * dy;

        edgeshell2_jacobian_adjugate_and_determinant_s(
                x0, x1, y0, y1, jacobian_adjugate, &jacobian_determinant);

        nitsche_gradient_adj(jacobian_adjugate,
                             jacobian_determinant,
                             GAMMA,
                             EPS_RIGHT,
                             hE,
                             u0_right,
                             g0_right,
                             element_vector);

        for (int d = 0; d < 2; d++) {
#pragma omp atomic update
            values[ev[d]] += element_vector[d];
        }
    }

    return SFEM_SUCCESS;
}

int nitsche_BC_apply(const ptrdiff_t nx,
                     const ptrdiff_t ny,
                     const ptrdiff_t *const strides,
                     const geom_t ox,
                     const geom_t oy,
                     const geom_t dx,
                     const geom_t dy,
                     const real_t *const SFEM_RESTRICT u,
                     real_t *const SFEM_RESTRICT values) {
    const scalar_t hE = dy;

// Left
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny; j++) {
        scalar_t jacobian_adjugate[2];
        scalar_t jacobian_determinant;
        scalar_t element_u[2];
        accumulator_t element_vector[2];
        idx_t ev[2];

        ev[0] = (j + 1) * strides[1];
        ev[1] = j * strides[1];

        for (int d = 0; d < 2; d++) {
            element_u[d] = u[ev[d]];
        }

        const geom_t x0 = ox;
        const geom_t x1 = ox;

        const geom_t y0 = oy + (j + 1) * dy;
        const geom_t y1 = oy + j * dy;

        edgeshell2_jacobian_adjugate_and_determinant_s(
                x0, x1, y0, y1, jacobian_adjugate, &jacobian_determinant);

        nitsche_apply_adj(jacobian_adjugate,
                          jacobian_determinant,
                          GAMMA,
                          EPS_LEFT,
                          hE,
                          element_u,
                          element_vector);

        for (int d = 0; d < 2; d++) {
#pragma omp atomic update
            values[ev[d]] += element_vector[d];
        }
    }

// Right
#pragma omp parallel for
    for (ptrdiff_t j = 0; j < ny; j++) {
        scalar_t jacobian_adjugate[2];
        scalar_t jacobian_determinant;
        scalar_t element_u[2];
        accumulator_t element_vector[2];
        idx_t ev[2];
        ev[0] = nx * strides[0] + j * strides[1];
        ev[1] = nx * strides[0] + (j + 1) * strides[1];

        for (int d = 0; d < 2; d++) {
            element_u[d] = u[ev[d]];
        }

        const geom_t x0 = ox + nx * dx;
        const geom_t x1 = ox + nx * dx;

        const geom_t y0 = oy + j * dy;
        const geom_t y1 = oy + (j + 1) * dy;

        edgeshell2_jacobian_adjugate_and_determinant_s(
                x0, x1, y0, y1, jacobian_adjugate, &jacobian_determinant);

        nitsche_apply_adj(jacobian_adjugate,
                          jacobian_determinant,
                          GAMMA,
                          EPS_RIGHT,
                          hE,
                          element_u,
                          element_vector);

        for (int d = 0; d < 2; d++) {
#pragma omp atomic update
            values[ev[d]] += element_vector[d];
        }
    }

    return SFEM_SUCCESS;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 4) {
        fprintf(stderr, "usage: %s <nx> <ny> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const ptrdiff_t nx = atol(argv[1]);
    const ptrdiff_t ny = atol(argv[2]);
    const char *output_folder = argv[3];

    const ptrdiff_t strides[2] = {1, nx + 1};
    const geom_t ox = -1;
    const geom_t oy = 1;
    const geom_t dx = 2. / nx;
    const geom_t dy = 2. / ny;

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    const ptrdiff_t ndofs = (nx + 1) * (ny + 1);
    real_t *u = (real_t *)calloc(ndofs, sizeof(real_t));
    real_t *rhs = (real_t *)calloc(ndofs, sizeof(real_t));

    bool use_nitsche = true;

    if (use_nitsche) {
        aa_quad4_laplacian_nitsche_BC_rhs(nx, ny, strides, ox, oy, dx, dy, rhs);
    } else {
        set_BC(nx, ny, strides, u);
        set_BC(nx, ny, strides, rhs);
    }

    auto op = sfem::make_op<real_t>(
            ndofs,
            ndofs,
            [=](const real_t *x, real_t *y) {
                aa_quad4_laplacian_nitsche_BC_apply(nx, ny, strides, ox, oy, dx, dy, x, y);

                if (use_nitsche) {
                    nitsche_BC_apply(nx, ny, strides, ox, oy, dx, dy, x, y);
                } else {
                    copy_at_BC(nx, ny, strides, x, y);
                }
            },
            sfem::EXECUTION_SPACE_HOST);

    auto solver = sfem::h_bcgs<real_t>();
    solver->set_n_dofs(ndofs);
    solver->set_max_it(2000);
    solver->set_op(op);
    solver->apply(rhs, u);

    char path[1024 * 10];
    sprintf(path, "%s/u.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, u, ndofs, ndofs);

    sprintf(path, "%s/rhs.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, rhs, ndofs, ndofs);

    free(u);
    free(rhs);
    return MPI_Finalize();
}
