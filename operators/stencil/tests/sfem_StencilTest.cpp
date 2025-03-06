
#include <stdio.h>
#include "sfem_test.h"

#include "sfem_API.hpp"

#include "stencil3.h"
#include "sshex8_skeleton_stencil.h"
#include "hex8_laplacian_inline_cpu.h"


bool verbose{true};

int test_stencil2() {
    auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    real_t s[3 * 3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};

    ptrdiff_t yc = 2 * 33;
    ptrdiff_t xc = 2 * 65;

    auto in = sfem::create_host_buffer<real_t>(xc * yc);
    blas->values(in->size(), 1, in->data());

    for (ptrdiff_t yi = 0; yi < yc; yi++) {
        for (ptrdiff_t xi = 0; xi < xc; xi++) {
            in->data()[yi * xc + xi] = xi * xi + yi * yi;
        }
    }

    auto out = sfem::create_host_buffer<real_t>((xc - 2) * (yc - 2));
    auto o   = out->data();

    double tick = MPI_Wtime();

    slice_stencil_3x3(xc - 2,
                      yc - 2,
                      s[0 * 3 + 0],
                      s[0 * 3 + 1],
                      s[0 * 3 + 2],
                      s[1 * 3 + 0],
                      s[1 * 3 + 1],
                      s[1 * 3 + 2],
                      s[2 * 3 + 0],
                      s[2 * 3 + 1],
                      s[2 * 3 + 2],
                      xc,
                      in->data(),
                      xc - 2,
                      o);

    double tock    = MPI_Wtime();
    double elapsed = tock - tick;

    if (verbose) printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s]\n", yc * xc, elapsed, 1e-6 * (xc * yc) / elapsed);

    for (ptrdiff_t yi = 0; yi < yc - 2; yi++) {
        for (ptrdiff_t xi = 0; xi < xc - 2; xi++) {
            SFEM_TEST_ASSERT(-4 == o[yi * (xc - 2) + xi]);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_stencil3_against_original() {
    auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    scalar_t x[8] = {0, 1, 1.5, 0, 0, 1, 1.5, 0};
    scalar_t y[8] = {0, 0, 1, 1,  0, 0, 2, 1};
    scalar_t z[8] = {0, 0, 0, 0,  1, 1, 1, 2};
    scalar_t fff[6] = {0, 0, 0, 0, 0, 0};

    hex8_fff(x, y, z, 0.5, 0.5, 0.5, fff);
    scalar_t element_matrix[8 * 8];
    hex8_laplacian_matrix_fff_integral(fff, element_matrix);

    real_t stencil[3 * 3 * 3];
    hex8_matrix_to_stencil(element_matrix, stencil);

    ptrdiff_t level = 16;
    ptrdiff_t zc    = level + 1;
    ptrdiff_t yc    = level + 1;
    ptrdiff_t xc    = level + 1;

    auto in = sfem::create_host_buffer<real_t>(xc * yc * zc);
    blas->values(in->size(), 1, in->data());

    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                const ptrdiff_t zstride                      = yc * xc;
                const ptrdiff_t ystride                      = xc;
                in->data()[zi * zstride + yi * ystride + xi] = xc * xc * xc + 0.02 * yc * yc * yc + 0.03 * zc * zc * zc;
            }
        }
    }

    auto out          = sfem::create_host_buffer<real_t>(xc * yc * zc);
    auto out_original = sfem::create_host_buffer<real_t>(xc * yc * zc);

    double tick = MPI_Wtime();
    
    sshex8_stencil(xc, yc, zc, stencil, in->data(), out->data());
    sshex8_surface_stencil(xc, yc, zc, 1, xc, xc*yc, element_matrix, in->data(), out->data());
    
    double tack = MPI_Wtime();

    sshex8_apply_element_matrix(level, element_matrix, in->data(), out_original->data());
    
    double tock = MPI_Wtime();

    double elapsed_stencil = (tack - tick);
    double elapsed_original = (tock - tack);

    if (verbose) {
        printf("#nodes %ld, stencil TTS: %g [s] TP: %g [MDOF/s], original TTS: %g [s] TP: %g [MDOF/s]\n",
               xc * yc * zc,
               elapsed_stencil,
               1e-6 * (xc * yc * zc) / elapsed_stencil,
               elapsed_original,
               1e-6 * (xc * yc * zc) / elapsed_original);
    }

    auto o  = out->data();
    auto oo = out_original->data();
    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                const ptrdiff_t zstride = xc * yc;
                const ptrdiff_t ystride = xc;

                const real_t actual   = o[zi * zstride + yi * ystride + xi];
                const real_t expected = oo[zi * zstride + yi * ystride + xi];
                SFEM_TEST_APPROXEQ(actual, expected, sizeof(real_t) == 4 ? 1e-2 : 1e-8);
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_stencil3() {
    auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    real_t stencil[3 * 3 * 3] = {0, 0,  0, 0,  -1, 0,  0, 0,  0,  //
                                 0, -1, 0, -1, 6,  -1, 0, -1, 0,  //
                                 0, 0,  0, 0,  -1, 0,  0, 0,  0};

#if 1
    scalar_t fff[6] = {1, 0, 0, 1, 0, 1};
    scalar_t element_matrix[8 * 8];
    hex8_laplacian_matrix_fff_integral(fff, element_matrix);

    if (sizeof(real_t) == 8) {
        hex8_matrix_to_stencil(element_matrix, stencil);
    }
#endif

    // ptrdiff_t zc = 257;
    // ptrdiff_t yc = 257;
    // ptrdiff_t xc = 257;

    ptrdiff_t zc = 129;
    ptrdiff_t yc = 259;
    ptrdiff_t xc = 257;

    auto in = sfem::create_host_buffer<real_t>(xc * yc * zc);
    blas->values(in->size(), 1, in->data());

    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                const ptrdiff_t zstride                      = yc * xc;
                const ptrdiff_t ystride                      = xc;
                in->data()[zi * zstride + yi * ystride + xi] = (xi * xi + yi * yi + zi * zi);
            }
        }
    }

    auto out = sfem::create_host_buffer<real_t>(xc * yc * zc);

    double tick = MPI_Wtime();

    int repeat = MAX(3, 100000 / (xc * yc * zc));

    for (int r = 0; r < repeat; r++) {
        // sshex8_stencil // (uncomment to see serial performance)
        par_sshex8_stencil
        (xc, yc, zc, stencil, in->data(), out->data());
        sshex8_surface_stencil(xc, yc, zc, 1, xc, xc*yc, element_matrix, in->data(), out->data());
    }

    double tock    = MPI_Wtime();
    double elapsed = (tock - tick) / repeat;

    if (verbose) {
        printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s] (repeat=%d)\n",
               xc * yc * zc,
               elapsed,
               1e-6 * (xc * yc * zc) / elapsed,
               repeat);
    }

    auto o = out->data();
    for (ptrdiff_t zi = 1; zi < zc - 1; zi++) {
        for (ptrdiff_t yi = 1; yi < yc - 1; yi++) {
            for (ptrdiff_t xi = 1; xi < xc - 1; xi++) {
                const ptrdiff_t zstride = xc * yc;
                const ptrdiff_t ystride = xc;

                const real_t val = o[zi * zstride + yi * ystride + xi];
                SFEM_TEST_APPROXEQ(-6 * repeat, val, sizeof(real_t) == 4 ? 1e-6 : 1e-8);
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_stencil2);
    SFEM_RUN_TEST(test_stencil3);
    SFEM_RUN_TEST(test_stencil3_against_original);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
