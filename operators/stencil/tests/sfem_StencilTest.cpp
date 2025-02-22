
#include <stdio.h>
#include "sfem_test.h"

#include "sfem_API.hpp"

#include "stencil3.h"

int test_stencil2() {
    auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    real_t s[3 * 3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};

    ptrdiff_t yc = 1000;
    ptrdiff_t xc = 1000;

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

    printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s]\n", yc * xc, elapsed, 1e-6 * (xc * yc) / elapsed);

    for (ptrdiff_t yi = 0; yi < yc - 2; yi++) {
        for (ptrdiff_t xi = 0; xi < xc - 2; xi++) {
            SFEM_TEST_ASSERT(-4 == o[yi * (xc - 2) + xi]);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_stencil3() {
    auto blas = sfem::blas<real_t>(sfem::EXECUTION_SPACE_HOST);

    real_t s[3 * 3 * 3] = {0, 0,  0, 0,  -1, 0,  0, 0,  0,  //
                           0, -1, 0, -1, 6,  -1, 0, -1, 0,  //
                           0, 0,  0, 0,  -1, 0,  0, 0,  0};

    ptrdiff_t zc = 1025;
    ptrdiff_t yc = 1025;
    ptrdiff_t xc = 1025;

    auto in = sfem::create_host_buffer<real_t>(xc * yc * zc);
    blas->values(in->size(), 1, in->data());

    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                in->data()[zi * yc * xc + yi * xc + xi] = xi * xi + yi * yi + zi * zi;
            }
        }
    }

    auto out = sfem::create_host_buffer<real_t>((xc - 2) * (yc - 2) * (zc - 2));
    auto o   = out->data();

    double tick = MPI_Wtime();

    par_slice_stencil_3x3x3
            // slice_stencil_3x3x3
            (xc - 2,
             yc - 2,
             zc - 2,
             s,  //
             xc,
             xc * yc,
             in->data(),
             //
             yc - 2,
             (xc - 2) * (xc - 2),
             o);

    double tock    = MPI_Wtime();
    double elapsed = tock - tick;

    printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s]\n", xc * yc * zc, elapsed, 1e-6 * (xc * yc * zc) / elapsed);

    for (ptrdiff_t zi = 0; zi < zc - 2; zi++) {
        for (ptrdiff_t yi = 0; yi < yc - 2; yi++) {
            for (ptrdiff_t xi = 0; xi < xc - 2; xi++) {
                real_t val = o[zi * (yc - 2) * (xc - 2) + yi * (xc - 2) + xi];
                SFEM_TEST_ASSERT(-6 == val);
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_stencil2);
    SFEM_RUN_TEST(test_stencil3);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
