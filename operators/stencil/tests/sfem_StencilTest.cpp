
#include <stdio.h>
#include "sfem_test.h"

#include "sfem_API.hpp"

#include "stencil3.h"

#include "hex8_laplacian_inline_cpu.h"

bool verbose{true};

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

    if (verbose) printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s]\n", yc * xc, elapsed, 1e-6 * (xc * yc) / elapsed);

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

#if 1
    if (sizeof(real_t) == 8) {
        scalar_t fff[6] = {1, 0, 0, 1, 0, 1};
        scalar_t A[8 * 8];
        hex8_laplacian_matrix_fff_integral(fff, A);

        // Elemental matrix to stencil
        const scalar_t x0 = A[13] + A[7];
        const scalar_t x1 = A[24] + A[5];
        const scalar_t x2 = A[12] + A[19] + A[25] + A[4];
        const scalar_t x3 = A[11] + A[20];
        const scalar_t x4 = A[18] + A[22];
        const scalar_t x5 = A[28] + A[2];
        const scalar_t x6 = A[29] + A[31] + A[3] + A[9];
        const scalar_t x7 = A[10] + A[32];
        const scalar_t x8 = A[16] + A[1] + A[27] + A[34];
        s[0]              = A[48];
        s[1]              = A[49] + A[56];
        s[2]              = A[57];
        s[3]              = A[40] + A[51];
        s[4]              = A[32] + A[41] + A[50] + A[59];
        s[5]              = A[33] + A[58];
        s[6]              = A[43];
        s[7]              = A[35] + A[42];
        s[8]              = A[34];
        s[9]              = A[16] + A[52];
        s[10]             = A[17] + A[24] + A[53] + A[60];
        s[11]             = A[25] + A[61];
        s[12]             = A[19] + A[44] + A[55] + A[8];
        s[13]             = A[0] + A[18] + A[27] + A[36] + A[45] + A[54] + A[63] + A[9];
        s[14]             = A[1] + A[26] + A[37] + A[62];
        s[15]             = A[11] + A[47];
        s[16]             = A[10] + A[39] + A[3] + A[46];
        s[17]             = A[2] + A[38];
        s[18]             = A[20];
        s[19]             = A[21] + A[28];
        s[20]             = A[29];
        s[21]             = A[12] + A[23];
        s[22]             = A[13] + A[22] + A[31] + A[4];
        s[23]             = A[30] + A[5];
        s[24]             = A[15];
        s[25]             = A[14] + A[7];
        s[26]             = A[6];
    }
#endif

    ptrdiff_t zc = 129;
    ptrdiff_t yc = 129;
    ptrdiff_t xc = 129;

    auto in = sfem::create_host_buffer<real_t>(xc * yc * zc);
    blas->values(in->size(), 1, in->data());

    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                in->data()[zi * yc * xc + yi * xc + xi] = (xi * xi + yi * yi + zi * zi);
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

    if (verbose) printf("#nodes %ld, TTS: %g [s] TP: %g [MDOF/s]\n", xc * yc * zc, elapsed, 1e-6 * (xc * yc * zc) / elapsed);

    for (ptrdiff_t zi = 0; zi < zc - 2; zi++) {
        for (ptrdiff_t yi = 0; yi < yc - 2; yi++) {
            for (ptrdiff_t xi = 0; xi < xc - 2; xi++) {
                real_t val = o[zi * (yc - 2) * (xc - 2) + yi * (xc - 2) + xi];
                SFEM_TEST_APPROXEQ(-6, val, sizeof(real_t) == 4 ? 1e-6 : 1e-8);
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
