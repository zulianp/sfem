#include "sfem_test.hpp"

#include "sfem_DSSBXYZE.hpp"
#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_laplacian.hpp"

#include <algorithm>
#include <cmath>

static double tolerance() { return sizeof(real_t) == sizeof(float) ? 2e-4 : 1e-10; }

static void fill_constant_valid_lanes(const int       level,
                                      const ptrdiff_t ne,
                                      const int       bsize,
                                      const real_t    value,
                                      real_t *const   data) {
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);
    std::fill(data, data + nvalues, (real_t)0);

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    data[sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz)] = value;
                }
            }
        }
    }
}

static void fill_signal_valid_lanes(const int       level,
                                    const ptrdiff_t ne,
                                    const int       bsize,
                                    const real_t    shift,
                                    real_t *const   data) {
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);
    std::fill(data, data + nvalues, (real_t)0);

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const double v = 0.125 + (double)shift + 0.07 * (double)(e + 1) + 0.011 * (double)(ix + 1) +
                                     0.017 * (double)(iy + 1) + 0.019 * (double)(iz + 1);
                    data[sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz)] = (real_t)v;
                }
            }
        }
    }
}

static double dot_valid_lanes(const int             level,
                              const ptrdiff_t       ne,
                              const int             bsize,
                              const real_t *const   a,
                              const real_t *const   b) {
    double ret = 0;

    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int iz = 0; iz <= level; ++iz) {
            for (int iy = 0; iy <= level; ++iy) {
                for (int ix = 0; ix <= level; ++ix) {
                    const ptrdiff_t idx = sfem_dss_bxyze_index(level, bsize, e, ix, iy, iz);
                    ret += (double)a[idx] * (double)b[idx];
                }
            }
        }
    }

    return ret;
}

static int test_laplacian_constant_is_zero() {
    const ptrdiff_t nx    = 3;
    const ptrdiff_t ny    = 2;
    const ptrdiff_t nz    = 2;
    const int       bsize = 4;

    for (int level = 1; level <= 5; ++level) {
        const ptrdiff_t ne      = nx * ny * nz;
        const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);

        auto x = sfem::create_host_buffer<real_t>((size_t)nvalues);
        auto y = sfem::create_host_buffer<real_t>((size_t)nvalues);

        fill_constant_valid_lanes(level, ne, bsize, (real_t)1, x->data());
        std::fill(y->data(), y->data() + nvalues, (real_t)-3);

        SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(level,
                                                              nx,
                                                              ny,
                                                              nz,
                                                              bsize,
                                                              (real_t)0.25,
                                                              (real_t)0.5,
                                                              (real_t)0.75,
                                                              (real_t)2,
                                                              x->data(),
                                                              y->data()),
                     SFEM_SUCCESS);

        const double tol = tolerance();
        for (ptrdiff_t i = 0; i < nvalues; ++i) {
            SFEM_TEST_APPROXEQ((double)y->data()[i], 0.0, tol);
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_laplacian_local_operator_is_symmetric() {
    const ptrdiff_t nx    = 2;
    const ptrdiff_t ny    = 2;
    const ptrdiff_t nz    = 1;
    const int       bsize = 4;

    for (int level = 1; level <= 4; ++level) {
        const ptrdiff_t ne      = nx * ny * nz;
        const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);

        auto x  = sfem::create_host_buffer<real_t>((size_t)nvalues);
        auto y  = sfem::create_host_buffer<real_t>((size_t)nvalues);
        auto ax = sfem::create_host_buffer<real_t>((size_t)nvalues);
        auto ay = sfem::create_host_buffer<real_t>((size_t)nvalues);

        fill_signal_valid_lanes(level, ne, bsize, (real_t)0.0, x->data());
        fill_signal_valid_lanes(level, ne, bsize, (real_t)0.37, y->data());

        SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_volumetric_apply(level,
                                                                         nx,
                                                                         ny,
                                                                         nz,
                                                                         bsize,
                                                                         (real_t)0.5,
                                                                         (real_t)0.75,
                                                                         (real_t)1.25,
                                                                         (real_t)1.5,
                                                                         x->data(),
                                                                         ax->data()),
                     SFEM_SUCCESS);
        SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_volumetric_apply(level,
                                                                         nx,
                                                                         ny,
                                                                         nz,
                                                                         bsize,
                                                                         (real_t)0.5,
                                                                         (real_t)0.75,
                                                                         (real_t)1.25,
                                                                         (real_t)1.5,
                                                                         y->data(),
                                                                         ay->data()),
                     SFEM_SUCCESS);

        const double lhs = dot_valid_lanes(level, ne, bsize, x->data(), ay->data());
        const double rhs = dot_valid_lanes(level, ne, bsize, y->data(), ax->data());

        SFEM_TEST_APPROXEQ(lhs, rhs, tolerance() * std::max(1.0, std::max(std::abs(lhs), std::abs(rhs))));
    }

    return SFEM_TEST_SUCCESS;
}

static int test_laplacian_frontend_buffer_wrapper() {
    const ptrdiff_t nx    = 3;
    const ptrdiff_t ny    = 2;
    const ptrdiff_t nz    = 1;
    const int       level = 3;
    const int       bsize = 8;
    const ptrdiff_t ne    = nx * ny * nz;

    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);
    auto            values  = sfem::create_host_buffer<real_t>((size_t)nvalues);
    fill_constant_valid_lanes(level, ne, bsize, (real_t)4, values->data());

    sfem::DSSBXYZEBuffer in;
    in.level              = level;
    in.element_block_size = bsize;
    in.nelements          = ne;
    in.nvalues            = nvalues;
    in.values             = values;

    sfem::DSSBXYZEBuffer out;
    SFEM_TEST_EQ(sfem::dss_bxyze_laplacian_cartesian_apply(nx,
                                                           ny,
                                                           nz,
                                                           (real_t)0.25,
                                                           (real_t)0.5,
                                                           (real_t)1.0,
                                                           (real_t)1.0,
                                                           in,
                                                           &out),
                 SFEM_SUCCESS);

    SFEM_TEST_EQ(out.level, level);
    SFEM_TEST_EQ(out.element_block_size, bsize);
    SFEM_TEST_EQ(out.nelements, ne);
    SFEM_TEST_EQ(out.nvalues, nvalues);
    SFEM_TEST_ASSERT(out.values != nullptr);

    for (ptrdiff_t i = 0; i < nvalues; ++i) {
        SFEM_TEST_APPROXEQ((double)out.values->data()[i], 0.0, tolerance());
    }

    return SFEM_TEST_SUCCESS;
}

static int test_laplacian_rejects_invalid_inputs() {
    real_t x[8] = {0};
    real_t y[8] = {0};

    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(0, 1, 1, 1, 1, 1, 1, 1, 1, x, y), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 0, 1, 1, 1, 1, 1, 1, 1, x, y), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 0, 1, 1, 1, 1, x, y), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 0, 1, 1, 1, x, y), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 1, 1, 1, 1, nullptr, y), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 1, 1, 1, 1, x, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 1, 1, 1, 1, x, x), SFEM_FAILURE);

    sfem::DSSBXYZEBuffer out;
    sfem::DSSBXYZEBuffer empty;
    SFEM_TEST_EQ(sfem::dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 1, 1, empty, &out), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem::dss_bxyze_laplacian_cartesian_apply(1, 1, 1, 1, 1, 1, 1, empty, nullptr), SFEM_FAILURE);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_laplacian_constant_is_zero);
    SFEM_RUN_TEST(test_laplacian_local_operator_is_symmetric);
    SFEM_RUN_TEST(test_laplacian_frontend_buffer_wrapper);
    SFEM_RUN_TEST(test_laplacian_rejects_invalid_inputs);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
