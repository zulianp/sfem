#include "sfem_test.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_SpectralElementLaplacian.hpp"
#include "sfem_aliases.hpp"
#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_laplacian.hpp"
#include "sfem_dss_bxyze_spectral_laplacian.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"

#include <algorithm>
#include <cmath>

static double tolerance() { return sizeof(real_t) == sizeof(float) ? 2e-4 : 1e-10; }
static double cartesian_tolerance() { return sizeof(real_t) == sizeof(float) ? 1e-3 : 1e-7; }

static void fill_global_signal(const ptrdiff_t n, real_t *const values) {
    for (ptrdiff_t i = 0; i < n; ++i) {
        values[i] = (real_t)(0.19 + 0.013 * (double)(i % 97) + 1e-5 * (double)i);
    }
}

static int compare_spectral_laplacian_case(const int level, const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz, const int bsize) {
    auto macro_mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), nx, ny, nz, 0, 0, 0, 1, 1, 1);
    auto mesh       = smesh::to_semistructured(level, macro_mesh, false, false);
    auto space      = sfem::FunctionSpace::create(mesh, 1);
    auto op         = sfem::SpectralElementLaplacian::create(space);

    SFEM_TEST_ASSERT(mesh != nullptr);
    SFEM_TEST_ASSERT(space != nullptr);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_EQ(op->initialize(), SFEM_SUCCESS);

    auto block = mesh->block(0);
    const ptrdiff_t ne      = block->n_elements();
    const ptrdiff_t ndofs   = space->n_dofs();
    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);

    auto x_global = sfem::create_host_buffer<real_t>((size_t)ndofs);
    auto y_global = sfem::create_host_buffer<real_t>((size_t)ndofs);
    auto x_bxyze  = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto y_bxyze  = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto y_cartesian = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto y_ref    = sfem::create_host_buffer<real_t>((size_t)nvalues);

    SFEM_TEST_ASSERT(x_global != nullptr);
    SFEM_TEST_ASSERT(y_global != nullptr);
    SFEM_TEST_ASSERT(x_bxyze != nullptr);
    SFEM_TEST_ASSERT(y_bxyze != nullptr);
    SFEM_TEST_ASSERT(y_cartesian != nullptr);
    SFEM_TEST_ASSERT(y_ref != nullptr);

    fill_global_signal(ndofs, x_global->data());
    std::fill(y_global->data(), y_global->data() + ndofs, (real_t)0);

    setenv("SFEM_USE_GLL", "0", 1);
    SFEM_TEST_EQ(op->apply(nullptr, x_global->data(), y_global->data()), SFEM_SUCCESS);

    const idx_t *const *const elements = reinterpret_cast<const idx_t *const *>(block->elements()->data());
    const geom_t *const *const points  = reinterpret_cast<const geom_t *const *>(mesh->points()->data());

    SFEM_TEST_EQ(sfem_dss_global_to_bxyze(level, ne, bsize, elements, x_global->data(), x_bxyze->data()), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_global_to_bxyze(level, ne, bsize, elements, y_global->data(), y_ref->data()), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_lagrange_spectral_laplacian_apply(
                         level, nx, ny, nz, bsize, elements, points, x_bxyze->data(), y_bxyze->data()),
                 SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_cartesian_apply_blocked(level,
                                                                  nx,
                                                                  ny,
                                                                  nz,
                                                                  bsize,
                                                                  bsize,
                                                                  (real_t)1 / (real_t)nx,
                                                                  (real_t)1 / (real_t)ny,
                                                                  (real_t)1 / (real_t)nz,
                                                                  (real_t)1,
                                                                  x_bxyze->data(),
                                                                  y_cartesian->data()),
                 SFEM_SUCCESS);

    double max_abs_diff = 0;
    double max_cartesian_abs_diff = 0;
    for (ptrdiff_t i = 0; i < nvalues; ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs((double)y_bxyze->data()[i] - (double)y_ref->data()[i]));
        max_cartesian_abs_diff =
                std::max(max_cartesian_abs_diff, std::abs((double)y_cartesian->data()[i] - (double)y_ref->data()[i]));
    }

    if (max_abs_diff >= tolerance() || max_cartesian_abs_diff >= cartesian_tolerance()) {
        fprintf(stderr,
                "level=%d nx=%ld ny=%ld nz=%ld B=%d spectral_diff=%g tolerance=%g cartesian_diff=%g cartesian_tolerance=%g\n",
                level,
                (long)nx,
                (long)ny,
                (long)nz,
                bsize,
                max_abs_diff,
                tolerance(),
                max_cartesian_abs_diff,
                cartesian_tolerance());
    }

    SFEM_TEST_ASSERT(max_abs_diff < tolerance());
    SFEM_TEST_ASSERT(max_cartesian_abs_diff < cartesian_tolerance());
    return SFEM_TEST_SUCCESS;
}

static int test_spectral_laplacian_cellwise_matches_global() {
    SFEM_TEST_EQ(compare_spectral_laplacian_case(2, 2, 2, 2, 4), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ(compare_spectral_laplacian_case(4, 2, 2, 1, 4), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ(compare_spectral_laplacian_case(8, 1, 1, 1, 8), SFEM_TEST_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_spectral_laplacian_cellwise_matches_global);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
