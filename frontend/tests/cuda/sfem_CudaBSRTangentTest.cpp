// The device BSR diagonal kernel against the host one, on a tangent assembled on the host (no
// generated operator assembles a BSR tangent on the device in 3D yet).

#include <algorithm>
#include <cmath>

#include "sfem_BSRTangent.hpp"
#include "sfem_cuda_BSR.hpp"
#include "../sfem_elasticity_test_problem.hpp"
#include "sfem_test.hpp"

using namespace sfem;

int test_device_diagonal_matches_host() {
    auto p = test::make_cube_problem("GeneratedNeoHookeanOgden", 3, real_t(0.1));

    const geom_t *const *const xyz = p.mesh->points()->data();
    real_t *const              x   = p.x->data();
    for (ptrdiff_t i = 0; i < p.mesh->n_nodes(); ++i) {
        x[i * 3 + 1] += real_t(0.03) * xyz[0][i];
        x[i * 3 + 2] -= real_t(0.02) * xyz[0][i] * xyz[1][i];
    }
    p.f->apply_constraints(x);

    auto tangent = BSRTangent::create(p.f, EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(tangent->assemble(x) == SFEM_SUCCESS);

    const ptrdiff_t rows = tangent->block_rows();
    const int       bs   = tangent->block_size();
    const ptrdiff_t nnz  = tangent->rowptr()[rows];

    auto rowptr = create_host_buffer<count_t>(rows + 1);
    auto colidx = create_host_buffer<idx_t>(nnz);
    auto values = create_host_buffer<real_t>(nnz * bs * bs);
    std::copy(tangent->rowptr(), tangent->rowptr() + rows + 1, rowptr->data());
    std::copy(tangent->colidx(), tangent->colidx() + nnz, colidx->data());
    const real_t *const h_values = static_cast<const real_t *>(tangent->values());
    std::copy(h_values, h_values + nnz * bs * bs, values->data());

    auto h_diag = create_host_buffer<real_t>(rows * bs);
    SFEM_TEST_ASSERT(tangent->diagonal(h_diag->data()) == SFEM_SUCCESS);

    auto d_rowptr = smesh::to_device(rowptr);
    auto d_colidx = smesh::to_device(colidx);
    auto d_values = smesh::to_device(values);
    auto d_diag   = create_buffer<real_t>(rows * bs, EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(d_bsr_extract_diagonal(rows, bs, d_rowptr->data(), d_colidx->data(), d_values->data(), d_diag->data()) ==
                     SFEM_SUCCESS);
    auto from_device = smesh::to_host(d_diag);

    for (ptrdiff_t i = 0; i < rows * bs; ++i) {
        SFEM_TEST_ASSERT(h_diag->data()[i] != 0);
        SFEM_TEST_ASSERT(from_device->data()[i] == h_diag->data()[i]);
    }

    // The product, against the host's.
    {
        const ptrdiff_t n = rows * bs;
        auto            v = create_host_buffer<real_t>(n);
        auto            y = create_host_buffer<real_t>(n);
        for (ptrdiff_t i = 0; i < n; ++i) v->data()[i] = std::sin(real_t(0.3 * i));
        tangent->op()->apply(v->data(), y->data());
        auto d_v = smesh::to_device(v);
        auto d_y = create_buffer<real_t>(n, EXECUTION_SPACE_DEVICE);
        const int product_status = d_bsr_spmv_native<count_t, idx_t, real_t, real_t>(
                rows, bs, d_rowptr->data(), d_colidx->data(), d_values->data(), real_t(0), d_v->data(), d_y->data(), nullptr);
        SFEM_TEST_ASSERT(product_status == SFEM_SUCCESS);
        auto   from_device = smesh::to_host(d_y);
        real_t worst = 0, scale = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(from_device->data()[i] - y->data()[i]));
            scale = std::max(scale, std::abs(y->data()[i]));
        }
        printf("  product: worst relative difference %.3g\n", (double)(worst / scale));
        SFEM_TEST_ASSERT(worst <= 1e-14 * scale);
    }

    // The largest absolute row sum, and the diagonal shifted by a fraction of a reference.
    const real_t h_sum = tangent->max_abs_row_sum();
    const real_t d_sum = d_bsr_max_abs_row_sum<count_t, idx_t, real_t, real_t>(rows, bs, d_rowptr->data(), d_colidx->data(), d_values->data());
    printf("  max |row| sum: host %.17g, device %.17g\n", (double)h_sum, (double)d_sum);
    SFEM_TEST_ASSERT(h_sum > 0 && std::abs(h_sum - d_sum) <= 1e-14 * h_sum);

    SFEM_TEST_ASSERT(tangent->shift_diagonal(real_t(0.25), h_diag->data()) == SFEM_SUCCESS);
    const int shifted_status =
            d_bsr_shift_diagonal<count_t, idx_t, real_t, real_t>(rows, bs, d_rowptr->data(), d_colidx->data(), real_t(0.25), d_diag->data(), d_values->data());
    SFEM_TEST_ASSERT(shifted_status == SFEM_SUCCESS);
    auto shifted = smesh::to_host(d_values);
    for (ptrdiff_t i = 0; i < nnz * bs * bs; ++i) {
        SFEM_TEST_ASSERT(shifted->data()[i] == static_cast<const real_t *>(tangent->values())[i]);
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_device_diagonal_matches_host);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
