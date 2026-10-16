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
    std::copy(tangent->values(), tangent->values() + nnz * bs * bs, values->data());

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
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_device_diagonal_matches_host);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
