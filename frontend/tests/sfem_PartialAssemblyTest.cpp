#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Env.hpp"
#include "sfem_Function.hpp"

// FIXME
#include "hex8_fff.h"
#include "sshex8_laplacian.h"

int test_hyperelasticity_partial_assembly() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const int dim = 3;
    const int N   = 4;

    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), N, N, N, 0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    fs->initialize_packed_mesh();

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "NeoHookeanOgden", es);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);

    // Active strain Fa: identity per element (AoS: 9 per element)
    const ptrdiff_t ne = mesh->n_elements();
    f->add_operator(op);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_buffer<real_t>(ndofs, es);
    auto            h     = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_mf  = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_bsr = sfem::create_buffer<real_t>(ndofs, es);

    // Zero displacement
    {
        auto du = sfem::to_host(u);
        for (ptrdiff_t i = 0; i < ndofs / dim; ++i) du->data()[i * dim + 0] = 0.1;
    }
    SFEM_TEST_ASSERT(f->update(u->data()) == SFEM_SUCCESS);

    // Deterministic input based on node coordinates
    {
        auto     dh = sfem::to_host(h);
        geom_t **pts{nullptr};
        if (fs->has_semi_structured_mesh()) {
            pts = fs->semi_structured_mesh().points()->data();
        } else {
            pts = fs->mesh_ptr()->points()->data();
        }

        const ptrdiff_t nn = fs->mesh_ptr()->n_nodes();
        for (ptrdiff_t i = 0; i < nn; ++i) {
            for (int b = 0; b < dim; ++b) {
                dh->data()[i * dim + b] = (real_t)((b + 1) * pts[0][i] * pts[0][i]);
            }
        }
    }

    auto mf  = sfem::create_linear_operator(MATRIX_FREE, f, u, es);
    auto bsr = sfem::create_linear_operator(BSR, f, u, es);
    SFEM_TEST_ASSERT(mf && bsr);

    mf->apply(h->data(), y_mf->data());
    bsr->apply(h->data(), y_bsr->data());

    // Basic sanity checks: results are finite and non-trivial
    auto   hy_mf    = sfem::to_host(y_mf);
    auto   hy_bsr   = sfem::to_host(y_bsr);

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_APPROXEQ(hy_mf->data()[i], hy_bsr->data()[i], (real_t)1e-10);
    }

    return SFEM_TEST_SUCCESS;
}

int test_hyperelasticity_active_strain_partial_assembly() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const int dim = 3;
    const int N   = 4;

    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::wrap(comm), N, N, N, 0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    fs->initialize_packed_mesh();

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, "NeoHookeanOgdenActiveStrainPacked", es);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);

    // Active strain Fa: identity per element (AoS: 9 per element)
    const ptrdiff_t ne = mesh->n_elements();
    auto            Fa = sfem::create_host_buffer<real_t>(9 * ne);
    for (ptrdiff_t e = 0; e < ne; ++e) {
        const real_t detFa = 2;
        for (int r = 0; r < dim; ++r) {
            for (int c = 0; c < dim; ++c) {
                Fa->data()[e * 9 + r * dim + c] = (r == c) ? ((real_t)1 / detFa) : (real_t)0;
            }
        }

        Fa->data()[e * 9] = detFa;
    }
    op->set_field("active_strain", Fa, 0);

    f->add_operator(op);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_buffer<real_t>(ndofs, es);
    auto            h     = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_mf  = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_bsr = sfem::create_buffer<real_t>(ndofs, es);

    // Zero displacement
    {
        auto du = sfem::to_host(u);
        for (ptrdiff_t i = 0; i < ndofs; ++i) du->data()[i] = 0;
    }
    SFEM_TEST_ASSERT(f->update(u->data()) == SFEM_SUCCESS);

    // Deterministic input based on node coordinates
    {
        auto     dh = sfem::to_host(h);
        geom_t **pts{nullptr};
        if (fs->has_semi_structured_mesh()) {
            pts = fs->semi_structured_mesh().points()->data();
        } else {
            pts = fs->mesh_ptr()->points()->data();
        }

        const ptrdiff_t nn = fs->mesh_ptr()->n_nodes();
        for (ptrdiff_t i = 0; i < nn; ++i) {
            for (int b = 0; b < dim; ++b) {
                dh->data()[i * dim + b] = (real_t)((b + 1) * pts[0][i] * pts[0][i]);
            }
        }
    }

    auto mf  = sfem::create_linear_operator(MATRIX_FREE, f, u, es);
    auto bsr = sfem::create_linear_operator(BSR, f, u, es);
    SFEM_TEST_ASSERT(mf && bsr);

    mf->apply(h->data(), y_mf->data());
    bsr->apply(h->data(), y_bsr->data());

    // Basic sanity checks: results are finite and non-trivial
    auto   hy_mf    = sfem::to_host(y_mf);
    auto   hy_bsr   = sfem::to_host(y_bsr);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_APPROXEQ(hy_mf->data()[i], hy_bsr->data()[i], (real_t)1e-10);
    }
    

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_hyperelasticity_partial_assembly);
    SFEM_RUN_TEST(test_hyperelasticity_active_strain_partial_assembly);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
