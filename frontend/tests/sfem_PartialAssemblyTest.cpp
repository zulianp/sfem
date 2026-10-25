#include <stdio.h>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "smesh_env.hpp"

// FIXME
#include "hex8_fff.hpp"
#include "sshex8_laplacian.hpp"

#include <algorithm>
#include <cmath>

// Independent of the assembled path: the Hessian apply y = H(u) h must match the central
// difference of the gradient, (g(u + eps h) - g(u - eps h)) / (2 eps).
int check_apply_against_gradient_fd(const std::shared_ptr<sfem::Function> &f,
                                    const sfem::SharedBuffer<real_t>      &u,
                                    const sfem::SharedBuffer<real_t>      &h,
                                    const sfem::SharedBuffer<real_t>      &y,
                                    const sfem::ExecutionSpace             es) {
    const ptrdiff_t ndofs = u->size();
    const real_t    eps   = 1e-6;

    auto up = sfem::create_buffer<real_t>(ndofs, es);
    auto um = sfem::create_buffer<real_t>(ndofs, es);
    auto gp = sfem::create_buffer<real_t>(ndofs, es);
    auto gm = sfem::create_buffer<real_t>(ndofs, es);

    {
        auto hu  = smesh::to_host(u);
        auto hh  = smesh::to_host(h);
        auto hup = smesh::to_host(up);
        auto hum = smesh::to_host(um);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            hup->data()[i] = hu->data()[i] + eps * hh->data()[i];
            hum->data()[i] = hu->data()[i] - eps * hh->data()[i];
        }
    }

    SFEM_TEST_ASSERT(f->gradient(up->data(), gp->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(f->gradient(um->data(), gm->data()) == SFEM_SUCCESS);

    auto hy  = smesh::to_host(y);
    auto hgp = smesh::to_host(gp);
    auto hgm = smesh::to_host(gm);

    real_t fd_max = 0, diff_max = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        const real_t fd = (hgp->data()[i] - hgm->data()[i]) / (2 * eps);
        fd_max          = std::max(fd_max, std::abs(fd));
        diff_max        = std::max(diff_max, std::abs(hy->data()[i] - fd));
    }

    SFEM_TEST_ASSERT(fd_max > 0);
    SFEM_TEST_APPROXEQ(diff_max / fd_max, (real_t)0, (real_t)1e-6);
    return SFEM_TEST_SUCCESS;
}

int test_hyperelasticity_partial_assembly(const std::string &op_name) {
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const int dim = 3;
    const int N   = 4;

    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(), N, N, N, 0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    fs->initialize_packed_mesh();

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, op_name.c_str(), es);
    SFEM_TEST_ASSERT(op != nullptr);
    SFEM_TEST_ASSERT(op->initialize() == SFEM_SUCCESS);

    // Active strain Fa: identity per element (AoS: 9 per element)
    f->add_operator(op);

    const ptrdiff_t ndofs = fs->n_dofs();
    auto            u     = sfem::create_buffer<real_t>(ndofs, es);
    auto            h     = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_mf  = sfem::create_buffer<real_t>(ndofs, es);
    auto            y_bsr = sfem::create_buffer<real_t>(ndofs, es);

    // Zero displacement
    {
        auto du = smesh::to_host(u);
        for (ptrdiff_t i = 0; i < ndofs / dim; ++i) du->data()[i * dim + 0] = 0.1;
    }
    SFEM_TEST_ASSERT(f->update(u->data()) == SFEM_SUCCESS);

    // Deterministic input based on node coordinates
    {
        auto     dh = smesh::to_host(h);
        geom_t **pts{nullptr};
        if (fs->has_semi_structured_mesh()) {
            pts = fs->mesh().points()->data();
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

    auto mf  = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, u, es);
    auto bsr = sfem::create_linear_operator(sfem::op_type::BSR, f, u, es);
    SFEM_TEST_ASSERT(mf && bsr);

    mf->apply(h->data(), y_mf->data());
    bsr->apply(h->data(), y_bsr->data());

    SFEM_TEST_ASSERT(check_apply_against_gradient_fd(f, u, h, y_mf, es) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_apply_against_gradient_fd(f, u, h, y_bsr, es) == SFEM_TEST_SUCCESS);

    auto hy_mf  = smesh::to_host(y_mf);
    auto hy_bsr = smesh::to_host(y_bsr);

    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_APPROXEQ(hy_mf->data()[i], hy_bsr->data()[i], (real_t)1e-10);
    }

    return SFEM_TEST_SUCCESS;
}

int test_hyperelasticity_active_strain_partial_assembly(const std::string &op_name) {
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    const int dim = 3;
    const int N   = 10;

    auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::world(), N, N, N, 0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    fs->initialize_packed_mesh();

    // std::cout << "Mesh:" << std::endl;
    // mesh->print(std::cout);
    // std::cout << std::endl;

    // mesh->write(smesh::Path("mesh_r"));

    // std::cout << "Packed mesh:" << std::endl;
    // fs->packed_mesh()->print(std::cout, 1);
    // std::cout << std::endl;

    auto f  = sfem::Function::create(fs);
    auto op = sfem::create_op(fs, op_name.c_str(), es);
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
        auto du = smesh::to_host(u);
        for (ptrdiff_t i = 0; i < ndofs; ++i) du->data()[i] = 0;
    }
    SFEM_TEST_ASSERT(f->update(u->data()) == SFEM_SUCCESS);

    // Deterministic input based on node coordinates
    {
        auto     dh = smesh::to_host(h);
        geom_t **pts{nullptr};
        if (fs->has_semi_structured_mesh()) {
            pts = fs->mesh().points()->data();
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

    auto mf  = sfem::create_linear_operator(sfem::op_type::MATRIX_FREE, f, u, es);
    auto bsr = sfem::create_linear_operator(sfem::op_type::BSR, f, u, es);
    SFEM_TEST_ASSERT(mf && bsr);

    mf->apply(h->data(), y_mf->data());
    bsr->apply(h->data(), y_bsr->data());

    SFEM_TEST_ASSERT(check_apply_against_gradient_fd(f, u, h, y_mf, es) == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_apply_against_gradient_fd(f, u, h, y_bsr, es) == SFEM_TEST_SUCCESS);

    auto hy_mf  = smesh::to_host(y_mf);
    auto hy_bsr = smesh::to_host(y_bsr);

    // The partial-assembly tensor is stored as metric_tensor_t (float): compare relative to its precision
    real_t y_max = 0;
    for (ptrdiff_t i = 0; i < ndofs; ++i) y_max = std::max(y_max, std::abs(hy_bsr->data()[i]));
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        SFEM_TEST_APPROXEQ(hy_mf->data()[i], hy_bsr->data()[i], (real_t)1e-6 * y_max);
    }

    return SFEM_TEST_SUCCESS;
}

int test_hyperelasticity_partial_assembly_all() {
    return test_hyperelasticity_partial_assembly("NeoHookeanOgden") +
           test_hyperelasticity_partial_assembly("MooneyRivlin");
}

int test_hyperelasticity_active_strain_partial_assembly_all() {
    return test_hyperelasticity_active_strain_partial_assembly("NeoHookeanOgdenActiveStrainPacked") +
           test_hyperelasticity_active_strain_partial_assembly("MooneyRivlinActiveStrainPacked");
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_hyperelasticity_partial_assembly_all);
    SFEM_RUN_TEST(test_hyperelasticity_active_strain_partial_assembly_all);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
