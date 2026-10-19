#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.hpp"

#include "smesh_semistructured.hpp"

using namespace sfem;

namespace {

    void device_zeros(const std::shared_ptr<Buffer<real_t>> &v) { d_memset(v->data(), 0, v->size() * sizeof(real_t)); }

    std::shared_ptr<Buffer<real_t>> host_zeros(const ptrdiff_t n) {
        auto h = create_host_buffer<real_t>(n);
        std::fill(h->data(), h->data() + h->size(), real_t(0));
        return h;
    }

    std::shared_ptr<Buffer<real_t>> fill_scalar_host(const FunctionSpace &fs) {
        auto            h      = create_host_buffer<real_t>(fs.n_dofs());
        geom_t **const  points = fs.mesh_ptr()->points()->data();
        const ptrdiff_t n      = fs.mesh_ptr()->n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t px = points[0][i];
            const real_t py = points[1][i];
            const real_t pz = points[2][i];
            h->data()[i]    = px * px + real_t(0.5) * py - real_t(0.25) * pz * pz + real_t(0.125) * px * py;
        }
        return h;
    }

    std::shared_ptr<Buffer<real_t>> fill_vector_host(const FunctionSpace &fs) {
        auto            h      = create_host_buffer<real_t>(fs.n_dofs());
        geom_t **const  points = fs.mesh_ptr()->points()->data();
        const ptrdiff_t n      = fs.mesh_ptr()->n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t px      = points[0][i];
            const real_t py      = points[1][i];
            const real_t pz      = points[2][i];
            h->data()[3 * i + 0] = px + real_t(0.25) * py * py;
            h->data()[3 * i + 1] = py - real_t(0.125) * px * pz;
            h->data()[3 * i + 2] = pz + real_t(0.5) * px * px;
        }
        return h;
    }

    int compare_host_device(const char                            *label,
                            const std::shared_ptr<Buffer<real_t>> &host,
                            const std::shared_ptr<Buffer<real_t>> &device,
                            const real_t                           tol) {
        auto hd = smesh::to_host(device);
        SFEM_TEST_ASSERT(host->size() == hd->size());

        real_t    max_abs = 0;
        ptrdiff_t n_fail  = 0;
        for (ptrdiff_t i = 0; i < (ptrdiff_t)host->size(); ++i) {
            const real_t a = host->data()[i];
            const real_t b = hd->data()[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                fprintf(stderr, "[Error] %s: non-finite at i=%ld a=%g b=%g\n", label, (long)i, (double)a, (double)b);
                return SFEM_TEST_FAILURE;
            }
            const real_t abs = std::fabs(a - b);
            const real_t den = std::max(std::fabs(a), std::fabs(b));
            const real_t rel = (den > (real_t)1e-14) ? abs / den : abs;
            max_abs          = std::max(max_abs, abs);
            if (abs > tol && rel > tol) {
                ++n_fail;
            }
        }

        if (n_fail > 0) {
            fprintf(stderr,
                    "[Error] %s: %ld entries fail (tol=%g, max_abs=%g)\n",
                    label,
                    (long)n_fail,
                    (double)tol,
                    (double)max_abs);
            return SFEM_TEST_FAILURE;
        }
        return SFEM_TEST_SUCCESS;
    }

    std::shared_ptr<FunctionSpace> checkerboard_ss_space(const int block_size) {
        auto hex = Mesh::create_hex8_checkerboard_cube(Communicator::self(), 2, 2, 2);
        auto ss  = smesh::to_semistructured(2, hex, true, false);
        return FunctionSpace::create(ss, block_size);
    }

    // GPU EM kernels are HEX8 / SSHEX8 only. Checkerboard SS is hex+tet.
    std::shared_ptr<FunctionSpace> hex_ss_space(const int block_size) {
        auto hex = Mesh::create_hex8_cube(Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1);
        auto ss  = smesh::to_semistructured(2, hex, true, false);
        return FunctionSpace::create(ss, block_size);
    }

    int apply_host_device(const std::shared_ptr<FunctionSpace>  &fs,
                          const char                            *op_name,
                          const std::shared_ptr<Buffer<real_t>> &x_host,
                          const real_t                           tol) {
        auto op_h = create_op(fs, op_name, EXECUTION_SPACE_HOST);
        auto op_d = create_op(fs, op_name, EXECUTION_SPACE_DEVICE);
        SFEM_TEST_ASSERT(op_h != nullptr);
        SFEM_TEST_ASSERT(op_d != nullptr);
        SFEM_TEST_ASSERT(op_h->initialize() == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op_d->initialize() == SFEM_SUCCESS);

        auto y_h = host_zeros(fs->n_dofs());
        auto y_d = create_buffer<real_t>(fs->n_dofs(), EXECUTION_SPACE_DEVICE);
        device_zeros(y_d);

        auto x_d = smesh::to_device(x_host);
        SFEM_TEST_ASSERT(op_h->apply(nullptr, x_host->data(), y_h->data()) == SFEM_SUCCESS);
        SFEM_TEST_ASSERT(op_d->apply(nullptr, x_d->data(), y_d->data()) == SFEM_SUCCESS);
        return compare_host_device(op_name, y_h, y_d, tol);
    }

}  // namespace

int test_checkerboard_ss_gpu_laplacian_vs_host() {
    auto fs = checkerboard_ss_space(1);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_EQ(fs->mesh().n_blocks(), static_cast<size_t>(2));
    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);
    return apply_host_device(fs, "Laplacian", fill_scalar_host(*fs), tol);
}

int test_ss_gpu_em_laplacian_vs_host() {
    auto fs = hex_ss_space(1);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_EQ(fs->mesh().n_blocks(), static_cast<size_t>(1));
    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);
    return apply_host_device(fs, "em:Laplacian", fill_scalar_host(*fs), tol);
}

int test_ss_gpu_em_linear_elasticity_vs_linear_elasticity() {
    auto fs = hex_ss_space(3);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_EQ(fs->mesh().n_blocks(), static_cast<size_t>(1));
    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-8) : real_t(1e-4);

    auto x_host = fill_vector_host(*fs);
    auto op_em  = create_op(fs, "em:LinearElasticity", EXECUTION_SPACE_DEVICE);
    auto op_le  = create_op(fs, "LinearElasticity", EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(op_em != nullptr);
    SFEM_TEST_ASSERT(op_le != nullptr);
    SFEM_TEST_ASSERT(op_em->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_le->initialize() == SFEM_SUCCESS);

    auto y_em = create_buffer<real_t>(fs->n_dofs(), EXECUTION_SPACE_DEVICE);
    auto y_le = create_buffer<real_t>(fs->n_dofs(), EXECUTION_SPACE_DEVICE);
    device_zeros(y_em);
    device_zeros(y_le);
    auto x_d = smesh::to_device(x_host);
    SFEM_TEST_ASSERT(op_em->apply(nullptr, x_d->data(), y_em->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_le->apply(nullptr, x_d->data(), y_le->data()) == SFEM_SUCCESS);
    return compare_host_device("em:LinearElasticity vs LinearElasticity", smesh::to_host(y_le), y_em, tol);
}

// LinearElasticity without the affine approximation on a distorted semi-structured mesh, where it
// differs from it by about 1%: the device's isoparametric kernels against the host's (which
// sfem_BSRTangentTest holds to the HEX8 operator) -- the action, the diagonal and the diagonal blocks.
int test_ss_gpu_linear_elasticity_isoparametric_vs_host() {
    auto hex = Mesh::create_hex8_cube(Communicator::self(), 2, 2, 2, 0, 0, 0, 1, 1, 1);
    {
        geom_t *const *const xyz = hex->points()->data();
        for (ptrdiff_t i = 0; i < hex->n_nodes(); ++i) {
            const geom_t x = xyz[0][i], y = xyz[1][i], z = xyz[2][i];
            xyz[0][i] += geom_t(0.15) * y * z;
            xyz[1][i] += geom_t(0.1) * x * x;
            xyz[2][i] -= geom_t(0.12) * x * y;
        }
    }
    auto            fs  = FunctionSpace::create(smesh::to_semistructured(2, hex, true, false), 3);
    const ptrdiff_t n   = fs->n_dofs();
    const real_t    tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-4);

    auto op_h = create_op(fs, "LinearElasticity", EXECUTION_SPACE_HOST);
    auto op_d = create_op(fs, "LinearElasticity", EXECUTION_SPACE_DEVICE);
    SFEM_TEST_ASSERT(op_h != nullptr && op_d != nullptr);
    // Before initialize, which builds what the chosen route reads.
    op_h->set_option("ASSUME_AFFINE", false);
    op_d->set_option("ASSUME_AFFINE", false);
    SFEM_TEST_ASSERT(op_h->initialize() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_d->initialize() == SFEM_SUCCESS);

    auto x_h = fill_vector_host(*fs);
    auto x_d = smesh::to_device(x_h);
    auto y_h = host_zeros(n);
    auto y_d = create_buffer<real_t>(n, EXECUTION_SPACE_DEVICE);
    device_zeros(y_d);
    SFEM_TEST_ASSERT(op_h->apply(nullptr, x_h->data(), y_h->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_d->apply(nullptr, x_d->data(), y_d->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(compare_host_device("isoparametric apply", y_h, y_d, tol) == SFEM_TEST_SUCCESS);

    auto diag_h = host_zeros(n);
    auto diag_d = create_buffer<real_t>(n, EXECUTION_SPACE_DEVICE);
    device_zeros(diag_d);
    SFEM_TEST_ASSERT(op_h->hessian_diag(nullptr, diag_h->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_d->hessian_diag(nullptr, diag_d->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(compare_host_device("isoparametric diagonal", diag_h, diag_d, tol) == SFEM_TEST_SUCCESS);

    auto blocks_h = host_zeros(n / 3 * 6);
    auto blocks_d = create_buffer<real_t>(n / 3 * 6, EXECUTION_SPACE_DEVICE);
    device_zeros(blocks_d);
    SFEM_TEST_ASSERT(op_h->hessian_block_diag_sym(nullptr, blocks_h->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op_d->hessian_block_diag_sym(nullptr, blocks_d->data()) == SFEM_SUCCESS);
    return compare_host_device("isoparametric diagonal blocks", blocks_h, blocks_d, tol);
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_checkerboard_ss_gpu_laplacian_vs_host);
    SFEM_RUN_TEST(test_ss_gpu_em_laplacian_vs_host);
    SFEM_RUN_TEST(test_ss_gpu_em_linear_elasticity_vs_linear_elasticity);
    SFEM_RUN_TEST(test_ss_gpu_linear_elasticity_isoparametric_vs_host);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
