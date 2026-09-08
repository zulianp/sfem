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
        geom_t **const  points = fs.mesh().points()->data();
        const ptrdiff_t n      = fs.mesh().n_nodes();
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
        geom_t **const  points = fs.mesh().points()->data();
        const ptrdiff_t n      = fs.mesh().n_nodes();
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t px  = points[0][i];
            const real_t py  = points[1][i];
            const real_t pz  = points[2][i];
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
            const real_t a   = host->data()[i];
            const real_t b   = hd->data()[i];
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
            fprintf(stderr, "[Error] %s: %ld entries fail (tol=%g, max_abs=%g)\n", label, (long)n_fail, (double)tol, (double)max_abs);
            return SFEM_TEST_FAILURE;
        }
        return SFEM_TEST_SUCCESS;
    }

    std::shared_ptr<FunctionSpace> checkerboard_ss_space(const int block_size) {
        auto hex = Mesh::create_hex8_checkerboard_cube(Communicator::self(), 2, 2, 2);
        auto ss  = smesh::to_semistructured(2, hex, true, false);
        return FunctionSpace::create(ss, block_size);
    }

    int apply_host_device(const std::shared_ptr<FunctionSpace>     &fs,
                          const char                               *op_name,
                          const std::shared_ptr<Buffer<real_t>>    &x_host,
                          const real_t                              tol) {
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

int test_checkerboard_ss_gpu_em_laplacian_vs_host() {
    auto fs = checkerboard_ss_space(1);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_EQ(fs->mesh().n_blocks(), static_cast<size_t>(2));
    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-10) : real_t(1e-5);
    return apply_host_device(fs, "em:Laplacian", fill_scalar_host(*fs), tol);
}

int test_checkerboard_ss_gpu_em_linear_elasticity_vs_host() {
    auto fs = checkerboard_ss_space(3);
    SFEM_TEST_ASSERT(fs->has_semi_structured_mesh());
    SFEM_TEST_EQ(fs->mesh().n_blocks(), static_cast<size_t>(2));
    const real_t tol = sizeof(real_t) == sizeof(double) ? real_t(1e-8) : real_t(1e-4);
    return apply_host_device(fs, "em:LinearElasticity", fill_vector_host(*fs), tol);
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_checkerboard_ss_gpu_laplacian_vs_host);
    SFEM_RUN_TEST(test_checkerboard_ss_gpu_em_laplacian_vs_host);
    SFEM_RUN_TEST(test_checkerboard_ss_gpu_em_linear_elasticity_vs_host);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
