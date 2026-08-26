#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

    void set_material_parameter(const std::shared_ptr<sfem::Op>   &op,
                                const std::shared_ptr<sfem::Mesh> &mesh,
                                const char *const                  name,
                                const real_t                       value) {
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), name, value);
        }
    }

    void fill_affine_field(const std::shared_ptr<sfem::Mesh> &mesh,
                           const real_t                       gradient[3][3],
                           real_t *const SFEM_RESTRICT        field) {
        const ptrdiff_t n_nodes = mesh->n_nodes();
        auto            points  = mesh->points()->data();
        const geom_t   *x       = points[0];
        const geom_t   *y       = points[1];
        const geom_t   *z       = points[2];

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < n_nodes; ++i) {
            const real_t X[3] = {x[i], y[i], z[i]};
            for (int d = 0; d < 3; ++d) {
                field[3 * i + d] = gradient[d][0] * X[0] + gradient[d][1] * X[1] + gradient[d][2] * X[2];
            }
        }
    }

    bool is_interior_node(const geom_t x, const geom_t y, const geom_t z) {
        constexpr geom_t eps = 1e-7;
        return x > eps && x < 1 - eps && y > eps && y < 1 - eps && z > eps && z < 1 - eps;
    }

    int check_homogeneous_deformation(const smesh::ElemType element_type) {
        auto mesh  = sfem::Mesh::create_cube(sfem::Communicator::self(), element_type, 2, 2, 2, 0, 0, 0, 1, 1, 1);
        auto space = sfem::FunctionSpace::create(mesh, 3);
        auto op    = sfem::create_op(space, "GeneratedMooneyRivlinKelvinVoigtNewmark", sfem::EXECUTION_SPACE_HOST);
        SFEM_TEST_ASSERT(op != nullptr);

        constexpr real_t mu      = 2.5;
        constexpr real_t lambda  = 7.0;
        constexpr real_t eta_s   = 0.4;
        constexpr real_t eta_b   = 0.2;
        constexpr real_t alpha_v = 3.25;

        set_material_parameter(op, mesh, "mu", mu);
        set_material_parameter(op, mesh, "lmbda", lambda);
        set_material_parameter(op, mesh, "eta_s", eta_s);
        set_material_parameter(op, mesh, "eta_b", eta_b);
        set_material_parameter(op, mesh, "newmark_velocity_alpha", alpha_v);

        const ptrdiff_t ndofs    = space->n_dofs();
        auto            current  = sfem::create_host_buffer<real_t>(ndofs);
        auto            previous = sfem::create_host_buffer<real_t>(ndofs);
        auto            residual = sfem::create_host_buffer<real_t>(ndofs);

        const real_t grad_u[3][3] = {
                {0.12, 0.07, -0.03},
                {0.04, -0.08, 0.05},
                {-0.02, 0.03, 0.10},
        };

        const real_t grad_fdot[3][3] = {
                {0.17, -0.04, 0.02},
                {0.03, 0.11, -0.05},
                {-0.01, 0.06, -0.09},
        };

        real_t grad_previous[3][3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                grad_previous[i][j] = grad_fdot[i][j] - alpha_v * grad_u[i][j];
            }
        }

        fill_affine_field(mesh, grad_u, current->data());
        fill_affine_field(mesh, grad_previous, previous->data());
        std::fill(residual->data(), residual->data() + ndofs, real_t(0));

        op->set_field("previous", previous, 0);
        SFEM_TEST_ASSERT(op->gradient(current->data(), residual->data()) == SFEM_SUCCESS);

        auto          points       = mesh->points()->data();
        const geom_t *x            = points[0];
        const geom_t *y            = points[1];
        const geom_t *z            = points[2];
        real_t        max_all      = 0;
        real_t        max_interior = 0;
        ptrdiff_t     n_interior   = 0;

        for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
            real_t node_norm2 = 0;
            for (int d = 0; d < 3; ++d) {
                const real_t r = residual->data()[3 * node + d];
                SFEM_TEST_ASSERT(std::isfinite(r));
                node_norm2 += r * r;
            }

            const real_t node_norm = std::sqrt(node_norm2);
            max_all                = std::max(max_all, node_norm);
            if (is_interior_node(x[node], y[node], z[node])) {
                ++n_interior;
                max_interior = std::max(max_interior, node_norm);
            }
        }

        const real_t tol = real_t(1e-9) * std::max(real_t(1), max_all);
        std::printf("homogeneous deformation %s: interior_nodes=%td max_interior=%.6e max_all=%.6e tol=%.6e\n",
                    sfem::type_to_string(element_type),
                    n_interior,
                    (double)max_interior,
                    (double)max_all,
                    (double)tol);

        SFEM_TEST_ASSERT(n_interior > 0);
        SFEM_TEST_ASSERT(max_all > 0);
        SFEM_TEST_ASSERT(max_interior <= tol);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int test_homogeneous_deformation_hex8() { return check_homogeneous_deformation(smesh::HEX8); }

int test_homogeneous_deformation_hex27() { return check_homogeneous_deformation(smesh::HEX27); }

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_homogeneous_deformation_hex8);
    SFEM_RUN_TEST(test_homogeneous_deformation_hex27);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
