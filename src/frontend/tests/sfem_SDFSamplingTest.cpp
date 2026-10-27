#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ContactConditions.hpp"
#include "sfem_test.hpp"
#include "smesh_grid.hpp"

// The cubic B-spline sampling of an SDF grid (resampling/surface/sfem_sdf_sampling.hpp): exact where it
// must be, h^4 where the field is smooth, and continuous to second derivatives across cells.
// The sampling itself is sresample's now, and so are its tests (tests/test_bspline.cpp there);
// what stays here is the contact surfaces reading it through ContactConditions.

// Every shell the contact path samples (QUADSHELL4 on hexahedra, TRISHELL3 on tetrahedra,
// EDGESHELL2 in 2D) reads the B-spline through ContactConditions. A tilted plane is reproduced,
// so the normal and the gap are exact at every node of the contact side, with either gap: the
// nodal one is the distance along the normal, -sdf / |grad sdf| (this sdf's gradient is not a unit
// vector), and the dual-weighted one, -sdf weighted with the dual basis, reproduces a linear field
// at the boundary nodes too.
static int check_tilted_plane(const std::shared_ptr<sfem::Mesh> &mesh,
                              const int                          dim,
                              const char                        *name,
                              const sfem::ContactConditions::Gap gap_type) {
    const real_t delta = 0.05, slope = 0.1;

    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    auto side = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t, const geom_t y, const geom_t) -> bool { return y > -1e-5 && y < 1e-5; });
    auto sdf = smesh::create_sdf(mesh->comm(),
                                 29,
                                 13,
                                 dim == 3 ? 29 : 5,
                                 -0.2,
                                 -0.3,
                                 dim == 3 ? -0.2 : -0.1,
                                 1.2,
                                 0.3,
                                 dim == 3 ? 1.2 : 0.1,
                                 [=](const geom_t x, const geom_t y, const geom_t) -> geom_t { return -delta - y + slope * x; });
    auto cc = sfem::ContactConditions::create(fs, sdf, side, sfem::EXECUTION_SPACE_HOST);
    cc->set_gap(gap_type);
    SFEM_TEST_ASSERT(cc->init() == SFEM_SUCCESS);

    const ptrdiff_t     n_c   = cc->n_constrained_dofs();
    const ptrdiff_t     ndofs = fs->n_dofs();
    std::vector<real_t> gap(n_c, 0), normals(dim * n_c, 0), unit(ndofs);
    SFEM_TEST_ASSERT(n_c > 0);
    SFEM_TEST_ASSERT(cc->signed_distance(gap.data()) == SFEM_SUCCESS);
    for (int d = 0; d < dim; ++d) {
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            unit[i] = (i % dim == d) ? 1 : 0;
        }
        cc->normal_project(unit.data(), normals.data() + d * n_c);
    }

    const idx_t *const   nodes  = cc->node_mapping()->data();
    geom_t *const *const points = mesh->points()->data();
    const real_t         len    = std::sqrt(1 + slope * slope);
    const real_t         n_ex[3] = {slope / len, -1 / len, 0};
    for (ptrdiff_t i = 0; i < n_c; ++i) {
        for (int d = 0; d < dim; ++d) {
            SFEM_TEST_APPROXEQ(normals[d * n_c + i], n_ex[d], 1e-5);
        }
        const real_t vertical = delta - slope * points[0][nodes[i]];
        SFEM_TEST_APPROXEQ(gap[i], gap_type == sfem::ContactConditions::Gap::NODAL ? vertical / len : vertical, 1e-5);
    }
    printf("%s (%s gap): %ld contact nodes, gap and normals exact at all\n",
           name,
           gap_type == sfem::ContactConditions::Gap::NODAL ? "nodal" : "dual-weighted",
           (long)n_c);
    return SFEM_TEST_SUCCESS;
}

// The normals of a contact surface moved by u are those of u alone, whatever it was sampled at
// before: re-linearising the obstacle (ContactConditions::update) replaces them. On a ball, so
// they differ from the ones at rest.
static int check_resampled_normals(const std::shared_ptr<sfem::Mesh> &mesh, const int dim, const char *name) {
    auto fs   = sfem::FunctionSpace::create(mesh, dim);
    auto side = sfem::Sideset::create_from_selector(
            mesh, [](const geom_t, const geom_t y, const geom_t) -> bool { return y > -1e-5 && y < 1e-5; });
    auto sdf = smesh::create_sdf(mesh->comm(),
                                 29,
                                 13,
                                 dim == 3 ? 29 : 5,
                                 -0.2,
                                 -0.3,
                                 dim == 3 ? -0.2 : -0.1,
                                 1.2,
                                 0.3,
                                 dim == 3 ? 1.2 : 0.1,
                                 [=](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
                                     const geom_t dx = x - 0.5, dy = y + 0.6, dz = dim == 3 ? z - 0.5 : 0;
                                     return 0.6 - std::sqrt(dx * dx + dy * dy + dz * dz);
                                 });

    const ptrdiff_t     ndofs = fs->n_dofs();
    std::vector<real_t> u(ndofs, 0);
    for (ptrdiff_t i = 0; i < ndofs; i += dim) {
        u[i] = 0.15;
    }

    auto at_rest_first = sfem::ContactConditions::create(fs, sdf, side, sfem::EXECUTION_SPACE_HOST);
    auto moved_only    = sfem::ContactConditions::create(fs, sdf, side, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_ASSERT(at_rest_first->init() == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(at_rest_first->update(u.data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(moved_only->update(u.data()) == SFEM_SUCCESS);

    const ptrdiff_t     n_c = at_rest_first->n_constrained_dofs();
    std::vector<real_t> unit(ndofs), a(n_c), b(n_c);
    real_t              diff = 0;
    for (int d = 0; d < dim; ++d) {
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            unit[i] = (i % dim == d) ? 1 : 0;
        }
        std::fill(a.begin(), a.end(), 0);
        std::fill(b.begin(), b.end(), 0);
        at_rest_first->normal_project(unit.data(), a.data());
        moved_only->normal_project(unit.data(), b.data());
        for (ptrdiff_t i = 0; i < n_c; ++i) {
            diff = std::max(diff, std::abs(a[i] - b[i]));
        }
    }
    printf("%s: normals after moving differ by %g from sampling there first\n", name, (double)diff);
    SFEM_TEST_ASSERT(diff < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int test_normals_follow_the_update() {
    auto comm = sfem::Communicator::self();
    SFEM_TEST_ASSERT(check_resampled_normals(sfem::Mesh::create_hex8_cube(comm, 4, 4, 4), 3, "QUADSHELL4") == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_resampled_normals(sfem::Mesh::create_tet4_cube(comm, 4, 4, 4), 3, "TRISHELL3") == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_resampled_normals(sfem::Mesh::create_quad4_square(comm, 4, 4), 2, "EDGESHELL2") == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int test_contact_surfaces_sample_planes() {
    auto comm = sfem::Communicator::self();
    for (auto gap : {sfem::ContactConditions::Gap::NODAL, sfem::ContactConditions::Gap::DUAL_WEIGHTED}) {
        SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_hex8_cube(comm, 4, 4, 4), 3, "QUADSHELL4", gap) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_tet4_cube(comm, 4, 4, 4), 3, "TRISHELL3", gap) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_quad4_square(comm, 4, 4), 2, "EDGESHELL2", gap) == SFEM_TEST_SUCCESS);
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_contact_surfaces_sample_planes);
    SFEM_RUN_TEST(test_normals_follow_the_update);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
