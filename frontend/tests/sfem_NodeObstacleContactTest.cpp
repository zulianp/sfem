// NodeObstacleContact against a plane, where every quantity is known in closed form.
//
// The obstacle is the half-space z < 0, its SDF -z (positive inside): at a node at height z the
// gap is z and the normal (0, 0, -1). The block's nodes start above it.

#include <cmath>
#include <memory>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_NodeObstacleContact.hpp"
#include "sfem_test.hpp"
#include "smesh_grid.hpp"

namespace {

    struct Fixture {
        std::shared_ptr<sfem::Mesh>                mesh;
        std::shared_ptr<sfem::NodeObstacleContact> contact;
        ptrdiff_t                                  nnodes{0};
        ptrdiff_t                                  ndofs{0};
    };

    Fixture make_fixture() {
        Fixture f;
        auto    comm = sfem::Communicator::world();
        f.mesh       = sfem::Mesh::create_hex8_cube(comm, 2, 2, 2, 0, 0, real_t(0.1), 1, 1, real_t(1.1));
        f.nnodes     = f.mesh->n_nodes();
        f.ndofs      = 3 * f.nnodes;

        auto sdf = smesh::create_sdf(comm, 41, 41, 41, -1, -1, -1, 2, 2, 2, [](const geom_t, const geom_t, const geom_t z) -> geom_t {
            return -z;
        });

        auto nodes = sfem::create_host_buffer<idx_t>(f.nnodes);
        for (ptrdiff_t i = 0; i < f.nnodes; ++i) nodes->data()[i] = (idx_t)i;
        f.contact = std::make_shared<sfem::NodeObstacleContact>(sdf, comm, nodes, f.mesh->points(), f.ndofs, 2);
        return f;
    }

    real_t pseudo_random(const ptrdiff_t i, const int salt) { return std::sin(real_t(0.7 * (i + 1) + salt)); }

}  // namespace

// The gap is the height, the frames are rotations whose first axis is the normal.
int test_gap_and_frames() {
    auto                f = make_fixture();
    std::vector<real_t> x(f.ndofs, 0);
    f.contact->update(x.data());

    const geom_t *const z   = f.mesh->points()->data()[2];
    const real_t *const gap = f.contact->gap()->data();
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        SFEM_TEST_ASSERT(std::abs(gap[i] - z[i]) < 1e-6);
    }

    std::vector<real_t> v(f.ndofs), w(f.ndofs);
    for (ptrdiff_t i = 0; i < f.ndofs; ++i) v[i] = w[i] = pseudo_random(i, 1);
    f.contact->to_frame(w.data());
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        // The normal component, and the length kept.
        SFEM_TEST_ASSERT(std::abs(w[i * 3] + v[i * 3 + 2]) < 1e-6);
        const real_t lv = v[i * 3] * v[i * 3] + v[i * 3 + 1] * v[i * 3 + 1] + v[i * 3 + 2] * v[i * 3 + 2];
        const real_t lw = w[i * 3] * w[i * 3] + w[i * 3 + 1] * w[i * 3 + 1] + w[i * 3 + 2] * w[i * 3 + 2];
        SFEM_TEST_ASSERT(std::abs(lv - lw) < 1e-12);
    }
    f.contact->from_frame(w.data());
    for (ptrdiff_t i = 0; i < f.ndofs; ++i) {
        SFEM_TEST_ASSERT(std::abs(w[i] - v[i]) < 1e-12);
    }

    std::vector<real_t> to_surface(f.ndofs);
    f.contact->gap_field(to_surface.data());
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        SFEM_TEST_ASSERT(std::abs(to_surface[i * 3] - 0) < 1e-6);
        SFEM_TEST_ASSERT(std::abs(to_surface[i * 3 + 2] + z[i]) < 1e-6);
    }
    return SFEM_TEST_SUCCESS;
}

// The reaction is the admissible part of the residual, -mu n with mu = max(0, -n . r): here
// max(0, r_z) along +z, nothing tangential.
int test_reaction() {
    auto                f = make_fixture();
    std::vector<real_t> x(f.ndofs, 0), r(f.ndofs), k(f.ndofs, 1), projected(f.ndofs), reaction(f.ndofs);
    for (ptrdiff_t i = 0; i < f.ndofs; ++i) r[i] = pseudo_random(i, 2);
    f.contact->update(x.data());

    real_t penetration = 0;
    f.contact->kkt_residual(r.data(), k.data(), projected.data(), &penetration);
    f.contact->reaction(reaction.data());

    int pushing = 0;
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        const real_t mu = std::max(real_t(0), r[i * 3 + 2]);
        SFEM_TEST_ASSERT(std::abs(reaction[i * 3]) < 1e-6 * mu + 1e-15);
        SFEM_TEST_ASSERT(std::abs(reaction[i * 3 + 1]) < 1e-6 * mu + 1e-15);
        SFEM_TEST_ASSERT(std::abs(reaction[i * 3 + 2] - mu) < 1e-6 * (1 + mu));
        for (int d = 0; d < 3; ++d) {
            SFEM_TEST_ASSERT(std::abs(r[i * 3 + d] - projected[i * 3 + d] - reaction[i * 3 + d]) < 1e-12);
        }
        pushing += mu > 0;
    }
    SFEM_TEST_ASSERT(pushing > 0);
    return SFEM_TEST_SUCCESS;
}

// Nodes whose next position x + dt v would be below the plane land on it; the others, and every
// tangential velocity, are left alone.
int test_project_velocity() {
    auto                f  = make_fixture();
    const real_t        dt = real_t(0.1);
    std::vector<real_t> x(f.ndofs, 0), v(f.ndofs), v0(f.ndofs);
    const geom_t *const z = f.mesh->points()->data()[2];
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        v[i * 3]     = pseudo_random(i, 3);
        v[i * 3 + 1] = pseudo_random(i, 4);
        v[i * 3 + 2] = -(z[i] + real_t(0.05) * pseudo_random(i, 5)) / dt;
    }
    v0 = v;
    f.contact->project_velocity(x.data(), v.data(), dt);

    // The sampled normal is the plane's to the grid's float precision, so a landing node's
    // tangential velocity moves by that much of its correction.
    int landed = 0;
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        if (z[i] + dt * v0[i * 3 + 2] < 0) {
            SFEM_TEST_ASSERT(std::abs(z[i] + dt * v[i * 3 + 2]) < 1e-6);
            SFEM_TEST_ASSERT(std::abs(v[i * 3] - v0[i * 3]) < 1e-5);
            SFEM_TEST_ASSERT(std::abs(v[i * 3 + 1] - v0[i * 3 + 1]) < 1e-5);
            landed++;
        } else {
            for (int d = 0; d < 3; ++d) SFEM_TEST_ASSERT(v[i * 3 + d] == v0[i * 3 + d]);
        }
    }
    SFEM_TEST_ASSERT(landed > 0 && landed < f.nnodes);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_gap_and_frames);
    SFEM_RUN_TEST(test_reaction);
    SFEM_RUN_TEST(test_project_velocity);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
