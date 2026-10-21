// NodeObstacleContact against a plane, where every quantity is known in closed form.
//
// The obstacle is the half-space z < 0, its SDF -z (positive inside): at a node at height z the
// gap is z and the normal (0, 0, -1). The block's nodes start above it.

#include <algorithm>
#include <cmath>
#include <limits>
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
        f.contact = std::make_shared<sfem::NodeObstacleContact>(sdf, comm, nodes, f.mesh->points(), f.ndofs, f.ndofs, 2);
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
// What the constraints see of the gap: the shift is a clearance, past the cutoff there is no
// obstacle, and a start inside it is released as the release goes from 0 to 1.
int test_shift_cutoff_release() {
    auto                f = make_fixture();
    const geom_t *const z = f.mesh->points()->data()[2];
    const real_t        shift = 0.05, cutoff = 0.5, sink = 0.2;

    f.contact->set_shift(shift);
    f.contact->set_cutoff(cutoff);
    std::vector<real_t> x(f.ndofs, 0);
    f.contact->update(x.data());
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
        const real_t g = f.contact->gap()->data()[i];
        if (z[i] - shift > cutoff) {
            SFEM_TEST_ASSERT(g >= real_t(SFEM_SDF_UNREACHABLE));
        } else {
            SFEM_TEST_ASSERT(std::abs(g - (z[i] - shift)) < 1e-6);
        }
    }

    // Sunk by `sink`, the bottom nodes start inside: released, nothing is; half released, half the
    // starting penetration is still admitted.
    for (ptrdiff_t i = 0; i < f.nnodes; ++i) x[i * 3 + 2] = -sink;
    f.contact->capture_penetration(x.data());
    for (const real_t release : {real_t(0), real_t(0.5), real_t(1)}) {
        f.contact->set_release(release);
        f.contact->update(x.data());
        for (ptrdiff_t i = 0; i < f.nnodes; ++i) {
            const real_t raw = z[i] - sink - shift;
            if (raw > cutoff) continue;
            const real_t p0 = std::max(real_t(0), -raw);
            SFEM_TEST_ASSERT(std::abs(f.contact->gap()->data()[i] - (raw + (1 - release) * p0)) < 1e-6);
        }
        if (release == 0) SFEM_TEST_ASSERT(f.contact->min_gap() > -1e-6);
        if (release == 1) SFEM_TEST_ASSERT(f.contact->min_gap() < -0.1);
    }
    return SFEM_TEST_SUCCESS;
}

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

#ifdef SFEM_ENABLE_CUDA
namespace {
    std::vector<real_t> host_copy(const real_t *const device, const ptrdiff_t n) {
        auto h = smesh::to_host(sfem::Buffer<real_t>::wrap(n, const_cast<real_t *>(device), sfem::MEMORY_SPACE_DEVICE));
        return std::vector<real_t>(h->data(), h->data() + n);
    }

    real_t worst(const std::vector<real_t> &a, const std::vector<real_t> &b) {
        real_t w = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            w     = std::max(w, std::abs(a[i] - b[i]));
            scale = std::max(scale, std::abs(a[i]));
        }
        return w / std::max(scale, real_t(1e-300));
    }
}  // namespace

// Every operation on the device against the host, on a sphere where normals and curvature vary
// from node to node and some nodes start inside.
int test_device_matches_host() {
    auto comm = sfem::Communicator::world();
    auto mesh = sfem::Mesh::create_hex8_cube(comm, 4, 4, 4, 0, 0, 0, 1, 1, 1);
    auto fs   = sfem::FunctionSpace::create(mesh, 3);
    const ptrdiff_t nnodes = mesh->n_nodes(), ndofs = 3 * nnodes;

    // A ball of radius 0.6 about (0.5, 0.5, -0.4), positive inside: its top cuts the block's base.
    auto sdf = smesh::create_sdf(comm, 61, 61, 61, -1, -1, -1.5, 2, 2, 1.5, [](const geom_t x, const geom_t y, const geom_t z) -> geom_t {
        return geom_t(0.6) - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z + 0.4) * (z + 0.4));
    });
    auto nodes = sfem::create_host_buffer<idx_t>(nnodes);
    for (ptrdiff_t i = 0; i < nnodes; ++i) nodes->data()[i] = (idx_t)i;
    sfem::NodeObstacleContact host(sdf, comm, nodes, mesh->points(), ndofs, ndofs, 2);
    sfem::NodeObstacleContact device(sdf, comm, nodes, mesh->points(), ndofs, ndofs, 2, sfem::EXECUTION_SPACE_DEVICE);

    auto to_device = [](const std::vector<real_t> &h) {
        return smesh::to_device(sfem::Buffer<real_t>::wrap(h.size(), const_cast<real_t *>(h.data())));
    };

    std::vector<real_t> x(ndofs), r(ndofs), k(ndofs), v(ndofs);
    for (ptrdiff_t i = 0; i < ndofs; ++i) {
        x[i] = real_t(0.05) * pseudo_random(i, 1);
        r[i] = pseudo_random(i, 2);
        k[i] = 1 + real_t(0.5) * pseudo_random(i, 3);
        v[i] = pseudo_random(i, 4);
    }
    auto dx = to_device(x), dr = to_device(r), dk = to_device(k), dv = to_device(v);

    host.update(x.data());
    device.update(dx->data());
    const real_t g = worst(std::vector<real_t>(host.gap()->data(), host.gap()->data() + nnodes), host_copy(device.gap()->data(), nnodes));
    printf("  gap %.3g\n", (double)g);
    SFEM_TEST_ASSERT(g < 1e-12);

    std::vector<real_t> w = v;
    auto                dw = to_device(v);
    host.to_frame(w.data());
    device.to_frame(dw->data());
    SFEM_TEST_ASSERT(worst(w, host_copy(dw->data(), ndofs)) < 1e-12);
    host.from_frame(w.data());
    device.from_frame(dw->data());
    SFEM_TEST_ASSERT(worst(w, host_copy(dw->data(), ndofs)) < 1e-12);

    std::vector<real_t> projected(ndofs);
    auto                dprojected = to_device(projected);
    real_t              pen_h = 0, pen_d = 0;
    const real_t        kkt_h = host.kkt_residual(r.data(), k.data(), projected.data(), &pen_h);
    const real_t        kkt_d = device.kkt_residual(dr->data(), dk->data(), dprojected->data(), &pen_d);
    printf("  kkt %.12g / %.12g, penetration %.6g / %.6g\n", (double)kkt_h, (double)kkt_d, (double)pen_h, (double)pen_d);
    SFEM_TEST_ASSERT(pen_h > 0);
    SFEM_TEST_ASSERT(std::abs(kkt_h - kkt_d) <= 1e-12 * kkt_h && std::abs(pen_h - pen_d) <= 1e-12 * pen_h);
    SFEM_TEST_ASSERT(worst(projected, host_copy(dprojected->data(), ndofs)) < 1e-12);

    std::vector<real_t> reaction(ndofs);
    auto                dreaction = to_device(reaction);
    host.reaction(reaction.data());
    device.reaction(dreaction->data());
    SFEM_TEST_ASSERT(worst(reaction, host_copy(dreaction->data(), ndofs)) < 1e-12);

    std::vector<real_t> ub(ndofs);
    auto                dub = to_device(ub);
    host.upper_bound(real_t(1e30), ub.data());
    device.upper_bound(real_t(1e30), dub->data());
    SFEM_TEST_ASSERT(worst(ub, host_copy(dub->data(), ndofs)) < 1e-12);

    // The curvature into the diagonal blocks of the node graph's BSR values, capped and not.
    auto graph = fs->node_to_node_graph();
    auto dgraph_rowptr = smesh::to_device(graph->rowptr());
    auto dgraph_colidx = smesh::to_device(graph->colidx());
    for (const bool capped : {false, true}) {
        std::vector<real_t> values(graph->nnz() * 9, 0);
        auto                dvalues = to_device(values);
        host.add_curvature(
                x.data(), r.data(), k.data(), capped, 1, graph->rowptr()->data(), graph->colidx()->data(), smesh::SMESH_DEFAULT, values.data());
        device.add_curvature(
                dx->data(), dr->data(), dk->data(), capped, 1, dgraph_rowptr->data(), dgraph_colidx->data(), smesh::SMESH_DEFAULT, dvalues->data());
        const real_t c = worst(values, host_copy(dvalues->data(), values.size()));
        printf("  curvature (capped %d) %.3g\n", (int)capped, (double)c);
        SFEM_TEST_ASSERT(c < 1e-12);
#if !SFEM_REAL_T_IS_FLOAT32
        // Into a tangent stored in single precision: the host's double values, rounded.
        auto hsingle = sfem::create_host_buffer<float>(values.size());
        std::fill(hsingle->data(), hsingle->data() + values.size(), 0.f);
        auto dsingle = smesh::to_device(hsingle);
        device.add_curvature(
                dx->data(), dr->data(), dk->data(), capped, 1, dgraph_rowptr->data(), dgraph_colidx->data(), smesh::SMESH_FLOAT32, dsingle->data());
        auto   single = smesh::to_host(dsingle);
        real_t vmax = 0, verr = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            vmax = std::max(vmax, std::abs(values[i]));
            verr = std::max(verr, std::abs(real_t(single->data()[i]) - values[i]));
        }
        printf("  curvature into single precision %.3g\n", (double)(verr / vmax));
        SFEM_TEST_ASSERT(vmax > 0 && verr <= std::numeric_limits<float>::epsilon() * vmax);
#endif
    }

    std::vector<real_t> field(ndofs);
    auto                dfield = to_device(field);
    host.gap_field(field.data());
    device.gap_field(dfield->data());
    SFEM_TEST_ASSERT(worst(field, host_copy(dfield->data(), ndofs)) < 1e-12);

    std::vector<real_t> vh = v;
    host.project_velocity(x.data(), vh.data(), real_t(0.1));
    device.project_velocity(dx->data(), dv->data(), real_t(0.1));
    SFEM_TEST_ASSERT(worst(vh, host_copy(dv->data(), ndofs)) < 1e-12);

    // The gap with a clearance, a cutoff and half of the start's penetration released.
    for (auto *c : {&host, &device}) {
        c->set_shift(real_t(0.01));
        c->set_cutoff(real_t(0.2));
    }
    host.capture_penetration(x.data());
    device.capture_penetration(dx->data());
    host.set_release(real_t(0.5));
    device.set_release(real_t(0.5));
    host.update(x.data());
    device.update(dx->data());
    const real_t go = worst(std::vector<real_t>(host.gap()->data(), host.gap()->data() + nnodes), host_copy(device.gap()->data(), nnodes));
    printf("  offset gap %.3g\n", (double)go);
    SFEM_TEST_ASSERT(go < 1e-12);
    return SFEM_TEST_SUCCESS;
}
#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_gap_and_frames);
    SFEM_RUN_TEST(test_shift_cutoff_release);
    SFEM_RUN_TEST(test_reaction);
    SFEM_RUN_TEST(test_project_velocity);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_matches_host);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
