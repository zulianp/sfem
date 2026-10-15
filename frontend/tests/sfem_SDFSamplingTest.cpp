#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_ContactConditions.hpp"
#include "node_interpolate.hpp"
#include "sfem_SDFObstacle.hpp"
#include "sfem_sdf_sampling.hpp"
#include "sfem_test.hpp"
#include "smesh_grid.hpp"

// The cubic B-spline sampling of an SDF grid (resampling/surface/sfem_sdf_sampling.hpp): exact where it
// must be, h^4 where the field is smooth, and continuous to second derivatives across cells.

namespace {
    using Field = std::function<real_t(const real_t, const real_t, const real_t)>;

    // An SDF grid of `field` on [0, 1]^3 with n nodes per axis, and its coefficients.
    struct Sampled {
        std::shared_ptr<smesh::Grid<geom_t>> sdf;
        std::shared_ptr<sfem::SDFBSpline>    spline;

        Sampled(const Field &field, const ptrdiff_t n) {
            sdf          = smesh::create_sdf(sfem::Communicator::self(),
                                    n,
                                    n,
                                    n,
                                    0,
                                    0,
                                    0,
                                    1,
                                    1,
                                    1,
                                    [&field](const geom_t x, const geom_t y, const geom_t z) -> geom_t { return field(x, y, z); });
            spline       = sfem::SDFBSpline::create(sdf);
        }

        template <int ORDER>
        bool sample(const real_t x, const real_t y, const real_t z, real_t *f, real_t *g, real_t *H) const {
            return sdf_sample<ORDER>(sdf->nlocal(),
                                     spline->stride->data(),
                                     sdf->origin(),
                                     sdf->delta(),
                                     spline->origin_coefficient(),
                                     x,
                                     y,
                                     z,
                                     f,
                                     g,
                                     H);
        }
    };

    const real_t sphere_radius = 0.3;

    real_t sphere(const real_t x, const real_t y, const real_t z) {
        return sphere_radius - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));
    }
}  // namespace

// Nodes are interpolated, and a linear field (a plane) is reproduced everywhere, gradient
// included; float storage of the coefficients sets the tolerance.
int test_reproduces_nodes_and_planes() {
    const Field   plane = [](const real_t x, const real_t y, const real_t z) { return 0.3 * x - 0.7 * y + 0.2 * z + 0.1; };
    const Sampled s(plane, 17);

    std::mt19937                           rng(7);
    std::uniform_real_distribution<real_t> u(0, 0.999);
    real_t                                 f, g[3], H[6];
    for (int k = 0; k < 1000; ++k) {
        const real_t x = u(rng), y = u(rng), z = u(rng);
        SFEM_TEST_ASSERT(s.sample<2>(x, y, z, &f, g, H));
        SFEM_TEST_APPROXEQ(f, plane(x, y, z), 1e-6);
        SFEM_TEST_APPROXEQ(g[0], 0.3, 1e-5);
        SFEM_TEST_APPROXEQ(g[1], -0.7, 1e-5);
        SFEM_TEST_APPROXEQ(g[2], 0.2, 1e-5);
        for (int i = 0; i < 6; ++i) {
            SFEM_TEST_APPROXEQ(H[i], 0, 1e-3);
        }
    }

    const Sampled  r(sphere, 17);
    const real_t   h = 1. / 16;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            for (int k = 0; k < 16; ++k) {
                SFEM_TEST_ASSERT(r.sample<0>(i * h, j * h, k * h, &f, nullptr, nullptr));
                SFEM_TEST_APPROXEQ(f, (real_t)r.sdf->data()[i + 17 * (j + 17 * k)], 1e-6);
            }
        }
    }
    return SFEM_TEST_SUCCESS;
}

// Near a sphere's surface (away from its centre, where the distance is not smooth) the sampled
// distance converges as h^4; trilinear sampling converged as h^2.
int test_converges_as_h4_on_a_sphere() {
    std::mt19937                           rng(11);
    std::uniform_real_distribution<real_t> u(-1, 1);
    std::vector<std::array<real_t, 3>>     points;
    while (points.size() < 2000) {
        const real_t p[3] = {0.5 + u(rng) * 0.45, 0.5 + u(rng) * 0.45, 0.5 + u(rng) * 0.45};
        const real_t r    = std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5) + (p[2] - 0.5) * (p[2] - 0.5));
        if (std::abs(r - sphere_radius) < 0.04) {
            points.push_back({p[0], p[1], p[2]});
        }
    }

    real_t previous = 0;
    for (const ptrdiff_t n : {13, 25, 49}) {
        const Sampled s(sphere, n);
        real_t        error = 0;
        for (const auto &p : points) {
            real_t f;
            SFEM_TEST_ASSERT(s.sample<0>(p[0], p[1], p[2], &f, nullptr, nullptr));
            error = std::max(error, std::abs(f - sphere(p[0], p[1], p[2])));
        }
        printf("n = %ld: max error %g", (long)n, error);
        if (previous > 0) {
            const real_t order = std::log2(previous / error);
            printf(", order %.2f", order);
            SFEM_TEST_ASSERT(order > 3.5);
        }
        printf("\n");
        previous = error;
    }
    return SFEM_TEST_SUCCESS;
}

// The gradient and the Hessian are continuous across the faces between cells.
int test_continuous_across_cells() {
    const ptrdiff_t n = 21;
    const Sampled   s(sphere, n);
    const real_t    h   = 1. / (n - 1);
    const real_t    eps = 1e-9;

    std::mt19937                           rng(3);
    std::uniform_real_distribution<real_t> u(0.2, 0.8);
    real_t                                 jump_g = 0, jump_H = 0;
    for (int k = 0; k < 500; ++k) {
        real_t p[3] = {u(rng), u(rng), u(rng)};
        const int axis = k % 3;
        p[axis]        = std::round(p[axis] / h) * h;  // on a face

        real_t a[3] = {p[0], p[1], p[2]}, b[3] = {p[0], p[1], p[2]};
        a[axis] -= eps;
        b[axis] += eps;
        real_t fa, ga[3], Ha[6], fb, gb[3], Hb[6];
        SFEM_TEST_ASSERT(s.sample<2>(a[0], a[1], a[2], &fa, ga, Ha));
        SFEM_TEST_ASSERT(s.sample<2>(b[0], b[1], b[2], &fb, gb, Hb));
        for (int d = 0; d < 3; ++d) jump_g = std::max(jump_g, std::abs(ga[d] - gb[d]));
        for (int d = 0; d < 6; ++d) jump_H = std::max(jump_H, std::abs(Ha[d] - Hb[d]));
    }
    printf("largest jump across a face: gradient %g, Hessian %g\n", jump_g, jump_H);
    SFEM_TEST_ASSERT(jump_g < 1e-6);
    SFEM_TEST_ASSERT(jump_H < 1e-4);
    return SFEM_TEST_SUCCESS;
}

// Outside the grid nothing is sampled.
int test_outside_is_reported() {
    const Sampled s(sphere, 9);
    real_t        f;
    SFEM_TEST_ASSERT(!s.sample<0>(-0.01, 0.5, 0.5, &f, nullptr, nullptr));
    SFEM_TEST_ASSERT(!s.sample<0>(0.5, 1.0, 0.5, &f, nullptr, nullptr));
    SFEM_TEST_ASSERT(s.sample<0>(0.5, 0.999, 0.5, &f, nullptr, nullptr));
    return SFEM_TEST_SUCCESS;
}

// The nodal sampling of node_interpolate.hpp at the points disp (xyz zero), all of them.
static void sample_points(const Sampled          &s,
                          const ptrdiff_t         np,
                          const real_t *const     p,  // AoS, 3 per point
                          real_t *const           gap,
                          real_t **const          normals,
                          real_t **const          curvature) {
    auto zero = smesh::create_host_buffer<geom_t>(3, np);
    if (gap || normals) {
        interpolate_displaced_gap(np,
                                  nullptr,
                                  zero->data(),
                                  p,
                                  s.sdf->nlocal(),
                                  s.spline->stride->data(),
                                  s.sdf->origin(),
                                  s.sdf->delta(),
                                  s.spline->origin_coefficient(),
                                  gap,
                                  normals);
    }
    if (curvature) {
        interpolate_displaced_curvature(np,
                                        nullptr,
                                        zero->data(),
                                        p,
                                        s.sdf->nlocal(),
                                        s.spline->stride->data(),
                                        s.sdf->origin(),
                                        s.sdf->delta(),
                                        s.spline->origin_coefficient(),
                                        curvature);
    }
}

// Moves the points onto the spline's zero level set: p += gap n, where s(p + t n) = s + t |grad s|.
static void to_surface(const Sampled &s, const ptrdiff_t np, real_t *const p) {
    auto gap     = smesh::create_host_buffer<real_t>(np);
    auto normals = smesh::create_host_buffer<real_t>(3, np);
    for (int it = 0; it < 6; ++it) {
        sample_points(s, np, p, gap->data(), normals->data(), nullptr);
        for (ptrdiff_t i = 0; i < np; ++i) {
            for (int d = 0; d < 3; ++d) {
                p[i * 3 + d] += gap->data()[i] * normals->data()[d][i];
            }
        }
    }
}

// The nodal curvature (H - n m^T - m n^T) / |grad s|, m = H n, from the prefiltered coefficients:
// on a sphere it converges to -(I - n n^T) / R as h^2; on the surface of a field that is not a
// distance (|grad s| varies, m != 0) it is minus the Hessian of the gap -s / |grad s| exactly;
// quadratics are reproduced (Hessian 2A) away from the grid's ends and planes have none anywhere.
// Points are sampled as X[idx] + disp with any idx and disp, and outside the grid the gap is
// unreachable, the normal and curvature zero.
int test_nodal_curvature() {
    // Sphere: convergence. Small, so that it stays clear of the prefilter's end layer.
    {
        const real_t R     = 0.15;
        const Field  ball  = [R](const real_t x, const real_t y, const real_t z) {
            return R - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + (z - 0.5) * (z - 0.5));
        };
        const ptrdiff_t np = 200;
        auto            p  = smesh::create_host_buffer<real_t>(3 * np);
        auto            nx = smesh::create_host_buffer<real_t>(3 * np);
        std::mt19937                           rng(5);
        std::normal_distribution<real_t>       u(0, 1);
        for (ptrdiff_t i = 0; i < np; ++i) {
            real_t d[3] = {u(rng), u(rng), u(rng)};
            const real_t len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            for (int k = 0; k < 3; ++k) {
                nx->data()[i * 3 + k] = -d[k] / len;  // grad of R - |x - c|
                p->data()[i * 3 + k]  = 0.5 + R * d[k] / len;
            }
        }

        real_t previous = 0;
        for (const ptrdiff_t n : {17, 33, 65}) {
            const Sampled s(ball, n);
            auto          C = smesh::create_host_buffer<real_t>(6, np);
            sample_points(s, np, p->data(), nullptr, nullptr, C->data());
            real_t error = 0;
            for (ptrdiff_t i = 0; i < np; ++i) {
                const real_t *const nn = &nx->data()[i * 3];
                int                 c  = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        const real_t exact = -((a == b) - nn[a] * nn[b]) / R;
                        error              = std::max(error, std::abs(C->data()[c][i] - exact));
                        c++;
                    }
                }
            }
            printf("sphere curvature, n = %ld: max error %g (1/R = %g)", (long)n, error, 1 / R);
            if (previous > 0) {
                const real_t order = std::log2(previous / error);
                printf(", order %.2f", order);
                SFEM_TEST_ASSERT(order > 1.7);
            }
            printf("\n");
            previous = error;
        }
    }

    // Ellipsoid level set (not a distance): the curvature is minus the Hessian of the sampled gap
    // -s / |grad s| on the surface, by central differences; the m terms matter here.
    {
        const Field ellipsoid = [](const real_t x, const real_t y, const real_t z) {
            return 0.04 - ((x - 0.5) * (x - 0.5) + 2 * (y - 0.5) * (y - 0.5) + 3 * (z - 0.5) * (z - 0.5));
        };
        const Sampled   s(ellipsoid, 33);
        const ptrdiff_t np = 50;
        auto            p  = smesh::create_host_buffer<real_t>(3 * np);
        std::mt19937                     rng(9);
        std::normal_distribution<real_t> u(0, 1);
        for (ptrdiff_t i = 0; i < np; ++i) {
            for (int k = 0; k < 3; ++k) {
                p->data()[i * 3 + k] = 0.5 + 0.1 * u(rng);
            }
        }
        to_surface(s, np, p->data());

        auto C = smesh::create_host_buffer<real_t>(6, np);
        auto g = smesh::create_host_buffer<real_t>(np);
        sample_points(s, np, p->data(), g->data(), nullptr, C->data());
        for (ptrdiff_t i = 0; i < np; ++i) {
            SFEM_TEST_ASSERT(std::abs(g->data()[i]) < 1e-12);
        }

        const real_t eps = 1e-4;
        auto         q   = smesh::create_host_buffer<real_t>(3 * np);
        auto         gq  = smesh::create_host_buffer<real_t>(np);
        auto         hess  = smesh::create_host_buffer<real_t>(np);
        real_t       error = 0, scale = 0, correction = 0;
        int          c = 0;
        for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
                // g at (+a, +b), (+a, -b), (-a, +b), (-a, -b)
                for (ptrdiff_t i = 0; i < np; ++i) hess->data()[i] = 0;
                for (int corner = 0; corner < 4; ++corner) {
                    const real_t sa = corner < 2 ? 1 : -1, sb = corner % 2 == 0 ? 1 : -1;
                    for (ptrdiff_t i = 0; i < np; ++i) {
                        for (int k = 0; k < 3; ++k) q->data()[i * 3 + k] = p->data()[i * 3 + k];
                        q->data()[i * 3 + a] += sa * eps;
                        q->data()[i * 3 + b] += sb * eps;
                    }
                    sample_points(s, np, q->data(), gq->data(), nullptr, nullptr);
                    for (ptrdiff_t i = 0; i < np; ++i) {
                        hess->data()[i] += sa * sb * gq->data()[i] / (4 * eps * eps);
                    }
                }
                for (ptrdiff_t i = 0; i < np; ++i) {
                    error = std::max(error, std::abs(C->data()[c][i] + hess->data()[i]));
                    scale = std::max(scale, std::abs(C->data()[c][i]));
                }
                c++;
            }
        }
        // How far the plain H / |grad s| is from it: what the m terms correct.
        {
            real_t     f, gr[3], H[6];
            const real_t *const pp = p->data();
            for (ptrdiff_t i = 0; i < np; ++i) {
                SFEM_TEST_ASSERT(s.sample<2>(pp[i * 3], pp[i * 3 + 1], pp[i * 3 + 2], &f, gr, H));
                const real_t qn = std::sqrt(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]);
                for (int k = 0; k < 6; ++k) correction = std::max(correction, std::abs(H[k] / qn - C->data()[k][i]));
            }
        }
        printf("ellipsoid: curvature vs -hess(gap) by differences %g (of %g); H / |grad s| is off by %g\n",
               error,
               scale,
               correction);
        SFEM_TEST_ASSERT(error < 1e-5 * scale);
        SFEM_TEST_ASSERT(correction > 100 * error);

        // A quadratic is reproduced away from the grid's ends: Hessian 2A.
        const real_t exact[6] = {-2, 0, 0, -4, 0, -6};
        std::mt19937                           r2(13);
        std::uniform_real_distribution<real_t> w(0.3, 0.7);
        for (int k = 0; k < 200; ++k) {
            real_t f, gr[3], H[6];
            SFEM_TEST_ASSERT(s.sample<2>(w(r2), w(r2), w(r2), &f, gr, H));
            for (int e = 0; e < 6; ++e) SFEM_TEST_APPROXEQ(H[e], exact[e], 1e-3);
        }
    }

    // A plane has no curvature, up to the grid's ends.
    {
        const Field     plane = [](const real_t x, const real_t y, const real_t z) { return 0.3 * x - 0.7 * y + 0.2 * z + 0.1; };
        const Sampled   s(plane, 17);
        const ptrdiff_t np = 500;
        auto            p  = smesh::create_host_buffer<real_t>(3 * np);
        std::mt19937                           rng(17);
        std::uniform_real_distribution<real_t> u(0, 0.999);
        for (ptrdiff_t i = 0; i < 3 * np; ++i) p->data()[i] = u(rng);
        auto C = smesh::create_host_buffer<real_t>(6, np);
        sample_points(s, np, p->data(), nullptr, nullptr, C->data());
        for (ptrdiff_t i = 0; i < np; ++i) {
            for (int c = 0; c < 6; ++c) SFEM_TEST_APPROXEQ(C->data()[c][i], 0, 1e-3);
        }
    }

    // X[idx] + disp, against the same points given compactly; and outside the grid.
    {
        const Sampled   s(sphere, 17);
        const ptrdiff_t nn = 64, np = 20;
        auto            X    = smesh::create_host_buffer<geom_t>(3, nn);
        auto            disp = smesh::create_host_buffer<real_t>(3 * nn);
        auto            idx  = smesh::create_host_buffer<idx_t>(np);
        auto            Xc   = smesh::create_host_buffer<geom_t>(3, np);
        auto            dc   = smesh::create_host_buffer<real_t>(3 * np);
        std::mt19937                           rng(23);
        std::uniform_real_distribution<real_t> u(0.1, 0.6);
        for (ptrdiff_t i = 0; i < nn; ++i) {
            for (int d = 0; d < 3; ++d) {
                X->data()[d][i]      = (geom_t)u(rng);
                disp->data()[i * 3 + d] = u(rng) / 3;
            }
        }
        for (ptrdiff_t i = 0; i < np; ++i) {
            idx->data()[i] = (idx_t)((7 * i + 3) % nn);
            for (int d = 0; d < 3; ++d) {
                Xc->data()[d][i]      = X->data()[d][idx->data()[i]];
                dc->data()[i * 3 + d] = disp->data()[idx->data()[i] * 3 + d];
            }
        }
        dc->data()[0] = 5;  // outside

        auto run = [&](const idx_t *const ix, geom_t **const xyz, const real_t *const dd, real_t *const gap, real_t **const nrm, real_t **const C) {
            interpolate_displaced_gap(np, ix, xyz, dd, s.sdf->nlocal(), s.spline->stride->data(), s.sdf->origin(), s.sdf->delta(), s.spline->origin_coefficient(), gap, nrm);
            interpolate_displaced_curvature(np, ix, xyz, dd, s.sdf->nlocal(), s.spline->stride->data(), s.sdf->origin(), s.sdf->delta(), s.spline->origin_coefficient(), C);
        };
        disp->data()[idx->data()[0] * 3] = 5;
        auto ga = smesh::create_host_buffer<real_t>(np), gb = smesh::create_host_buffer<real_t>(np);
        auto na = smesh::create_host_buffer<real_t>(3, np), nb = smesh::create_host_buffer<real_t>(3, np);
        auto Ca = smesh::create_host_buffer<real_t>(6, np), Cb = smesh::create_host_buffer<real_t>(6, np);
        run(idx->data(), X->data(), disp->data(), ga->data(), na->data(), Ca->data());
        run(nullptr, Xc->data(), dc->data(), gb->data(), nb->data(), Cb->data());
        for (ptrdiff_t i = 0; i < np; ++i) {
            SFEM_TEST_EQ(ga->data()[i], gb->data()[i]);
            for (int d = 0; d < 3; ++d) SFEM_TEST_EQ(na->data()[d][i], nb->data()[d][i]);
            for (int c = 0; c < 6; ++c) SFEM_TEST_EQ(Ca->data()[c][i], Cb->data()[c][i]);
        }

        // The sum is formed in real_t: the gap at point 1 is that of sdf_sample there.
        {
            const ptrdiff_t b = idx->data()[1];
            real_t          p[3], f, gr[3];
            for (int d = 0; d < 3; ++d) p[d] = (real_t)X->data()[d][b] + disp->data()[b * 3 + d];
            SFEM_TEST_ASSERT(s.sample<1>(p[0], p[1], p[2], &f, gr, nullptr));
            SFEM_TEST_APPROXEQ(ga->data()[1], -f / std::sqrt(gr[0] * gr[0] + gr[1] * gr[1] + gr[2] * gr[2]), 1e-15);
        }

        SFEM_TEST_EQ(ga->data()[0], real_t(SFEM_SDF_UNREACHABLE));
        for (int d = 0; d < 3; ++d) SFEM_TEST_EQ(na->data()[d][0], real_t(0));
        for (int c = 0; c < 6; ++c) SFEM_TEST_EQ(Ca->data()[c][0], real_t(0));
    }
    return SFEM_TEST_SUCCESS;
}

// Every shell the contact path samples (QUADSHELL4 on hexahedra, TRISHELL3 on tetrahedra,
// EDGESHELL2 in 2D) reads the B-spline through ContactConditions. A tilted plane is reproduced,
// so the normal is exact at every node of the contact side and the gap at the interior ones
// (the lumped projection averages over a boundary node's half support).
static int check_tilted_plane(const std::shared_ptr<sfem::Mesh> &mesh, const int dim, const char *name) {
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
    ptrdiff_t            n_interior = 0;
    for (ptrdiff_t i = 0; i < n_c; ++i) {
        for (int d = 0; d < dim; ++d) {
            SFEM_TEST_APPROXEQ(normals[d * n_c + i], n_ex[d], 1e-5);
        }
        const real_t x        = points[0][nodes[i]];
        const bool   interior = x > 1e-5 && x < 1 - 1e-5 && (dim == 2 || (points[2][nodes[i]] > 1e-5 && points[2][nodes[i]] < 1 - 1e-5));
        if (interior) {
            SFEM_TEST_APPROXEQ(gap[i], delta - slope * x, 1e-5);
            n_interior++;
        }
    }
    SFEM_TEST_ASSERT(n_interior > 0);
    printf("%s: %ld contact nodes, gap exact at %ld interior ones, normals at all\n", name, (long)n_c, (long)n_interior);
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
    SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_hex8_cube(comm, 4, 4, 4), 3, "QUADSHELL4") == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_tet4_cube(comm, 4, 4, 4), 3, "TRISHELL3") == SFEM_TEST_SUCCESS);
    SFEM_TEST_ASSERT(check_tilted_plane(sfem::Mesh::create_quad4_square(comm, 4, 4), 2, "EDGESHELL2") == SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_reproduces_nodes_and_planes);
    SFEM_RUN_TEST(test_converges_as_h4_on_a_sphere);
    SFEM_RUN_TEST(test_continuous_across_cells);
    SFEM_RUN_TEST(test_outside_is_reported);
    SFEM_RUN_TEST(test_nodal_curvature);
    SFEM_RUN_TEST(test_contact_surfaces_sample_planes);
    SFEM_RUN_TEST(test_normals_follow_the_update);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
