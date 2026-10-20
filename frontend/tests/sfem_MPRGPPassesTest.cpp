// The fused vector passes of an MPRGP iteration (project, chopped gradient, curvature, step,
// expansion) against plain loops written from the definitions, for a lower bound, an upper bound
// and both, on the host and, with CUDA, on the device. Dofs sit on a bound, within eps of one, and
// inside; the device runs more dofs than its grid has threads. Then a bound-constrained solve
// checked by its KKT conditions, computed here.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_mprgp.hpp"
#include "sfem_test.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_mprgp_impl.hpp"
#endif

namespace {
    using V = std::vector<real_t>;

    const ptrdiff_t n     = 300007;
    const real_t    eps   = 1e-6;
    const real_t    infty = 1e15;

    real_t pseudo_random(const ptrdiff_t i, const int stream) {
        const double v = std::sin(double(i) * 12.9898 + double(stream) * 78.233) * 43758.5453;
        return real_t(2 * (v - std::floor(v)) - 1);
    }

    struct Problem {
        bool has_lb, has_ub;
        V    lb, ub, x, g, d, Ad;
    };

    // Every eighth dof on the lower bound, the next on the upper, then within eps/2 of each, the
    // rest inside; the dofs on or near a bound do not move (d = 0).
    Problem make(const bool has_lb, const bool has_ub) {
        Problem p{has_lb, has_ub, V(n), V(n), V(n), V(n), V(n), V(n)};
        for (ptrdiff_t i = 0; i < n; ++i) {
            p.lb[i]  = -1 + real_t(0.1) * pseudo_random(i, 1);
            p.ub[i]  = 1 + real_t(0.1) * pseudo_random(i, 2);
            p.g[i]   = pseudo_random(i, 3);
            p.Ad[i]  = pseudo_random(i, 4);
            p.d[i]   = pseudo_random(i, 5);
            p.x[i]   = real_t(0.5) * pseudo_random(i, 6);
            const int k = int(i % 8);
            if (k < 4) p.d[i] = 0;
            if (k == 0 && has_lb) p.x[i] = p.lb[i];
            if (k == 1 && has_ub) p.x[i] = p.ub[i];
            if (k == 2 && has_lb) p.x[i] = p.lb[i] + eps / 2;
            if (k == 3 && has_ub) p.x[i] = p.ub[i] - eps / 2;
        }
        return p;
    }

    // The definitions, one dof at a time.
    real_t free_gradient(const Problem &p, const ptrdiff_t i, const real_t x, const real_t g) {
        if (p.has_lb && x <= p.lb[i]) return 0;
        if (p.has_ub && x >= p.ub[i]) return 0;
        return g;
    }

    real_t chopped_gradient(const Problem &p, const ptrdiff_t i, const real_t x, const real_t g) {
        if (p.has_lb && std::abs(p.lb[i] - x) < eps) return std::min(real_t(0), g);
        if (p.has_ub && std::abs(p.ub[i] - x) < eps) return std::max(real_t(0), g);
        return 0;
    }

    real_t projected(const Problem &p, const ptrdiff_t i, real_t x) {
        if (p.has_ub) x = std::min(x, p.ub[i]);
        if (p.has_lb) x = std::max(x, p.lb[i]);
        return x;
    }

    real_t relative(const real_t a, const real_t b) { return std::abs(a - b) / std::max(std::abs(b), real_t(1e-300)); }

    real_t worst(const V &a, const V &b) {
        real_t w = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            w     = std::max(w, std::abs(a[i] - b[i]));
            scale = std::max(scale, std::abs(b[i]));
        }
        return w / std::max(scale, real_t(1e-300));
    }

    // Where the passes run: the host's vectors as they are, or copies on the device.
    struct Space {
        sfem::MPRGP_Tpl<real_t>                    impl;
        sfem::ExecutionSpace                       es;
        std::vector<sfem::SharedBuffer<real_t>>    keep;
        sfem::SharedBuffer<real_t>                 work;

        real_t *put(V &v) {
            if (es == sfem::EXECUTION_SPACE_HOST) return v.data();
#ifdef SFEM_ENABLE_CUDA
            keep.push_back(smesh::to_device(sfem::Buffer<real_t>::wrap(v.size(), v.data())));
            return keep.back()->data();
#else
            return nullptr;
#endif
        }

        void get(const real_t *const from, V &to) {
            if (es == sfem::EXECUTION_SPACE_HOST) {
                std::copy(from, from + to.size(), to.begin());
                return;
            }
#ifdef SFEM_ENABLE_CUDA
            auto h = smesh::to_host(sfem::Buffer<real_t>::wrap(to.size(), const_cast<real_t *>(from), sfem::MEMORY_SPACE_DEVICE));
            std::copy(h->data(), h->data() + to.size(), to.begin());
#endif
        }

        real_t *work_buffer() {
            if (impl.work_size == 0) return nullptr;
            if (!work) work = sfem::create_buffer<real_t>(impl.work_size, es);
            return work->data();
        }
    };

    int check_passes(Space &s, const bool has_lb, const bool has_ub) {
        Problem p = make(has_lb, has_ub);
        V       lb = p.lb, ub = p.ub, x = p.x, g = p.g, d = p.d, Ad = p.Ad;
        real_t *dlb = has_lb ? s.put(lb) : nullptr, *dub = has_ub ? s.put(ub) : nullptr;
        real_t *dx = s.put(x), *dg = s.put(g), *dd = s.put(d), *dAd = s.put(Ad);
        const char *const name = has_lb && has_ub ? "both bounds" : (has_lb ? "lower bound" : "upper bound");

        // curvature
        real_t out[4];
        s.impl.curvature(n, dlb, dub, dx, dg, dd, dAd, infty, out, s.work_buffer());
        real_t dAd_r = 0, dd_r = 0, gd_r = 0, alpha_r = infty;
        for (ptrdiff_t i = 0; i < n; ++i) {
            dAd_r += p.d[i] * p.Ad[i];
            dd_r += p.d[i] * p.d[i];
            gd_r += p.g[i] * p.d[i];
            if (has_lb && p.d[i] > 0) alpha_r = std::min(alpha_r, (p.x[i] - p.lb[i]) / p.d[i]);
            if (has_ub && p.d[i] < 0) alpha_r = std::min(alpha_r, (p.x[i] - p.ub[i]) / p.d[i]);
        }
        printf("  %s, curvature: %.3g %.3g %.3g, alpha %.3g\n",
               name,
               (double)relative(out[0], dAd_r),
               (double)relative(out[1], dd_r),
               (double)relative(out[2], gd_r),
               (double)relative(out[3], alpha_r));
        SFEM_TEST_ASSERT(relative(out[0], dAd_r) < 1e-11 && relative(out[1], dd_r) < 1e-11 && relative(out[2], gd_r) < 1e-11);
        SFEM_TEST_ASSERT(out[3] == alpha_r);

        // chopped gradient
        V       gc(n, 0);
        real_t *dgc = s.put(gc);
        s.impl.chopped_gradient(n, dlb, dub, dx, dg, dgc, eps);
        s.get(dgc, gc);
        ptrdiff_t chopped = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            SFEM_TEST_ASSERT(gc[i] == chopped_gradient(p, i, p.x[i], p.g[i]));
            chopped += gc[i] != 0;
        }
        SFEM_TEST_ASSERT(chopped > n / 16);

        // step, along d and without one
        const real_t alpha = real_t(0.01);
        for (const bool along : {true, false}) {
            V       x1 = p.x, g1 = p.g, gf(n, 0);
            real_t *dgf = s.put(gf);
            s.impl.step(n, dlb, dub, alpha, along ? dd : nullptr, along ? dAd : nullptr, dx, dg, dgf, eps, out, s.work_buffer());
            V xs(n), gs(n), gfs(n);
            s.get(dx, xs);
            s.get(dg, gs);
            s.get(dgf, gfs);
            real_t Adgf = 0, gf2 = 0, gc2 = 0, gp2 = 0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (along) {
                    x1[i] = x1[i] - alpha * p.d[i];
                    g1[i] = g1[i] - alpha * p.Ad[i];
                }
                const real_t f = free_gradient(p, i, x1[i], g1[i]);
                const real_t c = chopped_gradient(p, i, x1[i], g1[i]);
                gf[i]          = f;
                if (along) Adgf += p.Ad[i] * f;
                gf2 += f * f;
                gc2 += c * c;
                gp2 += (f + c) * (f + c);
            }
            printf("  %s, step%s: x %.3g g %.3g gf %.3g, sums %.3g %.3g %.3g %.3g\n",
                   name,
                   along ? "" : " in place",
                   (double)worst(xs, x1),
                   (double)worst(gs, g1),
                   (double)worst(gfs, gf),
                   (double)(along ? relative(out[0], Adgf) : out[0]),
                   (double)relative(out[1], gf2),
                   (double)relative(out[2], gc2),
                   (double)relative(out[3], gp2));
            SFEM_TEST_ASSERT(worst(xs, x1) < 1e-14 && worst(gs, g1) < 1e-14 && worst(gfs, gf) < 1e-14);
            SFEM_TEST_ASSERT(along ? relative(out[0], Adgf) < 1e-11 : out[0] == 0);
            SFEM_TEST_ASSERT(relative(out[1], gf2) < 1e-11 && relative(out[2], gc2) < 1e-11 && relative(out[3], gp2) < 1e-11);
            SFEM_TEST_ASSERT(gc2 > 0 && gp2 > gf2);

            // Back to the start for what follows.
            x = p.x;
            g = p.g;
            if (s.es != sfem::EXECUTION_SPACE_HOST) {
                dx = s.put(x);
                dg = s.put(g);
            }
        }

        // expansion
        const real_t alpha_bar = real_t(0.3);
        s.impl.expand(n, dlb, dub, alpha, alpha_bar, dd, dAd, dx, dg);
        V xe(n);
        s.get(dx, xe);
        V         xr(n);
        ptrdiff_t clipped = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            const real_t xi = p.x[i] - alpha * p.d[i];
            const real_t gi = p.g[i] - alpha * p.Ad[i];
            const real_t v  = xi - alpha_bar * free_gradient(p, i, xi, gi);
            xr[i]           = projected(p, i, v);
            clipped += xr[i] != v;
        }
        printf("  %s, expansion: x %.3g, %ld clipped\n", name, (double)worst(xe, xr), (long)clipped);
        SFEM_TEST_ASSERT(worst(xe, xr) < 1e-14);

        // project, from outside the box
        V xo(n);
        for (ptrdiff_t i = 0; i < n; ++i) xo[i] = real_t(1.5) * pseudo_random(i, 7);
        V       xp  = xo;
        real_t *dxo = s.put(xo);
        s.impl.project(n, dlb, dub, dxo);
        s.get(dxo, xo);
        for (ptrdiff_t i = 0; i < n; ++i) SFEM_TEST_ASSERT(xo[i] == projected(p, i, xp[i]));
        return SFEM_TEST_SUCCESS;
    }

    int check_space(Space &s) {
        SFEM_TEST_ASSERT(s.impl.good());
        SFEM_TEST_ASSERT(check_passes(s, true, true) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(check_passes(s, true, false) == SFEM_TEST_SUCCESS);
        SFEM_TEST_ASSERT(check_passes(s, false, true) == SFEM_TEST_SUCCESS);
        return SFEM_TEST_SUCCESS;
    }
}  // namespace

int test_host_passes() {
    Space s;
    s.es = sfem::EXECUTION_SPACE_HOST;
    sfem::OpenMP_MPRGP<real_t>::build_mprgp(s.impl);
    return check_space(s);
}

#ifdef SFEM_ENABLE_CUDA
int test_device_passes() {
    Space s;
    s.es = sfem::EXECUTION_SPACE_DEVICE;
    sfem::CUDA_MPRGP<real_t>::build_mprgp(s.impl);
    return check_space(s);
}
#endif

// A chain pressed between two obstacles: min 1/2 x.Ax - b.x with lb <= x <= ub, A the Dirichlet
// Laplacian. At the solution the gradient vanishes inside, points out of the box on a bound.
int test_solution_meets_kkt() {
    constexpr ptrdiff_t m  = 400;
    auto            op = sfem::make_op<real_t>(
            m,
            m,
            [](const real_t *const in, real_t *const out) {
                for (ptrdiff_t i = 0; i < m; ++i) {
                    real_t v = 2 * in[i];
                    if (i > 0) v -= in[i - 1];
                    if (i < m - 1) v -= in[i + 1];
                    out[i] += v;
                }
            },
            sfem::EXECUTION_SPACE_HOST);
    auto lower = sfem::create_host_buffer<real_t>(m), upper = sfem::create_host_buffer<real_t>(m);
    V    b(m);
    for (ptrdiff_t i = 0; i < m; ++i) {
        const real_t s   = real_t(i) / real_t(m - 1);
        upper->data()[i] = real_t(0.02) + real_t(0.5) * (s - real_t(0.5)) * (s - real_t(0.5));
        lower->data()[i] = real_t(-0.03) - real_t(0.2) * s;
        b[i]             = real_t(0.002) * std::sin(real_t(9) * s);
    }
    auto qp = sfem::create_mprgp(op, sfem::EXECUTION_SPACE_HOST);
    qp->verbose    = false;
    qp->check_each = 1 << 30;
    qp->set_upper_bound(upper);
    qp->set_lower_bound(lower);
    qp->set_max_eig(4);
    qp->set_max_it(20 * m);
    qp->set_rtol(1e-12);
    qp->set_atol(1e-16);
    V x(m, 0);
    SFEM_TEST_ASSERT(qp->apply(b.data(), x.data()) == SFEM_SUCCESS);

    real_t    kkt = 0, scale = 0;
    ptrdiff_t on_lb = 0, on_ub = 0;
    for (ptrdiff_t i = 0; i < m; ++i) {
        real_t g = -b[i] + 2 * x[i];
        if (i > 0) g -= x[i - 1];
        if (i < m - 1) g -= x[i + 1];
        const real_t l = lower->data()[i], u = upper->data()[i];
        SFEM_TEST_ASSERT(x[i] >= l && x[i] <= u);
        real_t r = g;
        if (x[i] - l < 1e-12) {
            r = std::min(g, real_t(0));
            on_lb++;
        } else if (u - x[i] < 1e-12) {
            r = std::max(g, real_t(0));
            on_ub++;
        }
        kkt += r * r;
        scale += b[i] * b[i];
    }
    printf("  KKT %.3g of |b| %.3g, %ld on the lower bound, %ld on the upper, %d iterations\n",
           (double)std::sqrt(kkt),
           (double)std::sqrt(scale),
           (long)on_lb,
           (long)on_ub,
           qp->iterations());
    SFEM_TEST_ASSERT(on_lb > 0 && on_ub > 0);
    SFEM_TEST_ASSERT(std::sqrt(kkt) < 1e-10 * std::sqrt(scale));
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_host_passes);
#ifdef SFEM_ENABLE_CUDA
    SFEM_RUN_TEST(test_device_passes);
#endif
    SFEM_RUN_TEST(test_solution_meets_kkt);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
