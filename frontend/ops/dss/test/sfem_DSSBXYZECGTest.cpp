#include "sfem_test.hpp"

#include "sfem_DSSBXYZE.hpp"
#include "sfem_cg.hpp"
#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"
#include "lagrange.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

// Full-precision Gauss-Legendre rule on [0,1].
static void quadrature(const int Q, scalar_t *qx, scalar_t *qw) {
    for (int q = 0; q < Q; ++q) {
        long double z = std::cos(std::acos(-1.L) * (q + .75L) / (Q + .5L)), dp = 0;
        for (int it = 0; it < 32; ++it) {
            long double p0 = 1, p1 = z;
            for (int k = 2; k <= Q; ++k) {
                const long double p = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
                p0 = p1; p1 = p;
            }
            dp = Q * (z * p1 - p0) / (z * z - 1);
            const long double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 4 * std::numeric_limits<long double>::epsilon()) break;
        }
        qx[q] = (scalar_t)((1 + z) / 2);
        qw[q] = (scalar_t)(1 / ((1 - z * z) * dp * dp));
    }
}

template <int N>
static int check_laplacian_cg(const int bs, const int be) {
    constexpr int p = N - 1, Q = N, N3 = N * N * N, Q3 = Q * Q * Q;
    constexpr ptrdiff_t ne = 8, M = 2 * p + 1, ng = M * M * M;
    sfem_dss_bxyze_layout_t layout, global_layout;
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(p, ne, bs, be, &layout), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(p, ne, 1, 1, &global_layout), SFEM_SUCCESS);
    const ptrdiff_t nc = layout.nblocks * layout.block_stride;

    std::vector<idx_t> connectivity(N3 * ne);
    std::vector<const idx_t *> elements(N3);
    for (int v = 0; v < N3; ++v) elements[v] = connectivity.data() + v * ne;
    std::vector<geom_t> px(ng), py(ng), pz(ng);
    std::vector<int> boundary(ng), multiplicity(ng);
    std::vector<ptrdiff_t> cell_boundary;
    std::vector<real_t> truth(ng), rhs(ng), global_x(ng), residual(ng);
    const auto axis = [](const int i, const double scale, const double grade) {
        const int e = std::min(i / p, 1);
        const double a = .5 * e, b = a + .5, t = (double)(i - e * p) / p;
        return scale * ((1 - t) * (a + grade * a * a) + t * (b + grade * b * b));
    };
    for (int iz = 0; iz < M; ++iz) {
        for (int iy = 0; iy < M; ++iy) {
            for (int ix = 0; ix < M; ++ix) {
                const ptrdiff_t g = ix + M * (iy + M * iz);
                const double a = axis(ix, 1, .125), b = axis(iy, .75, .0625), c = axis(iz, .5, .03125);
                px[g] = (geom_t)(a + .25 * b - .125 * c);
                py[g] = (geom_t)(b + .125 * c); pz[g] = (geom_t)c;
                boundary[g] = ix == 0 || iy == 0 || iz == 0 || ix == M - 1 || iy == M - 1 || iz == M - 1;
                truth[g] = boundary[g] ? 0 : std::sin(.19 * (g + 1)) + .2 * std::cos(.37 * (g + 1));
            }
        }
    }
    for (ptrdiff_t e = 0; e < ne; ++e) {
        const int ex = e % 2, ey = (e / 2) % 2, ez = e / 4;
        for (int iz = 0; iz < N; ++iz) {
            for (int iy = 0; iy < N; ++iy) {
                for (int ix = 0; ix < N; ++ix) {
                    const int v = ix + N * (iy + N * iz);
                    const idx_t g = ex * p + ix + M * (ey * p + iy + M * (ez * p + iz));
                    connectivity[v * ne + e] = g;
                    ++multiplicity[g];
                    if (boundary[g]) cell_boundary.push_back((e / bs) * layout.block_stride + v * bs + e % bs);
                }
            }
        }
    }
    auto ids = sfem::create_host_buffer<ptrdiff_t>(cell_boundary.size());
    std::copy(cell_boundary.begin(), cell_boundary.end(), ids->data());
    auto S = sfem::create_host_buffer<scalar_t>(N * Q), D = sfem::create_host_buffer<scalar_t>(N * Q);
    auto qw = sfem::create_host_buffer<scalar_t>(Q);
    scalar_t qx[Q];
    quadrature(Q, qx, qw->data());
    SFEM_TEST_EQ(lagrange_eval<scalar_t>(p, Q, qx, S->data()), SFEM_SUCCESS);
    SFEM_TEST_EQ(lagrange_diff_eval<scalar_t>(p, Q, qx, D->data()), SFEM_SUCCESS);
    const geom_t *points[3] = {px.data(), py.data(), pz.data()};
    auto geometry = sfem::create_host_buffer<scalar_t>(6 * ne);
    std::vector<scalar_t> global_geometry(6 * ne);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, elements.data(), points, geometry->data()), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&global_layout, elements.data(), points, global_geometry.data()), SFEM_SUCCESS);
    sfem_dss_bxyze_laplacian_affine_volumetric_t prepared;
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, Q, &prepared), SFEM_SUCCESS);
    int threads = 1;
#ifdef SFEM_ENABLE_OPENMP
    threads = omp_get_max_threads();
#endif
    auto work = sfem::create_host_buffer<scalar_t>(threads * prepared.workspace_stride);
    auto cell_op = sfem::dss_bxyze_laplacian_affine_operator(prepared, 2, 2, 2, S, D, qw, geometry, work, ids);
    auto weights = sfem::dss_bxyze_structured_weights(layout, 2, 2, 2);
    SFEM_TEST_ASSERT(cell_op && weights);

    // Global element-by-element gather/8-contraction apply/scatter reference.
    constexpr ptrdiff_t scratch_size = sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, 1>();
    std::vector<real_t> u(N3), local_y(N3);
    std::vector<scalar_t> g(3 * Q3), scratch(scratch_size);
    auto global_op = sfem::make_op<real_t>(ng, ng, [&](const real_t *x, real_t *y) {
        std::fill(y, y + ng, (real_t)0);
        for (ptrdiff_t e = 0; e < ne; ++e) {
            for (int v = 0; v < N3; ++v) u[v] = x[elements[v][e]];
            sfem_dss_bxyze_lagrange_gradient<N, Q, 1>(
                    1, S->data(), D->data(), u.data(), g.data(), g.data() + Q3, g.data() + 2 * Q3, scratch.data());
            sfem_dss_bxyze_affine_metric_apply<Q, 1>(
                    1, qw->data(), global_geometry.data() + 6 * e, g.data(), g.data() + Q3, g.data() + 2 * Q3);
            sfem_dss_bxyze_lagrange_gradient_transpose<N, Q, 1>(
                    1, S->data(), D->data(), g.data(), g.data() + Q3, g.data() + 2 * Q3, local_y.data(), scratch.data());
            for (int v = 0; v < N3; ++v) y[elements[v][e]] += local_y[v];
        }
        for (ptrdiff_t i = 0; i < ng; ++i) if (boundary[i]) y[i] = x[i];
    }, sfem::EXECUTION_SPACE_HOST);
    SFEM_TEST_EQ(global_op->apply(truth.data(), rhs.data()), SFEM_SUCCESS);
    auto cell_rhs = sfem::create_host_buffer<real_t>(nc);
    auto cell_x = sfem::create_host_buffer<real_t>(nc);
    auto cell_y = sfem::create_host_buffer<real_t>(nc);
    SFEM_TEST_EQ(sfem_dss_global_to_bxyze(p, ne, bs, elements.data(), rhs.data(), cell_rhs->data()), SFEM_SUCCESS);
    auto cg = sfem::h_cellwise_cg<real_t>(cell_op, weights);
    auto global_cg = sfem::h_cg<real_t>();
    SFEM_TEST_ASSERT(cg != nullptr);
    global_cg->set_op(global_op);
    global_cg->set_apply_overwrites_output(true);
    const double eps = std::max((double)std::numeric_limits<real_t>::epsilon(),
                                (double)std::numeric_limits<scalar_t>::epsilon());
    const real_t rtol = eps < 1e-10 ? 1e-11 : 5e-5;
    for (auto solver : {cg, global_cg}) {
        solver->set_rtol(rtol); solver->set_atol(0); solver->set_max_it(10000); solver->verbose = false;
    }
    std::vector<double> weight_sum(ng);
    for (ptrdiff_t e = 0; e < ne; ++e) {
        for (int v = 0; v < N3; ++v) {
            const ptrdiff_t i = (e / bs) * layout.block_stride + v * bs + e % bs;
            const idx_t gid = elements[v][e];
            SFEM_TEST_APPROXEQ(weights->data()[i], 1.0 / multiplicity[gid], 8 * eps);
            weight_sum[gid] += weights->data()[i];
        }
    }
    for (const auto sum : weight_sum) SFEM_TEST_APPROXEQ(sum, 1., 8 * eps);
    const real_t bnorm = global_cg->blas->norm2(ng, rhs.data());
    SFEM_TEST_APPROXEQ(std::sqrt(cg->inner_product(nc, cell_rhs->data(), cell_rhs->data())),
                       bnorm, 64 * eps * bnorm);

    for (const bool fusion : {true, false}) {
        cg->set_fusion(fusion);
        global_cg->set_fusion(fusion);
        for (ptrdiff_t i = 0; i < ng; ++i) global_x[i] = .125 * truth[i];
        const int global_status = global_cg->apply(rhs.data(), global_x.data());
        SFEM_TEST_EQ(global_op->apply(global_x.data(), residual.data()), SFEM_SUCCESS);
        double global_rtr = 0;
        for (ptrdiff_t i = 0; i < ng; ++i) global_rtr += std::pow(rhs[i] - residual[i], 2);
        if (global_status != SFEM_SUCCESS) {
            std::printf("Global reference p=%d: status=%d iterations=%d true relres=%.3e\n",
                        p, global_status, global_cg->iterations(), std::sqrt(global_rtr) / bnorm);
        }
        SFEM_TEST_EQ(global_status, SFEM_SUCCESS);
        SFEM_TEST_ASSERT(std::sqrt(global_rtr) / bnorm < 100 * rtol);
        // Nonzero, consistent initial guess, also exercising workspace reuse.
        for (ptrdiff_t i = 0; i < ng; ++i) residual[i] = .125 * truth[i];
        SFEM_TEST_EQ(sfem_dss_global_to_bxyze(p, ne, bs, elements.data(), residual.data(), cell_x->data()), SFEM_SUCCESS);
        SFEM_TEST_EQ(cg->apply(cell_rhs->data(), cell_x->data()), SFEM_SUCCESS);
        SFEM_TEST_EQ(cell_op->apply(cell_x->data(), cell_y->data()), SFEM_SUCCESS);
        double norm = 0, error = 0, solution_difference = 0;
        for (ptrdiff_t e = 0; e < ne; ++e) {
            for (int v = 0; v < N3; ++v) {
                const ptrdiff_t i = (e / bs) * layout.block_stride + v * bs + e % bs;
                const idx_t gid = elements[v][e];
                const double r = cell_rhs->data()[i] - cell_y->data()[i];
                norm += weights->data()[i] * r * r;
                error = std::max(error, std::abs((double)cell_x->data()[i] - truth[gid]));
                solution_difference = std::max(solution_difference, std::abs((double)cell_x->data()[i] - global_x[gid]));
                SFEM_TEST_ASSERT(std::isfinite(cell_x->data()[i]));
                if (boundary[gid]) SFEM_TEST_EQ(cell_x->data()[i], (real_t)0);
            }
        }
        // Verify that consistent copies remain identical after the vector updates.
        std::vector<real_t> first(ng);
        std::vector<bool> seen(ng);
        for (ptrdiff_t e = 0; e < ne; ++e) {
            for (int v = 0; v < N3; ++v) {
                const ptrdiff_t i = (e / bs) * layout.block_stride + v * bs + e % bs;
                const idx_t gid = elements[v][e];
                if (seen[gid]) {
                    SFEM_TEST_EQ(cell_x->data()[i], first[gid]);
                } else {
                    first[gid] = cell_x->data()[i];
                    seen[gid] = true;
                }
            }
        }
        SFEM_TEST_ASSERT(std::sqrt(norm) / bnorm < 100 * rtol);
        SFEM_TEST_ASSERT(error < 1e6 * N * N * eps);
        SFEM_TEST_ASSERT(solution_difference < 1e6 * N * N * eps);
        std::printf("p=%d B=%d/%d threads=%d fusion=%d: CG=%d, global=%d, relres=%.3e, maxdiff=%.3e\n",
                    p, bs, be, threads, (int)fusion, cg->iterations(), global_cg->iterations(),
                    std::sqrt(norm) / bnorm, solution_difference);
    }
    // Incorrect shapes or short buffers must fail before apply.
    SFEM_TEST_ASSERT(!sfem::dss_bxyze_structured_weights(layout, 2, 2, 3));
    SFEM_TEST_ASSERT(!sfem::dss_bxyze_laplacian_affine_operator(prepared, 2, 2, 3, S, D, qw, geometry, work, ids));
    SFEM_TEST_ASSERT(!sfem::h_cellwise_cg<real_t>(global_op, weights));
#ifdef SFEM_ENABLE_OPENMP
    // Changing to a team larger than the supplied workspace fails cleanly.
    omp_set_num_threads(threads + 1);
    std::fill(cell_x->data(), cell_x->data() + nc, (real_t)0);
    SFEM_TEST_EQ(cg->apply(cell_rhs->data(), cell_x->data()), SFEM_FAILURE);
    omp_set_num_threads(threads);
#endif
    return SFEM_TEST_SUCCESS;
}

static int test_cellwise_cg() {
    for (const int threads : {1, 8}) {
#ifdef SFEM_ENABLE_OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#else
        if (threads != 1) continue;
#endif
        SFEM_TEST_EQ(check_laplacian_cg<3>(1, 1), SFEM_TEST_SUCCESS);
        SFEM_TEST_EQ(check_laplacian_cg<3>(8, 4), SFEM_TEST_SUCCESS);
        SFEM_TEST_EQ(check_laplacian_cg<5>(4, 2), SFEM_TEST_SUCCESS);
        SFEM_TEST_EQ(check_laplacian_cg<9>(8, 4), SFEM_TEST_SUCCESS);
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_cellwise_cg);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
