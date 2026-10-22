#include "sfem_test.hpp"

#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"
#include "sfem_dss_bxyze_geometry.hpp"

#include "hex8_laplacian_inline_cpu.hpp"
#include "lagrange.hpp"
#include "line_quadrature.hpp"
#include "sfem_macros.hpp"
#include "sshex8.hpp"
#include "lagrange_hex_laplacian_inline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

// Independent reference basis, evaluated in long double on equispaced nodes.
template <int N, int Q>
static void make_basis(scalar_t *const S, scalar_t *const D, long double *const qx) {
    for (int q = 0; q < Q; ++q) {
        qx[q] = ((long double)q + 0.25L) / ((long double)Q + 0.5L);
        for (int i = 0; i < N; ++i) {
            const long double xi = (long double)i / (N - 1);
            long double s = 1;
            long double d = 0;
            for (int j = 0; j < N; ++j) {
                if (j == i) continue;
                const long double xj = (long double)j / (N - 1);
                s *= (qx[q] - xj) / (xi - xj);

                long double term = 1 / (xi - xj);
                for (int k = 0; k < N; ++k) {
                    if (k == i || k == j) continue;
                    const long double xk = (long double)k / (N - 1);
                    term *= (qx[q] - xk) / (xi - xk);
                }
                d += term;
            }
            S[q * N + i] = (scalar_t)s;
            D[q * N + i] = (scalar_t)d;
        }
    }
}

template <int N, int Q, int B_EXEC>
static int check_gradient(const int bsave, const int lane_begin, const bool polynomial) {
    SFEM_TEST_ASSERT(lane_begin >= 0 && lane_begin + B_EXEC <= bsave);
    constexpr int guard = 7;
    constexpr ptrdiff_t nout = (ptrdiff_t)Q * Q * Q * B_EXEC;
    constexpr ptrdiff_t nwork = sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, B_EXEC>();
    constexpr scalar_t sentinel = (scalar_t)-777;
    scalar_t S[Q * N], D[Q * N];
    long double qx[Q];
    make_basis<N, Q>(S, D, qx);

    std::vector<real_t> input((size_t)N * N * N * bsave, std::numeric_limits<real_t>::quiet_NaN());
    std::vector<scalar_t> output((size_t)3 * nout + 4 * guard, sentinel);
    std::vector<scalar_t> scratch((size_t)nwork + 2 * guard, sentinel);
    scalar_t *const gx = output.data() + guard;
    scalar_t *const gy = gx + nout + guard;
    scalar_t *const gz = gy + nout + guard;
    scalar_t *const workspace = scratch.data() + guard;
    std::fill(workspace, workspace + nwork, std::numeric_limits<scalar_t>::quiet_NaN());

    for (int iz = 0; iz < N; ++iz) {
        for (int iy = 0; iy < N; ++iy) {
            for (int ix = 0; ix < N; ++ix) {
                const int v = (iz * N + iy) * N + ix;
                const long double x = (long double)ix / (N - 1);
                const long double y = (long double)iy / (N - 1);
                const long double z = (long double)iz / (N - 1);
                for (int b = 0; b < B_EXEC; ++b) {
                    const int lane = lane_begin + b;
                    long double value;
                    if (polynomial) {
                        const long double a = 0.5L + 0.125L * lane;
                        const long double c = -0.75L + 0.0625L * lane;
                        const long double d = 0.25L - 0.03125L * lane;
                        const long double h = 0.125L + 0.015625L * lane;
                        value = 0.375L + a * x + c * y + d * z + h * x * y * z +
                                0.0625L * std::pow(x, N - 1) * std::pow(y, N - 1) * std::pow(z, N - 1);
                    } else {
                        value = std::sin(0.13L * (v + 1) + 0.31L * (lane + 1)) + 0.01L * (ix * iy + iz);
                    }
                    input[(ptrdiff_t)v * bsave + lane] = (real_t)value;
                }
            }
        }
    }

    sfem_dss_bxyze_lagrange_gradient<N, Q, B_EXEC>(
            bsave, S, D, input.data() + lane_begin, gx, gy, gz, workspace);

    const double tol = 512 * N * N * std::max((double)std::numeric_limits<scalar_t>::epsilon(),
                                             (double)std::numeric_limits<real_t>::epsilon());
    std::vector<scalar_t> element_u((size_t)N * N * N);
    std::vector<scalar_t> reference((size_t)3 * Q * Q * Q);
    const scalar_t *const gradients[3] = {gx, gy, gz};
    for (int b = 0; b < B_EXEC; ++b) {
        for (int v = 0; v < N * N * N; ++v) {
            element_u[v] = (scalar_t)input[(ptrdiff_t)v * bsave + lane_begin + b];
        }
        lagrange_hex_triad_interpolate<N, Q, scalar_t>(D, S, S, element_u.data(), reference.data());
        lagrange_hex_triad_interpolate<N, Q, scalar_t>(S, D, S, element_u.data(), reference.data() + Q * Q * Q);
        lagrange_hex_triad_interpolate<N, Q, scalar_t>(S, S, D, element_u.data(), reference.data() + 2 * Q * Q * Q);

        for (int qz = 0; qz < Q; ++qz) {
            for (int qy = 0; qy < Q; ++qy) {
                for (int q = 0; q < Q; ++q) {
                    const int v = (qz * Q + qy) * Q + q;
                    const ptrdiff_t off = (ptrdiff_t)v * B_EXEC + b;
                    for (int c = 0; c < 3; ++c) {
                        const double expected = reference[c * Q * Q * Q + v];
                        SFEM_TEST_ASSERT(std::isfinite(gradients[c][off]));
                        SFEM_TEST_APPROXEQ((double)gradients[c][off], expected, tol * std::max(1.0, std::abs(expected)));
                    }

                    if (polynomial) {
                        const int lane = lane_begin + b;
                        const long double x = qx[q], y = qx[qy], z = qx[qz];
                        const long double a = 0.5L + 0.125L * lane;
                        const long double c = -0.75L + 0.0625L * lane;
                        const long double d = 0.25L - 0.03125L * lane;
                        const long double h = 0.125L + 0.015625L * lane;
                        const long double high = 0.0625L * (N - 1);
                        const long double expected[3] = {
                                a + h * y * z + high * std::pow(x, N - 2) * std::pow(y, N - 1) * std::pow(z, N - 1),
                                c + h * x * z + high * std::pow(x, N - 1) * std::pow(y, N - 2) * std::pow(z, N - 1),
                                d + h * x * y + high * std::pow(x, N - 1) * std::pow(y, N - 1) * std::pow(z, N - 2)};
                        for (int component = 0; component < 3; ++component) {
                            const double value = (double)expected[component];
                            SFEM_TEST_APPROXEQ((double)gradients[component][off], value, tol * std::max(1.0, std::abs(value)));
                        }
                    }
                }
            }
        }
    }

    // Arbitrary quadrature fields: check G^T independently of the metric stage.
    static_assert(nwork == sfem_dss_bxyze_lagrange_gradient_transpose_workspace_n_values<N, Q, B_EXEC>(),
                  "Forward and transpose must share the same scratch size");
    std::vector<scalar_t> flux((size_t)3 * nout);
    for (size_t i = 0; i < flux.size(); ++i) flux[i] = (scalar_t)std::sin(0.17 * (i + 1));
    std::vector<real_t> nodal((size_t)N * N * N * bsave + 2 * guard, (real_t)sentinel);
    real_t *const nodal_out = nodal.data() + guard;
    std::fill(workspace, workspace + nwork, std::numeric_limits<scalar_t>::quiet_NaN());
    sfem_dss_bxyze_lagrange_gradient_transpose<N, Q, B_EXEC>(
            bsave, S, D, flux.data(), flux.data() + nout, flux.data() + 2 * nout,
            nodal_out + lane_begin, workspace);
    scalar_t ones[Q];
    std::fill(ones, ones + Q, (scalar_t)1);
    std::vector<scalar_t> nodal_reference((size_t)N * N * N);
    for (int b = 0; b < B_EXEC; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < Q * Q * Q; ++q) {
                reference[c * Q * Q * Q + q] = flux[c * nout + (ptrdiff_t)q * B_EXEC + b];
            }
        }
        std::fill(nodal_reference.begin(), nodal_reference.end(), (scalar_t)0);
        lagrange_hex_triad_integrate_add<N, Q, scalar_t>(D, S, S, ones, reference.data(), nodal_reference.data());
        lagrange_hex_triad_integrate_add<N, Q, scalar_t>(S, D, S, ones, reference.data() + Q * Q * Q,
                                                        nodal_reference.data());
        lagrange_hex_triad_integrate_add<N, Q, scalar_t>(S, S, D, ones, reference.data() + 2 * Q * Q * Q,
                                                        nodal_reference.data());
        long double lhs = 0, rhs = 0, scale = 0;
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < Q * Q * Q; ++q) {
                const ptrdiff_t off = (ptrdiff_t)q * B_EXEC + b;
                const long double term = (long double)gradients[c][off] * flux[c * nout + off];
                lhs += term;
                scale += std::abs(term);
            }
        }
        for (int v = 0; v < N * N * N; ++v) {
            const ptrdiff_t off = (ptrdiff_t)v * bsave + lane_begin + b;
            SFEM_TEST_ASSERT(std::isfinite(nodal_out[off]));
            SFEM_TEST_APPROXEQ((double)nodal_out[off], (double)nodal_reference[v],
                              tol * std::max(1.0, std::abs((double)nodal_reference[v])));
            const long double term = (long double)input[off] * nodal_out[off];
            rhs += term;
            scale += std::abs(term);
        }
        SFEM_TEST_APPROXEQ((double)lhs, (double)rhs, tol * std::max(1.0, (double)scale));
    }
    for (int v = 0; v < N * N * N; ++v) {
        for (int b = 0; b < bsave; ++b) {
            if (b < lane_begin || b >= lane_begin + B_EXEC) SFEM_TEST_EQ(nodal_out[(ptrdiff_t)v * bsave + b], (real_t)sentinel);
        }
    }
    for (int i = 0; i < guard; ++i) {
        SFEM_TEST_EQ(nodal[i], (real_t)sentinel);
        SFEM_TEST_EQ(nodal[guard + (ptrdiff_t)N * N * N * bsave + i], (real_t)sentinel);
    }

    for (ptrdiff_t i = 0; i < nwork; ++i) {
        SFEM_TEST_ASSERT(std::isfinite(workspace[i]));
    }
    for (int i = 0; i < guard; ++i) {
        for (int region = 0; region < 4; ++region) {
            SFEM_TEST_EQ(output[region * (nout + guard) + i], sentinel);
        }
        SFEM_TEST_EQ(scratch[i], sentinel);
        SFEM_TEST_EQ(scratch[guard + nwork + i], sentinel);
    }
    return SFEM_TEST_SUCCESS;
}

static int test_gradient_polynomial_reproduction() {
    SFEM_TEST_EQ((check_gradient<2, 2, 1>(1, 0, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<3, 3, 4>(8, 4, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<3, 6, 4>(12, 4, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<5, 5, 8>(16, 8, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<5, 12, 4>(12, 4, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<9, 9, 4>(8, 4, true)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<9, 24, 8>(16, 0, true)), SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

static int test_gradient_matches_scalar_full_batches() {
    SFEM_TEST_EQ((check_gradient<3, 6, 4>(8, 4, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<5, 12, 8>(16, 8, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<9, 9, 8>(16, 8, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<5, 3, 4>(8, 4, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<2, 1, 4>(4, 0, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<3, 3, 2>(4, 2, false)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_gradient<9, 9, 16>(32, 16, false)), SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

template <int N, int Q, int BE>
static int check_affine_apply(const int bsave, const int nthreads) {
    constexpr int guard = 7;
    constexpr scalar_t sentinel = (scalar_t)-777;
    constexpr ptrdiff_t ngrad = (ptrdiff_t)Q * Q * Q * BE;
    constexpr ptrdiff_t used = 3 * ngrad + sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, BE>();
    sfem_dss_bxyze_layout_t layout;
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(N - 1, (ptrdiff_t)nthreads * bsave, bsave, BE, &layout), SFEM_SUCCESS);
    sfem_dss_bxyze_laplacian_affine_volumetric_t op;
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, Q, &op), SFEM_SUCCESS);
    const ptrdiff_t stride = op.workspace_stride;
    SFEM_TEST_ASSERT(stride >= used);
    std::vector<scalar_t> pool((size_t)nthreads * stride + 2 * guard, sentinel);
    std::vector<real_t> x((size_t)layout.nblocks * layout.block_stride);
    std::vector<real_t> y(x.size() + 2 * guard, (real_t)sentinel);
    real_t *const nodal_y = y.data() + guard;
    constexpr int N3 = N * N * N;
    std::vector<idx_t> connectivity((size_t)N3 * layout.nelements);
    std::vector<const idx_t *> elements(N3);
    std::vector<geom_t> px(connectivity.size()), py(px.size()), pz(px.size());
    for (int v = 0; v < N3; ++v) elements[v] = connectivity.data() + v * layout.nelements;
    scalar_t S[Q * N], D[Q * N], qw[Q];
    long double qx[Q];
    make_basis<N, Q>(S, D, qx);
    for (int q = 0; q < Q; ++q) qw[q] = (scalar_t)(0.625 + 0.125 * q);
    for (ptrdiff_t e = 0; e < layout.nelements; ++e) {
        // J = [a,1/4,-1/8; 0,b,1/8; 0,0,c], different on every cell.
        const long double a = 1 + 0.125L * e, b = 0.75L + 0.0625L * e, c = 0.5L + 0.03125L * e;
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < N; ++i) {
                    const long double xi = (long double)i / (N - 1);
                    const long double eta = (long double)j / (N - 1);
                    const long double zeta = (long double)k / (N - 1);
                    x[sfem_dss_bxyze_index(N - 1, bsave, e, i, j, k)] =
                            (real_t)((e + 1) * xi + (2 * e + 3) * eta + (3 * e + 5) * zeta + xi * eta * zeta);
                    const int v = (k * N + j) * N + i;
                    const idx_t gid = (idx_t)(e * N3 + v);
                    connectivity[(ptrdiff_t)v * layout.nelements + e] = gid;
                    px[gid] = (geom_t)(4 * e + a * xi + 0.25L * eta - 0.125L * zeta);
                    py[gid] = (geom_t)(b * eta + 0.125L * zeta);
                    pz[gid] = (geom_t)(c * zeta);
                }
            }
        }
    }
    const geom_t *points[3] = {px.data(), py.data(), pz.data()};
    const ptrdiff_t ngeom = sfem_dss_bxyze_affine_geometry_n_values(&layout);
    SFEM_TEST_ASSERT(ngeom > 0);
    std::vector<scalar_t> geometry((size_t)ngeom);
    SFEM_TEST_EQ(sfem_dss_bxyze_affine_geometry(&layout, elements.data(), points, geometry.data()), SFEM_SUCCESS);
#ifdef SFEM_ENABLE_OPENMP
    const int previous_threads = omp_get_max_threads(), previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);
#endif
    const int status = sfem_dss_bxyze_laplacian_affine_volumetric_apply(
            &op, S, D, qw, geometry.data(), x.data(), nodal_y, pool.data() + guard);
    // Reuse the prepared specialization and workspace with a different input.
    for (real_t &value : x) value *= 2;
    const int reused_status = sfem_dss_bxyze_laplacian_affine_volumetric_apply(
            &op, S, D, qw, geometry.data(), x.data(), nodal_y, pool.data() + guard);
#ifdef SFEM_ENABLE_OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
    SFEM_TEST_EQ(status, SFEM_SUCCESS);
    SFEM_TEST_EQ(reused_status, SFEM_SUCCESS);
    const double tol = 512 * N * N * std::max((double)std::numeric_limits<scalar_t>::epsilon(),
                                             (double)std::numeric_limits<real_t>::epsilon());
    // Independent reference: solve J^T v = grad(u), then J f = w det(J) v.
    // This avoids reproducing the six-entry metric multiplication under test.
    const auto expected_flux = [&](const ptrdiff_t e, const int q, const int qy, const int qz, double *const out) {
        const long double a = 1 + 0.125L * e, b = 0.75L + 0.0625L * e, c = 0.5L + 0.03125L * e;
        const long double rx = 2 * (e + 1 + qx[qy] * qx[qz]);
        const long double ry = 2 * (2 * e + 3 + qx[q] * qx[qz]);
        const long double rz = 2 * (3 * e + 5 + qx[q] * qx[qy]);
        const long double vx = rx / a;
        const long double vy = (ry - 0.25L * vx) / b;
        const long double vz = (rz + 0.125L * vx - 0.125L * vy) / c;
        const long double wd = (long double)qw[q] * qw[qy] * qw[qz] * a * b * c;
        const long double fz = wd * vz / c;
        const long double fy = (wd * vy - 0.125L * fz) / b;
        out[0] = (double)((wd * vx - 0.25L * fy + 0.125L * fz) / a);
        out[1] = (double)fy;
        out[2] = (double)fz;
    };
    std::vector<bool> visited((size_t)nthreads, false);
    for (int t = 0; t < nthreads; ++t) {
        const scalar_t *const w = pool.data() + guard + t * stride;
        SFEM_TEST_ASSERT(std::isfinite(w[0]));
        // One storage block per thread, without assuming which thread owns it.
        ptrdiff_t first = -1;
        for (ptrdiff_t eb = 0; eb < layout.nblocks; ++eb) {
            const ptrdiff_t candidate = eb * bsave + bsave - BE;
            double reference[3];
            expected_flux(candidate, 0, 0, 0, reference);
            bool matches = true;
            for (int c = 0; c < 3; ++c) {
                matches &= std::abs((double)w[c * ngrad] - reference[c]) <=
                           tol * std::max(1.0, std::abs(reference[c]));
            }
            if (matches) {
                SFEM_TEST_ASSERT(first < 0);
                first = candidate;
            }
        }
        SFEM_TEST_ASSERT(first >= 0);
        SFEM_TEST_ASSERT(!visited[first / bsave]);
        visited[first / bsave] = true;
        for (int qz = 0; qz < Q; ++qz) {
            for (int qy = 0; qy < Q; ++qy) {
                for (int q = 0; q < Q; ++q) {
                    for (int b = 0; b < BE; ++b) {
                        const ptrdiff_t off = (((ptrdiff_t)qz * Q + qy) * Q + q) * BE + b;
                        const ptrdiff_t e = first + b;
                        double expected[3];
                        expected_flux(e, q, qy, qz, expected);
                        for (int c = 0; c < 3; ++c) {
                            SFEM_TEST_ASSERT(std::isfinite(w[c * ngrad + off]));
                            SFEM_TEST_APPROXEQ((double)w[c * ngrad + off], expected[c],
                                              tol * std::max(1.0, std::abs(expected[c])));
                        }
                    }
                }
            }
        }
        for (ptrdiff_t i = used; i < stride; ++i) SFEM_TEST_EQ(w[i], sentinel);
    }
    for (int i = 0; i < guard; ++i) {
        SFEM_TEST_EQ(pool[i], sentinel);
        SFEM_TEST_EQ(pool[guard + nthreads * stride + i], sentinel);
        SFEM_TEST_EQ(y[i], (real_t)sentinel);
        SFEM_TEST_EQ(y[guard + x.size() + i], (real_t)sentinel);
    }
    // Compare every element with the original scalar nine-contraction Laplacian.
    std::vector<scalar_t> element_u(N3), reference(N3);
    long double energy = 0;
    double max_relative_error = 0;
    for (ptrdiff_t e = 0; e < layout.nelements; ++e) {
        for (int v = 0; v < N3; ++v) {
            const ptrdiff_t off = (e / bsave) * layout.block_stride + (ptrdiff_t)v * bsave + e % bsave;
            element_u[v] = (scalar_t)x[off];
        }
        scalar_t metric[6];
        for (int c = 0; c < 6; ++c) metric[c] = geometry[((e / bsave) * 6 + c) * bsave + e % bsave];
        std::fill(reference.begin(), reference.end(), (scalar_t)0);
        lagrange_hex_laplacian_apply<N, Q, scalar_t>(S, D, metric, qw, element_u.data(), reference.data());
        double difference = 0, norm = 0;
        for (int v = 0; v < N3; ++v) {
            const ptrdiff_t off = (e / bsave) * layout.block_stride + (ptrdiff_t)v * bsave + e % bsave;
            SFEM_TEST_ASSERT(std::isfinite(nodal_y[off]));
            difference = std::max(difference, std::abs((double)nodal_y[off] - reference[v]));
            norm = std::max(norm, std::abs((double)reference[v]));
            energy += (long double)x[off] * nodal_y[off];
        }
        // Eight and nine contractions sum in different orders; use the element
        // infinity norm so cancellation in individual entries does not set scale.
        const double relative_error = difference / std::max(1.0, norm);
        SFEM_TEST_APPROXEQ(relative_error, 0.0, tol);
        max_relative_error = std::max(max_relative_error, relative_error);
    }
    SFEM_TEST_ASSERT(energy > 0);
    std::printf("affine apply p=%d Q=%d B_save=%d B_exec=%d threads=%d: relative error %.3e\n",
                N - 1, Q, bsave, BE, nthreads, max_relative_error);
    return SFEM_TEST_SUCCESS;
}

static int test_affine_apply() {
    int threads = 1;
#ifdef SFEM_ENABLE_OPENMP
    threads = 2;
#endif
    SFEM_TEST_EQ((check_affine_apply<3, 6, 1>(4, 1)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<3, 3, 2>(4, threads)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<5, 12, 4>(8, threads)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<5, 5, 8>(8, threads)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<9, 9, 8>(16, threads)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<9, 24, 16>(16, threads)), SFEM_TEST_SUCCESS);
    SFEM_TEST_EQ((check_affine_apply<3, 6, 2>(64, threads)), SFEM_TEST_SUCCESS);
    // Non-tabulated storage sizes keep the generic path and the same contract.
    SFEM_TEST_EQ((check_affine_apply<3, 6, 2>(6, threads)), SFEM_TEST_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

static int test_affine_apply_validation() {
    sfem_dss_bxyze_layout_t layout;
    sfem_dss_bxyze_laplacian_affine_volumetric_t op = {};
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_apply(
                         &op, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(2, 8, 8, 4, &layout), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, 3, &op), SFEM_SUCCESS);
    SFEM_TEST_ASSERT(op.kernel != nullptr);
    SFEM_TEST_EQ(op.workspace_stride, sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(&layout, 3));
    // Setup captures metadata, independently of later changes to the source.
    layout.nelements = 0;
    SFEM_TEST_EQ(op.layout.nelements, (ptrdiff_t)8);
    layout = op.layout;
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, 3, nullptr), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, 4, &op), SFEM_FAILURE);
    SFEM_TEST_ASSERT(op.kernel == nullptr);
    SFEM_TEST_EQ(op.workspace_stride, (ptrdiff_t)0);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(nullptr, 3, &op), SFEM_FAILURE);
    auto invalid = layout;
    ++invalid.block_stride;
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&invalid, 3, &op), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(2, 7, 8, 4, &invalid), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&invalid, 3, &op), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(2, 12, 6, 4, &invalid), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&invalid, 3, &op), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(3, 8, 8, 4, &invalid), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&invalid, 4, &op), SFEM_FAILURE);
    SFEM_TEST_EQ(sfem_dss_bxyze_create_layout(2, 0, 8, 4, &layout), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, 3, &op), SFEM_SUCCESS);
    SFEM_TEST_EQ(sfem_dss_bxyze_laplacian_affine_volumetric_apply(
                         &op, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), SFEM_SUCCESS);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_gradient_polynomial_reproduction);
    SFEM_RUN_TEST(test_gradient_matches_scalar_full_batches);
    SFEM_RUN_TEST(test_affine_apply);
    SFEM_RUN_TEST(test_affine_apply_validation);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
