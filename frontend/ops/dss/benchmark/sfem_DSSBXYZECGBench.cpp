#include "sfem_API.hpp"
#include "sfem_DSSBXYZE.hpp"
#include "sfem_cg.hpp"
#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"
#include "lagrange.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace {

// Paired steady-state solves of the same discrete Dirichlet problem. Both use
// ConjugateGradient, identical callback paths, fusion and reduction policies.
// The global operator uses structured node IDs, eight contractions in each
// direction, and atomics only for shared element-boundary nodes;
// the cellwise operator includes DSS. Setup, conversion, initial-guess reset and
// validation are outside timers. Warmup also allocates/touches CG work vectors.
// Every sample starts from the same x0; AB/BA ordering is balanced. All samples
// are retained, with no selection of a winning B or best repetition.
// With deterministic BLAS enabled, its existing temporary chunk allocations
// remain inside both solvers. The default uses OpenMP SIMD reductions instead.

void require(const bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

ptrdiff_t product(const ptrdiff_t a, const ptrdiff_t b) {
    require(a >= 0 && b >= 0 && (!b || a <= std::numeric_limits<ptrdiff_t>::max() / b),
            "problem dimensions overflow ptrdiff_t");
    return a * b;
}

struct Options {
    std::vector<int> levels{2, 4, 8}, sizes{4}, blocks{8}, exec_blocks{4}, fusions{1};
#ifdef SFEM_ENABLE_OPENMP
    std::vector<int> threads{1, 8};
#else
    std::vector<int> threads{1};
#endif
    int repeat{4}, warmup{1}, max_it{20000}, fixed_iterations{0}, deterministic{0}, seed{42};
    int warmup_iterations{10};
    double rtol{1e-8}, initial_scale{0};
    std::string method{"paired"};
    std::string rhs{"smooth"};
    std::string output{"frontend/ops/dss/benchmark/dss_bxyze_cg_bench.csv"};
};

int integer(const std::string &s, const int minimum = 0) {
    char *end = nullptr;
    errno = 0;
    const long n = std::strtol(s.c_str(), &end, 10);
    require(!errno && end != s.c_str() && !*end && n >= minimum && n <= std::numeric_limits<int>::max(),
            "invalid integer option");
    return (int)n;
}

std::vector<int> list(const std::string &s) {
    std::vector<int> result;
    std::stringstream stream(s);
    std::string part;
    while (std::getline(stream, part, ',')) {
        const int n = integer(part);
        if (std::find(result.begin(), result.end(), n) == result.end()) result.push_back(n);
    }
    require(!result.empty(), "empty option list");
    return result;
}

void usage(const char *name) {
    std::fprintf(stderr,
        "usage: %s [--levels 2,4,8] [--sizes 4] [--blocks 8] [--exec-blocks 4]\n"
        "  [--threads 1,8] [--fusions 1] [--repeat 4] [--warmup 1]\n"
        "  [--rtol 1e-8] [--max-it 20000] [--deterministic 0] [--rhs smooth|random]\n"
        "  [--seed 42] [--output file.csv]\n"
        "  [--method paired|global|cellwise] [--warmup-iterations 10] [--initial-scale 0]\n"
        "  [--fixed-iterations K] (isolated mode: K CG steps and true residual)\n"
        "Sizes are macro-elements per axis. Q=p+1 Gauss-Legendre; graded affine shear.\n"
        "Homogeneous Dirichlet; discrete manufactured RHS; x0=initial_scale*truth; atol=0.\n"
        "Even repeat >=2 and warmup >=1 required; all blocks/batches must be full.\n"
        "No preconditioner. Global8 includes zero/gather/scatter; cellwise includes DSS.\n"
        "Writes raw paired samples and file_summary.csv (medians and iteration ranges).\n"
        "All B configurations are reported independently; no tuning/selection on timed samples.\n"
        "Time/iteration includes CG initialization; setup/reset/validation are excluded.\n"
        "Global/cellwise mode: one isolated solve, bounded warmup; repeat is ignored.\n"
        "Fixed-iteration runs report attempt_seconds; tts_s is NaN because convergence is not required.\n"
        "Single MPI rank. Deterministic=1 affects dot products, not fused residual reductions.\n", name);
}

Options options(const int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        require(i + 1 < argc, "missing option value");
        const std::string value(argv[++i]);
        if (arg == "--levels") o.levels = list(value);
        else if (arg == "--sizes") o.sizes = list(value);
        else if (arg == "--blocks") o.blocks = list(value);
        else if (arg == "--exec-blocks") o.exec_blocks = list(value);
        else if (arg == "--threads") o.threads = list(value);
        else if (arg == "--fusions") o.fusions = list(value);
        else if (arg == "--repeat") o.repeat = integer(value, 2);
        else if (arg == "--warmup") o.warmup = integer(value, 1);
        else if (arg == "--max-it") o.max_it = integer(value, 1);
        else if (arg == "--fixed-iterations") o.fixed_iterations = integer(value, 1);
        else if (arg == "--deterministic") o.deterministic = integer(value);
        else if (arg == "--seed") o.seed = integer(value);
        else if (arg == "--method") o.method = value;
        else if (arg == "--warmup-iterations") o.warmup_iterations = integer(value, 1);
        else if (arg == "--rhs") o.rhs = value;
        else if (arg == "--output") o.output = value;
        else if (arg == "--rtol" || arg == "--initial-scale") {
            char *end = nullptr;
            const double parsed = std::strtod(value.c_str(), &end);
            require(end != value.c_str() && !*end && std::isfinite(parsed) && parsed >= 0 && parsed < 1,
                    "invalid tolerance or initial scale");
            if (arg == "--rtol") { require(parsed > 0, "rtol must be positive"); o.rtol = parsed; }
            else o.initial_scale = parsed;
        } else throw std::runtime_error("unknown option: " + arg);
    }
    require(o.repeat % 2 == 0, "repeat must be even to balance execution order");
    require(o.deterministic <= 1 && !o.output.empty(), "invalid deterministic flag or output path");
    require(o.rhs == "smooth" || o.rhs == "random", "rhs must be smooth or random");
    require(o.method == "paired" || o.method == "global" || o.method == "cellwise", "invalid method");
    if (o.fixed_iterations) {
        require(o.method != "paired", "fixed iterations require isolated mode");
        o.max_it = o.fixed_iterations;
        o.rtol = 0;
    }
    if (o.method != "paired") require(o.levels.size() == 1 && o.sizes.size() == 1 && o.blocks.size() == 1 &&
            o.exec_blocks.size() == 1 && o.threads.size() == 1 && o.fusions.size() == 1,
            "isolated mode requires one configuration");
    for (const int p : o.levels) require(p == 2 || p == 4 || p == 8, "supported degrees: 2,4,8");
    for (const int f : o.fusions) require(f <= 1, "fusion must be 0 or 1");
    for (const int t : o.threads) {
        require(t > 0, "threads must be positive");
#ifndef SFEM_ENABLE_OPENMP
        require(t == 1, "OpenMP is disabled");
#endif
    }
    for (const int n : o.sizes) {
        require(n > 0, "size must be positive");
        const ptrdiff_t ne = product(product(n, n), n);
        bool has_pair = false;
        for (const int bs : o.blocks) {
            require(bs > 0 && ne % bs == 0, "nelements must be divisible by positive B_save");
            for (const int be : o.exec_blocks) {
                require(be == 1 || be == 2 || be == 4 || be == 8 || be == 16, "unsupported B_exec");
                if (be <= bs) {
                    require(bs % be == 0, "B_save must be divisible by B_exec");
                    has_pair = true;
                }
            }
        }
        require(has_pair, "no valid B_save/B_exec pair");
    }
    return o;
}

void set_threads(const int n) {
#ifdef SFEM_ENABLE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(n);
    int actual = 0;
#pragma omp parallel
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    require(actual == n, "OpenMP did not create the requested team");
#else
    require(n == 1, "OpenMP is disabled");
#endif
}

// Full-precision Gauss-Legendre, with moment checks, on [0,1].
void quadrature(const int Q, scalar_t *qx, scalar_t *qw) {
    const auto legendre = [Q](const long double z, long double &p, long double &dp) {
        long double p0 = 1, p1 = z;
        for (int k = 2; k <= Q; ++k) {
            const long double pk = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
            p0 = p1; p1 = pk;
        }
        p = p1;
        dp = Q * (z * p1 - p0) / (z * z - 1);
    };
    for (int i = 0; i < (Q + 1) / 2; ++i) {
        long double z = std::cos(std::acos(-1.L) * (i + .75L) / (Q + .5L)), p, dp;
        bool converged = false;
        for (int k = 0; k < 64; ++k) {
            legendre(z, p, dp);
            const long double dz = p / dp;
            z -= dz;
            if (std::abs(dz) <= 8 * std::numeric_limits<long double>::epsilon()) { converged = true; break; }
        }
        require(converged, "quadrature iteration failed");
        legendre(z, p, dp);
        qx[i] = (scalar_t)((1 - z) / 2); qx[Q - 1 - i] = (scalar_t)((1 + z) / 2);
        qw[i] = qw[Q - 1 - i] = (scalar_t)(1 / ((1 - z * z) * dp * dp));
    }
    for (int k = 0; k < 2 * Q; ++k) {
        long double moment = 0;
        for (int q = 0; q < Q; ++q) moment += qw[q] * std::pow((long double)qx[q], k);
        require(std::abs(moment - 1.L / (k + 1)) < 256 * std::numeric_limits<scalar_t>::epsilon(),
                "quadrature moment check failed");
    }
}

struct Problem {
    int p, N, size;
    ptrdiff_t ne, ng, nc;
    std::vector<idx_t> connectivity;
    std::vector<const idx_t *> elements;
    std::array<std::vector<geom_t>, 3> points;
    std::vector<ptrdiff_t> boundary;
    std::vector<int> multiplicity;
    std::vector<real_t> truth, initial, rhs;
    sfem::SharedBuffer<scalar_t> S, D, qw, geometry;
    double r0_norm{0}, b_norm{0};

    Problem(const int degree, const int n, const Options &o) : p(degree), N(p + 1), size(n) {
        ne = product(product(n, n), n);
        const ptrdiff_t M = product(n, p) + 1, N3 = N * N * N;
        ng = product(product(M, M), M); nc = product(ne, N3);
        require(ng - 1 <= std::numeric_limits<idx_t>::max(), "global node ids exceed idx_t");
        connectivity.resize(nc); elements.resize(N3);
        for (int v = 0; v < N3; ++v) elements[v] = connectivity.data() + v * ne;
        for (auto &axis : points) axis.resize(ng);
        multiplicity.resize(ng); truth.resize(ng); rhs.resize(ng);
        if (o.initial_scale != 0) initial.resize(ng);
        std::mt19937_64 random(o.seed);
        std::uniform_real_distribution<double> uniform(-1, 1);
        const auto axis = [=](const ptrdiff_t i, const double scale, const double grade) {
            const ptrdiff_t e = std::min(i / p, (ptrdiff_t)n - 1);
            const double a = (double)e / n, b = (double)(e + 1) / n, t = (double)(i - e * p) / p;
            return (1 - t) * (scale * a + grade * a * a) + t * (scale * b + grade * b * b);
        };
        const double pi = std::acos(-1.);
        for (ptrdiff_t iz = 0; iz < M; ++iz) for (ptrdiff_t iy = 0; iy < M; ++iy) for (ptrdiff_t ix = 0; ix < M; ++ix) {
            const ptrdiff_t g = ix + M * (iy + M * iz);
            const double a = axis(ix, 1, .125), b = axis(iy, .75, .0625), c = axis(iz, .5, .03125);
            points[0][g] = (geom_t)(a + .25 * b - .125 * c);
            points[1][g] = (geom_t)(b + .125 * c); points[2][g] = (geom_t)c;
            const bool constrained = ix == 0 || iy == 0 || iz == 0 || ix == M - 1 || iy == M - 1 || iz == M - 1;
            if (constrained) boundary.push_back(g);
            const double value = o.rhs == "random" ? uniform(random) :
                    std::sin(pi * ix / (M - 1)) * std::sin(pi * iy / (M - 1)) * std::sin(pi * iz / (M - 1)) *
                    (1 + .1 * std::cos(2 * pi * ix / (M - 1)));
            truth[g] = constrained ? 0 : (real_t)value;
            if (!initial.empty()) initial[g] = (real_t)o.initial_scale * truth[g];
        }
        for (ptrdiff_t e = 0; e < ne; ++e) {
            const ptrdiff_t ex = e % n, ey = (e / n) % n, ez = e / ((ptrdiff_t)n * n);
            for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
                const int v = ix + N * (iy + N * iz);
                const ptrdiff_t g = ex * p + ix + M * (ey * p + iy + M * (ez * p + iz));
                connectivity[v * ne + e] = (idx_t)g;
                ++multiplicity[g];
            }
        }
        S = sfem::create_host_buffer<scalar_t>(N * N); D = sfem::create_host_buffer<scalar_t>(N * N);
        qw = sfem::create_host_buffer<scalar_t>(N); geometry = sfem::create_host_buffer<scalar_t>(product(6, ne));
        std::vector<scalar_t> qx(N);
        quadrature(N, qx.data(), qw->data());
        require(lagrange_eval<scalar_t>(p, N, qx.data(), S->data()) == SFEM_SUCCESS &&
                lagrange_diff_eval<scalar_t>(p, N, qx.data(), D->data()) == SFEM_SUCCESS, "basis setup failed");
        sfem_dss_bxyze_layout_t layout;
        const geom_t *xyz[3] = {points[0].data(), points[1].data(), points[2].data()};
        require(sfem_dss_bxyze_create_layout(p, ne, 1, 1, &layout) == SFEM_SUCCESS &&
                sfem_dss_bxyze_affine_geometry(&layout, elements.data(), xyz, geometry->data()) == SFEM_SUCCESS,
                "global geometry setup failed");
        for (auto &axis : points) std::vector<geom_t>().swap(axis);
    }
};

struct GlobalOperator final : sfem::Operator<real_t> {
    using Kernel = void (*)(GlobalOperator &, const real_t *, real_t *);
    const Problem &p;
    int threads;
    ptrdiff_t scalar_stride, nodal_stride, index_stride;
    std::vector<scalar_t> scratch;
    std::vector<real_t> nodal;
    std::vector<idx_t> indices;
    Kernel kernel{nullptr};

    template <int N, bool Atomic>
    static void apply_tpl(GlobalOperator &data, const real_t *x, real_t *y) {
        constexpr int N3 = N * N * N;
        const auto &p = data.p;
        const ptrdiff_t grid_n = (ptrdiff_t)p.p * p.size + 1;
        const ptrdiff_t grid_n2 = grid_n * grid_n;
#pragma omp parallel
        {
            int tid = 0;
#ifdef SFEM_ENABLE_OPENMP
            tid = omp_get_thread_num();
#endif
            real_t *u = data.nodal.data() + tid * data.nodal_stride, *v = u + N3;
            idx_t *ids = data.indices.data() + tid * data.index_stride;
            scalar_t *gx = data.scratch.data() + tid * data.scalar_stride;
            scalar_t *gy = gx + N3, *gz = gy + N3, *work = gz + N3;
#pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < p.ng; ++i) y[i] = 0;
#pragma omp for schedule(static)
            for (ptrdiff_t e = 0; e < p.ne; ++e) {
                const ptrdiff_t ex = e % p.size;
                const ptrdiff_t ey = (e / p.size) % p.size;
                const ptrdiff_t ez = e / ((ptrdiff_t)p.size * p.size);
                const ptrdiff_t origin = p.p * (ex + grid_n * ey + grid_n2 * ez);
                for (int iz = 0; iz < N; ++iz) {
                    for (int iy = 0; iy < N; ++iy) {
                        const ptrdiff_t row = origin + (ptrdiff_t)iy * grid_n + (ptrdiff_t)iz * grid_n2;
                        const int local_row = (iz * N + iy) * N;
                        for (int ix = 0; ix < N; ++ix) {
                            const int i = local_row + ix;
                            ids[i] = (idx_t)(row + ix);
                            u[i] = x[ids[i]];
                        }
                    }
                }
                sfem_dss_bxyze_lagrange_gradient<N, N, 1>(1, p.S->data(), p.D->data(), u, gx, gy, gz, work);
                sfem_dss_bxyze_affine_metric_apply<N, 1>(1, p.qw->data(), p.geometry->data() + 6 * e, gx, gy, gz);
                sfem_dss_bxyze_lagrange_gradient_transpose<N, N, 1>(1, p.S->data(), p.D->data(), gx, gy, gz, v, work);
                if constexpr (Atomic) {
                    for (int iz = 0; iz < N; ++iz) {
                        for (int iy = 0; iy < N; ++iy) {
                            const int row = (iz * N + iy) * N;
                            if (iz > 0 && iz < N - 1 && iy > 0 && iy < N - 1) {
#pragma omp atomic update
                                y[ids[row]] += v[row];
                                for (int ix = 1; ix < N - 1; ++ix) y[ids[row + ix]] += v[row + ix];
#pragma omp atomic update
                                y[ids[row + N - 1]] += v[row + N - 1];
                            } else {
                                for (int ix = 0; ix < N; ++ix) {
#pragma omp atomic update
                                    y[ids[row + ix]] += v[row + ix];
                                }
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < N3; ++i) y[ids[i]] += v[i];
                }
            }
        }
    }

    GlobalOperator(const Problem &problem, const int nt) : p(problem), threads(nt) {
        sfem_dss_bxyze_layout_t layout;
        require(sfem_dss_bxyze_create_layout(p.p, p.ne, 1, 1, &layout) == SFEM_SUCCESS, "global layout failed");
        scalar_stride = sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(&layout, p.N);
        require(scalar_stride > 0, "invalid global workspace");
        const auto padded = [](const ptrdiff_t n, const size_t bytes) {
            const ptrdiff_t gap = std::max<ptrdiff_t>(1, 128 / bytes);
            return ((n + gap - 1) / gap + 1) * gap;
        };
        nodal_stride = padded(2 * p.N * p.N * p.N, sizeof(real_t));
        index_stride = padded(p.N * p.N * p.N, sizeof(idx_t));
        scratch.resize(product(nt, scalar_stride)); nodal.resize(product(nt, nodal_stride));
        indices.resize(product(nt, index_stride));
        switch (p.p) {
            case 2: kernel = nt > 1 ? apply_tpl<3, true> : apply_tpl<3, false>; break;
            case 4: kernel = nt > 1 ? apply_tpl<5, true> : apply_tpl<5, false>; break;
            case 8: kernel = nt > 1 ? apply_tpl<9, true> : apply_tpl<9, false>; break;
        }
        require(kernel != nullptr, "unsupported degree");
    }
    ptrdiff_t rows() const override { return p.ng; }
    ptrdiff_t cols() const override { return p.ng; }
    sfem::ExecutionSpace execution_space() const override { return sfem::EXECUTION_SPACE_HOST; }
    int apply(const real_t *x, real_t *y) override {
#ifdef SFEM_ENABLE_OPENMP
        if (omp_get_max_threads() > threads) return SFEM_FAILURE;
#endif
        kernel(*this, x, y);
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < (ptrdiff_t)p.boundary.size(); ++i) y[p.boundary[i]] = x[p.boundary[i]];
        return SFEM_SUCCESS;
    }
};

// Install the same callback structure/call depth for both paths; specialization
// removes the weight load/multiply entirely from the global path. No unit-weight
// array is passed to the global solver to artificially equalize its work.
template <bool Weighted>
void callbacks(const std::shared_ptr<sfem::ConjugateGradient<real_t>> &cg,
               const sfem::SharedBuffer<real_t> &weights = nullptr) {
    cg->inner_product = [weights](const ptrdiff_t n, const real_t *x, const real_t *y) {
        const real_t *w = nullptr;
        if constexpr (Weighted) w = weights->data();
        return sfem::openmp_dot<real_t, Weighted>(n, x, y, w);
    };
    cg->impl.update_x_r_and_rtr = [weights](const ptrdiff_t n, const real_t alpha,
                                           const real_t *p, const real_t *Ap, real_t *x, real_t *r) {
        const real_t *w = nullptr;
        if constexpr (Weighted) w = weights->data();
        return sfem::OpenMP_CG<real_t>::template update_x_r_and_rtr_tpl<Weighted>(n, alpha, p, Ap, x, r, w);
    };
}

struct Run {
    std::shared_ptr<sfem::Operator<real_t>> op;
    std::shared_ptr<sfem::ConjugateGradient<real_t>> cg;
    sfem::SharedBuffer<real_t> b, x, initial;
};

Run global_run(const Problem &p, const int threads) {
    Run r;
    r.op = std::make_shared<GlobalOperator>(p, threads);
    r.cg = sfem::h_cg<real_t>(); r.cg->set_op(r.op); callbacks<false>(r.cg);
    r.b = sfem::create_host_buffer<real_t>(p.ng); r.x = sfem::create_host_buffer<real_t>(p.ng);
    std::copy(p.rhs.begin(), p.rhs.end(), r.b->data());
    if (!p.initial.empty()) {
        r.initial = sfem::create_host_buffer<real_t>(p.ng);
        std::copy(p.initial.begin(), p.initial.end(), r.initial->data());
    }
    return r;
}

Run cell_run(const Problem &p, const int bs, const int be, const int threads) {
    sfem_dss_bxyze_layout_t layout;
    sfem_dss_bxyze_laplacian_affine_volumetric_t prepared;
    require(sfem_dss_bxyze_create_layout(p.p, p.ne, bs, be, &layout) == SFEM_SUCCESS &&
            sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, p.N, &prepared) == SFEM_SUCCESS,
            "cellwise layout/setup failed");
    auto geometry = sfem::create_host_buffer<scalar_t>(product(6, p.ne));
    // Reorder the exact same six metrics, rather than recomputing geometry.
    for (ptrdiff_t e = 0; e < p.ne; ++e) for (int h = 0; h < 6; ++h)
        geometry->data()[(e / bs) * 6 * bs + h * bs + e % bs] = p.geometry->data()[6 * e + h];
    std::vector<ptrdiff_t> offsets;
    for (ptrdiff_t e = 0; e < p.ne; ++e) for (int v = 0; v < p.N * p.N * p.N; ++v) {
        if (std::binary_search(p.boundary.begin(), p.boundary.end(), (ptrdiff_t)p.elements[v][e]))
            offsets.push_back((e / bs) * layout.block_stride + v * bs + e % bs);
    }
    auto ids = sfem::create_host_buffer<ptrdiff_t>(offsets.size());
    std::copy(offsets.begin(), offsets.end(), ids->data());
    auto work = sfem::create_host_buffer<scalar_t>(product(threads, prepared.workspace_stride));
    auto weights = sfem::dss_bxyze_structured_weights(layout, p.size, p.size, p.size);
    Run r;
    r.op = sfem::dss_bxyze_laplacian_affine_operator(
            prepared, p.size, p.size, p.size, p.S, p.D, p.qw, geometry, work, ids);
    require(r.op && weights, "cellwise operator/weights setup failed");
    r.cg = sfem::h_cellwise_cg<real_t>(r.op, weights);
    require(r.cg != nullptr, "cellwise CG setup failed");
    callbacks<true>(r.cg, weights);
    r.b = sfem::create_host_buffer<real_t>(p.nc); r.x = sfem::create_host_buffer<real_t>(p.nc);
    require(sfem_dss_global_to_bxyze(p.p, p.ne, bs, p.elements.data(), p.rhs.data(), r.b->data()) == SFEM_SUCCESS,
            "cellwise RHS conversion failed");
    if (!p.initial.empty()) {
        r.initial = sfem::create_host_buffer<real_t>(p.nc);
        require(sfem_dss_global_to_bxyze(p.p, p.ne, bs, p.elements.data(), p.initial.data(), r.initial->data()) == SFEM_SUCCESS,
                "cellwise initial guess conversion failed");
    }
    // Validate every copy and inverse multiplicity before any solve is timed.
    require(sfem_dss_global_to_bxyze(p.p, p.ne, bs, p.elements.data(), p.truth.data(), r.x->data()) == SFEM_SUCCESS,
            "truth conversion failed");
    std::vector<real_t> y(p.nc);
    require(r.op->apply(r.x->data(), y.data()) == SFEM_SUCCESS, "initial cellwise validation apply failed");
    double difference = 0, magnitude = 0;
    const double eps = std::max((double)std::numeric_limits<scalar_t>::epsilon(),
                                (double)std::numeric_limits<real_t>::epsilon());
    for (ptrdiff_t e = 0; e < p.ne; ++e) for (int v = 0; v < p.N * p.N * p.N; ++v) {
        const ptrdiff_t i = (e / bs) * layout.block_stride + v * bs + e % bs;
        const idx_t g = p.elements[v][e];
        require(std::abs(weights->data()[i] - 1. / p.multiplicity[g]) <= 8 * eps, "wrong cellwise multiplicity");
        require(std::isfinite(y[i]), "non-finite assembled cellwise output");
        difference = std::max(difference, std::abs((double)y[i] - p.rhs[g]));
        magnitude = std::max(magnitude, std::abs((double)p.rhs[g]));
    }
    require(difference <= 512 * p.N * eps * std::max(1., magnitude), "global and cellwise operators disagree");
    return r;
}

void configure(Run &r, const Options &o, const int fusion) {
    r.cg->set_rtol((real_t)o.rtol); r.cg->set_atol(0); r.cg->set_max_it(o.max_it);
    r.cg->set_verbose(false); r.cg->set_fusion(fusion != 0); r.cg->set_apply_overwrites_output(true);
}

struct Sample { double seconds{0}, per_iteration{0}, relres{0}, truth_error{0}; int iterations{0}; };

Sample solve(Run &r) {
    if (r.initial) std::copy(r.initial->data(), r.initial->data() + r.initial->size(), r.x->data());
    else std::fill(r.x->data(), r.x->data() + r.x->size(), (real_t)0);
    const double start = smesh::time_seconds();
    const int status = r.cg->apply(r.b->data(), r.x->data());
    const double elapsed = smesh::time_seconds() - start;
    require(status == SFEM_SUCCESS, "CG did not converge; increase --max-it or inspect the problem");
    // CG's existing counter is zero-based on successful nonzero-residual solves.
    const int iterations = r.cg->iterations() + 1;
    require(elapsed > 0 && iterations > 0, "invalid CG timing/counter");
    return {elapsed, elapsed / iterations, 0, 0, iterations};
}

// Always evaluate the true residual with one serial, assembled global operator,
// independently of each solver's weighted/recursive residual. Outside timers.
void validate(const Problem &p, GlobalOperator &reference, const Options &o,
              const real_t *x, std::vector<real_t> &scratch, Sample &s) {
    require(reference.apply(x, scratch.data()) == SFEM_SUCCESS, "residual reference apply failed");
    long double rtr = 0;
    for (ptrdiff_t i = 0; i < p.ng; ++i) {
        require(std::isfinite(x[i]) && std::isfinite(scratch[i]), "non-finite solution/residual");
        const long double r = (long double)p.rhs[i] - scratch[i];
        rtr += r * r;
        s.truth_error = std::max(s.truth_error, std::abs((double)x[i] - p.truth[i]));
    }
    for (const ptrdiff_t i : p.boundary) require(x[i] == 0, "nonzero Dirichlet solution value");
    s.relres = (double)std::sqrt(rtr) / p.r0_norm;
    if (!o.fixed_iterations) require(s.relres <= o.rtol, "true residual exceeds the requested rtol");
}

double extract(const Problem &p, const int bs, const Run &cell, std::vector<real_t> &x) {
    std::fill(x.begin(), x.end(), std::numeric_limits<real_t>::quiet_NaN());
    double duplicate_error = 0;
    const ptrdiff_t stride = (ptrdiff_t)bs * p.N * p.N * p.N;
    for (ptrdiff_t e = 0; e < p.ne; ++e) for (int v = 0; v < p.N * p.N * p.N; ++v) {
        const real_t value = cell.x->data()[(e / bs) * stride + v * bs + e % bs];
        const idx_t g = p.elements[v][e];
        require(std::isfinite(value), "non-finite cellwise solution");
        if (std::isfinite(x[g])) duplicate_error = std::max(duplicate_error, std::abs((double)value - x[g]));
        x[g] = value;
    }
    require(duplicate_error == 0, "cellwise copies lost consistency");
    return duplicate_error;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : .5 * (values[n / 2 - 1] + values[n / 2]);
}

void write(FILE *out, const std::string &row) {
    require(std::fputs(row.c_str(), out) >= 0 && std::fflush(out) == 0, "CSV write/flush failed");
}

const char *prefix_header = "degree,Q,nx,ny,nz,nelements,n_global_dofs,n_cellwise_dofs,B_save,B_exec,threads,"
        "fusion,deterministic_blas,rtol,atol,max_it,rhs,seed,repeat,warmup,real_bytes,scalar_bytes,initial_scale,";

std::string prefix(const Problem &p, const Options &o, const int bs, const int be, const int threads, const int fusion) {
    std::ostringstream s;
    s << std::scientific << std::setprecision(12) << p.p << ',' << p.N << ',' << p.size << ',' << p.size << ',' << p.size << ','
      << p.ne << ',' << p.ng << ',' << p.nc << ',' << bs << ',' << be << ',' << threads << ',' << fusion << ','
      << o.deterministic << ',' << o.rtol << ",0," << o.max_it << ',' << o.rhs << ',' << o.seed << ','
      << o.repeat << ',' << o.warmup << ',' << sizeof(real_t) << ',' << sizeof(scalar_t) << ',' << o.initial_scale << ',';
    return s.str();
}

void measure(const Problem &p, const Options &o, const int bs, const int be, const int threads,
             const int fusion, Run &global, FILE *raw, FILE *summary) {
    std::fprintf(stderr, "Starting p=%d size=%d t=%d fusion=%d B=%d/%d\n", p.p, p.size, threads, fusion, bs, be);
    Run cell = cell_run(p, bs, be, threads);
    configure(global, o, fusion); configure(cell, o, fusion);
    for (int w = 0; w < o.warmup; ++w) {
        if (w % 2 == 0) { solve(global); solve(cell); }
        else { solve(cell); solve(global); }
    }
    std::array<std::vector<Sample>, 2> samples;
    std::vector<real_t> cell_global(p.ng), scratch(p.ng);
    GlobalOperator reference(p, 1);
    double max_difference = 0, max_duplicate_error = 0;
    const auto pre = prefix(p, o, bs, be, threads, fusion);
    for (int r = 0; r < o.repeat; ++r) {
        Sample g, c;
        if (r % 2 == 0) { g = solve(global); c = solve(cell); }
        else { c = solve(cell); g = solve(global); }
        const double duplicates = extract(p, bs, cell, cell_global);
        // Serial reference assembly has no atomics; restore the requested team
        // before either solver runs again. Thread management is never timed.
        set_threads(1);
        validate(p, reference, o, global.x->data(), scratch, g);
        validate(p, reference, o, cell_global.data(), scratch, c);
        set_threads(threads);
        double difference = 0;
        for (ptrdiff_t i = 0; i < p.ng; ++i)
            difference = std::max(difference, std::abs((double)global.x->data()[i] - cell_global[i]));
        max_difference = std::max(max_difference, difference);
        max_duplicate_error = std::max(max_duplicate_error, duplicates);
        samples[0].push_back(g); samples[1].push_back(c);
        std::ostringstream row;
        row << std::scientific << std::setprecision(12) << pre << r << ',' << (r % 2 ? "cellwise_first" : "global_first") << ','
            << g.iterations << ',' << c.iterations << ',' << g.seconds << ',' << c.seconds << ','
            << g.per_iteration << ',' << c.per_iteration << ',' << g.relres << ',' << c.relres << ','
            << g.truth_error << ',' << c.truth_error << ',' << difference << ',' << duplicates << ','
            << g.seconds / c.seconds << ',' << g.per_iteration / c.per_iteration << '\n';
        write(raw, row.str());
        std::fprintf(stderr, "  sample %d/%d: global %.6g s, cellwise %.6g s, true relres %.3e / %.3e\n",
                     r + 1, o.repeat, g.seconds, c.seconds, g.relres, c.relres);
    }
    std::array<double, 2> tts{}, tpi{}, residual{}, truth_error{};
    std::array<int, 2> itmin{}, itmax{};
    std::ostringstream row;
    row << std::scientific << std::setprecision(12) << pre;
    for (int m = 0; m < 2; ++m) {
        std::vector<double> times, per_iteration;
        itmin[m] = std::numeric_limits<int>::max();
        for (const auto &s : samples[m]) {
            times.push_back(s.seconds); per_iteration.push_back(s.per_iteration);
            itmin[m] = std::min(itmin[m], s.iterations); itmax[m] = std::max(itmax[m], s.iterations);
            residual[m] = std::max(residual[m], s.relres); truth_error[m] = std::max(truth_error[m], s.truth_error);
        }
        tts[m] = median(times); tpi[m] = median(per_iteration);
        row << *std::min_element(times.begin(), times.end()) << ',' << tts[m] << ',' << tpi[m] << ','
            << itmin[m] << ',' << itmax[m] << ',' << residual[m] << ',' << truth_error[m] << ',';
    }
    row << max_difference << ',' << max_duplicate_error << ',' << tts[0] / tts[1] << ',' << tpi[0] / tpi[1] << '\n';
    write(summary, row.str());
    std::fprintf(stderr, "p=%d size=%d t=%d fusion=%d B=%d/%d: global %.6g s (%d..%d it), "
                 "cellwise %.6g s (%d..%d it), speedup %.3fx, per-iteration %.3fx\n",
                 p.p, p.size, threads, fusion, bs, be, tts[0], itmin[0], itmax[0], tts[1], itmin[1], itmax[1],
                 tts[0] / tts[1], tpi[0] / tpi[1]);
}

void setup_rhs(Problem &p) {
    set_threads(1);
    GlobalOperator reference(p, 1);
    require(reference.apply(p.truth.data(), p.rhs.data()) == SFEM_SUCCESS, "RHS setup failed");
    std::vector<real_t> y;
    if (!p.initial.empty()) {
        y.resize(p.ng);
        require(reference.apply(p.initial.data(), y.data()) == SFEM_SUCCESS, "initial residual setup failed");
    }
    long double rtr = 0, btb = 0;
    for (ptrdiff_t i = 0; i < p.ng; ++i) {
        const long double r = (long double)p.rhs[i] - (y.empty() ? 0 : y[i]);
        rtr += r * r; btb += (long double)p.rhs[i] * p.rhs[i];
    }
    p.r0_norm = (double)std::sqrt(rtr); p.b_norm = (double)std::sqrt(btb);
    require(p.r0_norm > 0 && std::isfinite(p.r0_norm), "zero/non-finite initial residual");
}

size_t peak_rss_bytes() {
    struct rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
#ifdef __APPLE__
    return (size_t)usage.ru_maxrss;
#else
    return (size_t)usage.ru_maxrss * 1024;
#endif
}

// Recreate only for the independent serial validation, after releasing the CG
// scratch. The cellwise apply itself never accesses this connectivity.
void restore_connectivity(Problem &p) {
    const int N = p.N, n = p.size, N3 = N * N * N;
    const ptrdiff_t M = (ptrdiff_t)n * p.p + 1;
    p.connectivity.resize(p.nc); p.elements.resize(N3);
    for (int v = 0; v < N3; ++v) p.elements[v] = p.connectivity.data() + v * p.ne;
    for (ptrdiff_t e = 0; e < p.ne; ++e) {
        const ptrdiff_t ex = e % n, ey = (e / n) % n, ez = e / ((ptrdiff_t)n * n);
        for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
            const int v = ix + N * (iy + N * iz);
            p.connectivity[v * p.ne + e] = (idx_t)(ex * p.p + ix + M * (ey * p.p + iy + M * (ez * p.p + iz)));
        }
    }
}

int isolated(const Options &o) {
    const int degree = o.levels[0], size = o.sizes[0], bs = o.blocks[0], be = o.exec_blocks[0];
    const int threads = o.threads[0], fusion = o.fusions[0];
    std::fprintf(stderr, "Setup %s: p=%d size=%d threads=%d B=%d/%d\n",
                 o.method.c_str(), degree, size, threads, bs, be);
    set_threads(1);
    Problem p(degree, size, o);
    setup_rhs(p);
    set_threads(threads);
    Run r = o.method == "global" ? global_run(p, threads) : cell_run(p, bs, be, threads);
    configure(r, o, fusion);
    std::vector<int>().swap(p.multiplicity);
    // Both isolated operators now use implicit structured topology during apply.
    std::vector<idx_t>().swap(p.connectivity);
    std::vector<const idx_t *>().swap(p.elements);
    const auto reset = [&] {
        if (r.initial) std::copy(r.initial->data(), r.initial->data() + r.initial->size(), r.x->data());
        else std::fill(r.x->data(), r.x->data() + r.x->size(), (real_t)0);
    };
    // Warm the same number of iterations and allocate CG workspace without
    // spending another full solve. A normal iteration-limit return is expected.
    const int warm_it = std::min(o.warmup_iterations, o.max_it);
    r.cg->set_max_it(warm_it);
    reset();
    const double warm_start = smesh::time_seconds();
    const int warm_status = r.cg->apply(r.b->data(), r.x->data());
    require(warm_status == SFEM_SUCCESS || r.cg->iterations() == warm_it, "operator failed during warmup");
    std::fprintf(stderr, "Warmup complete: %d iterations, %.3f s\n",
                 warm_status == SFEM_SUCCESS ? r.cg->iterations() + 1 : warm_it, smesh::time_seconds() - warm_start);
    r.cg->set_max_it(o.max_it);
    reset();
    const double start = smesh::time_seconds();
    auto *const cg = r.cg.get();
    cg->interceptor = [cg, start, last = start](real_t *) mutable {
        const int iteration = cg->iterations() + 1;
        if (iteration % 10) return;
        const double now = smesh::time_seconds();
        if (now - last >= 30) {
            std::fprintf(stderr, "CG progress: iteration=%d elapsed=%.1f s\n", iteration, now - start);
            last = now;
        }
    };
    // Closure setup precedes the timed call; progress has the same policy in
    // both methods and performs a clock read only every ten iterations.
    const double tick = smesh::time_seconds();
    const int status = cg->apply(r.b->data(), r.x->data());
    Sample s;
    s.seconds = smesh::time_seconds() - tick;
    s.iterations = cg->iterations() + (status == SFEM_SUCCESS ? 1 : 0);
    s.per_iteration = s.iterations > 0 ? s.seconds / s.iterations : std::numeric_limits<double>::quiet_NaN();
    const size_t solve_rss = peak_rss_bytes();
    std::fprintf(stderr, "Solve finished: status=%d iterations=%d elapsed=%.6g s; validating\n", status, s.iterations, s.seconds);
    // Validation needs the solution, operator, truth and RHS, not the Krylov
    // vectors or weights. Release them before allocating global check buffers.
    r.cg.reset(); r.b.reset(); r.initial.reset();
    if (o.method == "cellwise") restore_connectivity(p);
    set_threads(1);
    GlobalOperator reference(p, 1);
    std::vector<real_t> canonical, scratch(p.ng);
    double duplicate_error = 0;
    const real_t *x = r.x->data();
    bool valid = true;
    try {
        if (o.method == "cellwise") {
            canonical.resize(p.ng);
            duplicate_error = extract(p, bs, r, canonical);
            x = canonical.data();
        }
        validate(p, reference, o, x, scratch, s);
    } catch (const std::exception &e) {
        valid = false;
        std::fprintf(stderr, "Validation failed: %s\n", e.what());
    }
    const bool fixed_complete = o.fixed_iterations && status != SFEM_SUCCESS && s.iterations == o.fixed_iterations;
    const bool converged = !o.fixed_iterations && status == SFEM_SUCCESS;
    const bool accepted = valid && (fixed_complete || converged);
    const char *state = !valid ? "validation_failed" : fixed_complete ? "fixed_iterations" :
                        converged ? "converged" : "not_converged";
    using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
    File out(std::fopen(o.output.c_str(), "w"), &std::fclose);
    require(out != nullptr, "cannot open isolated result CSV");
    write(out.get(), std::string(prefix_header) +
            "method,status,iterations,attempt_seconds,tts_s,s_per_iteration,true_relres_r0,truth_error_inf,"
            "duplicate_difference_inf,peak_solve_rss_bytes,peak_process_rss_bytes,warmup_iterations\n");
    std::ostringstream row;
    row << std::scientific << std::setprecision(12) << prefix(p, o, bs, be, threads, fusion)
        << o.method << ',' << state << ',' << s.iterations << ',' << s.seconds << ','
        << (accepted && converged ? s.seconds : std::numeric_limits<double>::quiet_NaN()) << ',' << s.per_iteration << ','
        << s.relres << ',' << s.truth_error << ',' << duplicate_error << ',' << solve_rss << ',' << peak_rss_bytes() << ','
        << warm_it << '\n';
    write(out.get(), row.str());
    std::fprintf(stderr, "Saved %s: %s, true relres=%.3e\n", o.output.c_str(), state, s.relres);
    return accepted ? SFEM_SUCCESS : SFEM_FAILURE;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(argv[0]); return SFEM_SUCCESS; }
    try {
        const Options o = options(argc, argv);
        require(setenv("SFEM_DETERMINISTIC_BLAS", o.deterministic ? "1" : "0", 1) == 0, "cannot select BLAS policy");
        require(sfem::blas_deterministic() == (o.deterministic != 0), "BLAS policy initialized too early");
        auto context = sfem::initialize(argc, argv);
        require(context->communicator()->size() == 1, "benchmark requires one MPI rank");
        if (o.method != "paired") return isolated(o);
        const std::string path = (o.output.size() >= 4 && o.output.substr(o.output.size() - 4) == ".csv"
                                  ? o.output.substr(0, o.output.size() - 4) : o.output) + "_summary.csv";
        using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
        File raw(std::fopen(o.output.c_str(), "w"), &std::fclose), summary(std::fopen(path.c_str(), "w"), &std::fclose);
        require(raw && summary, "cannot open output CSVs");
        write(raw.get(), std::string(prefix_header) +
              "sample,order,global_iterations,cellwise_iterations,global_tts_s,cellwise_tts_s,"
              "global_s_per_iteration,cellwise_s_per_iteration,global_true_relres_r0,cellwise_true_relres_r0,"
              "global_truth_error_inf,cellwise_truth_error_inf,solution_difference_inf,duplicate_difference_inf,"
              "speedup_tts,speedup_per_iteration\n");
        write(summary.get(), std::string(prefix_header) +
              "global_tts_min_s,global_tts_median_s,global_s_per_iteration_median,global_iterations_min,global_iterations_max,"
              "global_true_relres_r0_max,global_truth_error_inf_max,"
              "cellwise_tts_min_s,cellwise_tts_median_s,cellwise_s_per_iteration_median,cellwise_iterations_min,cellwise_iterations_max,"
              "cellwise_true_relres_r0_max,cellwise_truth_error_inf_max,solution_difference_inf_max,duplicate_difference_inf_max,"
              "speedup_tts_medians,speedup_per_iteration_medians\n");
        std::fprintf(stderr, "CG comparison: rtol=%.3g, atol=0, repeats=%d, warmup=%d, deterministic BLAS=%d, RHS=%s\n",
                     o.rtol, o.repeat, o.warmup, o.deterministic, o.rhs.c_str());
        int cases = 0;
        for (const int p : o.levels) for (const int size : o.sizes) {
            // One serial RHS/reference setup, shared by all layouts and threads.
            set_threads(1);
            Problem problem(p, size, o);
            setup_rhs(problem);
            for (const int threads : o.threads) {
                set_threads(threads);
                Run global = global_run(problem, threads);
                for (const int bs : o.blocks) for (const int be : o.exec_blocks) if (be <= bs) {
                    for (const int fusion : o.fusions) {
                        measure(problem, o, bs, be, threads, fusion, global, raw.get(), summary.get());
                        ++cases;
                    }
                }
            }
        }
        std::fprintf(stderr, "Saved %d validated configurations to %s\nSummary: %s\n", cases, o.output.c_str(), path.c_str());
        return SFEM_SUCCESS;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "sfem_DSSBXYZECGBench: %s\n", e.what());
        return SFEM_FAILURE;
    }
}
