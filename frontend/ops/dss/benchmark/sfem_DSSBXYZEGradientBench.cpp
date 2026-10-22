#include "sfem_API.hpp"
#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"
#include "smesh_base.hpp"
#include "smesh_semistructured.hpp"

#include "hex8_laplacian_inline_cpu.hpp"
#include "lagrange.hpp"
#include "line_quadrature.hpp"
#include "sfem_macros.hpp"
#include "sshex8.hpp"
#include "lagrange_hex_laplacian_inline.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SFEM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace {

// Reference gradients only: no metric, weights, transpose evaluation or DSS.
// Timings include input gather (both global paths), all three output writes and the
// OpenMP region. Setup, conversion, validation and heap allocation are excluded.
// The original global reference uses nine directional contractions. Global8
// gathers one element and calls the shared eight-contraction kernel with B_EXEC=1
// and storage stride 1. BXYZE uses the same kernel on batches of elements.
// CSV global_* and speedup_* columns refer to the original nine-contraction path;
// global8_* and speedup_vs_global8_* columns describe the additional reference.
// Outputs are materialized for every element and quadrature point in all paths.
// Only full storage blocks and full execution batches are supported.

std::string default_output_path() {
    const std::string source(__FILE__);
    const auto slash = source.find_last_of("/\\");
    return (slash == std::string::npos ? "frontend/ops/dss/benchmark/" : source.substr(0, slash + 1)) +
           "dss_bxyze_gradient_bench.csv";
}

struct Options {
    std::vector<int> levels{2, 4, 8};
    std::vector<int> sizes{8, 16};
    std::vector<int> blocks{4, 8, 16, 32, 64};
    std::vector<int> exec_blocks{1, 2, 4, 8, 16};
#ifdef SFEM_ENABLE_OPENMP
    std::vector<int> threads{1, 8};
#else
    std::vector<int> threads{1};
#endif
    int repeat{12};
    int warmup{6};
    int seed{42};
    std::string output{default_output_path()};
};

bool parse_integer(const std::string &text, const int minimum, int &value) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (errno || end == text.c_str() || *end || parsed < minimum || parsed > std::numeric_limits<int>::max()) return false;
    value = (int)parsed;
    return true;
}

bool parse_list(const char *text, std::vector<int> &values) {
    std::stringstream stream(text);
    std::string item;
    values.clear();
    while (std::getline(stream, item, ',')) {
        int value;
        if (!parse_integer(item, 1, value)) return false;
        if (std::find(values.begin(), values.end(), value) == values.end()) values.push_back(value);
    }
    return !values.empty();
}

void usage(const char *name) {
    std::fprintf(stderr,
                 "usage: %s [--levels 2,4,8] [--sizes 8,16] [--blocks 4,8,16,32,64]\n"
                 "          [--exec-blocks 1,2,4,8,16] [--threads 1,8] [--repeat 12]\n"
                 "          [--warmup 6] [--seed 42] [--output file.csv]\n"
                 "Q=p+1 Gauss-Legendre; sizes are macro-elements per axis.\n"
                 "Measures global 9 contractions, global 8 contractions and cellwise 8 contractions.\n"
                 "Multiples of 6 repeats balance all six execution orders.\n"
                 "Pairs with B_exec>B_save are skipped. Requires nelements%%B_save=0 and B_save%%B_exec=0.\n",
                 name);
}

bool parse_options(int argc, char **argv, Options &opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (i + 1 == argc) return false;
        const char *value = argv[++i];
        bool valid = true;
        if (arg == "--levels") valid = parse_list(value, opts.levels);
        else if (arg == "--sizes") valid = parse_list(value, opts.sizes);
        else if (arg == "--blocks") valid = parse_list(value, opts.blocks);
        else if (arg == "--exec-blocks") valid = parse_list(value, opts.exec_blocks);
        else if (arg == "--threads") valid = parse_list(value, opts.threads);
        else if (arg == "--repeat") valid = parse_integer(value, 1, opts.repeat);
        else if (arg == "--warmup") valid = parse_integer(value, 0, opts.warmup);
        else if (arg == "--seed") valid = parse_integer(value, 0, opts.seed);
        else if (arg == "--output") opts.output = value;
        else return false;
        if (!valid) return false;
    }
    for (const int p : opts.levels) {
        if (p != 2 && p != 4 && p != 8) return false;
    }
    bool has_pair = false;
    for (const int be : opts.exec_blocks) {
        if (be != 1 && be != 2 && be != 4 && be != 8 && be != 16) return false;
        for (const int bs : opts.blocks) has_pair |= be <= bs;
    }
#ifndef SFEM_ENABLE_OPENMP
    for (const int threads : opts.threads) if (threads != 1) return false;
#endif
    return has_pair && !opts.output.empty();
}

ptrdiff_t checked_product(const ptrdiff_t a, const ptrdiff_t b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<ptrdiff_t>::max() / b)) {
        throw std::length_error("benchmark dimensions overflow ptrdiff_t");
    }
    return a * b;
}

void validate_full_batches(const Options &opts) {
    for (const int bs : opts.blocks) {
        for (const int be : opts.exec_blocks) {
            if (be <= bs && bs % be != 0) {
                throw std::runtime_error("full batches require B_save % B_exec == 0: B_save=" +
                                         std::to_string(bs) + ", B_exec=" + std::to_string(be));
            }
        }
        for (const int size : opts.sizes) {
            const ptrdiff_t ne = checked_product(checked_product(size, size), size);
            if (ne % bs != 0) {
                throw std::runtime_error("full storage blocks require nelements % B_save == 0: size=" +
                                         std::to_string(size) + ", B_save=" + std::to_string(bs));
            }
        }
    }
}

int thread_id() {
#ifdef SFEM_ENABLE_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void set_threads(const int requested) {
#ifdef SFEM_ENABLE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(requested);
    int actual = 0;
#pragma omp parallel
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    if (actual != requested) throw std::runtime_error("OpenMP did not create the requested number of threads");
#else
    if (requested != 1) throw std::runtime_error("OpenMP is disabled");
#endif
}

void legendre(const int n, const long double z, long double &p, long double &dp) {
    long double p0 = 1, p1 = z;
    for (int k = 2; k <= n; ++k) {
        const long double pk = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = pk;
    }
    p = p1;
    dp = n * (z * p1 - p0) / (z * z - 1);
}

template <int Q>
void gauss_legendre_unit(std::array<scalar_t, Q> &points, std::array<scalar_t, Q> &weights) {
    const long double pi = std::acos(-1.0L);
    for (int i = 0; i < (Q + 1) / 2; ++i) {
        long double z = std::cos(pi * (i + 0.75L) / (Q + 0.5L));
        long double p, dp;
        bool converged = false;
        for (int it = 0; it < 64; ++it) {
            legendre(Q, z, p, dp);
            const long double dz = p / dp;
            z -= dz;
            if (std::abs(dz) <= 8 * std::numeric_limits<long double>::epsilon()) {
                converged = true;
                break;
            }
        }
        if (!converged) throw std::runtime_error("Gauss-Legendre root iteration failed");
        legendre(Q, z, p, dp);
        const scalar_t w = (scalar_t)(1 / ((1 - z * z) * dp * dp));
        points[i] = (scalar_t)((1 - z) / 2);
        points[Q - 1 - i] = (scalar_t)((1 + z) / 2);
        weights[i] = weights[Q - 1 - i] = w;
    }
    // Check the actual scalar_t rule, including its constant integral and nodes.
    for (int q = 0; q < Q; ++q) {
        if (!(points[q] > 0 && points[q] < 1 && weights[q] > 0)) {
            throw std::runtime_error("invalid Gauss-Legendre rule");
        }
    }
    for (int power = 0; power < 2 * Q; ++power) {
        long double integral = 0;
        for (int q = 0; q < Q; ++q) integral += weights[q] * std::pow((long double)points[q], power);
        if (std::abs(integral - 1.0L / (power + 1)) > 256 * std::numeric_limits<scalar_t>::epsilon()) {
            throw std::runtime_error("Gauss-Legendre moment check failed");
        }
    }
}

template <int P>
struct Problem {
    static constexpr int N = P + 1;
    static constexpr int Q = N;
    static constexpr int N3 = N * N * N;
    static constexpr int Q3 = Q * Q * Q;
    sfem::SharedMesh mesh;
    ptrdiff_t nelements{0}, ndofs{0};
    const idx_t *const *elements{nullptr};
    std::array<scalar_t, Q> qx, qw;
    std::array<scalar_t, Q * N> S, D;
    std::vector<real_t> x_global;
    std::vector<scalar_t> global_output;
};

template <int P>
void setup_problem(Problem<P> &p, const int size, const Options &opts,
                   const std::shared_ptr<sfem::Communicator> &comm) {
    const ptrdiff_t ne = checked_product(checked_product(size, size), size);
    checked_product(checked_product(ne, Problem<P>::Q3), 3);
    auto macro = sfem::Mesh::create_hex8_cube(comm, size, size, size, 0, 0, 0, 1, 1, 1);
    if (!macro) throw std::runtime_error("macro mesh creation failed");
    p.mesh = smesh::to_semistructured(P, macro, false, false);
    if (!p.mesh || p.mesh->n_blocks() != 1) throw std::runtime_error("semi-structured mesh creation failed");
    p.nelements = p.mesh->block(0)->n_elements();
    p.ndofs = p.mesh->n_nodes();
    p.elements = p.mesh->block(0)->elements()->data();
    if (p.nelements != ne) throw std::runtime_error("unexpected mesh element count");

    gauss_legendre_unit<Problem<P>::Q>(p.qx, p.qw);
    if (lagrange_eval<scalar_t>(P, p.Q, p.qx.data(), p.S.data()) != SFEM_SUCCESS ||
        lagrange_diff_eval<scalar_t>(P, p.Q, p.qx.data(), p.D.data()) != SFEM_SUCCESS) {
        throw std::runtime_error("basis setup failed");
    }
    p.x_global.resize((size_t)p.ndofs);
    std::mt19937_64 random((unsigned int)opts.seed);
    std::uniform_real_distribution<double> uniform(-1, 1);
    for (auto &value : p.x_global) value = (real_t)uniform(random);
    p.global_output.resize((size_t)checked_product(checked_product(ne, p.Q3), 3));
}

struct ThreadWorkspace {
    scalar_t *element_u;
    real_t *global8_element_u;
    scalar_t *global8;
    scalar_t *batched;
};

// Leave a gap between thread slices, also for unaligned allocation bases.
template <typename T = scalar_t>
ptrdiff_t padded_stride(const ptrdiff_t n) {
    const ptrdiff_t padding = 128 / sizeof(T);
    return ((n + padding - 1) / padding + 1) * padding;
}

template <int P>
void apply_global(Problem<P> &p, const std::vector<ThreadWorkspace> &workspaces) {
    constexpr int N = Problem<P>::N, Q = Problem<P>::Q, Q3 = Problem<P>::Q3;
    const ptrdiff_t component_stride = p.nelements * Q3;
    const idx_t *const *elements = p.elements;
    const real_t *const x = p.x_global.data();
    scalar_t *const output = p.global_output.data();
    const scalar_t *const S = p.S.data();
    const scalar_t *const D = p.D.data();
#pragma omp parallel
    {
        scalar_t *const u = workspaces[thread_id()].element_u;
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < p.nelements; ++e) {
            for (int v = 0; v < Problem<P>::N3; ++v) u[v] = (scalar_t)x[elements[v][e]];
            scalar_t *const gx = output + e * Q3;
            scalar_t *const gy = gx + component_stride;
            scalar_t *const gz = gy + component_stride;
            lagrange_hex_triad_interpolate<N, Q, scalar_t>(D, S, S, u, gx);
            lagrange_hex_triad_interpolate<N, Q, scalar_t>(S, D, S, u, gy);
            lagrange_hex_triad_interpolate<N, Q, scalar_t>(S, S, D, u, gz);
        }
    }
}

template <int P>
void apply_global8(const Problem<P> &p, scalar_t *const output,
                   const std::vector<ThreadWorkspace> &workspaces) {
    constexpr int N = Problem<P>::N, Q = Problem<P>::Q, Q3 = Problem<P>::Q3;
    const ptrdiff_t component_stride = p.nelements * Q3;
    const idx_t *const *elements = p.elements;
    const real_t *const x = p.x_global.data();
    const scalar_t *const S = p.S.data();
    const scalar_t *const D = p.D.data();
#pragma omp parallel
    {
        const auto &local = workspaces[thread_id()];
        real_t *const u = local.global8_element_u;
        scalar_t *const workspace = local.global8;
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < p.nelements; ++e) {
            for (int v = 0; v < Problem<P>::N3; ++v) u[v] = x[elements[v][e]];
            scalar_t *const gx = output + e * Q3;
            scalar_t *const gy = gx + component_stride;
            scalar_t *const gz = gy + component_stride;
            sfem_dss_bxyze_lagrange_gradient<N, Q, 1>(1, S, D, u, gx, gy, gz, workspace);
        }
    }
}

template <int P, int BE>
void apply_batched(const Problem<P> &p, const sfem_dss_bxyze_layout_t &layout,
                   const real_t *const x, scalar_t *const output,
                   const std::vector<ThreadWorkspace> &workspaces) {
    constexpr int N = Problem<P>::N, Q = Problem<P>::Q, Q3 = Problem<P>::Q3;
    const ptrdiff_t component_stride = layout.ntasks * Q3 * BE;
    const scalar_t *const S = p.S.data();
    const scalar_t *const D = p.D.data();
#pragma omp parallel
    {
        scalar_t *const workspace = workspaces[thread_id()].batched;
#pragma omp for schedule(static)
        for (ptrdiff_t eb = 0; eb < layout.nblocks; ++eb) {
            for (int lane = 0; lane < layout.element_block_size; lane += BE) {
                const ptrdiff_t task = eb * layout.chunks_per_block + lane / BE;
                const real_t *const xb = x + eb * layout.block_stride + lane;
                scalar_t *const gx = output + task * Q3 * BE;
                scalar_t *const gy = gx + component_stride;
                scalar_t *const gz = gy + component_stride;
                sfem_dss_bxyze_lagrange_gradient<N, Q, BE>(
                        layout.element_block_size, S, D, xb, gx, gy, gz, workspace);
            }
        }
    }
}

struct Error {
    double absolute{0};
    double scaled{0};
};

template <int P>
Error compare_global8_outputs(const Problem<P> &p, const scalar_t *const output) {
    double absolute = 0, scaled = 0;
    int invalid = 0;
    const ptrdiff_t nvalues = (ptrdiff_t)p.global_output.size();
#pragma omp parallel for schedule(static) reduction(max : absolute, scaled) reduction(| : invalid)
    for (ptrdiff_t i = 0; i < nvalues; ++i) {
        const double reference = p.global_output[i];
        const double value = output[i];
        invalid |= !std::isfinite(reference) || !std::isfinite(value);
        const double difference = std::abs(value - reference);
        absolute = std::max(absolute, difference);
        scaled = std::max(scaled, difference / std::max(1.0, std::abs(reference)));
    }
    const double tolerance = 512 * Problem<P>::N * Problem<P>::N * std::numeric_limits<scalar_t>::epsilon();
    if (invalid || scaled > tolerance) throw std::runtime_error("global9 and global8 gradients disagree");
    return {absolute, scaled};
}

template <int P, int BE>
Error compare_outputs(const Problem<P> &p, const sfem_dss_bxyze_layout_t &layout, const scalar_t *output) {
    constexpr int Q3 = Problem<P>::Q3;
    const ptrdiff_t scalar_stride = p.nelements * Q3;
    const ptrdiff_t batch_stride = layout.ntasks * Q3 * BE;
    double absolute = 0, scaled = 0;
    int invalid = 0;
#pragma omp parallel for schedule(static) reduction(max : absolute, scaled) reduction(| : invalid)
    for (ptrdiff_t e = 0; e < p.nelements; ++e) {
        const ptrdiff_t eb = e / layout.element_block_size;
        const int lane = (int)(e % layout.element_block_size);
        const ptrdiff_t task = eb * layout.chunks_per_block + lane / BE;
        const int b = lane % BE;
        for (int c = 0; c < 3; ++c) {
            for (int q = 0; q < Q3; ++q) {
                const double reference = p.global_output[c * scalar_stride + e * Q3 + q];
                const double value = output[c * batch_stride + (task * Q3 + q) * BE + b];
                invalid |= !std::isfinite(reference) || !std::isfinite(value);
                const double difference = std::abs(value - reference);
                absolute = std::max(absolute, difference);
                scaled = std::max(scaled, difference / std::max(1.0, std::abs(reference)));
            }
        }
    }
    const double tolerance = 512 * Problem<P>::N * Problem<P>::N * std::numeric_limits<scalar_t>::epsilon();
    if (invalid || scaled > tolerance) throw std::runtime_error("global and BXYZE gradients disagree");
    return {absolute, scaled};
}

void observe_output(const scalar_t *output) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(output) : "memory");
#else
    (void)output;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

template <class Apply>
double time_apply(Apply apply, const scalar_t *output) {
    const double tick = smesh::time_seconds();
    apply();
    observe_output(output);
    return smesh::time_seconds() - tick;
}

struct Timings {
    double minimum;
    double median;
};

Timings summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    const double median = n % 2 ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2;
    if (!(samples.front() > 0)) throw std::runtime_error("timer resolution is insufficient for this case");
    return {samples.front(), median};
}

template <int P, int BE>
void run_layout(Problem<P> &p, const int size, const int bs, const int threads, const Options &opts,
                const std::vector<real_t> &x_bxyze, FILE *out) {
    constexpr int N = Problem<P>::N, Q = Problem<P>::Q, Q3 = Problem<P>::Q3;
    sfem_dss_bxyze_layout_t layout;
    if (sfem_dss_bxyze_create_layout(P, p.nelements, bs, BE, &layout) != SFEM_SUCCESS) {
        throw std::runtime_error("invalid BXYZE layout");
    }
    constexpr ptrdiff_t nwork = sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, BE>();
    constexpr ptrdiff_t nwork_global8 = sfem_dss_bxyze_lagrange_gradient_workspace_n_values<N, Q, 1>();
    const ptrdiff_t global_stride = padded_stride(Problem<P>::N3);
    const ptrdiff_t global8_input_stride = padded_stride<real_t>(Problem<P>::N3);
    const ptrdiff_t global8_work_stride = padded_stride(nwork_global8);
    const ptrdiff_t work_stride = padded_stride(nwork);
    const ptrdiff_t output_values = checked_product(checked_product(checked_product(layout.ntasks, Q3), BE), 3);
    std::vector<scalar_t> batched_output((size_t)output_values);
    std::vector<scalar_t> global8_output(p.global_output.size());
    std::vector<scalar_t> global_workspace((size_t)checked_product(threads, global_stride));
    std::vector<real_t> global8_input_workspace((size_t)checked_product(threads, global8_input_stride));
    std::vector<scalar_t> global8_workspace((size_t)checked_product(threads, global8_work_stride));
    std::vector<scalar_t> batched_workspace((size_t)checked_product(threads, work_stride));
    std::vector<ThreadWorkspace> workspaces((size_t)threads);
    for (int t = 0; t < threads; ++t) {
        workspaces[t] = {global_workspace.data() + t * global_stride,
                         global8_input_workspace.data() + t * global8_input_stride,
                         global8_workspace.data() + t * global8_work_stride,
                         batched_workspace.data() + t * work_stride};
    }
    auto global = [&]() { apply_global(p, workspaces); };
    auto global8 = [&]() { apply_global8(p, global8_output.data(), workspaces); };
    auto batched = [&]() { apply_batched<P, BE>(p, layout, x_bxyze.data(), batched_output.data(), workspaces); };

    global();
    global8();
    batched();
    const Error error = compare_outputs<P, BE>(p, layout, batched_output.data());
    const Error error_global8 = compare_global8_outputs(p, global8_output.data());

    // Rotate through every permutation: each method occupies every timing
    // position equally when repeat is a multiple of six.
    constexpr int orders[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                 {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    for (int r = 0; r < opts.warmup; ++r) {
        for (const int method : orders[r % 6]) {
            switch (method) {
                case 0: global(); break;
                case 1: global8(); break;
                case 2: batched(); break;
            }
        }
    }
    std::vector<double> global_samples((size_t)opts.repeat), global8_samples((size_t)opts.repeat),
            batched_samples((size_t)opts.repeat);
    for (int r = 0; r < opts.repeat; ++r) {
        for (const int method : orders[r % 6]) {
            switch (method) {
                case 0: global_samples[r] = time_apply(global, p.global_output.data()); break;
                case 1: global8_samples[r] = time_apply(global8, global8_output.data()); break;
                case 2: batched_samples[r] = time_apply(batched, batched_output.data()); break;
            }
        }
    }
    const Timings g = summarize(global_samples), g8 = summarize(global8_samples), b = summarize(batched_samples);
    const double dofs = 1e-9 * (double)p.ndofs;
    std::fprintf(out,
                 "%d,%d,%d,gauss_legendre,%d,%d,%d,%td,%td,%d,%d,%d,%d,%d,%d,%zu,%zu,%td,%td,"
                 "%.9e,%.9e,%.9e,%.9e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.9e,%.9e,"
                 "%td,%.9e,%.9e,%.6f,%.6f,%.6f,%.6f,%.9e,%.9e\n",
                 P, N, Q, size, size, size, p.nelements, p.ndofs, bs, BE, threads, opts.repeat, opts.warmup,
                 opts.seed, sizeof(real_t), sizeof(scalar_t), nwork, output_values,
                 g.minimum, g.median, b.minimum, b.median,
                 dofs / g.minimum, dofs / g.median, dofs / b.minimum, dofs / b.median,
                 g.minimum / b.minimum, g.median / b.median, error.absolute, error.scaled,
                 nwork_global8, g8.minimum, g8.median, dofs / g8.minimum, dofs / g8.median,
                 g8.minimum / b.minimum, g8.median / b.median, error_global8.absolute, error_global8.scaled);
    if (std::fflush(out)) throw std::runtime_error("CSV write failed");
}

template <int P>
int run_problem(const int size, const Options &opts, FILE *out,
                 const std::shared_ptr<sfem::Communicator> &comm) {
    Problem<P> p;
    setup_problem(p, size, opts, comm);
    int cases = 0;
    for (const int threads : opts.threads) {
        set_threads(threads);
        std::fprintf(stderr, "gradient benchmark: p=%d Q=%d size=%d elements=%td dofs=%td threads=%d\n",
                     P, p.Q, size, p.nelements, p.ndofs, threads);
        for (const int bs : opts.blocks) {
            sfem_dss_bxyze_layout_t storage;
            if (sfem_dss_bxyze_create_layout(P, p.nelements, bs, 1, &storage) != SFEM_SUCCESS) {
                throw std::runtime_error("invalid storage layout");
            }
            const ptrdiff_t nvalues = storage.block_stride * storage.nblocks;
            std::vector<real_t> x_bxyze((size_t)nvalues);
            if (sfem_dss_global_to_bxyze(P, p.nelements, bs, p.elements, p.x_global.data(), x_bxyze.data()) != SFEM_SUCCESS) {
                throw std::runtime_error("global to BXYZE conversion failed");
            }
            for (const int be : opts.exec_blocks) {
                if (be > bs) continue;
                switch (be) {
                    case 1: run_layout<P, 1>(p, size, bs, threads, opts, x_bxyze, out); break;
                    case 2: run_layout<P, 2>(p, size, bs, threads, opts, x_bxyze, out); break;
                    case 4: run_layout<P, 4>(p, size, bs, threads, opts, x_bxyze, out); break;
                    case 8: run_layout<P, 8>(p, size, bs, threads, opts, x_bxyze, out); break;
                    case 16: run_layout<P, 16>(p, size, bs, threads, opts, x_bxyze, out); break;
                    default: throw std::runtime_error("unsupported execution block size");
                }
                ++cases;
            }
        }
    }
    return cases;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(argv[0]); return SFEM_SUCCESS; }
    Options opts;
    if (!parse_options(argc, argv, opts)) { usage(argv[0]); return SFEM_FAILURE; }
    try {
        validate_full_batches(opts);
        auto context = sfem::initialize(argc, argv);
        if (context->communicator()->size() != 1) throw std::runtime_error("benchmark requires one MPI rank");
        set_threads(opts.threads.front());
        std::unique_ptr<FILE, decltype(&std::fclose)> out(std::fopen(opts.output.c_str(), "w"), &std::fclose);
        if (!out) throw std::runtime_error("cannot open CSV output: " + opts.output);
        std::fprintf(out.get(),
                     "degree,N,Q,quadrature,nx,ny,nz,nelements,n_global_dofs,B_save,B_exec,threads,repeat,warmup,seed,"
                     "real_bytes,scalar_bytes,workspace_scalars_per_thread,batched_output_scalars,"
                     "global_tts_min_s,global_tts_median_s,cellwise_tts_min_s,cellwise_tts_median_s,"
                     "global_gdof_s_best,global_gdof_s_median,cellwise_gdof_s_best,cellwise_gdof_s_median,"
                     "speedup_best,speedup_median,max_abs_diff,max_scaled_diff,"
                     "global8_workspace_scalars_per_thread,global8_tts_min_s,global8_tts_median_s,"
                     "global8_gdof_s_best,global8_gdof_s_median,speedup_vs_global8_best,speedup_vs_global8_median,"
                     "global8_max_abs_diff,global8_max_scaled_diff\n");
        int cases = 0;
        for (const int level : opts.levels) {
            for (const int size : opts.sizes) {
                switch (level) {
                    case 2: cases += run_problem<2>(size, opts, out.get(), context->communicator()); break;
                    case 4: cases += run_problem<4>(size, opts, out.get(), context->communicator()); break;
                    case 8: cases += run_problem<8>(size, opts, out.get(), context->communicator()); break;
                }
            }
        }
        std::fprintf(stderr, "Saved %d validated comparisons to %s\n", cases, opts.output.c_str());
        return SFEM_SUCCESS;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "sfem_DSSBXYZEGradientBench: %s\n", error.what());
        return SFEM_FAILURE;
    }
}
