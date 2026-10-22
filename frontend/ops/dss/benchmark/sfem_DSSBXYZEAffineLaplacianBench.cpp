#include "sfem_API.hpp"
#include "sfem_dss_bxyze_affine_laplacian.hpp"
#include "sfem_dss_bxyze_geometry.hpp"
#include "smesh_base.hpp"
#include "smesh_semistructured.hpp"

#include "lagrange.hpp"

#include <algorithm>
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

// Both paths call the public affine volumetric apply (8 + 8 contractions).
// Baseline: B_save = B_exec = 1; candidates: the requested BXYZE layouts.
// Timings include the complete apply and its OpenMP region, before DSS.
// Mesh, quadrature, bases, geometry, dispatch, allocation, conversion and
// validation are outside the timer. Outputs are overwritten on every call.
// The best CSV selects the lowest candidate median in each (p,size,threads)
// sweep; it is a summary of the sweep, not an independent confirmation run.

std::string default_output_path() {
    const std::string source(__FILE__);
    const auto slash = source.find_last_of("/\\");
    return (slash == std::string::npos ? "frontend/ops/dss/benchmark/" : source.substr(0, slash + 1)) +
           "dss_bxyze_affine_laplacian_bench.csv";
}

struct Options {
    std::vector<int> levels{2, 4, 8}, sizes{8, 16}, blocks{4, 8, 16, 32, 64}, exec_blocks{1, 2, 4, 8, 16};
#ifdef SFEM_ENABLE_OPENMP
    std::vector<int> threads{1, 8};
#else
    std::vector<int> threads{1};
#endif
    int repeat{12}, warmup{6}, seed{42};
    std::string output{default_output_path()};
};

bool parse_integer(const char *text, const int minimum, int &value) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < minimum || parsed > std::numeric_limits<int>::max()) return false;
    value = (int)parsed;
    return true;
}

bool parse_list(const char *text, std::vector<int> &values) {
    std::stringstream stream(text);
    std::string item;
    values.clear();
    while (std::getline(stream, item, ',')) {
        int value;
        if (!parse_integer(item.c_str(), 1, value)) return false;
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
                 "Graded affine hexahedra with shear; no DSS or format conversion in timings.\n"
                 "Always compares with B_save=B_exec=1 using the same public apply.\n"
                 "Even repeat counts balance AB/BA order. Full blocks/batches required.\n"
                 "Writes all pairs and a companion file_best.csv with the best medians.\n", name);
}

bool parse_options(const int argc, char **argv, Options &opts) {
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
    for (const int p : opts.levels) if (p != 2 && p != 4 && p != 8) return false;
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

void validate_dimensions(const Options &opts) {
    for (const int size : opts.sizes) {
        const ptrdiff_t ne = checked_product(checked_product(size, size), size);
        for (const int bs : opts.blocks) {
            if (ne % bs) throw std::runtime_error("nelements must be divisible by B_save");
            for (const int be : opts.exec_blocks) {
                if (be <= bs && bs % be) throw std::runtime_error("B_save must be divisible by B_exec");
            }
        }
        for (const int p : opts.levels) {
            checked_product(ne, (ptrdiff_t)(p + 1) * (p + 1) * (p + 1));
            const ptrdiff_t n = checked_product(size, p) + 1;
            checked_product(checked_product(n, n), n);
        }
    }
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

// Generate full-precision rules on [0,1], avoiding the rounded tabulated rules.
void gauss_legendre_unit(const int Q, std::vector<scalar_t> &qx, std::vector<scalar_t> &qw) {
    qx.resize(Q);
    qw.resize(Q);
    const long double pi = std::acos(-1.0L);
    const auto legendre = [Q](const long double z, long double &p, long double &dp) {
        long double p0 = 1, p1 = z;
        for (int k = 2; k <= Q; ++k) {
            const long double pk = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
            p0 = p1;
            p1 = pk;
        }
        p = p1;
        dp = Q * (z * p1 - p0) / (z * z - 1);
    };
    for (int i = 0; i < (Q + 1) / 2; ++i) {
        long double z = std::cos(pi * (i + 0.75L) / (Q + 0.5L)), p, dp;
        bool converged = false;
        for (int k = 0; k < 64; ++k) {
            legendre(z, p, dp);
            const long double dz = p / dp;
            z -= dz;
            if (std::abs(dz) <= 8 * std::numeric_limits<long double>::epsilon()) { converged = true; break; }
        }
        if (!converged) throw std::runtime_error("Gauss-Legendre iteration failed");
        legendre(z, p, dp);
        qx[i] = (scalar_t)((1 - z) / 2);
        qx[Q - 1 - i] = (scalar_t)((1 + z) / 2);
        qw[i] = qw[Q - 1 - i] = (scalar_t)(1 / ((1 - z * z) * dp * dp));
    }
    for (int q = 0; q < Q; ++q) {
        if (!(qx[q] > 0 && qx[q] < 1 && qw[q] > 0)) throw std::runtime_error("invalid quadrature rule");
    }
    for (int power = 0; power < 2 * Q; ++power) {
        long double moment = 0;
        for (int q = 0; q < Q; ++q) moment += qw[q] * std::pow((long double)qx[q], power);
        if (std::abs(moment - 1.0L / (power + 1)) > 256 * std::numeric_limits<scalar_t>::epsilon()) {
            throw std::runtime_error("Gauss-Legendre moment check failed");
        }
    }
}

struct Problem {
    int p, N, Q, size;
    ptrdiff_t nelements, ndofs, cell_dofs;
    sfem::SharedMesh mesh;
    std::vector<scalar_t> qx, qw, S, D;
    std::vector<real_t> x_global;

    Problem(const int degree, const int n, const int seed, const std::shared_ptr<sfem::Communicator> &comm)
        : p(degree), N(p + 1), Q(N), size(n) {
        auto macro = sfem::Mesh::create_hex8_cube(comm, size, size, size, 0, 0, 0, 1, 1, 1);
        if (!macro) throw std::runtime_error("macro mesh creation failed");
        // Grade each coordinate before applying a fixed shear. Since the grading
        // is separable on the macro grid, each individual hex remains affine.
        auto points = macro->points()->data();
        for (ptrdiff_t i = 0; i < macro->n_nodes(); ++i) {
            const double x = points[0][i], y = points[1][i], z = points[2][i];
            const double a = x + 0.125 * x * x, b = 0.75 * y + 0.0625 * y * y, c = 0.5 * z + 0.03125 * z * z;
            points[0][i] = (geom_t)(a + 0.25 * b - 0.125 * c);
            points[1][i] = (geom_t)(b + 0.125 * c);
            points[2][i] = (geom_t)c;
        }
        mesh = smesh::to_semistructured(p, macro, false, false);
        if (!mesh || mesh->n_blocks() != 1) throw std::runtime_error("semi-structured mesh creation failed");
        nelements = mesh->block(0)->n_elements();
        ndofs = mesh->n_nodes();
        cell_dofs = checked_product(nelements, (ptrdiff_t)N * N * N);
        if (nelements != (ptrdiff_t)size * size * size) throw std::runtime_error("unexpected element count");
        gauss_legendre_unit(Q, qx, qw);
        S.resize((size_t)Q * N);
        D.resize(S.size());
        if (lagrange_eval<scalar_t>(p, Q, qx.data(), S.data()) != SFEM_SUCCESS ||
            lagrange_diff_eval<scalar_t>(p, Q, qx.data(), D.data()) != SFEM_SUCCESS) {
            throw std::runtime_error("basis setup failed");
        }
        x_global.resize((size_t)ndofs);
        std::mt19937_64 random((unsigned int)seed);
        std::uniform_real_distribution<double> uniform(-1, 1);
        for (auto &value : x_global) value = (real_t)uniform(random);
    }
};

struct Prepared {
    sfem_dss_bxyze_laplacian_affine_volumetric_t op{};
    std::vector<real_t> x, y;
    std::vector<scalar_t> geometry, workspace;

    Prepared(const Problem &p, const int bs, const int be, const int threads) {
        sfem_dss_bxyze_layout_t layout;
        if (sfem_dss_bxyze_create_layout(p.p, p.nelements, bs, be, &layout) != SFEM_SUCCESS ||
            sfem_dss_bxyze_laplacian_affine_volumetric_setup(&layout, p.Q, &op) != SFEM_SUCCESS) {
            throw std::runtime_error("affine operator setup failed");
        }
        const ptrdiff_t ngeom = sfem_dss_bxyze_affine_geometry_n_values(&layout);
        if (ngeom < 0) throw std::runtime_error("invalid geometry layout");
        geometry.resize((size_t)ngeom);
        const idx_t *const *const elements = p.mesh->block(0)->elements()->data();
        const geom_t *const *const points = p.mesh->points()->data();
        if (sfem_dss_bxyze_affine_geometry(&layout, elements, points, geometry.data()) != SFEM_SUCCESS) {
            throw std::runtime_error("affine geometry setup failed");
        }
        x.resize((size_t)checked_product(layout.nblocks, layout.block_stride));
        y.resize(x.size(), std::numeric_limits<real_t>::quiet_NaN());
        if (sfem_dss_global_to_bxyze(p.p, p.nelements, bs, elements, p.x_global.data(), x.data()) != SFEM_SUCCESS) {
            throw std::runtime_error("global to BXYZE conversion failed");
        }
        workspace.resize((size_t)checked_product(threads, op.workspace_stride));
    }

    int apply(const Problem &p) {
        return sfem_dss_bxyze_laplacian_affine_volumetric_apply(
                &op, p.S.data(), p.D.data(), p.qw.data(), geometry.data(), x.data(), y.data(), workspace.data());
    }
};

struct Error { double absolute{0}, relative{0}; };

Error compare_outputs(const Problem &p, const Prepared &baseline, const Prepared &candidate) {
    const auto &layout = candidate.op.layout;
    const int bs = layout.element_block_size;
    const ptrdiff_t n3 = (ptrdiff_t)p.N * p.N * p.N;
    double absolute = 0, relative = 0;
    int invalid = 0;
#pragma omp parallel for schedule(static) reduction(max : absolute, relative) reduction(| : invalid)
    for (ptrdiff_t e = 0; e < p.nelements; ++e) {
        double difference = 0, norm = 0;
        const ptrdiff_t base = (e / bs) * layout.block_stride + e % bs;
        for (ptrdiff_t v = 0; v < n3; ++v) {
            const double ref = baseline.y[e * n3 + v], value = candidate.y[base + v * bs];
            invalid |= !std::isfinite(ref) || !std::isfinite(value);
            difference = std::max(difference, std::abs(value - ref));
            norm = std::max(norm, std::abs(ref));
        }
        absolute = std::max(absolute, difference);
        relative = std::max(relative, difference / std::max(1.0, norm));
    }
    const double tolerance = 512 * p.N * p.N * std::max(std::numeric_limits<scalar_t>::epsilon(),
                                                       (scalar_t)std::numeric_limits<real_t>::epsilon());
    if (invalid || relative > tolerance) throw std::runtime_error("B=1 and batched nodal outputs disagree");
    return {absolute, relative};
}

void observe_output(const real_t *output) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(output) : "memory");
#else
    (void)output;
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

double time_apply(const Problem &p, Prepared &data) {
    const double tick = smesh::time_seconds();
    const int status = data.apply(p);
    observe_output(data.y.data());
    const double elapsed = smesh::time_seconds() - tick;
    if (status != SFEM_SUCCESS) throw std::runtime_error("affine apply failed");
    return elapsed;
}

struct Timings { double minimum{0}, median{0}; };

Timings summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    if (!(samples.front() > 0)) throw std::runtime_error("insufficient timer resolution");
    return {samples.front(), n % 2 ? samples[n / 2] : 0.5 * (samples[n / 2 - 1] + samples[n / 2])};
}

struct Result {
    int bs, be;
    ptrdiff_t workspace_stride;
    Timings baseline, candidate;
    Error error;
};

Result measure_pair(const Problem &p, Prepared &baseline, const int bs, const int be,
                    const int threads, const Options &opts) {
    Prepared candidate(p, bs, be, threads);
    if (baseline.apply(p) != SFEM_SUCCESS || candidate.apply(p) != SFEM_SUCCESS) {
        throw std::runtime_error("validation apply failed");
    }
    const Error error = compare_outputs(p, baseline, candidate);
    std::vector<double> a((size_t)opts.repeat), b(a.size());
    for (int r = 0; r < opts.warmup; ++r) {
        Prepared &first = r % 2 ? candidate : baseline;
        Prepared &second = r % 2 ? baseline : candidate;
        if (first.apply(p) != SFEM_SUCCESS || second.apply(p) != SFEM_SUCCESS) throw std::runtime_error("warmup failed");
    }
    for (int r = 0; r < opts.repeat; ++r) {
        if (r % 2) { b[r] = time_apply(p, candidate); a[r] = time_apply(p, baseline); }
        else { a[r] = time_apply(p, baseline); b[r] = time_apply(p, candidate); }
    }
    return {bs, be, candidate.op.workspace_stride, summarize(a), summarize(b), error};
}

void write_header(FILE *out) {
    std::fprintf(out,
                 "degree,N,Q,quadrature,geometry,nx,ny,nz,nelements,n_global_dofs,n_cellwise_dofs,"
                 "baseline_B_save,baseline_B_exec,B_save,B_exec,threads,repeat,warmup,seed,real_bytes,scalar_bytes,"
                 "baseline_workspace_scalars_per_thread,workspace_scalars_per_thread,"
                 "baseline_tts_min_s,baseline_tts_median_s,cellwise_tts_min_s,cellwise_tts_median_s,"
                 "baseline_gdof_s_best,baseline_gdof_s_median,cellwise_gdof_s_best,cellwise_gdof_s_median,"
                 "baseline_cell_gdof_s_median,cellwise_cell_gdof_s_median,speedup_best,speedup_median,"
                 "max_abs_diff,max_relative_inf_diff,is_best\n");
}

void write_result(FILE *out, const Problem &p, const Options &opts, const int threads,
                  const ptrdiff_t baseline_stride, const Result &r, const bool best) {
    const double global = 1e-9 * p.ndofs, cell = 1e-9 * p.cell_dofs;
    std::fprintf(out,
                 "%d,%d,%d,gauss_legendre,graded_affine_shear,%d,%d,%d,%td,%td,%td,1,1,%d,%d,%d,%d,%d,%d,%zu,%zu,%td,%td,"
                 "%.9e,%.9e,%.9e,%.9e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.9e,%.9e,%d\n",
                 p.p, p.N, p.Q, p.size, p.size, p.size, p.nelements, p.ndofs, p.cell_dofs,
                 r.bs, r.be, threads, opts.repeat, opts.warmup, opts.seed, sizeof(real_t), sizeof(scalar_t),
                 baseline_stride, r.workspace_stride, r.baseline.minimum, r.baseline.median,
                 r.candidate.minimum, r.candidate.median,
                 global / r.baseline.minimum, global / r.baseline.median,
                 global / r.candidate.minimum, global / r.candidate.median,
                 cell / r.baseline.median, cell / r.candidate.median,
                 r.baseline.minimum / r.candidate.minimum, r.baseline.median / r.candidate.median,
                 r.error.absolute, r.error.relative, (int)best);
}

int run_problem(const Problem &p, const Options &opts, FILE *out, FILE *best_out) {
    int count = 0;
    for (const int threads : opts.threads) {
        set_threads(threads);
        Prepared baseline(p, 1, 1, threads);
        std::vector<Result> results;
        std::fprintf(stderr, "affine Laplacian: p=%d Q=%d size=%d elements=%td dofs=%td threads=%d\n",
                     p.p, p.Q, p.size, p.nelements, p.ndofs, threads);
        for (const int bs : opts.blocks) {
            for (const int be : opts.exec_blocks) {
                if (be <= bs) results.push_back(measure_pair(p, baseline, bs, be, threads, opts));
            }
        }
        const auto best = std::min_element(results.begin(), results.end(), [](const Result &a, const Result &b) {
            return a.candidate.median < b.candidate.median;
        });
        for (const auto &r : results) write_result(out, p, opts, threads, baseline.op.workspace_stride, r, &r == &*best);
        write_result(best_out, p, opts, threads, baseline.op.workspace_stride, *best, true);
        if (std::fflush(out) || std::fflush(best_out)) throw std::runtime_error("CSV write failed");
        std::fprintf(stderr, "  best B_save=%d B_exec=%d: %.6g s, %.3f GDoF/s, %.3fx vs B=1\n",
                     best->bs, best->be, best->candidate.median, 1e-9 * p.ndofs / best->candidate.median,
                     best->baseline.median / best->candidate.median);
        count += (int)results.size();
    }
    return count;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(argv[0]); return SFEM_SUCCESS; }
    Options opts;
    if (!parse_options(argc, argv, opts)) { usage(argv[0]); return SFEM_FAILURE; }
    try {
        validate_dimensions(opts);
        auto context = sfem::initialize(argc, argv);
        if (context->communicator()->size() != 1) throw std::runtime_error("benchmark requires one MPI rank");
        set_threads(opts.threads.front());
        const auto suffix = opts.output.size() >= 4 && opts.output.substr(opts.output.size() - 4) == ".csv";
        const std::string best_path = (suffix ? opts.output.substr(0, opts.output.size() - 4) : opts.output) + "_best.csv";
        using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
        File out(std::fopen(opts.output.c_str(), "w"), &std::fclose);
        File best_out(std::fopen(best_path.c_str(), "w"), &std::fclose);
        if (!out || !best_out) throw std::runtime_error("cannot open benchmark CSV output");
        write_header(out.get());
        write_header(best_out.get());
        int cases = 0;
        for (const int level : opts.levels) {
            for (const int size : opts.sizes) {
                Problem problem(level, size, opts.seed, context->communicator());
                cases += run_problem(problem, opts, out.get(), best_out.get());
            }
        }
        std::fprintf(stderr, "Saved %d validated comparisons to %s\nBest configurations: %s\n",
                     cases, opts.output.c_str(), best_path.c_str());
        return SFEM_SUCCESS;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "sfem_DSSBXYZEAffineLaplacianBench: %s\n", error.what());
        return SFEM_FAILURE;
    }
}
