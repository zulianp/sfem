#include "sfem_API.hpp"
#include "sfem_dss_bxyze_affine_laplacian.hpp"
#include "sfem_dss_bxyze_geometry.hpp"
#include "sfem_dss_bxyze_affine_laplacian_inline.hpp"
#include "sfem_dss_bxyze_structured.hpp"
#include "smesh_base.hpp"
#include "smesh_semistructured.hpp"

#include "lagrange.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <iomanip>
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

// Same precomputed S, D, weights and six affine metric components in both paths.
// Global8: gather, eight-contraction gradient, metric, eight-contraction adjoint,
// then scatter. Structured indexing computes global IDs from the logical grid;
// the original elements-based path remains available for comparison. Output
// zeroing is timed; with multiple threads, shared element-boundary nodes use
// atomic scatter while element-interior nodes can be updated directly.
// Cellwise: public apply, measured alone and followed immediately by structured
// DSS. The latter produces the same assembled values, replicated in cell copies.
// Setup, allocation, conversion and validation are outside all timers. Three
// methods run in all six orders. Best apply and apply+DSS medians are selected
// independently within each sweep, without an independent confirmation run.

std::string default_output_path(const bool structured_indexing, const bool boundary_atomics) {
    const std::string source(__FILE__);
    const auto slash = source.find_last_of("/\\");
    return (slash == std::string::npos ? "frontend/ops/dss/benchmark/" : source.substr(0, slash + 1)) +
           "dss_bxyze_affine_laplacian_global" + (structured_indexing ? "_structured" : "") +
           (boundary_atomics ? "_boundary" : "") + "_bench.csv";
}

struct Options {
    std::vector<int> levels{2, 4, 8}, sizes{16, 64}, blocks{1, 2, 4, 8, 16, 32, 64}, exec_blocks{1, 2, 4, 8, 16};
#ifdef SFEM_ENABLE_OPENMP
    std::vector<int> threads{1, 8};
#else
    std::vector<int> threads{1};
#endif
    int repeat{0}, warmup{-1}, seed{42};  // Unset counts select the per-mesh defaults in main.
    bool structured_indexing{true};
    bool boundary_atomics{true};
    std::string output;
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
                 "usage: %s [--levels 2,4,8] [--sizes 16,64] [--blocks 1,2,4,8,16,32,64]\n"
                 "          [--exec-blocks 1,2,4,8,16] [--threads 1,8] [--repeat R]\n"
                 "          [--warmup W] [--seed 42] [--global-indexing structured|elements]\n"
                 "          [--global-scatter boundary|all] [--output file.csv]\n"
                 "Q=p+1 Gauss-Legendre; sizes are macro-elements per axis.\n"
                 "Defaults: 64^3 uses 6 repeats / 1 warmup; other sizes use 12 / 6.\n"
                 "Explicit --repeat / --warmup override these counts for all sizes.\n"
                 "Graded affine hexahedra with shear; format conversion is outside timings.\n"
                 "Global8 uses structured IDs and boundary-only atomics by default.\n"
                 "Global8 (zero/gather/compute/scatter) versus cellwise apply and apply+DSS.\n"
                 "Repeat counts divisible by six balance all orders. Full blocks/batches required.\n"
                 "Writes all configurations and file_best.csv with separate apply/DSS optima.\n", name);
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
        else if (arg == "--global-indexing") {
            if (std::string(value) == "structured") opts.structured_indexing = true;
            else if (std::string(value) == "elements") opts.structured_indexing = false;
            else valid = false;
        }
        else if (arg == "--global-scatter") {
            if (std::string(value) == "boundary") opts.boundary_atomics = true;
            else if (std::string(value) == "all") opts.boundary_atomics = false;
            else valid = false;
        }
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
    if (opts.output.empty()) opts.output = default_output_path(opts.structured_indexing, opts.boundary_atomics);
    return has_pair;
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

// The benchmark's HEX8 elements have x-fastest structured ordering, but
// to_semistructured numbers global nodes by entity (vertices, edges, faces,
// interiors). This index applies to a separately permuted global vector.
ptrdiff_t structured_gid(const Problem &p, const ptrdiff_t e, const int v) {
    const ptrdiff_t ex = e % p.size;
    const ptrdiff_t ey = (e / p.size) % p.size;
    const ptrdiff_t ez = e / ((ptrdiff_t)p.size * p.size);
    const int ix = v % p.N, iy = (v / p.N) % p.N, iz = v / (p.N * p.N);
    const ptrdiff_t n = (ptrdiff_t)p.p * p.size + 1;
    return (p.p * ex + ix) + n * ((p.p * ey + iy) + n * (p.p * ez + iz));
}

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


template <typename T>
ptrdiff_t padded_stride(const ptrdiff_t values) {
    const ptrdiff_t gap = std::max<ptrdiff_t>(1, 128 / sizeof(T));
    return ((values + gap - 1) / gap + 1) * gap;
}

struct GlobalPrepared;
using GlobalKernel = void (*)(const Problem &, GlobalPrepared &);

template <int N, bool AtomicScatter, bool StructuredIndexing, bool BoundaryAtomics>
void global8_apply_tpl(const Problem &p, GlobalPrepared &data);

template <int N>
GlobalKernel select_global_kernel(const bool atomic, const bool structured, const bool boundary) {
    if (structured) {
        if (atomic) return boundary ? global8_apply_tpl<N, true, true, true>
                                    : global8_apply_tpl<N, true, true, false>;
        return global8_apply_tpl<N, false, true, false>;
    }
    if (atomic) return boundary ? global8_apply_tpl<N, true, false, true>
                                : global8_apply_tpl<N, true, false, false>;
    return global8_apply_tpl<N, false, false, false>;
}

struct GlobalPrepared {
    GlobalKernel kernel{nullptr};
    bool structured_indexing, boundary_atomics;
    ptrdiff_t workspace_stride, nodal_stride, index_stride;
    std::vector<real_t> x_structured, y, nodal_workspace;
    std::vector<idx_t> index_workspace;
    std::vector<scalar_t> geometry, workspace;

    GlobalPrepared(const Problem &p, const int threads, const bool structured, const bool boundary)
        : structured_indexing(structured), boundary_atomics(boundary) {
        sfem_dss_bxyze_layout_t layout;
        if (sfem_dss_bxyze_create_layout(p.p, p.nelements, 1, 1, &layout) != SFEM_SUCCESS) {
            throw std::runtime_error("global geometry layout setup failed");
        }
        geometry.resize((size_t)sfem_dss_bxyze_affine_geometry_n_values(&layout));
        if (sfem_dss_bxyze_affine_geometry(&layout, p.mesh->block(0)->elements()->data(),
                                          p.mesh->points()->data(), geometry.data()) != SFEM_SUCCESS) {
            throw std::runtime_error("global affine geometry setup failed");
        }
        if (structured_indexing) {
            const ptrdiff_t n = (ptrdiff_t)p.p * p.size + 1;
            if (p.ndofs != checked_product(checked_product(n, n), n)) {
                throw std::runtime_error("structured global node count mismatch");
            }
            const idx_t invalid = std::numeric_limits<idx_t>::max();
            const idx_t *const *const elements = p.mesh->block(0)->elements()->data();
            std::vector<idx_t> lex_to_old((size_t)p.ndofs, invalid);
            for (ptrdiff_t e = 0; e < p.nelements; ++e) {
                for (int v = 0; v < p.N * p.N * p.N; ++v) {
                    const ptrdiff_t gid = structured_gid(p, e, v);
                    const idx_t old = elements[v][e];
                    if (gid < 0 || gid >= p.ndofs || (ptrdiff_t)old < 0 || (ptrdiff_t)old >= p.ndofs ||
                        (lex_to_old[(size_t)gid] != invalid && lex_to_old[(size_t)gid] != old)) {
                        throw std::runtime_error("structured global numbering disagrees with mesh connectivity");
                    }
                    lex_to_old[(size_t)gid] = old;
                }
            }
            x_structured.resize((size_t)p.ndofs);
            std::vector<bool> seen_old((size_t)p.ndofs, false);
            for (ptrdiff_t gid = 0; gid < p.ndofs; ++gid) {
                const idx_t old = lex_to_old[(size_t)gid];
                if (old == invalid) throw std::runtime_error("incomplete structured global numbering");
                if (seen_old[(size_t)old]) throw std::runtime_error("structured global numbering is not a permutation");
                seen_old[(size_t)old] = true;
                x_structured[(size_t)gid] = p.x_global[(size_t)old];
            }
        }
        workspace_stride = sfem_dss_bxyze_laplacian_affine_volumetric_workspace_n_values(&layout, p.Q);
        if (workspace_stride < 0) throw std::runtime_error("global workspace setup failed");
        const ptrdiff_t n3 = (ptrdiff_t)p.N * p.N * p.N;
        nodal_stride = padded_stride<real_t>(2 * n3);
        index_stride = padded_stride<idx_t>(n3);
        workspace.resize((size_t)checked_product(threads, workspace_stride));
        nodal_workspace.resize((size_t)checked_product(threads, nodal_stride));
        index_workspace.resize((size_t)checked_product(threads, index_stride));
        y.resize((size_t)p.ndofs, std::numeric_limits<real_t>::quiet_NaN());
        // Degree, indexing and scatter policy are selected outside the timed apply.
        switch (p.p) {
            case 2: kernel = select_global_kernel<3>(threads > 1, structured, boundary); break;
            case 4: kernel = select_global_kernel<5>(threads > 1, structured, boundary); break;
            case 8: kernel = select_global_kernel<9>(threads > 1, structured, boundary); break;
            default: throw std::runtime_error("unsupported global degree");
        }
    }

    void apply(const Problem &p) { kernel(p, *this); }
};

template <int N, bool AtomicScatter, bool StructuredIndexing, bool BoundaryAtomics>
void global8_apply_tpl(const Problem &p, GlobalPrepared &data) {
    constexpr int Q = N, N3 = N * N * N, Q3 = Q * Q * Q;
    const real_t *const x = StructuredIndexing ? data.x_structured.data() : p.x_global.data();
    real_t *const y = data.y.data();
    const scalar_t *const S = p.S.data(), *const D = p.D.data(), *const qw = p.qw.data();
    const ptrdiff_t grid_n = (ptrdiff_t)p.p * p.size + 1;
    const ptrdiff_t grid_n2 = grid_n * grid_n;
#pragma omp parallel
    {
        int tid = 0;
#ifdef SFEM_ENABLE_OPENMP
        tid = omp_get_thread_num();
#endif
        real_t *const u = data.nodal_workspace.data() + tid * data.nodal_stride;
        real_t *const element_y = u + N3;
        idx_t *const ev = data.index_workspace.data() + tid * data.index_stride;
        scalar_t *const gx = data.workspace.data() + tid * data.workspace_stride;
        scalar_t *const gy = gx + Q3, *const gz = gy + Q3, *const scratch = gz + Q3;
        // Every call computes y=A*x rather than accumulating previous calls.
        // The implicit barrier completes zeroing before any scatter.
#pragma omp for schedule(static)
        for (ptrdiff_t i = 0; i < p.ndofs; ++i) y[i] = 0;
#pragma omp for schedule(static)
        for (ptrdiff_t e = 0; e < p.nelements; ++e) {
            if constexpr (StructuredIndexing) {
                const ptrdiff_t ex = e % p.size;
                const ptrdiff_t ey = (e / p.size) % p.size;
                const ptrdiff_t ez = e / ((ptrdiff_t)p.size * p.size);
                const ptrdiff_t origin = p.p * (ex + grid_n * ey + grid_n2 * ez);
                for (int iz = 0; iz < N; ++iz) {
                    for (int iy = 0; iy < N; ++iy) {
                        const ptrdiff_t row = origin + (ptrdiff_t)iy * grid_n + (ptrdiff_t)iz * grid_n2;
                        const int vbase = (iz * N + iy) * N;
                        for (int ix = 0; ix < N; ++ix) {
                            const int v = vbase + ix;
                            ev[v] = (idx_t)(row + ix);
                            u[v] = x[ev[v]];
                        }
                    }
                }
            } else {
                const idx_t *const *const elements = p.mesh->block(0)->elements()->data();
                for (int v = 0; v < N3; ++v) {
                    ev[v] = elements[v][e];
                    u[v] = x[ev[v]];
                }
            }
            sfem_dss_bxyze_lagrange_gradient<N, Q, 1>(1, S, D, u, gx, gy, gz, scratch);
            sfem_dss_bxyze_affine_metric_apply<Q, 1>(1, qw, data.geometry.data() + 6 * e, gx, gy, gz);
            sfem_dss_bxyze_lagrange_gradient_transpose<N, Q, 1>(1, S, D, gx, gy, gz, element_y, scratch);
            if constexpr (AtomicScatter && BoundaryAtomics) {
                // A node strictly inside this element has exactly one writer.
                for (int iz = 0; iz < N; ++iz) {
                    for (int iy = 0; iy < N; ++iy) {
                        const int row = (iz * N + iy) * N;
                        if (iz > 0 && iz < N - 1 && iy > 0 && iy < N - 1) {
#pragma omp atomic update
                            y[ev[row]] += element_y[row];
                            for (int ix = 1; ix < N - 1; ++ix) {
                                y[ev[row + ix]] += element_y[row + ix];
                            }
#pragma omp atomic update
                            y[ev[row + N - 1]] += element_y[row + N - 1];
                        } else {
                            for (int ix = 0; ix < N; ++ix) {
#pragma omp atomic update
                                y[ev[row + ix]] += element_y[row + ix];
                            }
                        }
                    }
                }
            } else {
                for (int v = 0; v < N3; ++v) {
                    if constexpr (AtomicScatter) {
#pragma omp atomic update
                        y[ev[v]] += element_y[v];
                    } else {
                        y[ev[v]] += element_y[v];
                    }
                }
            }
        }
    }
}

int apply_dss(const Problem &p, Prepared &data) {
    const int status = data.apply(p);
    if (status != SFEM_SUCCESS) return status;
    return sfem_dss_bxyze_structured_inplace(
            p.p, p.size, p.size, p.size, data.op.layout.element_block_size, data.y.data());
}

struct Error { double absolute{0}, relative{0}; };

// Compare all cell copies, including every face/edge/corner. Before DSS use the
// B=1 local reference; after DSS use the assembled global reference at its gid.
// Never scatter-add an already assembled cellwise result (that would count
// shared nodes multiple times).
Error compare_outputs(const Problem &p, const Prepared &candidate, const real_t *reference,
                      const bool assembled, const bool structured_global = false) {
    const auto &layout = candidate.op.layout;
    const int bs = layout.element_block_size;
    const ptrdiff_t n3 = (ptrdiff_t)p.N * p.N * p.N;
    const idx_t *const *const elements = p.mesh->block(0)->elements()->data();
    double absolute = 0, relative = 0;
    int invalid = 0;
#pragma omp parallel for schedule(static) reduction(max : absolute, relative) reduction(| : invalid)
    for (ptrdiff_t e = 0; e < p.nelements; ++e) {
        double difference = 0, norm = 0;
        const ptrdiff_t base = (e / bs) * layout.block_stride + e % bs;
        for (ptrdiff_t v = 0; v < n3; ++v) {
            const ptrdiff_t gid = assembled ? (structured_global ? structured_gid(p, e, (int)v) : elements[v][e])
                                            : e * n3 + v;
            const double ref = reference[gid];
            const double value = candidate.y[base + v * bs];
            invalid |= !std::isfinite(ref) || !std::isfinite(value);
            difference = std::max(difference, std::abs(value - ref));
            norm = std::max(norm, std::abs(ref));
        }
        absolute = std::max(absolute, difference);
        relative = std::max(relative, difference / std::max(1.0, norm));
    }
    const double tolerance = 512 * p.N * std::max(std::numeric_limits<scalar_t>::epsilon(),
                                                  (scalar_t)std::numeric_limits<real_t>::epsilon());
    if (invalid || relative > tolerance) {
        throw std::runtime_error(assembled ? "cellwise DSS and global assembly disagree"
                                          : "B=1 and batched local outputs disagree");
    }
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
    Timings global, apply, apply_dss;
    Error local_error, assembled_error;
};

Result measure(const Problem &p, GlobalPrepared &global, const Prepared &local_reference,
               const int bs, const int be, const int threads, const Options &opts) {
    Prepared candidate(p, bs, be, threads);
    if (candidate.apply(p) != SFEM_SUCCESS) throw std::runtime_error("validation apply failed");
    const Error local_error = compare_outputs(p, candidate, local_reference.y.data(), false);
    global.apply(p);
    if (apply_dss(p, candidate) != SFEM_SUCCESS) throw std::runtime_error("validation DSS failed");
    Error assembled_error = compare_outputs(p, candidate, global.y.data(), true, global.structured_indexing);

    const auto run = [&](const int method) {
        if (method == 0) { global.apply(p); return SFEM_SUCCESS; }
        return method == 1 ? candidate.apply(p) : apply_dss(p, candidate);
    };
    constexpr int orders[6][3] = {{0,1,2}, {1,2,0}, {2,0,1}, {2,1,0}, {1,0,2}, {0,2,1}};
    for (int r = 0; r < opts.warmup; ++r) {
        for (const int method : orders[r % 6]) {
            if (run(method) != SFEM_SUCCESS) throw std::runtime_error("warmup failed");
        }
    }
    std::array<std::vector<double>, 3> samples;
    for (auto &values : samples) values.resize((size_t)opts.repeat);
    for (int r = 0; r < opts.repeat; ++r) {
        for (const int method : orders[r % 6]) {
            const double tick = smesh::time_seconds();
            const int status = run(method);
            observe_output(method == 0 ? global.y.data() : candidate.y.data());
            const double elapsed = smesh::time_seconds() - tick;
            if (status != SFEM_SUCCESS) throw std::runtime_error("timed apply failed");
            samples[method][r] = elapsed;
        }
    }
    // Recheck after repeated calls as well, including non-deterministic atomic
    // accumulation order. The input stays fixed; every call overwrites output.
    if (apply_dss(p, candidate) != SFEM_SUCCESS) throw std::runtime_error("final DSS failed");
    const Error final_error = compare_outputs(p, candidate, global.y.data(), true, global.structured_indexing);
    assembled_error.absolute = std::max(assembled_error.absolute, final_error.absolute);
    assembled_error.relative = std::max(assembled_error.relative, final_error.relative);
    return {bs, be, candidate.op.workspace_stride, summarize(samples[0]), summarize(samples[1]),
            summarize(samples[2]), local_error, assembled_error};
}

void write_header(FILE *out) {
    std::fprintf(out,
        "degree,N,Q,quadrature,geometry,nx,ny,nz,nelements,n_global_dofs,n_cellwise_dofs,"
        "B_save,B_exec,threads,repeat,warmup,seed,real_bytes,scalar_bytes,"
        "forward_contractions,transpose_contractions,global_indexing,global_scatter,global_scatter_policy,global_zero_timed,conversion_timed,"
        "global_workspace_scalars_per_thread,global_workspace_reals_per_thread,"
        "global_workspace_indices_per_thread,cellwise_workspace_scalars_per_thread,"
        "global8_tts_min_s,global8_tts_median_s,cellwise_apply_tts_min_s,cellwise_apply_tts_median_s,"
        "cellwise_apply_dss_tts_min_s,cellwise_apply_dss_tts_median_s,"
        "global8_gdof_s_best,global8_gdof_s_median,cellwise_apply_gdof_s_best,cellwise_apply_gdof_s_median,"
        "cellwise_apply_dss_gdof_s_best,cellwise_apply_dss_gdof_s_median,"
        "speedup_apply_best,speedup_apply_median,speedup_apply_dss_best,speedup_apply_dss_median,"
        "local_max_abs_diff,local_max_relative_inf_diff,assembled_max_abs_diff,assembled_max_relative_inf_diff,"
        "is_best_apply,is_best_apply_dss,selection\n");
}

void write_result(FILE *out, const Problem &p, const Options &opts, const int threads,
                  const GlobalPrepared &global, const Result &r, const bool best_apply,
                  const bool best_dss, const char *selection) {
    const double dofs = 1e-9 * p.ndofs;
    std::ostringstream row;
    row << std::scientific << std::setprecision(9)
        << p.p << ',' << p.N << ',' << p.Q << ",gauss_legendre,graded_affine_shear,"
        << p.size << ',' << p.size << ',' << p.size << ',' << p.nelements << ',' << p.ndofs << ',' << p.cell_dofs << ','
        << r.bs << ',' << r.be << ',' << threads << ',' << opts.repeat << ',' << opts.warmup << ',' << opts.seed << ','
        << sizeof(real_t) << ',' << sizeof(scalar_t) << ",8,8,"
        << (global.structured_indexing ? "structured" : "elements") << ','
        << (threads > 1 ? (global.boundary_atomics ? "atomic_boundary" : "atomic_all") : "serial") << ','
        << (global.boundary_atomics ? "boundary" : "all") << ",1,0,"
        << global.workspace_stride << ',' << global.nodal_stride << ',' << global.index_stride << ',' << r.workspace_stride << ','
        << r.global.minimum << ',' << r.global.median << ',' << r.apply.minimum << ',' << r.apply.median << ','
        << r.apply_dss.minimum << ',' << r.apply_dss.median << ','
        << dofs / r.global.minimum << ',' << dofs / r.global.median << ','
        << dofs / r.apply.minimum << ',' << dofs / r.apply.median << ','
        << dofs / r.apply_dss.minimum << ',' << dofs / r.apply_dss.median << ','
        << r.global.minimum / r.apply.minimum << ',' << r.global.median / r.apply.median << ','
        << r.global.minimum / r.apply_dss.minimum << ',' << r.global.median / r.apply_dss.median << ','
        << r.local_error.absolute << ',' << r.local_error.relative << ','
        << r.assembled_error.absolute << ',' << r.assembled_error.relative << ','
        << (int)best_apply << ',' << (int)best_dss << ',' << selection << '\n';
    if (std::fputs(row.str().c_str(), out) < 0) throw std::runtime_error("CSV write failed");
}

int run_problem(const Problem &p, const Options &opts, FILE *out, FILE *best_out) {
    int count = 0;
    for (const int threads : opts.threads) {
        set_threads(threads);
        GlobalPrepared global(p, threads, opts.structured_indexing, opts.boundary_atomics);
        Prepared local_reference(p, 1, 1, threads);
        if (local_reference.apply(p) != SFEM_SUCCESS) throw std::runtime_error("local reference failed");
        std::vector<Result> results;
        std::fprintf(stderr, "affine global8 comparison: p=%d Q=%d size=%d elements=%td dofs=%td threads=%d indexing=%s\n",
                     p.p, p.Q, p.size, p.nelements, p.ndofs, threads,
                     opts.structured_indexing ? "structured" : "elements");
        for (const int bs : opts.blocks) {
            for (const int be : opts.exec_blocks) {
                if (be <= bs) {
                    results.push_back(measure(p, global, local_reference, bs, be, threads, opts));
                    const auto &r = results.back();
                    std::fprintf(stderr, "  B_save=%d B_exec=%d: apply=%.6g s, apply+DSS=%.6g s\n",
                                 bs, be, r.apply.median, r.apply_dss.median);
                }
            }
        }
        const auto best_apply = std::min_element(results.begin(), results.end(), [](const Result &a, const Result &b) {
            return a.apply.median < b.apply.median;
        });
        const auto best_dss = std::min_element(results.begin(), results.end(), [](const Result &a, const Result &b) {
            return a.apply_dss.median < b.apply_dss.median;
        });
        for (const auto &r : results) {
            write_result(out, p, opts, threads, global, r, &r == &*best_apply, &r == &*best_dss, "sweep");
        }
        write_result(best_out, p, opts, threads, global, *best_apply, true, best_apply == best_dss, "best_apply");
        write_result(best_out, p, opts, threads, global, *best_dss, best_apply == best_dss, true, "best_apply_dss");
        if (std::fflush(out) || std::fflush(best_out)) throw std::runtime_error("CSV flush failed");
        std::fprintf(stderr,
                     "  best apply: B_save=%d B_exec=%d, %.6g s, %.3f GDoF/s, %.3fx vs global8\n"
                     "  best total: B_save=%d B_exec=%d, %.6g s, %.3f GDoF/s, %.3fx vs global8\n",
                     best_apply->bs, best_apply->be, best_apply->apply.median, 1e-9 * p.ndofs / best_apply->apply.median,
                     best_apply->global.median / best_apply->apply.median,
                     best_dss->bs, best_dss->be, best_dss->apply_dss.median, 1e-9 * p.ndofs / best_dss->apply_dss.median,
                     best_dss->global.median / best_dss->apply_dss.median);
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
        const std::string best_path = (opts.output.size() >= 4 && opts.output.substr(opts.output.size() - 4) == ".csv"
                                      ? opts.output.substr(0, opts.output.size() - 4) : opts.output) + "_best.csv";
        using File = std::unique_ptr<FILE, decltype(&std::fclose)>;
        File out(std::fopen(opts.output.c_str(), "w"), &std::fclose);
        File best_out(std::fopen(best_path.c_str(), "w"), &std::fclose);
        if (!out || !best_out) throw std::runtime_error("cannot open benchmark CSV output");
        write_header(out.get());
        write_header(best_out.get());
        int cases = 0;
        for (const int level : opts.levels) {
            for (const int size : opts.sizes) {
                Options measurement = opts;
                measurement.repeat = opts.repeat ? opts.repeat : (size == 64 ? 6 : 12);
                measurement.warmup = opts.warmup >= 0 ? opts.warmup : (size == 64 ? 1 : 6);
                Problem problem(level, size, opts.seed, context->communicator());
                cases += run_problem(problem, measurement, out.get(), best_out.get());
            }
        }
        std::fprintf(stderr, "Saved %d validated configurations to %s\nBest configurations: %s\n",
                     cases, opts.output.c_str(), best_path.c_str());
        return SFEM_SUCCESS;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "sfem_DSSBXYZEAffineLaplacianGlobalBench: %s\n", error.what());
        return SFEM_FAILURE;
    }
}
