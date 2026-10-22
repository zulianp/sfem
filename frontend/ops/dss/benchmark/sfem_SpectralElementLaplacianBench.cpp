#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_SpectralElementLaplacian.hpp"
#include "sfem_aliases.hpp"
#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_laplacian.hpp"
#include "sfem_dss_bxyze_spectral_laplacian.hpp"
#include "smesh_base.hpp"
#include "smesh_mesh.hpp"
#include "smesh_semistructured.hpp"
#include "smesh_sshex8.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile double spectral_laplacian_bench_sink = 0;

std::string output_path_next_to_source(const char *const filename) {
    const std::string source_path(__FILE__);
    const auto        slash = source_path.find_last_of("/\\");

    if (slash != std::string::npos) {
        return source_path.substr(0, slash + 1) + filename;
    }

    return std::string("frontend/ops/dss/benchmark/") + filename;
}

std::string default_output_path() {
    return output_path_next_to_source("spectral_element_laplacian_bench.csv");
}

std::string default_fixed_dofs_output_path() {
    return output_path_next_to_source("spectral_element_laplacian_fixed_dofs_bench.csv");
}

struct Options {
    std::vector<int>       levels{2, 4, 8};
    std::vector<ptrdiff_t> sizes{16, 32, 64};
    std::vector<int>       blocks{64};
    std::vector<int>       exec_blocks{64};
    int                    repeat{5};
    int                    warmup{1};
    bool                   hierarchical_ordering{false};
    bool                   use_gll{false};
    bool                   fixed_dofs{false};
    ptrdiff_t              fixed_resolution{128};
    std::string            output_path{default_output_path()};
    bool                   output_path_explicit{false};
};

std::vector<int> parse_int_list(const char *const text) {
    std::vector<int> values;

    if (!text || !text[0]) {
        return values;
    }

    std::stringstream ss(text);
    std::string       item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::atoi(item.c_str()));
        }
    }

    return values;
}

std::vector<ptrdiff_t> parse_ptrdiff_list(const char *const text) {
    std::vector<ptrdiff_t> values;

    if (!text || !text[0]) {
        return values;
    }

    std::stringstream ss(text);
    std::string       item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back((ptrdiff_t)std::atoll(item.c_str()));
        }
    }

    return values;
}

bool parse_bool(const char *const text) {
    if (!text || !text[0]) {
        return false;
    }

    return std::atoi(text) != 0 || std::strcmp(text, "true") == 0 || std::strcmp(text, "on") == 0 ||
           std::strcmp(text, "yes") == 0;
}

void apply_environment(Options &opts) {
    const auto levels = parse_int_list(std::getenv("SFEM_SPECTRAL_LAPLACIAN_LEVELS"));
    if (!levels.empty()) {
        opts.levels = levels;
    }

    const auto sizes = parse_ptrdiff_list(std::getenv("SFEM_SPECTRAL_LAPLACIAN_SIZES"));
    if (!sizes.empty()) {
        opts.sizes = sizes;
    }

    const auto blocks = parse_int_list(std::getenv("SFEM_SPECTRAL_LAPLACIAN_BLOCKS"));
    if (!blocks.empty()) {
        opts.blocks = blocks;
    }

    const auto exec_blocks = parse_int_list(std::getenv("SFEM_SPECTRAL_LAPLACIAN_EXEC_BLOCKS"));
    if (!exec_blocks.empty()) {
        opts.exec_blocks = exec_blocks;
    }

    if (const char *repeat = std::getenv("SFEM_SPECTRAL_LAPLACIAN_REPEAT")) {
        opts.repeat = std::max(1, std::atoi(repeat));
    } else if (const char *repeat_compat = std::getenv("SFEM_REPEAT")) {
        opts.repeat = std::max(1, std::atoi(repeat_compat));
    }

    if (const char *warmup = std::getenv("SFEM_SPECTRAL_LAPLACIAN_WARMUP")) {
        opts.warmup = std::max(0, std::atoi(warmup));
    }

    if (const char *hierarchical = std::getenv("SFEM_SPECTRAL_LAPLACIAN_HIERARCHICAL")) {
        opts.hierarchical_ordering = parse_bool(hierarchical);
    }

    if (const char *gll = std::getenv("SFEM_SPECTRAL_LAPLACIAN_GLL")) {
        opts.use_gll = parse_bool(gll);
    }

    if (const char *fixed_dofs = std::getenv("SFEM_SPECTRAL_LAPLACIAN_FIXED_DOFS")) {
        opts.fixed_dofs = parse_bool(fixed_dofs);
    }

    if (const char *fixed_resolution = std::getenv("SFEM_SPECTRAL_LAPLACIAN_FIXED_RESOLUTION")) {
        opts.fixed_resolution = std::max<ptrdiff_t>(1, (ptrdiff_t)std::atoll(fixed_resolution));
    }

    if (const char *output = std::getenv("SFEM_SPECTRAL_LAPLACIAN_OUTPUT")) {
        if (output[0]) {
            opts.output_path          = output;
            opts.output_path_explicit = true;
        }
    }
}

void print_usage(const char *const argv0) {
    std::fprintf(stderr,
                 "usage: %s [--levels 2,4,8] [--sizes 16,32,64] [--blocks 64] [--exec-blocks 64] "
                 "[--repeat N] [--warmup N] "
                 "[--hierarchical 0|1] [--gll 0|1] [--fixed-dofs] [--fixed-resolution N] "
                 "[--output path] [--stdout]\n",
                 argv0);
    std::fprintf(stderr,
                 "env: SFEM_SPECTRAL_LAPLACIAN_LEVELS, SFEM_SPECTRAL_LAPLACIAN_SIZES, "
                 "SFEM_SPECTRAL_LAPLACIAN_BLOCKS, SFEM_SPECTRAL_LAPLACIAN_EXEC_BLOCKS, "
                 "SFEM_SPECTRAL_LAPLACIAN_REPEAT/SFEM_REPEAT, SFEM_SPECTRAL_LAPLACIAN_WARMUP, "
                 "SFEM_SPECTRAL_LAPLACIAN_HIERARCHICAL, SFEM_SPECTRAL_LAPLACIAN_GLL, "
                 "SFEM_SPECTRAL_LAPLACIAN_FIXED_DOFS, SFEM_SPECTRAL_LAPLACIAN_FIXED_RESOLUTION, "
                 "SFEM_SPECTRAL_LAPLACIAN_OUTPUT\n");
}

bool parse_args(const int argc, char **const argv, Options &opts) {
    apply_environment(opts);

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }

        if (arg == "--stdout") {
            opts.output_path          = "-";
            opts.output_path_explicit = true;
            continue;
        }

        if (arg == "--fixed-dofs") {
            opts.fixed_dofs = true;
            continue;
        }

        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }

        const char *const value = argv[++i];
        if (arg == "--levels") {
            opts.levels = parse_int_list(value);
        } else if (arg == "--sizes") {
            opts.sizes = parse_ptrdiff_list(value);
        } else if (arg == "--blocks") {
            opts.blocks = parse_int_list(value);
        } else if (arg == "--exec-blocks") {
            opts.exec_blocks = parse_int_list(value);
        } else if (arg == "--repeat") {
            opts.repeat = std::max(1, std::atoi(value));
        } else if (arg == "--warmup") {
            opts.warmup = std::max(0, std::atoi(value));
        } else if (arg == "--hierarchical") {
            opts.hierarchical_ordering = parse_bool(value);
        } else if (arg == "--gll") {
            opts.use_gll = parse_bool(value);
        } else if (arg == "--fixed-resolution") {
            opts.fixed_resolution = std::max<ptrdiff_t>(1, (ptrdiff_t)std::atoll(value));
        } else if (arg == "--output") {
            opts.output_path          = value;
            opts.output_path_explicit = true;
        } else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (opts.fixed_dofs && !opts.output_path_explicit) {
        opts.output_path = default_fixed_dofs_output_path();
    }

    return !opts.levels.empty() && !opts.blocks.empty() && !opts.exec_blocks.empty() && (opts.fixed_dofs || !opts.sizes.empty());
}

ptrdiff_t unique_dofs(const int level, const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    return (nx * level + 1) * (ny * level + 1) * (nz * level + 1);
}

void fill_input(const ptrdiff_t n, real_t *const values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        values[i] = (real_t)(0.25 + 1e-6 * (double)(i % 104729));
    }
}

void zero(const ptrdiff_t n, real_t *const values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        values[i] = (real_t)0;
    }
}

double sample_output(const ptrdiff_t n, const real_t *const values) {
    if (n <= 0) {
        return 0;
    }

    return (double)values[0] + (double)values[n / 2] + (double)values[n - 1];
}

struct Timings {
    double min_seconds{1e100};
    double avg_seconds{0};
    double max_seconds{0};
};

int time_full_apply(sfem::Op       &op,
                    const ptrdiff_t ndofs,
                    const real_t   *const x,
                    real_t         *const y,
                    const int       repeat,
                    Timings        &timings) {
    double sum_seconds = 0;

    for (int r = 0; r < repeat; ++r) {
        const double tick = smesh::time_seconds();
        zero(ndofs, y);
        const int err = op.apply(nullptr, x, y);
        const double dt = smesh::time_seconds() - tick;

        if (err != SFEM_SUCCESS) {
            return err;
        }

        timings.min_seconds = std::min(timings.min_seconds, dt);
        timings.max_seconds = std::max(timings.max_seconds, dt);
        sum_seconds += dt;
    }

    timings.avg_seconds = sum_seconds / repeat;
    return SFEM_SUCCESS;
}

int time_kernel_apply(sfem::Op       &op,
                      const ptrdiff_t ndofs,
                      const real_t   *const x,
                      real_t         *const y,
                      const int       repeat,
                      Timings        &timings) {
    zero(ndofs, y);

    double sum_seconds = 0;

    for (int r = 0; r < repeat; ++r) {
        const double tick = smesh::time_seconds();
        const int    err  = op.apply(nullptr, x, y);
        const double dt   = smesh::time_seconds() - tick;

        if (err != SFEM_SUCCESS) {
            return err;
        }

        timings.min_seconds = std::min(timings.min_seconds, dt);
        timings.max_seconds = std::max(timings.max_seconds, dt);
        sum_seconds += dt;
    }

    timings.avg_seconds = sum_seconds / repeat;
    return SFEM_SUCCESS;
}

int time_cellwise_apply(const int                  level,
                        const ptrdiff_t            nx,
                        const ptrdiff_t            ny,
                        const ptrdiff_t            nz,
                        const int                  bsize,
                        const idx_t *const *const  elements,
                        const geom_t *const *const points,
                        const real_t *const        x_bxyze,
                        real_t *const              y_bxyze,
                        const int                  repeat,
                        Timings                   &timings) {
    double sum_seconds = 0;

    for (int r = 0; r < repeat; ++r) {
        const double tick = smesh::time_seconds();
        const int    err  = sfem_dss_bxyze_lagrange_spectral_laplacian_apply(
                level, nx, ny, nz, bsize, elements, points, x_bxyze, y_bxyze);
        const double dt = smesh::time_seconds() - tick;

        if (err != SFEM_SUCCESS) {
            return err;
        }

        timings.min_seconds = std::min(timings.min_seconds, dt);
        timings.max_seconds = std::max(timings.max_seconds, dt);
        sum_seconds += dt;
    }

    timings.avg_seconds = sum_seconds / repeat;
    return SFEM_SUCCESS;
}

int time_cartesian_apply(const sfem_dss_bxyze_laplacian_cartesian_layout_t *const layout,
                         const real_t        hx,
                         const real_t        hy,
                         const real_t        hz,
                         const real_t *const x_bxyze,
                         real_t *const       y_bxyze,
                         const int           repeat,
                         Timings            &timings) {
    double sum_seconds = 0;

    for (int r = 0; r < repeat; ++r) {
        const double tick = smesh::time_seconds();
        const int    err  = sfem_dss_bxyze_laplacian_cartesian_apply_layout(
                layout, hx, hy, hz, (real_t)1, x_bxyze, y_bxyze);
        const double dt = smesh::time_seconds() - tick;

        if (err != SFEM_SUCCESS) {
            return err;
        }

        timings.min_seconds = std::min(timings.min_seconds, dt);
        timings.max_seconds = std::max(timings.max_seconds, dt);
        sum_seconds += dt;
    }

    timings.avg_seconds = sum_seconds / repeat;
    return SFEM_SUCCESS;
}

int run_case(FILE                           *const out,
             const int                            level,
             const ptrdiff_t                      size,
             const int                            bsave,
             const int                            bexec,
             const Options                       &opts,
             const std::shared_ptr<sfem::Communicator> &comm) {
    const ptrdiff_t nx = size;
    const ptrdiff_t ny = size;
    const ptrdiff_t nz = size;

    if (bsave <= 0) {
        std::fprintf(stderr, "skipping invalid B_save=%d\n", bsave);
        return SFEM_SUCCESS;
    }

    if (bexec <= 0 || bexec > bsave) {
        std::fprintf(stderr, "skipping invalid B_save=%d B_exec=%d\n", bsave, bexec);
        return SFEM_SUCCESS;
    }

    if (!opts.use_gll && level != 2 && level != 4 && level != 8) {
        std::fprintf(stderr, "skipping unsupported Lagrange SpectralElementLaplacian level=%d\n", level);
        return SFEM_SUCCESS;
    }

    if (opts.use_gll) {
        std::fprintf(stderr, "cell-wise same-microkernel benchmark is implemented for the Lagrange branch only\n");
        return SFEM_FAILURE;
    }

    if (opts.use_gll && level != 2 && level != 4 && level != 8 && level != 16) {
        std::fprintf(stderr, "skipping unsupported GLL SpectralElementLaplacian level=%d\n", level);
        return SFEM_SUCCESS;
    }

    auto macro_mesh = sfem::Mesh::create_hex8_cube(comm, nx, ny, nz, 0, 0, 0, 1, 1, 1);
    if (!macro_mesh) {
        std::fprintf(stderr, "failed to create HEX8 macro mesh size=%ld\n", (long)size);
        return SFEM_FAILURE;
    }

    auto mesh = smesh::to_semistructured(level, macro_mesh, opts.hierarchical_ordering, opts.use_gll);
    if (!mesh) {
        std::fprintf(stderr, "failed to create semistructured mesh level=%d size=%ld\n", level, (long)size);
        return SFEM_FAILURE;
    }

    auto space = sfem::FunctionSpace::create(mesh, 1);
    if (!space) {
        std::fprintf(stderr, "failed to create FunctionSpace level=%d size=%ld\n", level, (long)size);
        return SFEM_FAILURE;
    }

    auto op = sfem::SpectralElementLaplacian::create(space);
    if (!op || op->initialize() != SFEM_SUCCESS) {
        std::fprintf(stderr, "failed to initialize SpectralElementLaplacian level=%d size=%ld\n", level, (long)size);
        return SFEM_FAILURE;
    }

    const ptrdiff_t nelements      = mesh->block(0)->n_elements();
    const int       nxe            = smesh::sshex8_nxe(level);
    const ptrdiff_t ndofs          = space->n_dofs();
    const ptrdiff_t expected_dofs  = unique_dofs(level, nx, ny, nz);
    const ptrdiff_t element_values = nelements * (ptrdiff_t)nxe;
    const ptrdiff_t bxyze_values   = sfem_dss_bxyze_n_values(level, nelements, bsave);
    const idx_t *const *const elements =
            reinterpret_cast<const idx_t *const *>(mesh->block(0)->elements()->data());
    const geom_t *const *const points = reinterpret_cast<const geom_t *const *>(mesh->points()->data());

    if (bxyze_values <= 0) {
        std::fprintf(stderr, "invalid BXYZE size level=%d size=%ld B_save=%d\n", level, (long)size, bsave);
        return SFEM_FAILURE;
    }

    auto x = sfem::create_host_buffer<real_t>((size_t)ndofs);
    auto y = sfem::create_host_buffer<real_t>((size_t)ndofs);
    auto x_bxyze = sfem::create_host_buffer<real_t>((size_t)bxyze_values);
    auto y_bxyze = sfem::create_host_buffer<real_t>((size_t)bxyze_values);
    if (!x || !y || !x_bxyze || !y_bxyze) {
        std::fprintf(stderr,
                     "allocation failed level=%d size=%ld B=%d ndofs=%ld bxyze_values=%ld\n",
                     level,
                     (long)size,
                     bsave,
                     (long)ndofs,
                     (long)bxyze_values);
        return SFEM_FAILURE;
    }

    fill_input(ndofs, x->data());
    if (sfem_dss_global_to_bxyze(level, nelements, bsave, elements, x->data(), x_bxyze->data()) != SFEM_SUCCESS) {
        std::fprintf(stderr, "failed global->BXYZE conversion level=%d size=%ld B_save=%d\n", level, (long)size, bsave);
        return SFEM_FAILURE;
    }

    const real_t hx = (real_t)1 / (real_t)nx;
    const real_t hy = (real_t)1 / (real_t)ny;
    const real_t hz = (real_t)1 / (real_t)nz;

    sfem_dss_bxyze_laplacian_cartesian_layout_t cartesian_layout;
    if (sfem_dss_bxyze_laplacian_cartesian_create_layout(
                level, nx, ny, nz, bsave, bexec, &cartesian_layout) != SFEM_SUCCESS) {
        std::fprintf(stderr,
                     "failed to create Cartesian BXYZE Laplacian layout level=%d size=%ld B_save=%d B_exec=%d\n",
                     level,
                     (long)size,
                     bsave,
                     bexec);
        return SFEM_FAILURE;
    }

    setenv("SFEM_USE_GLL", opts.use_gll ? "1" : "0", 1);
    for (int r = 0; r < opts.warmup; ++r) {
        zero(ndofs, y->data());
        if (op->apply(nullptr, x->data(), y->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        if (sfem_dss_bxyze_lagrange_spectral_laplacian_apply(
                    level, nx, ny, nz, bsave, elements, points, x_bxyze->data(), y_bxyze->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        if (sfem_dss_bxyze_laplacian_cartesian_apply_layout(
                    &cartesian_layout, hx, hy, hz, (real_t)1, x_bxyze->data(), y_bxyze->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
    }

    Timings full_apply;
    int     err = time_full_apply(*op, ndofs, x->data(), y->data(), opts.repeat, full_apply);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    Timings kernel_apply;
    err = time_kernel_apply(*op, ndofs, x->data(), y->data(), opts.repeat, kernel_apply);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    Timings cellwise_apply;
    err = time_cellwise_apply(level,
                              nx,
                              ny,
                              nz,
                              bsave,
                              elements,
                              points,
                              x_bxyze->data(),
                              y_bxyze->data(),
                              opts.repeat,
                              cellwise_apply);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    Timings cartesian_apply;
    err = time_cartesian_apply(&cartesian_layout,
                               hx,
                               hy,
                               hz,
                               x_bxyze->data(),
                               y_bxyze->data(),
                               opts.repeat,
                               cartesian_apply);
    if (err != SFEM_SUCCESS) {
        return err;
    }

    spectral_laplacian_bench_sink += sample_output(ndofs, y->data());
    spectral_laplacian_bench_sink += sample_output(bxyze_values, y_bxyze->data());

    const double melem_s              = 1e-6 * (double)nelements / full_apply.min_seconds;
    const double gdof_s_unique        = 1e-9 * (double)ndofs / full_apply.min_seconds;
    const double gdof_s_element       = 1e-9 * (double)element_values / full_apply.min_seconds;
    const double kernel_melem_s       = 1e-6 * (double)nelements / kernel_apply.min_seconds;
    const double kernel_gdof_s_unique = 1e-9 * (double)ndofs / kernel_apply.min_seconds;
    const double kernel_gdof_s_element = 1e-9 * (double)element_values / kernel_apply.min_seconds;
    const double cellwise_melem_s       = 1e-6 * (double)nelements / cellwise_apply.min_seconds;
    const double cellwise_gdof_s_unique = 1e-9 * (double)ndofs / cellwise_apply.min_seconds;
    const double cellwise_gdof_s_element = 1e-9 * (double)element_values / cellwise_apply.min_seconds;
    const double cartesian_melem_s       = 1e-6 * (double)nelements / cartesian_apply.min_seconds;
    const double cartesian_gdof_s_unique = 1e-9 * (double)ndofs / cartesian_apply.min_seconds;
    const double cartesian_gdof_s_element = 1e-9 * (double)element_values / cartesian_apply.min_seconds;

    std::fprintf(out,
                 "SpectralElementLaplacian,%s,%d,%ld,%ld,%ld,%d,%d,%d,%d,%d,%d,%ld,%d,%ld,%ld,%ld,%ld,"
                 "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
                 "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                 opts.use_gll ? "gll" : "lagrange",
                 level,
                 (long)nx,
                 (long)ny,
                 (long)nz,
                 opts.hierarchical_ordering ? 1 : 0,
                 opts.use_gll ? 1 : 0,
                 bsave,
                 bexec,
                 opts.repeat,
                 opts.warmup,
                 (long)nelements,
                 nxe,
                 (long)element_values,
                 (long)ndofs,
                 (long)expected_dofs,
                 (long)bxyze_values,
                 full_apply.min_seconds,
                 full_apply.avg_seconds,
                 full_apply.max_seconds,
                 kernel_apply.min_seconds,
                 kernel_apply.avg_seconds,
                 kernel_apply.max_seconds,
                 cellwise_apply.min_seconds,
                 cellwise_apply.avg_seconds,
                 cellwise_apply.max_seconds,
                 cartesian_apply.min_seconds,
                 cartesian_apply.avg_seconds,
                 cartesian_apply.max_seconds,
                 gdof_s_unique,
                 gdof_s_element,
                 melem_s,
                 kernel_gdof_s_unique,
                 kernel_gdof_s_element,
                 kernel_melem_s,
                 cellwise_gdof_s_unique,
                 cellwise_gdof_s_element,
                 cellwise_melem_s,
                 cartesian_gdof_s_unique,
                 cartesian_gdof_s_element,
                 cartesian_melem_s);

    return SFEM_SUCCESS;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);

    if (ctx->communicator()->size() != 1) {
        std::fprintf(stderr, "sfem_SpectralElementLaplacianBench: MPI execution is not supported yet\n");
        return SFEM_FAILURE;
    }

    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return SFEM_FAILURE;
    }

    struct FileCloser {
        void operator()(FILE *const file) const {
            if (file && file != stdout) {
                std::fclose(file);
            }
        }
    };

    std::unique_ptr<FILE, FileCloser> output_file;
    FILE                             *out = stdout;

    if (opts.output_path != "-") {
        FILE *const file = std::fopen(opts.output_path.c_str(), "w");
        if (!file) {
            std::fprintf(stderr,
                         "sfem_SpectralElementLaplacianBench: failed to open %s: %s\n",
                         opts.output_path.c_str(),
                         std::strerror(errno));
            return SFEM_FAILURE;
        }

        output_file.reset(file);
        out = file;
        std::fprintf(stderr, "sfem_SpectralElementLaplacianBench: writing %s\n", opts.output_path.c_str());
    }

    std::fprintf(out,
                 "operator,basis,level,nx,ny,nz,hierarchical,use_gll,B,B_exec,repeat,warmup,"
                 "nelements,nxe,n_element_dofs,nunique,expected_nunique,n_bxyze_values,"
                 "tts_apply_min_s,tts_apply_avg_s,tts_apply_max_s,"
                 "tts_kernel_min_s,tts_kernel_avg_s,tts_kernel_max_s,"
                 "tts_cellwise_min_s,tts_cellwise_avg_s,tts_cellwise_max_s,"
                 "tts_cartesian_min_s,tts_cartesian_avg_s,tts_cartesian_max_s,"
                 "gdof_s_unique,gdof_s_element,melem_s,"
                 "kernel_gdof_s_unique,kernel_gdof_s_element,kernel_melem_s,"
                 "cellwise_gdof_s_unique,cellwise_gdof_s_element,cellwise_melem_s,"
                 "cartesian_gdof_s_unique,cartesian_gdof_s_element,cartesian_melem_s\n");

    if (opts.fixed_dofs) {
        int n_cases = 0;
        for (const int level : opts.levels) {
            if (level <= 0) {
                std::fprintf(stderr, "skipping invalid level=%d\n", level);
                continue;
            }

            if (opts.fixed_resolution % level != 0) {
                std::fprintf(stderr,
                             "skipping level=%d: fixed_resolution=%ld is not divisible by level\n",
                             level,
                             (long)opts.fixed_resolution);
                continue;
            }

            const ptrdiff_t size = opts.fixed_resolution / level;
            if (size <= 0) {
                std::fprintf(stderr, "skipping level=%d: computed size=%ld\n", level, (long)size);
                continue;
            }

            for (const int bsize : opts.blocks) {
                for (const int bexec : opts.exec_blocks) {
                    if (run_case(out, level, size, bsize, bexec, opts, ctx->communicator()) != SFEM_SUCCESS) {
                        return SFEM_FAILURE;
                    }
                }
            }
            ++n_cases;
        }

        if (n_cases == 0) {
            std::fprintf(stderr,
                         "sfem_SpectralElementLaplacianBench: no fixed-DoF cases generated "
                         "(fixed_resolution=%ld)\n",
                         (long)opts.fixed_resolution);
            return SFEM_FAILURE;
        }

    } else {
        for (const int level : opts.levels) {
            for (const ptrdiff_t size : opts.sizes) {
                for (const int bsize : opts.blocks) {
                    for (const int bexec : opts.exec_blocks) {
                        if (run_case(out, level, size, bsize, bexec, opts, ctx->communicator()) != SFEM_SUCCESS) {
                            return SFEM_FAILURE;
                        }
                    }
                }
            }
        }
    }

    std::fprintf(stderr, "sfem_SpectralElementLaplacianBench: sink=%g\n", spectral_laplacian_bench_sink);

    return SFEM_SUCCESS;
}
