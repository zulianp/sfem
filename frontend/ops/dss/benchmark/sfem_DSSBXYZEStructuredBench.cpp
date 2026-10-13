#include "sfem_API.hpp"
#include "sfem_aliases.hpp"
#include "sfem_dss_bxyze.hpp"
#include "sfem_dss_bxyze_structured.hpp"
#include "smesh_base.hpp"

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

std::string default_output_path() {
    const std::string source_path(__FILE__);
    const auto        slash = source_path.find_last_of("/\\");

    if (slash != std::string::npos) {
        return source_path.substr(0, slash + 1) + "dss_bxyze_structured_bench.csv";
    }

    return "frontend/ops/dss/benchmark/dss_bxyze_structured_bench.csv";
}

struct Options {
    std::vector<int>       levels{1, 2, 3, 4};
    std::vector<ptrdiff_t> sizes{16, 32, 64};
    std::vector<int>       blocks{1, 4, 8, 16, 32};
    int                    repeat{5};
    int                    warmup{1};
    std::string            output_path{default_output_path()};
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

void apply_environment(Options &opts) {
    const auto levels = parse_int_list(std::getenv("SFEM_DSS_LEVELS"));
    if (!levels.empty()) {
        opts.levels = levels;
    }

    const auto sizes = parse_ptrdiff_list(std::getenv("SFEM_DSS_SIZES"));
    if (!sizes.empty()) {
        opts.sizes = sizes;
    }

    const auto blocks = parse_int_list(std::getenv("SFEM_DSS_BLOCKS"));
    if (!blocks.empty()) {
        opts.blocks = blocks;
    }

    if (const char *repeat = std::getenv("SFEM_DSS_REPEAT")) {
        opts.repeat = std::max(1, std::atoi(repeat));
    } else if (const char *repeat_compat = std::getenv("SFEM_REPEAT")) {
        opts.repeat = std::max(1, std::atoi(repeat_compat));
    }

    if (const char *warmup = std::getenv("SFEM_DSS_WARMUP")) {
        opts.warmup = std::max(0, std::atoi(warmup));
    }

    if (const char *output = std::getenv("SFEM_DSS_OUTPUT")) {
        if (output[0]) {
            opts.output_path = output;
        }
    }
}

void print_usage(const char *const argv0) {
    std::fprintf(stderr,
                 "usage: %s [--levels 1,2,3] [--sizes 16,32,64] [--blocks 1,4,8,16] "
                 "[--repeat N] [--warmup N] [--output path] [--stdout]\n",
                 argv0);
    std::fprintf(stderr,
                 "env: SFEM_DSS_LEVELS, SFEM_DSS_SIZES, SFEM_DSS_BLOCKS, "
                 "SFEM_DSS_REPEAT/SFEM_REPEAT, SFEM_DSS_WARMUP, SFEM_DSS_OUTPUT\n");
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
            opts.output_path = "-";
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
        } else if (arg == "--repeat") {
            opts.repeat = std::max(1, std::atoi(value));
        } else if (arg == "--warmup") {
            opts.warmup = std::max(0, std::atoi(value));
        } else if (arg == "--output") {
            opts.output_path = value;
        } else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    return !opts.levels.empty() && !opts.sizes.empty() && !opts.blocks.empty();
}

ptrdiff_t unique_dofs(const int level, const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    return (nx * level + 1) * (ny * level + 1) * (nz * level + 1);
}

double interface_pairs(const int level, const ptrdiff_t nx, const ptrdiff_t ny, const ptrdiff_t nz) {
    const double n  = (double)level + 1.0;
    const double n2 = n * n;

    return n2 * ((double)nx * (double)ny * (double)(nz - 1) +
                 (double)nx * (double)(ny - 1) * (double)nz +
                 (double)(nx - 1) * (double)ny * (double)nz);
}

void fill_bxyze(const ptrdiff_t nvalues, real_t *const values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nvalues; ++i) {
        values[i] = (real_t)(1.0 + 1e-6 * (double)(i % 104729));
    }
}

void copy_buffer(const ptrdiff_t nvalues, const real_t *const src, real_t *const dst) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nvalues; ++i) {
        dst[i] = src[i];
    }
}

int run_case(FILE *const out, const int level, const ptrdiff_t size, const int bsize, const Options &opts) {
    const ptrdiff_t nx = size;
    const ptrdiff_t ny = size;
    const ptrdiff_t nz = size;
    const ptrdiff_t ne = nx * ny * nz;

    const ptrdiff_t nvalues = sfem_dss_bxyze_n_values(level, ne, bsize);
    if (nvalues <= 0) {
        std::fprintf(stderr, "invalid BXYZE size for level=%d size=%ld B=%d\n", level, (long)size, bsize);
        return SFEM_FAILURE;
    }

    auto original = sfem::create_host_buffer<real_t>((size_t)nvalues);
    auto work     = sfem::create_host_buffer<real_t>((size_t)nvalues);

    if (!original || !work) {
        std::fprintf(stderr, "allocation failed for level=%d size=%ld B=%d\n", level, (long)size, bsize);
        return SFEM_FAILURE;
    }

    fill_bxyze(nvalues, original->data());

    for (int r = 0; r < opts.warmup; ++r) {
        copy_buffer(nvalues, original->data(), work->data());
        if (sfem_dss_bxyze_structured_inplace(level, nx, ny, nz, bsize, work->data()) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
    }

    double min_seconds = 1e100;
    double max_seconds = 0;
    double sum_seconds = 0;

    for (int r = 0; r < opts.repeat; ++r) {
        copy_buffer(nvalues, original->data(), work->data());

        const double tick = smesh::time_seconds();
        const int    err  = sfem_dss_bxyze_structured_inplace(level, nx, ny, nz, bsize, work->data());
        const double dt   = smesh::time_seconds() - tick;

        if (err != SFEM_SUCCESS) {
            return err;
        }

        min_seconds = std::min(min_seconds, dt);
        max_seconds = std::max(max_seconds, dt);
        sum_seconds += dt;
    }

    const double avg_seconds   = sum_seconds / opts.repeat;
    const double ncell_dofs    = (double)nvalues;
    const double nunique       = (double)unique_dofs(level, nx, ny, nz);
    const double pairs         = interface_pairs(level, nx, ny, nz);
    const double useful_bytes  = pairs * 4.0 * (double)sizeof(real_t);
    const double gdof_cell_s   = 1e-9 * ncell_dofs / min_seconds;
    const double gdof_s_unique = 1e-9 * nunique / min_seconds;
    const double gb_s_useful   = 1e-9 * useful_bytes / min_seconds;

    std::fprintf(out,
                 "%d,%ld,%ld,%ld,%d,%d,%d,%ld,%ld,%.0f,%.0f,%.9e,%.9e,%.9e,%.6f,%.6f,%.6f\n",
                 level,
                 (long)nx,
                 (long)ny,
                 (long)nz,
                 bsize,
                 opts.repeat,
                 opts.warmup,
                 (long)ne,
                 (long)nvalues,
                 nunique,
                 pairs,
                 min_seconds,
                 avg_seconds,
                 max_seconds,
                 gdof_cell_s,
                 gdof_s_unique,
                 gb_s_useful);

    return SFEM_SUCCESS;
}

}  // namespace

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize(argc, argv);

    if (ctx->communicator()->size() != 1) {
        std::fprintf(stderr, "sfem_DSSBXYZEStructuredBench: MPI execution is not supported yet\n");
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
                         "sfem_DSSBXYZEStructuredBench: failed to open %s: %s\n",
                         opts.output_path.c_str(),
                         std::strerror(errno));
            return SFEM_FAILURE;
        }

        output_file.reset(file);
        out = file;
        std::fprintf(stderr, "sfem_DSSBXYZEStructuredBench: writing %s\n", opts.output_path.c_str());
    }

    std::fprintf(out,
                 "level,nx,ny,nz,B,repeat,warmup,nelements,nvalues,nunique,interface_pairs,"
                 "tts_min_s,tts_avg_s,tts_max_s,gdof_cell_s,gdof_s_unique,gb_s_useful\n");

    for (const int level : opts.levels) {
        for (const ptrdiff_t size : opts.sizes) {
            for (const int bsize : opts.blocks) {
                if (run_case(out, level, size, bsize, opts) != SFEM_SUCCESS) {
                    return SFEM_FAILURE;
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
