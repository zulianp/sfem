#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>

#include <mpi.h>

#include "sfem_API.hpp"
#include "sfem_CRS.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_device_buffer.hpp"
#include "smesh_path.hpp"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_cuda_crs_SpMV.hpp"
#endif

static ptrdiff_t crs_num_cols(const sfem::SharedBuffer<sfem::idx_t>& colidx) {
    ptrdiff_t cols = 0;
    for (ptrdiff_t k = 0; k < colidx->size(); ++k) {
        cols = std::max(cols, static_cast<ptrdiff_t>(colidx->data()[k]) + 1);
    }
    return cols;
}

static void bench_synchronize(const sfem::ExecutionSpace es) {
#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        sfem::device_synchronize();
    }
#else
    (void)es;
#endif
}

static void scale_vector(const sfem::ExecutionSpace es, const ptrdiff_t n, const real_t alpha, real_t* const SFEM_RESTRICT x) {
#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        d_scal(n, alpha, x);
        return;
    }
#else
    (void)es;
#endif
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

static sfem::SharedBuffer<real_t> load_input_vector(const smesh::Path&         x_path,
                                                    const ptrdiff_t            cols,
                                                    const sfem::ExecutionSpace es) {
    if (strcmp("gen:ones", x_path.c_str()) == 0) {
        auto                        x      = sfem::create_host_buffer<real_t>(cols);
        real_t* const SFEM_RESTRICT x_data = x->data();
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < cols; ++i) {
            x_data[i] = 1;
        }
        return x;
    }

    auto x = sfem::Buffer<real_t>::from_file(smesh::Path(x_path));
    if (!x) {
        fprintf(stderr, "Failed to read vector from %s\n", x_path.c_str());
        return nullptr;
    }
    if (x->size() != static_cast<size_t>(cols)) {
        fprintf(stderr, "Vector size %zu does not match matrix columns %td\n", x->size(), cols);
        return nullptr;
    }

    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        x = smesh::to_device(x);
    }

    return x;
}

static std::shared_ptr<sfem::CRS<sfem::count_t, sfem::idx_t, real_t>> make_crs_operator(
        const sfem::ExecutionSpace              es,
        const ptrdiff_t                         rows,
        const ptrdiff_t                         cols,
        const sfem::SharedBuffer<sfem::count_t> rowptr,
        const sfem::SharedBuffer<sfem::idx_t>   colidx,
        const sfem::SharedBuffer<real_t>        values,
        const int                               transpose) {
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
#ifdef SFEM_ENABLE_CUDA
        if (transpose) {
            auto crs_host = sfem::h_crs_spmv<sfem::count_t, sfem::idx_t, real_t>(
                    rows, cols, rowptr, colidx, values, static_cast<real_t>(0));
            crs_host = crs_host->transpose();
            return sfem::d_crs_spmv(rows,
                                    cols,
                                    smesh::to_device(crs_host->row_ptr),
                                    smesh::to_device(crs_host->col_idx),
                                    smesh::to_device(crs_host->values),
                                    static_cast<real_t>(0));
        }

        return sfem::d_crs_spmv(
                rows, cols, smesh::to_device(rowptr), smesh::to_device(colidx), smesh::to_device(values), static_cast<real_t>(0));
#else
        fprintf(stderr, "Device execution requires SFEM_ENABLE_CUDA\n");
        return nullptr;
#endif
    }

    auto crs = sfem::h_crs_spmv<sfem::count_t, sfem::idx_t, real_t>(rows, cols, rowptr, colidx, values, static_cast<real_t>(0));
    if (transpose) {
        crs = crs->transpose();
    }
    return crs;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 6) {
        fprintf(stderr, "usage: %s <alpha> <transpose> <crs_folder> <x.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_REPEAT = 1;
    SFEM_READ_ENV(SFEM_REPEAT, atoi);

    auto es = smesh::Env::read("EXECUTION_SPACE_HOST", sfem::EXECUTION_SPACE_HOST);

    const real_t      alpha     = atof(argv[1]);
    const int         transpose = atoi(argv[2]);
    const smesh::Path crs_folder(argv[3]);
    const smesh::Path x_path(argv[4]);
    const smesh::Path output_path(argv[5]);

    double tick = smesh::time_seconds();

    auto rowptr = sfem::Buffer<sfem::count_t>::from_file(crs_folder / "rowptr.raw");
    auto colidx = sfem::Buffer<sfem::idx_t>::from_file(crs_folder / "colidx.raw");
    auto values = sfem::Buffer<real_t>::from_file(crs_folder / "values.raw");

    if (!rowptr || !colidx || !values) {
        fprintf(stderr, "Failed to read CRS from %s\n", crs_folder.c_str());
        return EXIT_FAILURE;
    }

    const ptrdiff_t rows = rowptr->size() - 1;
    const ptrdiff_t cols = crs_num_cols(colidx);
    const ptrdiff_t nnz  = colidx->size();

    auto x = load_input_vector(x_path, cols, es);
    if (!x) {
        return EXIT_FAILURE;
    }

    auto crs = make_crs_operator(es, rows, cols, rowptr, colidx, values, transpose);
    if (!crs) {
        return EXIT_FAILURE;
    }

    sfem::SharedBuffer<real_t> y = sfem::create_buffer<real_t>(rows, es);

    auto blas = sfem::blas<real_t>(es);

    blas->scal(cols, alpha, x->data());

    bench_synchronize(es);
    double spmv_tick = smesh::time_seconds();

    for (int repeat = 0; repeat < SFEM_REPEAT; repeat++) {
        crs->apply(x->data(), y->data());
    }

    bench_synchronize(es);
    double spmv_tock      = smesh::time_seconds();
    double avg_time       = (spmv_tock - spmv_tick) / SFEM_REPEAT;
    double avg_throughput = (rows / avg_time) * (sizeof(real_t) * 1e-9);

    printf("spmv:  %g %g %ld %ld %ld\n", avg_time, avg_throughput, 0l, rows, nnz);

    if (smesh::to_host(y)->to_file(output_path) != SFEM_SUCCESS) {
        fprintf(stderr, "Failed to write output to %s\n", output_path.c_str());
        return EXIT_FAILURE;
    }

    double tock = smesh::time_seconds();
    if (!rank) {
        printf("bench_spmv.exe.cpp (%s)\n", es == sfem::EXECUTION_SPACE_HOST ? "host" : "device");
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
