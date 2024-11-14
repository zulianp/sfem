#include "matrixio_array.h"
#include "sfem_defs.h"

#include "sfem_mask.h"

#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc != 4) {
        fprintf(stderr, "usage: %s <n> <input> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    int size;
    MPI_Comm_size(comm, &size);

    if (size > 1) {
        fprintf(stderr, "No parallel!\n");
        return SFEM_FAILURE;
    }

    int SFEM_SKIP_WRITE = 0;
    SFEM_READ_ENV(SFEM_SKIP_WRITE, atoi);

    const ptrdiff_t n = atol(argv[1]);
    const char *path_input = argv[2];
    const char *path_output = argv[3];

    idx_t *data = nullptr;
    ptrdiff_t nlocal = 0, nglobal = 0;
    if (array_create_from_file(
                comm, path_input, SFEM_MPI_IDX_T, (void **)&data, &nlocal, &nglobal)) {
        return SFEM_FAILURE;
    }

    mask_t *m = mask_create(n);

    double tick = MPI_Wtime();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nlocal; i++) {
        mask_set(data[i], m);
    }

    double tock = MPI_Wtime();
    double elapsed = tock - tick;

    ptrdiff_t output_bytes = mask_count(n) * sizeof(mask_t);
    ptrdiff_t intput_bytes = nlocal * sizeof(idx_t);
    printf("Mask set TTS %g [s] BW %g [GB/s]\n",
           elapsed,
           1e-9 * (intput_bytes + output_bytes) / elapsed);

    if (0) {
        for (ptrdiff_t i = 0; i < n; i++) {
            int val = mask_get(i, m);
            printf("%ld %d\n", i, val);
        }
        printf("\n");
    }

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < nlocal; i++) {
        int must_be_true = mask_get(data[i], m);
        assert(must_be_true);
    }
#endif

    if (!SFEM_SKIP_WRITE) {
        // There will be an issue with the padding of the mask if run in parallel
        if (array_write(comm, path_output, SFEM_MPI_MASK_T, m, mask_count(n), mask_count(n))) {
            return SFEM_FAILURE;
        }
    }

    mask_destroy(m);

    return MPI_Finalize();
}
