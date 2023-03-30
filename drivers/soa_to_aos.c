#include <glob.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

typedef int idx_t;
#define MPI_IDX_T MPI_INT

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    const char *help = "usage: %s <soa_pattern> <n_bytes_x_entry> <output_aos.raw>\n";

    if (argc != 4) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *path_soa_pattern = argv[1];
    int n_bytes_x_entry = atoi(argv[2]);
    const char *path_output_aos = argv[3];

    glob_t gl;
    glob(path_soa_pattern, GLOB_MARK, NULL, &gl);

    int n_arrays = gl.gl_pathc;

    printf("n_arrays (%d):\n", n_arrays);
    for (int np = 0; np < n_arrays; np++) {
        printf("%s\n", gl.gl_pathv[np]);
    }

    double tick = MPI_Wtime();

    MPI_Datatype values_mpi_t = MPI_CHAR;
    char **arrays;
    arrays = (char **)malloc(n_arrays * sizeof(char*));

    ptrdiff_t check_bytes = 0;
    ptrdiff_t _nope_, n_bytes = 0;
    for (int np = 0; np < n_arrays; np++) {
        array_create_from_file(comm, gl.gl_pathv[np], values_mpi_t, (void **)&arrays[np], &_nope_, &n_bytes);
        if(!check_bytes) {
            check_bytes = n_bytes;
        } else {
            if(check_bytes != n_bytes) {
                fprintf(stderr, "Bad input! arrays do not have same length %ld != %ld\n", (long)check_bytes, (long)n_bytes);
            }
        }
    }

    ptrdiff_t n_values = n_bytes / n_bytes_x_entry;
    if ((n_values * n_bytes_x_entry) != n_bytes) {
        fprintf(stderr, "Bad input! %ld != %d * %ld\n", (long)n_bytes, n_bytes_x_entry, n_values);
        return EXIT_FAILURE;
    }

    ptrdiff_t n_bytes_aos = n_bytes * n_arrays * sizeof(char);
    char *aos = (char *)malloc(n_bytes_aos);

    for (int np = 0; np < n_arrays; np++) {
        for (ptrdiff_t i = 0; i < n_values; i++) {
            memcpy(&aos[(i * n_arrays + np) * n_bytes_x_entry], &arrays[np][i * n_bytes_x_entry], n_bytes_x_entry);
        }
    }

    array_write(comm, path_output_aos, values_mpi_t, (void*)aos, n_bytes_aos, n_bytes_aos);

    for (int np = 0; np < n_arrays; np++) {
        free(arrays[np]);
    }

    free(aos);
    globfree(&gl);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
