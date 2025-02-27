#include "sfem_Buffer.hpp"

// Sloppily adapted from stream bench

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MB(nn) (nn * 1000 * 1000)
typedef double array_element_t;

static double mysecond() { return MPI_Wtime(); }
#define REPEAT 5

static void bench(const size_t                   n,
                  array_element_t *SFEM_RESTRICT a,
                  array_element_t *SFEM_RESTRICT b,
                  array_element_t *SFEM_RESTRICT c) {
    size_t          i, j;
    int             k;
    array_element_t scalar = 3.0;
    double          t;
    double          times[4][REPEAT];

    static const char *label[4] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

    static double avgtime[4] = {0}, maxtime[4] = {0}, mintime[4] = {10000000, 10000000, 10000000, 10000000};

    static double bytes[4];
    bytes[0] = 2 * sizeof(array_element_t) * n;
    bytes[1] = 2 * sizeof(array_element_t) * n;
    bytes[2] = 3 * sizeof(array_element_t) * n;
    bytes[3] = 3 * sizeof(array_element_t) * n;

    for (k = 0; k < REPEAT; k++) {
        times[0][k] = mysecond();

#pragma omp parallel for
        for (j = 0; j < n; j++) c[j] = a[j];

        times[0][k] = mysecond() - times[0][k];
        times[1][k] = mysecond();

#pragma omp parallel for
        for (j = 0; j < n; j++) b[j] = scalar * c[j];

        times[1][k] = mysecond() - times[1][k];
        times[2][k] = mysecond();

#pragma omp parallel for
        for (j = 0; j < n; j++) c[j] = a[j] + b[j];

        times[2][k] = mysecond() - times[2][k];
        times[3][k] = mysecond();

#pragma omp parallel for
        for (j = 0; j < n; j++) a[j] = b[j] + scalar * c[j];

        times[3][k] = mysecond() - times[3][k];
    }

    /*  --- SUMMARY --- */

    for (k = 1; k < REPEAT; k++) /* note -- skip first iteration */
    {
        for (j = 0; j < 4; j++) {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf("Function    Best Rate GB/s  Avg time     Min time     Max time\n");
    for (j = 0; j < 4; j++) {
        avgtime[j] = avgtime[j] / (double)(REPEAT - 1);

        printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n",
               label[j],
               1.0E-09 * bytes[j] / mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t n = MB(500);
    if (argc > 1) {
        n = MB(atol(argv[1]));
    }

    double GBs = 1e-9 * (n * sizeof(array_element_t));
    printf("Array size %g [GB]\n", GBs);

    auto buff_a = sfem::create_host_buffer<array_element_t>(n);
    auto buff_b = sfem::create_host_buffer<array_element_t>(n);
    auto buff_c = sfem::create_host_buffer<array_element_t>(n);

    auto a = buff_a->data();
    auto b = buff_b->data();
    auto c = buff_c->data();

    bench(n, a, b, c);

    // TODO add proper test
    int err = 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        err += a[i] != b[i];
    }

    MPI_Finalize();
    return err > 0;
}
