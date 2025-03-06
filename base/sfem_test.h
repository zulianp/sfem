#ifndef SFEM_TEST_H
#define SFEM_TEST_H

#include <assert.h>
#include <stdio.h>
#include <exception>
#include <math.h>

#include "sfem_Tracer.hpp"
#include "sfem_base.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SFEM_TEST_SUCCESS 0
#define SFEM_TEST_FAILURE 1
#define SFEM_TEST_SKIPPED 2

static void sfem_print_test_info() {
    printf("=======================\n");
    printf("SFEM_TESTING Info:\n");

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!rank) {
        printf("MPI_Comm_size=%d\n", size);
    }

#ifdef _OPENMP
#pragma omp parallel
    {
        size_t start, len;
        int    id  = omp_get_thread_num();
        int    num = omp_get_num_threads();

        if (!id) {
            printf("OMP_NUM_THREADS=%d\n", num);
        }
    }
#endif
    printf("=======================\n");
}

#define SFEM_UNIT_TEST_INIT(argc, argv) \
    MPI_Init(&argc, &argv);             \
    sfem_print_test_info();             \
    int err = 0;
#define SFEM_RUN_TEST(test_)                                                                                    \
    do {                                                                                                        \
        SFEM_TRACE_SCOPE(#test_);                                                                               \
        int this_test = 0;                                                                                      \
        try {                                                                                                   \
            if ((this_test = test_())) fprintf(stderr, "TEST: %s failed! %s:%d\n", #test_, __FILE__, __LINE__); \
        } catch (const std::exception &ex) {                                                                    \
            fprintf(stderr, "Exception: %s, in test %s! %s:%d\n", ex.what(), #test_, __FILE__, __LINE__);       \
            this_test = SFEM_TEST_FAILURE;                                                                      \
        }                                                                                                       \
        err += this_test;                                                                                       \
    } while (0)

#define SFEM_UNIT_TEST_FINALIZE()                                                     \
    do {                                                                              \
        if (err) fprintf(stderr, "The number of tests failed in unit is %d!\n", err); \
        MPI_Finalize();                                                               \
    } while (0)
#define SFEM_UNIT_TEST_ERR() (err)

static inline int sfem_test_assert(const int expr, const char *expr_string, const char *file, const int line) {
    if (!expr) {
        fprintf(stderr, "\nAssertion failure: %s\nAt %s:%d\n\n", expr_string, file, line);
        assert(0);
        return SFEM_TEST_FAILURE;
    }

    return 0;
}

static inline int sfem_test_assert_approxeq(double      a,
                                            double      b,
                                            double      tol,
                                            const char *file,
                                            const int   line) {
    if (fabs(a - b) > tol) {
        fprintf(stderr, "\nAssertion failure: %g != %g (diff %g > %g) \nAt %s:%d\n\n", a, b, fabs(a - b), tol, file, line);
        assert(0);
        return SFEM_TEST_FAILURE;
    }

    return 0;
}

#define SFEM_TEST_ASSERT(expr)                                                    \
    if (sfem_test_assert(expr, #expr, __FILE__, __LINE__) == SFEM_TEST_FAILURE) { \
        return SFEM_TEST_FAILURE;                                                 \
    }

#define SFEM_TEST_APPROXEQ(a, b, tol)                                           \
    if (sfem_test_assert_approxeq(a, b, tol, __FILE__, __LINE__) == SFEM_TEST_FAILURE) { \
        return SFEM_TEST_FAILURE;                                               \
    }

#ifdef __cplusplus
}
#endif

#endif  // SFEM_TEST
