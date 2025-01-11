#ifndef SFEM_TEST_H
#define SFEM_TEST_H

#include <assert.h>
#include <stdio.h>

#include "sfem_Tracer.hpp"
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SFEM_TEST_SUCCESS 0
#define SFEM_TEST_FAILURE 1
#define SFEM_UNIT_TEST_INIT(argc,argv) MPI_Init(&argc, &argv); int err = 0;
#define SFEM_RUN_TEST(test_)                                                                          \
    do {                                                                                              \
        SFEM_TRACE_SCOPE(#test_);                                                                     \
        if ((err = test_())) fprintf(stderr, "TEST: %s failed! %s:%d\n", #test_, __FILE__, __LINE__); \
    } while (0)
#define SFEM_UNIT_TEST_FINALIZE() MPI_Finalize()
#define SFEM_UNIT_TEST_ERR() (err)

static inline int sfem_test_assert(const int expr, const char *expr_string, const char *file, const int line) {
    if (!expr) {
        fprintf(stderr, "\nAssertion failure: %s\nAt %s:%d\n\n", expr_string, file, line);
        assert(0);
        return SFEM_TEST_FAILURE;
    }

    return 0;
}
#define SFEM_TEST_ASSERT(expr)                                                    \
    if (sfem_test_assert(expr, #expr, __FILE__, __LINE__) == SFEM_TEST_FAILURE) { \
        return SFEM_TEST_FAILURE;                                                 \
    }

#ifdef __cplusplus
}
#endif

#endif  // SFEM_TEST
