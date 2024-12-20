#ifndef SFEM_TEST_H
#define SFEM_TEST_H

#include <assert.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SFEM_UNIT_TEST_INIT() int err = 0;
#define SFEM_RUN_TEST(test_)                                                                          \
    do {                                                                                              \
        if ((err = test_())) fprintf(stderr, "TEST: %s failed! %s:%d\n", #test_, __FILE__, __LINE__); \
    } while (0)
#define SFEM_UNIT_TEST_FINALIZE() (err)

static inline int sfem_test_assert(const int expr, const char *expr_string, const char *file, const int line) {
    if (!expr) {
        fprintf(stderr, "\nAssertion failure: %s\nAt %s:%d\n\n", expr_string, file, line);
        assert(0);
        return 1;
    }

    return 0;
}
#define SFEM_TEST_ASSERT(expr)                               \
    if (sfem_test_assert(expr, #expr, __FILE__, __LINE__)) { \
        return 1;                                            \
    }

#ifdef __cplusplus
}
#endif

#endif  // SFEM_TEST
