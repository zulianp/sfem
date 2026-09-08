#ifndef SFEM_TEST_H
#define SFEM_TEST_H

// SFEM test utilities are a thin wrapper around the smesh test macros.
// The only SFEM-specific addition is the initialization path, which routes
// through sfem::initialize (so CUDA device ops get registered when enabled).

#include "sfem_base.hpp"
#include "sfem_context.hpp"

#include "smesh_test.hpp"

#define SFEM_TEST_SUCCESS SMESH_TEST_SUCCESS
#define SFEM_TEST_FAILURE SMESH_TEST_FAILURE
#define SFEM_TEST_SKIPPED SMESH_TEST_SKIPPED

#define SFEM_UNIT_TEST_INIT(argc, argv)            \
    auto context__ = sfem::initialize(argc, argv); \
    smesh_print_test_info();                       \
    const int smesh_test_argc__ = argc;            \
    char **smesh_test_argv__    = argv;            \
    int err                     = 0;

#define SFEM_RUN_TEST(test_) SMESH_RUN_TEST(test_)
#define SFEM_UNIT_TEST_FINALIZE() SMESH_UNIT_TEST_FINALIZE()
#define SFEM_UNIT_TEST_ERR() SMESH_UNIT_TEST_ERR()

#define SFEM_TEST_ASSERT(expr) SMESH_TEST_ASSERT(expr)
#define SFEM_TEST_APPROXEQ(a, b, tol) SMESH_TEST_APPROXEQ(a, b, tol)
#define SFEM_ASSERT_ARRAY_APPROX_EQ(n__, a__, b__, tol__) SMESH_ASSERT_ARRAY_APPROX_EQ(n__, a__, b__, tol__)
#define SFEM_TEST_EQ(a, b) SMESH_TEST_EQ(a, b)
#define SFEM_ASSERT_ARRAY_EQ(n__, a__, b__) SMESH_ASSERT_ARRAY_EQ(n__, a__, b__)

#endif  // SFEM_TEST
