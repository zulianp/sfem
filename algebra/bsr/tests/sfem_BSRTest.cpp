#include "sfem_BSR.hpp"
#include "sfem_test.hpp"

#include <initializer_list>
#include <vector>

template <typename T>
static sfem::SharedBuffer<T> make_buffer(std::initializer_list<T> values) {
    auto out = sfem::create_host_buffer<T>(values.size());
    T*   d   = out->data();

    ptrdiff_t i = 0;
    for (const T value : values) {
        d[i++] = value;
    }

    return out;
}

// A is a 2x2 block matrix with 2x2 blocks (4x4 scalar):
//   block(0,0) = [[1,2],[3,4]]   block(0,1) = [[5,6],[7,8]]
//   block(1,0) = [[9,10],[11,12]]
// Blocks are stored row-major within the block.
static sfem::SharedBuffer<sfem::count_t> a_rowptr() { return make_buffer<sfem::count_t>({0, 2, 3}); }
static sfem::SharedBuffer<sfem::idx_t>   a_colidx() { return make_buffer<sfem::idx_t>({0, 1, 0}); }
static sfem::SharedBuffer<sfem::real_t>  a_values() {
    return make_buffer<sfem::real_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
}

int test_bsr_transpose() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto ar = a_rowptr();
    auto ac = a_colidx();
    auto av = a_values();

    auto b_rowptr = sfem::create_host_buffer<R>(0);
    auto b_colidx = sfem::create_host_buffer<C>(0);
    auto b_values = sfem::create_host_buffer<T>(0);

    SFEM_TEST_ASSERT(sfem::bsr_transpose(2, 2, 2, 2, ar, ac, av, b_rowptr, b_colidx, b_values) == SFEM_SUCCESS);

    const R expected_rowptr[] = {0, 2, 3};
    const C expected_colidx[] = {0, 1, 0};
    // Transposed blocks (row-major): A(0,0)^T, A(1,0)^T, A(0,1)^T
    const T expected_values[] = {1, 3, 2, 4, 9, 11, 10, 12, 5, 7, 6, 8};

    SFEM_TEST_EQ(b_rowptr->size(), static_cast<size_t>(3));
    SFEM_TEST_EQ(b_colidx->size(), static_cast<size_t>(3));
    SFEM_TEST_EQ(b_values->size(), static_cast<size_t>(12));
    SFEM_ASSERT_ARRAY_EQ(3, b_rowptr->data(), expected_rowptr);
    SFEM_ASSERT_ARRAY_EQ(3, b_colidx->data(), expected_colidx);
    SFEM_ASSERT_ARRAY_APPROX_EQ(12, b_values->data(), expected_values, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int test_bsr_transpose_apply() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto a = sfem::h_bsr_spmv<R, C, T>(2, 2, 2, a_rowptr(), a_colidx(), a_values(), static_cast<T>(0));

    auto at = a->transpose();

    SFEM_TEST_EQ(at->row_block_size(), 2);
    SFEM_TEST_EQ(at->col_block_size(), 2);
    SFEM_TEST_EQ(at->rows(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(at->cols(), static_cast<ptrdiff_t>(4));

    // A (dense, row-major 4x4):
    //   1  2  5  6
    //   3  4  7  8
    //   9 10  0  0
    //  11 12  0  0
    const T x[4] = {1, 2, 3, 4};

    // A^T * x
    T y_t[4] = {0, 0, 0, 0};
    at->apply(x, y_t);

    const T expected_t[] = {1 * 1 + 3 * 2 + 9 * 3 + 11 * 4,
                            2 * 1 + 4 * 2 + 10 * 3 + 12 * 4,
                            5 * 1 + 7 * 2 + 0 * 3 + 0 * 4,
                            6 * 1 + 8 * 2 + 0 * 3 + 0 * 4};
    SFEM_ASSERT_ARRAY_APPROX_EQ(4, y_t, expected_t, 1e-6);

    // Transpose twice recovers the original action: A * x
    auto att = at->transpose();
    T    y[4] = {0, 0, 0, 0};
    att->apply(x, y);

    const T expected[] = {1 * 1 + 2 * 2 + 5 * 3 + 6 * 4,
                          3 * 1 + 4 * 2 + 7 * 3 + 8 * 4,
                          9 * 1 + 10 * 2 + 0 * 3 + 0 * 4,
                          11 * 1 + 12 * 2 + 0 * 3 + 0 * 4};
    SFEM_ASSERT_ARRAY_APPROX_EQ(4, y, expected, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_bsr_transpose);
    SFEM_RUN_TEST(test_bsr_transpose_apply);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
