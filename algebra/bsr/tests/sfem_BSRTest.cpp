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

int test_bsr_mm_rectangular() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    // 1x1 blocks: identical sparsity/numerics to the CRS mm test.
    auto a_rowptr = make_buffer<R>({0, 2, 3, 5});
    auto a_colidx = make_buffer<C>({0, 2, 1, 0, 3});
    auto a_values = make_buffer<T>({1, 2, 3, 4, 5});

    auto b_rowptr = make_buffer<R>({0, 1, 3, 4, 6});
    auto b_colidx = make_buffer<C>({1, 0, 1, 0, 0, 1});
    auto b_values = make_buffer<T>({7, 8, 9, 10, 11, 12});

    auto c_rowptr = sfem::create_host_buffer<R>(0);
    auto c_colidx = sfem::create_host_buffer<C>(0);
    auto c_values = sfem::create_host_buffer<T>(0);

    SFEM_TEST_ASSERT(sfem::bsr_mm(2, 1, 1, 1, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, c_rowptr, c_colidx, c_values) ==
                     SFEM_SUCCESS);

    const R expected_rowptr[] = {0, 2, 4, 6};
    SFEM_TEST_EQ(c_rowptr->size(), static_cast<size_t>(4));
    SFEM_ASSERT_ARRAY_EQ(4, c_rowptr->data(), expected_rowptr);

    std::vector<T> dense(6, 0);
    for (ptrdiff_t r = 0; r < 3; ++r) {
        for (R k = c_rowptr->data()[r]; k < c_rowptr->data()[r + 1]; ++k) {
            const C col = c_colidx->data()[k];
            SFEM_TEST_ASSERT(col >= 0 && col < 2);
            dense[static_cast<size_t>(r) * 2 + col] = c_values->data()[k];
        }
    }

    const T expected_dense[] = {20, 7, 24, 27, 55, 88};
    SFEM_ASSERT_ARRAY_APPROX_EQ(6, dense.data(), expected_dense, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int test_bsr_mm_apply() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto a = sfem::h_bsr_spmv<R, C, T>(2, 2, 2, a_rowptr(), a_colidx(), a_values(), static_cast<T>(0));

    // B has two 2x2 identity blocks in column 0.
    auto b_rowptr = make_buffer<R>({0, 1, 2});
    auto b_colidx = make_buffer<C>({0, 0});
    auto b_values = make_buffer<T>({1, 0, 0, 1, 1, 0, 0, 1});

    auto b = sfem::h_bsr_spmv<R, C, T>(2, 1, 2, b_rowptr, b_colidx, b_values, static_cast<T>(0));
    auto c = a->mm(b);

    SFEM_TEST_EQ(c->rows(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(c->cols(), static_cast<ptrdiff_t>(2));

    // C(:,0) = A(:,0) + A(:,1), C(1,0) = A(1,0)
    const T expected_blocks[] = {6, 8, 10, 12, 9, 10, 11, 12};
    SFEM_TEST_EQ(c->values->size(), static_cast<size_t>(8));
    SFEM_ASSERT_ARRAY_APPROX_EQ(8, c->values->data(), expected_blocks, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int test_bsr_rap() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto r_rowptr = make_buffer<R>({0, 2, 4});
    auto r_colidx = make_buffer<C>({0, 1, 1, 2});
    auto r_values = make_buffer<T>({1, 0.5, 0.5, 1});

    auto a_rowptr = make_buffer<R>({0, 2, 5, 7});
    auto a_colidx = make_buffer<C>({0, 1, 0, 1, 2, 1, 2});
    auto a_values = make_buffer<T>({4, 1, 1, 3, 2, 2, 5});

    auto p_rowptr = make_buffer<R>({0, 1, 3, 4});
    auto p_colidx = make_buffer<C>({0, 0, 1, 1});
    auto p_values = make_buffer<T>({1, 0.5, 0.5, 1});

    auto r = sfem::h_bsr_spmv<R, C, T>(2, 3, 1, r_rowptr, r_colidx, r_values, 0);
    auto a = sfem::h_bsr_spmv<R, C, T>(3, 3, 1, a_rowptr, a_colidx, a_values, 0);
    auto p = sfem::h_bsr_spmv<R, C, T>(3, 2, 1, p_rowptr, p_colidx, p_values, 0);
    auto c = sfem::rap(r, a, p);

    SFEM_TEST_EQ(c->rows(), static_cast<ptrdiff_t>(2));
    SFEM_TEST_EQ(c->cols(), static_cast<ptrdiff_t>(2));

    std::vector<T> dense(4, 0);
    for (ptrdiff_t i = 0; i < c->rows(); ++i) {
        for (R k = c->row_ptr->data()[i]; k < c->row_ptr->data()[i + 1]; ++k) {
            const C col = c->col_idx->data()[k];
            SFEM_TEST_ASSERT(col >= 0 && col < c->cols());
            dense[static_cast<size_t>(i) * c->cols() + col] = c->values->data()[k];
        }
    }

    const T expected_dense[] = {5.75, 2.25, 2.25, 7.75};
    SFEM_ASSERT_ARRAY_APPROX_EQ(4, dense.data(), expected_dense, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_bsr_transpose);
    SFEM_RUN_TEST(test_bsr_transpose_apply);
    SFEM_RUN_TEST(test_bsr_mm_rectangular);
    SFEM_RUN_TEST(test_bsr_mm_apply);
    SFEM_RUN_TEST(test_bsr_rap);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
