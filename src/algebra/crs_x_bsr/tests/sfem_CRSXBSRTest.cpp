#include "sfem_CRS_X_BSR.hpp"
#include "sfem_test.hpp"

#include <cassert>
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

template <typename R, typename C, typename TStorage, typename T>
static std::vector<TStorage> dense_blocks(const std::shared_ptr<sfem::BSR<R, C, TStorage, T>>& bsr) {
    const ptrdiff_t block_rows       = bsr->row_ptr->size() - 1;
    const ptrdiff_t block_cols       = bsr->block_cols_;
    const int       row_block_size   = bsr->row_block_size_;
    const int       col_block_size   = bsr->col_block_size_;
    const int       block_matrix_size = row_block_size * col_block_size;

    std::vector<TStorage> dense(block_rows * block_cols * block_matrix_size, 0);

    for (ptrdiff_t i = 0; i < block_rows; i++) {
        for (R k = bsr->row_ptr->data()[i]; k < bsr->row_ptr->data()[i + 1]; k++) {
            const C col = bsr->col_idx->data()[k];
            assert(col >= 0 && col < block_cols);

            const TStorage* const block = &bsr->values->data()[k * block_matrix_size];
            TStorage* const       dst   = &dense[(i * block_cols + col) * block_matrix_size];
            for (int d = 0; d < block_matrix_size; d++) {
                dst[d] = block[d];
            }
        }
    }

    return dense;
}

int test_crs_x_bsr_mm() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto crs_rowptr = make_buffer<R>({0, 2, 4});
    auto crs_colidx = make_buffer<C>({0, 2, 1, 2});
    auto crs_values = make_buffer<T>({2, 3, 4, 5});
    auto crs        = sfem::h_crs_spmv<R, C, T>(2, 3, crs_rowptr, crs_colidx, crs_values, 0);

    auto bsr_rowptr = make_buffer<R>({0, 1, 3, 4});
    auto bsr_colidx = make_buffer<C>({0, 0, 1, 1});
    auto bsr_values = make_buffer<T>({1, 2, 3, 4, 5, 6, 7, 8, 1, 0, 0, 1, 2, 1, 0, 3});
    auto bsr        = sfem::h_bsr_spmv<R, C, T>(3, 2, 2, bsr_rowptr, bsr_colidx, bsr_values, 0);

    auto c = sfem::mm(crs, bsr);

    SFEM_TEST_EQ(c->rows(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(c->cols(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(c->row_block_size(), 2);
    SFEM_TEST_EQ(c->col_block_size(), 2);

    const T expected[] = {2, 4, 6, 8, 6, 3, 0, 9, 20, 24, 28, 32, 14, 5, 0, 19};
    const auto dense   = dense_blocks(c);
    SFEM_ASSERT_ARRAY_APPROX_EQ(16, dense.data(), expected, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int test_bsr_x_crs_mm() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto bsr_rowptr = make_buffer<R>({0, 2, 4});
    auto bsr_colidx = make_buffer<C>({0, 2, 1, 2});
    auto bsr_values = make_buffer<T>({1, 2, 3, 4, 5, 6, 7, 8, 1, 0, 0, 1, 2, 1, 0, 3});
    auto bsr        = sfem::h_bsr_spmv<R, C, T>(2, 3, 2, bsr_rowptr, bsr_colidx, bsr_values, 0);

    auto crs_rowptr = make_buffer<R>({0, 1, 3, 4});
    auto crs_colidx = make_buffer<C>({0, 0, 1, 1});
    auto crs_values = make_buffer<T>({2, 3, 4, 5});
    auto crs        = sfem::h_crs_spmv<R, C, T>(3, 2, crs_rowptr, crs_colidx, crs_values, 0);

    auto c = sfem::mm(bsr, crs);

    SFEM_TEST_EQ(c->rows(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(c->cols(), static_cast<ptrdiff_t>(4));
    SFEM_TEST_EQ(c->row_block_size(), 2);
    SFEM_TEST_EQ(c->col_block_size(), 2);

    const T expected[] = {2, 4, 6, 8, 25, 30, 35, 40, 3, 0, 0, 3, 14, 5, 0, 19};
    const auto dense   = dense_blocks(c);
    SFEM_ASSERT_ARRAY_APPROX_EQ(16, dense.data(), expected, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int test_mixed_rap() {
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

    auto r = sfem::h_crs_spmv<R, C, T>(2, 3, r_rowptr, r_colidx, r_values, 0);
    auto a = sfem::h_bsr_spmv<R, C, T>(3, 3, 1, a_rowptr, a_colidx, a_values, 0);
    auto p = sfem::h_crs_spmv<R, C, T>(3, 2, p_rowptr, p_colidx, p_values, 0);
    auto c = sfem::rap(r, a, p);

    SFEM_TEST_EQ(c->rows(), static_cast<ptrdiff_t>(2));
    SFEM_TEST_EQ(c->cols(), static_cast<ptrdiff_t>(2));

    const T expected[] = {5.75, 2.25, 2.25, 7.75};
    const auto dense   = dense_blocks(c);
    SFEM_ASSERT_ARRAY_APPROX_EQ(4, dense.data(), expected, 1e-6);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char* argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_crs_x_bsr_mm);
    SFEM_RUN_TEST(test_bsr_x_crs_mm);
    SFEM_RUN_TEST(test_mixed_rap);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
