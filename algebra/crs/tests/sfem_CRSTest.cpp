#include "sfem_CRS.hpp"
#include "sfem_test.hpp"

#include <initializer_list>
#include <vector>

template <typename T>
static sfem::SharedBuffer<T> make_buffer(std::initializer_list<T> values) {
    auto out = sfem::create_host_buffer<T>(values.size());
    T   *d   = out->data();

    ptrdiff_t i = 0;
    for (const T value : values) {
        d[i++] = value;
    }

    return out;
}

int test_crs_mm_rectangular() {
    using R = sfem::count_t;
    using C = sfem::idx_t;
    using T = sfem::real_t;

    auto a_rowptr = make_buffer<R>({0, 2, 3, 5});
    auto a_colidx = make_buffer<C>({0, 2, 1, 0, 3});
    auto a_values = make_buffer<T>({1, 2, 3, 4, 5});

    auto b_rowptr = make_buffer<R>({0, 1, 3, 4, 6});
    auto b_colidx = make_buffer<C>({1, 0, 1, 0, 0, 1});
    auto b_values = make_buffer<T>({7, 8, 9, 10, 11, 12});

    auto c_rowptr = sfem::create_host_buffer<R>(0);
    auto c_colidx = sfem::create_host_buffer<C>(0);
    auto c_values = sfem::create_host_buffer<T>(0);

    SFEM_TEST_ASSERT(sfem::crs_mm(2, a_rowptr, a_colidx, a_values, b_rowptr, b_colidx, b_values, c_rowptr, c_colidx, c_values) ==
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

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

    SFEM_RUN_TEST(test_crs_mm_rectangular);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
