#include "ssquad4_interpolate.h"
#include "sfem_API.hpp"

#include <stdio.h>

#include "sfem_test.h"

int test_incidence_count() {
    auto elements = sfem::h_buffer<idx_t>(9, 2);

    for (ptrdiff_t i = 0; i < 9; i++) {
        elements->data()[i][0] = i;
        elements->data()[i][1] = 6 + i;
    }

    auto count = sfem::h_buffer<uint16_t>(9 + 6);
    ssquad4_element_node_incidence_count(2, 1, 2, elements->data(), count->data());

    for (ptrdiff_t i = 0; i < 6; i++) {
        SFEM_TEST_ASSERT(count->data()[i] == 1);
        SFEM_TEST_ASSERT(count->data()[9 + i] == 1);
    }

    for (ptrdiff_t i = 0; i < 3; i++) {
        SFEM_TEST_ASSERT(count->data()[i + 6] == 2);
    }

    return 0;
}

int test_level1_to_level2() {
    real_t from[4] = {0, 2, 2, 0};
    real_t to[9];

    // In this example we use the same element indices array for both discretization
    // the correct acceess is handled through strides. Specific ordering is expected!
    auto elements = sfem::h_buffer<idx_t>(9, 1);

    elements->data()[0][0] = 0;
    elements->data()[1][0] = 4;
    elements->data()[2][0] = 1;

    elements->data()[3][0] = 7;
    elements->data()[4][0] = 8;
    elements->data()[5][0] = 5;

    elements->data()[6][0] = 3;
    elements->data()[7][0] = 6;
    elements->data()[8][0] = 2;

    SFEM_TEST_ASSERT(ssquad4_prolongate(1,                 // nelements,
                                        1,                 // rom_level
                                        2,                 // from_level_stride
                                        elements->data(),  // from_elements
                                        2,                 // to_level
                                        1,                 // to_level_stride
                                        elements->data(),  // to_elements
                                        1,                 // vec_size
                                        from,
                                        to) == SFEM_SUCCESS);

    auto e = elements->data();
    for (int xi = 0; xi < 3; xi++) {
        for (int yi = 0; yi < 3; yi++) {
            const idx_t idx = e[yi * 3 + xi][0];
            SFEM_TEST_ASSERT(fabs(to[idx] - xi) < 1e-12);
        }
    }

    auto count_to = sfem::h_buffer<uint16_t>(9);
    ssquad4_element_node_incidence_count(2, 1, 1, elements->data(), count_to->data());

    for (ptrdiff_t i = 0; i < 9; i++) {
        SFEM_TEST_ASSERT(count_to->data()[i] == 1);
    }

    auto count_from = sfem::h_buffer<uint16_t>(4);
    ssquad4_element_node_incidence_count(1, 2, 1, elements->data(), count_from->data());

    for (ptrdiff_t i = 0; i < 4; i++) {
        SFEM_TEST_ASSERT(count_from->data()[i] == 1);
    }

    return 0;
}

int test_level2_to_level4() {
    real_t from[9] = {0, 2, 4, 0, 2, 4, 0, 2, 4};
    real_t to[25];

    auto from_elements = sfem::h_buffer<idx_t>(9, 1);
    for (ptrdiff_t i = 0; i < 9; i++) {
        from_elements->data()[i][0] = i;
    }

    auto to_elements = sfem::h_buffer<idx_t>(25, 1);
    for (ptrdiff_t i = 0; i < 25; i++) {
        to_elements->data()[i][0] = i;
    }

    auto e = to_elements->data();
    SFEM_TEST_ASSERT(ssquad4_prolongate(1,                      // nelements
                                        2,                      // from_level
                                        1,                      // from_level_stride
                                        from_elements->data(),  // from_elements
                                        4,                      // to_level
                                        1,                      // to_level_stride
                                        to_elements->data(),    // to_elements
                                        1,                      // vec_size
                                        from,
                                        to) == SFEM_SUCCESS);

    for (int yi = 0; yi < 5; yi++) {
        for (int xi = 0; xi < 5; xi++) {
            const idx_t idx = e[yi * 5 + xi][0];
            SFEM_TEST_ASSERT(fabs(to[idx] - xi) < 1e-12);
        }
    }

    real_t from2[9] = {0, 0, 0, 2, 2, 2, 4, 4, 4};
    SFEM_TEST_ASSERT(ssquad4_prolongate(1,                      // nelements,
                                        2,                      // from_level
                                        1,                      // from_level_stride
                                        from_elements->data(),  // from_elements
                                        4,                      // to_level
                                        1,                      // to_level_stride
                                        to_elements->data(),    // to_elements
                                        1,                      // vec_size
                                        from2,
                                        to) == SFEM_SUCCESS);

    for (int yi = 0; yi < 5; yi++) {
        for (int xi = 0; xi < 5; xi++) {
            const idx_t idx = e[yi * 5 + xi][0];
            SFEM_TEST_ASSERT(fabs(to[idx] - yi) < 1e-12);
        }
    }

    return 0;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
    SFEM_RUN_TEST(test_incidence_count);
    SFEM_RUN_TEST(test_level2_to_level4);
    SFEM_RUN_TEST(test_level1_to_level2);
    return SFEM_UNIT_TEST_FINALIZE();
}
