// #include "sshex8_interpolate.h"
#include "proteus_hex8_interpolate.h"
#include "sfem_API.hpp"

#include <stdio.h>

#include "sfem_test.h"

static int test_incidence_count() {
    auto elements = sfem::h_buffer<idx_t>(27, 2);

    for (ptrdiff_t i = 0; i < 27; i++) {
        elements->data()[i][0] = i;
        elements->data()[i][1] = 18 + i;
    }

    auto count = sfem::h_buffer<uint16_t>(27 + 18);
    sshex8_element_node_incidence_count(2, 1, 2, elements->data(), count->data());

    for (ptrdiff_t i = 0; i < 18; i++) {
        SFEM_TEST_ASSERT(count->data()[i] == 1);
        SFEM_TEST_ASSERT(count->data()[27 + i] == 1);
    }

    for (ptrdiff_t i = 0; i < 9; i++) {
        SFEM_TEST_ASSERT(count->data()[i + 18] == 2);
    }

    return SFEM_TEST_SUCCESS;
}

static int test_restrict_level2_to_level1() {
    auto to   = sfem::h_buffer<real_t>(8);
    auto from = sfem::h_buffer<real_t>(27);

    for (ptrdiff_t i = 0; i < from->size(); i++) {
        from->data()[i] = 1;
    }

    // In this example we use the same element indices array for both discretization
    // the correct acceess is handled through strides. Specific ordering is expected!
    auto elements = sfem::h_buffer<idx_t>(27, 1);

    auto e = elements->data();

    // (0,0,0)-(1,0,0)
    e[0][0] = 0;
    e[1][0] = 8;
    e[2][0] = 1;

    // (0,0.5,0)-(1,0.5,0)
    e[3][0] = 9;
    e[4][0] = 10;
    e[5][0] = 11;

    // (0,1,0)-(1,1,0)
    e[6][0] = 2;
    e[7][0] = 12;
    e[8][0] = 3;

    // (0,0,0.5)-(1,0,0.5)
    e[9][0]  = 13;
    e[10][0] = 14;
    e[11][0] = 15;

    // (0,0.5,0.5)-(1,0.5,0.5)
    e[12][0] = 16;
    e[13][0] = 17;
    e[14][0] = 18;

    // (0,1,0.5)-(1,1,0.5)
    e[15][0] = 19;
    e[16][0] = 20;
    e[17][0] = 21;

    // (0,1,1)-(1,1,1)
    e[18][0] = 4;
    e[19][0] = 22;
    e[20][0] = 5;

    // (0,1,1)-(1,1,1)
    e[21][0] = 23;
    e[22][0] = 24;
    e[23][0] = 25;

    // (0,1,1)-(1,1,1)
    e[24][0] = 6;
    e[25][0] = 26;
    e[26][0] = 7;

    auto count = sfem::h_buffer<uint16_t>(27);
    sshex8_element_node_incidence_count(2, 1, 1, elements->data(), count->data());

    SFEM_TEST_ASSERT(sshex8_restrict(1,                 // nelements,
                                     2,                 // from_level
                                     1,                 // from_level_stride
                                     elements->data(),  // from_elements
                                     count->data(),
                                     1,                 // to_level
                                     2,                 // to_level_stride
                                     elements->data(),  // to_elements
                                     1,                 // vec_size
                                     from->data(),
                                     to->data()) == SFEM_SUCCESS);

     for (ptrdiff_t i = 0; i < 8; i++) {
         SFEM_TEST_ASSERT(fabs(to->data()[i] - 3.375) < 1e-14);
     }

    return SFEM_TEST_SUCCESS;
}

static int test_level1_to_level4() {
    auto from = sfem::h_buffer<real_t>(8);
    auto to   = sfem::h_buffer<real_t>(125);

    for (ptrdiff_t i = 0; i < from->size(); i++) {
        from->data()[i] = 1;
    }

    auto from_elements = sfem::h_buffer<idx_t>(8, 1);
    auto to_elements   = sfem::h_buffer<idx_t>(125, 1);

    for (ptrdiff_t i = 0; i < from_elements->extent(0); i++) {
        from_elements->data()[i][0] = i;
    }

    for (ptrdiff_t i = 0; i < to_elements->extent(0); i++) {
        to_elements->data()[i][0] = i;
    }

    SFEM_TEST_ASSERT(sshex8_prolongate(1,                      // nelements,
                                       1,                      // from_level
                                       1,                      // from_level_stride
                                       from_elements->data(),  // from_elements
                                       4,                      // to_level
                                       1,                      // to_level_stride
                                       to_elements->data(),    // to_elements
                                       1,                      // vec_size
                                       from->data(),
                                       to->data()) == SFEM_SUCCESS);

    for (ptrdiff_t i = 0; i < to->size(); i++) {
        SFEM_TEST_ASSERT(fabs(to->data()[i] - 1) < 1e-12);
    }

    for (int zi = 0; zi < 2; zi++) {
        for (int yi = 0; yi < 2; yi++) {
            for (int xi = 0; xi < 2; xi++) {
                from->data()[zi * 4 + yi * 2 + xi] = xi * 4;
            }
        }
    }

    SFEM_TEST_ASSERT(sshex8_prolongate(1,                      // nelements,
                                       1,                      // from_level
                                       1,                      // from_level_stride
                                       from_elements->data(),  // from_elements
                                       4,                      // to_level
                                       1,                      // to_level_stride
                                       to_elements->data(),    // to_elements
                                       1,                      // vec_size
                                       from->data(),
                                       to->data()) == SFEM_SUCCESS);

    for (int zi = 0; zi < 5; zi++) {
        for (int yi = 0; yi < 5; yi++) {
            for (int xi = 0; xi < 5; xi++) {
                SFEM_TEST_ASSERT(fabs(to->data()[zi * 25 + yi * 5 + xi] - xi) < 1e-12);
            }
        }
    }

    return SFEM_TEST_SUCCESS;
}

static int test_level1_to_level2() {
    auto from = sfem::h_buffer<real_t>(8);
    auto to   = sfem::h_buffer<real_t>(27);

    // In this example we use the same element indices array for both discretization
    // the correct acceess is handled through strides. Specific ordering is expected!
    auto elements = sfem::h_buffer<idx_t>(27, 1);

    auto e = elements->data();

    // (0,0,0)-(1,0,0)
    e[0][0] = 0;
    e[1][0] = 8;
    e[2][0] = 1;

    // (0,0.5,0)-(1,0.5,0)
    e[3][0] = 9;
    e[4][0] = 10;
    e[5][0] = 11;

    // (0,1,0)-(1,1,0)
    e[6][0] = 2;
    e[7][0] = 12;
    e[8][0] = 3;

    // (0,0,0.5)-(1,0,0.5)
    e[9][0]  = 13;
    e[10][0] = 14;
    e[11][0] = 15;

    // (0,0.5,0.5)-(1,0.5,0.5)
    e[12][0] = 16;
    e[13][0] = 17;
    e[14][0] = 18;

    // (0,1,0.5)-(1,1,0.5)
    e[15][0] = 19;
    e[16][0] = 20;
    e[17][0] = 21;

    // (0,1,1)-(1,1,1)
    e[18][0] = 4;
    e[19][0] = 22;
    e[20][0] = 5;

    // (0,1,1)-(1,1,1)
    e[21][0] = 23;
    e[22][0] = 24;
    e[23][0] = 25;

    // (0,1,1)-(1,1,1)
    e[24][0] = 6;
    e[25][0] = 26;
    e[26][0] = 7;

    for (int zi = 0; zi < 2; zi++) {
        for (int yi = 0; yi < 2; yi++) {
            for (int xi = 0; xi < 2; xi++) {
                const idx_t idx = e[2 * zi * 9 + 2 * yi * 3 + 2 * xi][0];
                SFEM_TEST_ASSERT(idx < 8);
                from->data()[idx] = xi * 2;
            }
        }
    }

    SFEM_TEST_ASSERT(sshex8_prolongate(1,  // nelements,
                                       1,  // from_level
                                       2,  // from_level_stride
                                       e,  // from_elements
                                       2,  // to_level
                                       1,  // to_level_stride
                                       e,  // to_elements
                                       1,  // vec_size
                                       from->data(),
                                       to->data()) == SFEM_SUCCESS);

    for (int zi = 0; zi < 3; zi++) {
        for (int yi = 0; yi < 3; yi++) {
            for (int xi = 0; xi < 3; xi++) {
                const idx_t idx = e[zi * 9 + yi * 3 + xi][0];
                SFEM_TEST_ASSERT(fabs(to->data()[idx] - xi) < 1e-12);
            }
        }
    }

    auto count_to = sfem::h_buffer<uint16_t>(27);
    sshex8_element_node_incidence_count(2, 1, 1, elements->data(), count_to->data());

    for (ptrdiff_t i = 0; i < 27; i++) {
        SFEM_TEST_ASSERT(count_to->data()[i] == 1);
    }

    auto count_from = sfem::h_buffer<uint16_t>(8);
    sshex8_element_node_incidence_count(1, 2, 1, elements->data(), count_from->data());

    for (ptrdiff_t i = 0; i < 8; i++) {
        SFEM_TEST_ASSERT(count_from->data()[i] == 1);
    }

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
    SFEM_RUN_TEST(test_incidence_count);
    SFEM_RUN_TEST(test_restrict_level2_to_level1);
    SFEM_RUN_TEST(test_level1_to_level2);
    SFEM_RUN_TEST(test_level1_to_level4);
    return SFEM_UNIT_TEST_FINALIZE();
}
