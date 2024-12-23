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

    count->print(std::cout);

    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
    SFEM_RUN_TEST(test_incidence_count);
    return SFEM_UNIT_TEST_FINALIZE();
}