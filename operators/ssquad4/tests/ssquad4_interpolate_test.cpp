#include "ssquad4_interpolate.h"
#include "sfem_API.hpp"

#include <stdio.h>

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

    int err = ssquad4_prolongate(1,                 // nelements,
                                 1,                 // rom_level
                                 2,                 // from_level_stride
                                 elements->data(),  // from_elements
                                 2,                 // to_level
                                 1,                 // to_level_stride
                                 elements->data(),  // to_elements
                                 1,                 // vec_size
                                 from,
                                 to);

    auto e = elements->data();
    for (int xi = 0; xi < 3; xi++) {
        for (int yi = 0; yi < 3; yi++) {
            const idx_t idx = e[yi * 3 + xi][0];
            err |= !(fabs(to[idx] - xi) < 1e-12);
            assert(!err);
        }
    }

    return err;
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

    auto e   = to_elements->data();
    int  err = ssquad4_prolongate(1,                      // nelements,
                                 2,                      // from_level
                                 1,                      // from_level_stride
                                 from_elements->data(),  // from_elements
                                 4,                      // to_level
                                 1,                      // to_level_stride
                                 to_elements->data(),    // to_elements
                                 1,                      // vec_size
                                 from,
                                 to);

    for (int yi = 0; yi < 5; yi++) {
        for (int xi = 0; xi < 5; xi++) {
            const idx_t idx = e[yi * 5 + xi][0];
            err |= !(fabs(to[idx] - xi) < 1e-12);
            assert(!err);
        }
    }

    real_t from2[9] = {0, 0, 0, 2, 2, 2, 4, 4, 4};
    err |= ssquad4_prolongate(1,                      // nelements,
                              2,                      // from_level
                              1,                      // from_level_stride
                              from_elements->data(),  // from_elements
                              4,                      // to_level
                              1,                      // to_level_stride
                              to_elements->data(),    // to_elements
                              1,                      // vec_size
                              from2,
                              to);

    for (int yi = 0; yi < 5; yi++) {
        for (int xi = 0; xi < 5; xi++) {
            const idx_t idx = e[yi * 5 + xi][0];
            err |= !(fabs(to[idx] - yi) < 1e-12);
        }
    }

    return err;
}

int main(int argc, char *argv[]) { 
	return test_level2_to_level4() | test_level1_to_level2(); 
}
