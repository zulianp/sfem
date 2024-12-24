
#include "sfem_API.hpp"

#include <stdio.h>

#include "sfem_test.h"

int test_sshex8_hierarchical_renumbering()
{

	auto elements = sfem::h_buffer<idx_t>(27, 2);

	for (ptrdiff_t i = 0; i < 27; i++) {
	    elements->data()[i][0] = i;
	    elements->data()[i][1] = 18 + i;
	}




	return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT();
    SFEM_RUN_TEST(test_sshex8_hierarchical_renumbering);
    return SFEM_UNIT_TEST_FINALIZE();
}
