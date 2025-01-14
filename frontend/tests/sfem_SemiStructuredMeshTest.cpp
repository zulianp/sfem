#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"


int test_derefine() {
    MPI_Comm comm = MPI_COMM_WORLD;

    int L = 8;
    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 2, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 2, 1, 1);

    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(L);

    // Do this before any other operation
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    // Create the points to avoid duplication in coarse levels (lazy init)!
    // fs->semi_structured_mesh().points();    

    auto levels = fs->semi_structured_mesh().derefinement_levels();
    assert(levels[3] == L);
    assert(levels[2] == 4);
    assert(levels[1] == 2);
    assert(levels[0] == 1);

    auto l2_mesh = fs->semi_structured_mesh().derefine(levels[2]);
    auto l1_mesh = fs->semi_structured_mesh().derefine(levels[1]);

    // Recursive way
    auto l1_mesh_from_l2 = l2_mesh->derefine(levels[1]);

    std::string output_dir = "test_derefine";
    sfem::create_directory(output_dir.c_str());

    SFEM_TEST_ASSERT(l2_mesh->export_as_standard((output_dir + "/l2_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(l1_mesh->export_as_standard((output_dir + "/l1_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(l1_mesh_from_l2->export_as_standard((output_dir + "/l1_mesh_from_l2").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((output_dir + "/og_mesh").c_str()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_prolong_restrict()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    
    int L = 8;
    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 2, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 2, 1, 1);

    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(L);

    // Do this before any other operation
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    auto coarse_mesh = fs->semi_structured_mesh().derefine(levels[2]);

}


int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_derefine);
    SFEM_RUN_TEST(test_prolong_restrict);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
