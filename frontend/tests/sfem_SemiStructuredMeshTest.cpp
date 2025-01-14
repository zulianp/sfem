#include <stdio.h>

#include "sfem_test.h"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"

int test_derefine(const std::shared_ptr<sfem::Mesh> &m, const std::string &output_dir) {
    int L = 8;

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

    sfem::create_directory(output_dir.c_str());

    SFEM_TEST_ASSERT(m->write((output_dir + "/input_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(l2_mesh->export_as_standard((output_dir + "/l2_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(l1_mesh->export_as_standard((output_dir + "/l1_mesh").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(l1_mesh_from_l2->export_as_standard((output_dir + "/l1_mesh_from_l2").c_str()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(fs->semi_structured_mesh().export_as_standard((output_dir + "/og_mesh").c_str()) == SFEM_SUCCESS);

    return SFEM_TEST_SUCCESS;
}

int test_derefine_cube() {
    MPI_Comm comm = MPI_COMM_WORLD;

    int L                    = 8;
    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 2, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 2, 1, 1);

    return test_derefine(m, "test_derefine_cube");
}

int test_derefine_mesh() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     m    = sfem::Mesh::create_from_file(comm, "pinion");
    return test_derefine(m, "test_derefine_mesh");
}

int test_prolongation(const std::shared_ptr<sfem::Mesh> &m, const std::string &output_dir) {
    int  L  = 16;
    auto fs = sfem::FunctionSpace::create(m, 1);
    fs->promote_to_semi_structured(L);

    // // Do this before any other operation
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    auto levels = fs->semi_structured_mesh().derefinement_levels();

    auto coarse_fs = fs->derefine(levels[2]);  // Choose any level

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    auto es = sfem::EXECUTION_SPACE_HOST;
    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    auto prolongation = create_hierarchical_prolongation(coarse_fs, fs, es);
    auto coarse_field = sfem::create_host_buffer<real_t>(coarse_fs->n_dofs());

    {
        auto            data  = coarse_field->data();
        const ptrdiff_t ndofs = coarse_fs->n_dofs();

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < ndofs; i++) {
            data[i] = 1;
        }
    }

#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        coarse_field = sfem::to_device(coarse_field);
    }
#endif

    auto fine_field = sfem::create_buffer<real_t>(fs->n_dofs(), es);
    prolongation->apply(coarse_field->data(), fine_field->data());

#ifdef SFEM_ENABLE_CUDA
    if (es == sfem::EXECUTION_SPACE_DEVICE) {
        fine_field = sfem::to_host(fine_field);
    }
#endif

    {
        auto            data  = fine_field->data();
        const ptrdiff_t ndofs = fs->n_dofs();

        for (ptrdiff_t i = 0; i < ndofs; i++) {
            SFEM_TEST_ASSERT(fabs(data[i] - 1) < 1e-8);
        }
    }

    return SFEM_TEST_SUCCESS;
}

int test_prolongation_cube() {
    MPI_Comm comm = MPI_COMM_WORLD;

    int L                    = 8;
    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    auto m = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 2, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, 2, 1, 1);

    return test_prolongation(m, "test_derefine_cube");
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_derefine_cube);
    // SFEM_RUN_TEST(test_derefine_mesh);
    SFEM_RUN_TEST(test_prolongation_cube);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
