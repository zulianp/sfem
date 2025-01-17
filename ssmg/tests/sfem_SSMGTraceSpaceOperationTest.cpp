#include <memory>

#include "sfem_test.h"

#include "sfem_Function.hpp"

#include "sfem_Buffer.hpp"
#include "sfem_base.h"
#include "sfem_crs_SpMV.hpp"
#include "spmv.h"

#include "matrixio_array.h"

#include "sfem_API.hpp"
#include "sfem_ssgmg.hpp"

#include "sfem_hex8_mesh_graph.h"
#include "ssquad4_interpolate.h"

#ifdef SFEM_ENABLE_CUDA
#include "sfem_Function_incore_cuda.hpp"
#include "sfem_cuda_blas.h"
#include "sfem_cuda_solver.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

int test_trace_space_operations(const std::shared_ptr<sfem::FunctionSpace> &coarse_fs,
                                const std::shared_ptr<sfem::FunctionSpace> &fine_fs,
                                const std::shared_ptr<sfem::Sideset>       &sideset,
                                const std::string                          &name,
                                const sfem::ExecutionSpace                  es) {
    auto coarse_x = sfem::create_host_buffer<real_t>(coarse_fs->n_dofs());
    auto fine_x   = sfem::create_buffer<real_t>(fine_fs->n_dofs(), es);

    auto &&fine_ssmesh   = fine_fs->semi_structured_mesh();
    auto &&coarse_ssmesh = coarse_fs->semi_structured_mesh();

    {
        auto            points     = coarse_ssmesh.points()->data();
        const ptrdiff_t n          = coarse_fs->n_dofs();
        const int       block_size = coarse_fs->block_size();

        auto data = coarse_x->data();
        for (ptrdiff_t i = 0; i < n; i++) {
            data[i * block_size] = (points[0][i] + 1) * (points[1][i] + 1) * (points[2][i] + 1);
        }
    }

#ifdef SFEM_ENABLE_CUDA
    coarse_x = sfem::to_device(coarse_x);
#endif

    const int nexs       = (fine_ssmesh.level() + 1) * (fine_ssmesh.level() + 1);
    auto      fine_sides = sfem::create_host_buffer<idx_t>(nexs, sideset->parent()->size());

    SFEM_TEST_ASSERT(sshex8_extract_surface_from_sideset(fine_ssmesh.level(),
                                                         fine_ssmesh.element_data(),
                                                         sideset->parent()->size(),
                                                         sideset->parent()->data(),
                                                         sideset->lfi()->data(),
                                                         fine_sides->data()) == SFEM_SUCCESS);

    std::cout << "fine_sides: " << fine_sides->extent(0) << " x " << fine_sides->extent(1) << "\n";

    SFEM_TEST_ASSERT(ssquad4_prolongate(fine_sides->extent(1),                        // nelements,
                                        coarse_ssmesh.level(),                        // rom_level
                                        fine_ssmesh.level() / coarse_ssmesh.level(),  // from_level_stride
                                        fine_sides->data(),                           // from_elements
                                        fine_ssmesh.level(),                          // to_level
                                        1,                                            // to_level_stride
                                        fine_sides->data(),                           // to_elements
                                        fine_fs->block_size(),                        // vec_size
                                        coarse_x->data(),
                                        fine_x->data()) == SFEM_SUCCESS);

#if 1
    sfem::create_directory(name.c_str());
    sfem::create_directory((name + "/fields").c_str());

    SFEM_TEST_ASSERT(fine_fs->semi_structured_mesh().export_as_standard((name + "/mesh").c_str()) == SFEM_SUCCESS);

    sfem::Output out(fine_fs);
    out.enable_AoS_to_SoA(true);

    out.set_output_dir((name + "/fields").c_str());
    SFEM_TEST_ASSERT(out.write("u", fine_x->data()) == SFEM_SUCCESS);
#endif
    
    return SFEM_TEST_SUCCESS;
}

int test_trace_prolongation() {
    MPI_Comm comm = MPI_COMM_WORLD;
    auto     es   = sfem::EXECUTION_SPACE_HOST;

    const char *SFEM_EXECUTION_SPACE{nullptr};
    SFEM_READ_ENV(SFEM_EXECUTION_SPACE, );

    if (SFEM_EXECUTION_SPACE) {
        es = sfem::execution_space_from_string(SFEM_EXECUTION_SPACE);
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 4;
    // SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int SFEM_BASE_RESOLUTION = 1;
    SFEM_READ_ENV(SFEM_BASE_RESOLUTION, atoi);

    geom_t Lx = 1;
    auto   m  = sfem::Mesh::create_hex8_cube(
            comm, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, SFEM_BASE_RESOLUTION * 1, 0, 0, 0, Lx, 1, 1);

    int  block_size = 1;
    auto fs         = sfem::FunctionSpace::create(m, block_size);
    fs->promote_to_semi_structured(SFEM_ELEMENT_REFINE_LEVEL);
    fs->semi_structured_mesh().apply_hierarchical_renumbering();

    auto sideset = sfem::Sideset::create_from_selector(
            m, [=](const geom_t /*x*/, const geom_t y, const geom_t z) -> bool { return y > -1e-5 && y < 1e-5; });

    return test_trace_space_operations(fs->derefine(2), fs, sideset, "test_trace_prolongation", es);
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);

#ifdef SFEM_ENABLE_CUDA
    sfem::register_device_ops();
#endif

    SFEM_RUN_TEST(test_trace_prolongation);

    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
