#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_OpFactory.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

// Every element a generated operator meshes has to be reachable, whatever the
// caller asked for.
//
// The wrapper picks affine or isoparametric geometry once per operator, from a
// runtime flag, and then calls one dimension-generic entry point.  That entry
// point switches on the element type and serves only the elements whose kernel
// was generated in the mode it stands for -- and the two modes do not cover the
// same elements.  A constant-P1 simplex has no isoparametric kernel at all,
// because its affine kernel already computes the same numbers, so TET4 and TRI3
// are unreachable with the flag off.  A quadrilateral has no affine kernel in
// 2D, so QUAD4 is unreachable with the flag on.  Whichever way the flag is set,
// some element the operator meshes falls through to `unsupported_dispatch`.
//
// So this drives each element in both settings.  The geometry mode is the
// operator's business, not the caller's, wherever the element leaves no choice.
namespace {

    std::shared_ptr<sfem::FunctionSpace> space_for(const std::string &mesh, const int block_size) {
        auto comm = sfem::Communicator::self();
        if (mesh == "tet4") {
            return sfem::FunctionSpace::create(sfem::Mesh::create_tet4_cube(comm, 1, 1, 1), block_size);
        }
        if (mesh == "hex8") {
            return sfem::FunctionSpace::create(sfem::Mesh::create_hex8_cube(comm, 1, 1, 1), block_size);
        }
        if (mesh == "tri3") {
            return sfem::FunctionSpace::create(sfem::Mesh::create_tri3_square(comm, 1, 1, 0, 0, 1, 1), block_size);
        }
        return sfem::FunctionSpace::create(sfem::Mesh::create_quad4_square(comm, 1, 1, 0, 0, 1, 1), block_size);
    }

    int failures = 0;

    void drive(const std::string &op_name, const std::string &mesh, const int block_size, const bool affine) {
        auto space = space_for(mesh, block_size);
        auto op    = sfem::Factory::create_op(space, op_name.c_str());
        if (!op) {
            fprintf(stderr, "%s: no op on %s\n", op_name.c_str(), mesh.c_str());
            ++failures;
            return;
        }
        op->set_option("ASSUME_AFFINE", affine);

        const ptrdiff_t ndofs = space->n_dofs();
        auto            state = sfem::create_host_buffer<real_t>(ndofs);
        auto            out   = sfem::create_host_buffer<real_t>(ndofs);
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            state->data()[i] = real_t(0.001) * real_t(i % 7);
        }

        if (op->gradient(state->data(), out->data()) != SFEM_SUCCESS) {
            fprintf(stderr,
                    "UNROUTABLE %s gradient on %s with ASSUME_AFFINE=%d\n",
                    op_name.c_str(),
                    mesh.c_str(),
                    (int)affine);
            ++failures;
            return;
        }

        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            if (!std::isfinite(out->data()[i])) {
                fprintf(stderr,
                        "NON-FINITE %s gradient on %s with ASSUME_AFFINE=%d\n",
                        op_name.c_str(),
                        mesh.c_str(),
                        (int)affine);
                ++failures;
                return;
            }
        }
    }

}  // namespace

int test_every_element_routes_in_both_geometry_modes() {
    failures = 0;
    for (const bool affine : {false, true}) {
        drive("GeneratedLaplace", "tet4", 1, affine);
        drive("GeneratedLaplace", "hex8", 1, affine);
        drive("GeneratedLaplace", "tri3", 1, affine);
        drive("GeneratedLaplace", "quad4", 1, affine);
        drive("GeneratedLinearElasticity", "tet4", 3, affine);
        drive("GeneratedLinearElasticity", "hex8", 3, affine);
        drive("GeneratedLinearElasticity", "tri3", 2, affine);
        drive("GeneratedLinearElasticity", "quad4", 2, affine);
        drive("GeneratedNeoHookeanOgden", "tet4", 3, affine);
        drive("GeneratedNeoHookeanOgden", "hex8", 3, affine);
    }
    fprintf(stderr, "unroutable combinations: %d\n", failures);
    SFEM_TEST_ASSERT(failures == 0);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_every_element_routes_in_both_geometry_modes);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
