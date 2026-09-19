// The material's whole residual, as one kernel.
//
// Mooney-Rivlin Kelvin-Voigt is written as two units -- an energy for the
// elastic response, a residual for the viscous one -- and each publishes its
// own kernels, which the operator calls one after the other into the same
// vector.  A residual merit cannot be built on that: it squares a node's
// complete value, and after the first traversal no node is finished.
//
// `total_residual_collection` sums the two 1-forms into one, which the
// pipeline emits as an ordinary residual unit.  This test holds that kernel to
// the only thing that makes the sum legitimate: it must compute exactly what
// the two units compute between them, which is what `Function::gradient`
// assembles today.

#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_Function.hpp"
#include "sfem_GeneratedMooneyRivlinKelvinVoigt_c_abi.hpp"
#include "sfem_OpFactory.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

    static const real_t kMu        = real_t(3);
    static const real_t kLambda    = real_t(1);
    static const real_t kEtaShear  = real_t(0.05);
    static const real_t kEtaBulk   = real_t(0.01);
    static const real_t kDtShift   = real_t(2);
    static const int    kBlockSize = 3;

    void seed(const ptrdiff_t n, const int salt, real_t *const values) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            values[i] = real_t(1e-3) * std::sin(real_t(0.5 * (i + 1) + salt));
        }
    }

    /// The combined kernel must equal the two units the operator runs apart.
    int test_the_combined_residual_is_the_operators_gradient() {
        auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3);
        auto space = sfem::FunctionSpace::create(mesh, kBlockSize);

        const ptrdiff_t ndofs = space->n_dofs();

        auto op = sfem::Factory::create_op(space, "GeneratedMooneyRivlinKelvinVoigt");
        SFEM_TEST_ASSERT(op != nullptr);
        for (const auto &block : mesh->blocks()) {
            op->set_value_in_block(block->name(), "mu", kMu);
            op->set_value_in_block(block->name(), "lmbda", kLambda);
            op->set_value_in_block(block->name(), "eta_s", kEtaShear);
            op->set_value_in_block(block->name(), "eta_b", kEtaBulk);
            op->set_value_in_block(block->name(), "u_dt_shift", kDtShift);
        }

        auto previous = sfem::create_host_buffer<real_t>(ndofs);
        seed(ndofs, 9, previous->data());
        op->set_field("previous", previous, 0);

        auto state = sfem::create_host_buffer<real_t>(ndofs);
        seed(ndofs, 1, state->data());

        auto f = sfem::Function::create(space);
        f->add_operator(op);

        // The two units, run the way the operator runs them.
        auto reference = sfem::create_host_buffer<real_t>(ndofs);
        SFEM_TEST_ASSERT(f->gradient(state->data(), reference->data()) == SFEM_SUCCESS);

        // The combined unit, one traversal.
        auto combined = sfem::create_host_buffer<real_t>(ndofs);
        const ptrdiff_t nelements = mesh->n_elements();
        const ptrdiff_t nnodes    = mesh->n_nodes();

        SFEM_TEST_ASSERT(mooney_rivlin_kelvin_voigt_residual_merit_residual_3d_i_msoa(
                                 smesh::HEX8,
                                 smesh::TypeToEnum<real_t>::value(),
                                 nelements,
                                 nnodes,
                                 mesh->elements(0)->data(),
                                 const_cast<const geom_t *const *>(mesh->points()->data()),
                                 kEtaBulk,
                                 kEtaShear,
                                 kLambda,
                                 kMu,
                                 kDtShift,
                                 kBlockSize,
                                 state->data() + 0,
                                 state->data() + 1,
                                 state->data() + 2,
                                 kBlockSize,
                                 previous->data() + 0,
                                 previous->data() + 1,
                                 previous->data() + 2,
                                 kBlockSize,
                                 combined->data() + 0,
                                 combined->data() + 1,
                                 combined->data() + 2) == SFEM_SUCCESS);

        real_t worst = 0;
        real_t scale = 0;
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            worst = std::max(worst, std::abs(combined->data()[i] - reference->data()[i]));
            scale = std::max(scale, std::abs(reference->data()[i]));
        }
        // Otherwise the agreement below is agreement about zero.
        SFEM_TEST_ASSERT(scale > real_t(1e-12));
        if (!(worst <= real_t(1e-11) * scale)) {
            fprintf(stderr,
                    "combined residual differs: worst %g against scale %g\n",
                    (double)worst,
                    (double)scale);
        }
        SFEM_TEST_ASSERT(worst <= real_t(1e-11) * scale);
        return SFEM_TEST_SUCCESS;
    }

    /// Both units have to be inside the combined kernel.
    ///
    /// The negative control for the test above.  The viscous term is small
    /// beside the elastic one, so a combined kernel that had silently dropped
    /// it would still look close; running the operator with the viscosity off
    /// must move the answer by more than the tolerance that test allows.
    int test_the_viscous_term_is_inside_the_combined_residual() {
        auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 3, 3, 3);
        auto space = sfem::FunctionSpace::create(mesh, kBlockSize);

        const ptrdiff_t ndofs = space->n_dofs();
        auto state    = sfem::create_host_buffer<real_t>(ndofs);
        auto previous = sfem::create_host_buffer<real_t>(ndofs);
        seed(ndofs, 1, state->data());
        seed(ndofs, 9, previous->data());

        std::vector<real_t> with(ndofs, 0);
        std::vector<real_t> without(ndofs, 0);

        for (int arm = 0; arm < 2; ++arm) {
            const real_t eta_s = arm == 0 ? kEtaShear : real_t(0);
            const real_t eta_b = arm == 0 ? kEtaBulk : real_t(0);
            real_t *const out  = arm == 0 ? with.data() : without.data();
            SFEM_TEST_ASSERT(mooney_rivlin_kelvin_voigt_residual_merit_residual_3d_i_msoa(
                                     smesh::HEX8,
                                     smesh::TypeToEnum<real_t>::value(),
                                     mesh->n_elements(),
                                     mesh->n_nodes(),
                                     mesh->elements(0)->data(),
                                     const_cast<const geom_t *const *>(mesh->points()->data()),
                                     eta_b,
                                     eta_s,
                                     kLambda,
                                     kMu,
                                     kDtShift,
                                     kBlockSize,
                                     state->data() + 0,
                                     state->data() + 1,
                                     state->data() + 2,
                                     kBlockSize,
                                     previous->data() + 0,
                                     previous->data() + 1,
                                     previous->data() + 2,
                                     kBlockSize,
                                     out + 0,
                                     out + 1,
                                     out + 2) == SFEM_SUCCESS);
        }

        real_t worst = 0;
        real_t scale = 0;
        for (ptrdiff_t i = 0; i < ndofs; ++i) {
            worst = std::max(worst, std::abs(with[i] - without[i]));
            scale = std::max(scale, std::abs(with[i]));
        }
        SFEM_TEST_ASSERT(scale > real_t(1e-12));
        SFEM_TEST_ASSERT(worst > real_t(1e-11) * scale);
        return SFEM_TEST_SUCCESS;
    }

}  // namespace

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_combined_residual_is_the_operators_gradient);
    SFEM_RUN_TEST(test_the_viscous_term_is_inside_the_combined_residual);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
