#include "sfem_test.hpp"

#include "sfem_API.hpp"
#include "sfem_OpFactory.hpp"

#include "sfem_GeneratedNeoHookeanOgden.hpp"
#include "sfem_GeneratedNeoHookeanOgden_c_abi.hpp"
#include "sfem_GeneratedTwoPhaseFlow.hpp"
#include "sfem_GeneratedTwoPhaseFlow_c_abi.hpp"
#include "generated/neumann/op/sfem_GeneratedNeumann.hpp"
#include "generated/neumann/op/sfem_GeneratedNeumann_c_abi.hpp"
#include "generated/neumann_general/op/sfem_GeneratedNeumannGeneral.hpp"
#include "generated/neumann_general/op/sfem_GeneratedNeumannGeneral_c_abi.hpp"
#include "generated/poro_hyperelasticity/op/sfem_GeneratedPoroHyperelasticity.hpp"
#include "generated/poro_hyperelasticity/op/sfem_GeneratedPoroHyperelasticity_c_abi.hpp"
#include "generated/stokes/op/sfem_GeneratedStokes.hpp"
#include "generated/stokes/op/sfem_GeneratedStokes_c_abi.hpp"

#include <memory>
#include <type_traits>

namespace {
    template <typename T>
    void require_op_type() {
        static_assert(std::is_base_of<sfem::Op, T>::value, "generated wrapper must derive from sfem::Op");
    }

    std::shared_ptr<sfem::FunctionSpace> hex8_space(const int block_size) {
        auto mesh = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
        return sfem::FunctionSpace::create(mesh, block_size);
    }
}  // namespace

int test_generated_wrapper_headers_compile() {
    require_op_type<sfem::GeneratedNeoHookeanOgden>();
    require_op_type<sfem::GeneratedTwoPhaseFlow>();
    require_op_type<sfem::GeneratedPoroHyperelasticity>();
    require_op_type<sfem::GeneratedStokes>();
    require_op_type<sfem::GeneratedNeumann>();
    require_op_type<sfem::GeneratedNeumannGeneral>();
    return SFEM_TEST_SUCCESS;
}

int test_generated_wrapper_factory_registration() {
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeoHookeanOgden") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(2), "GeneratedTwoPhaseFlow") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(4), "GeneratedPoroHyperelasticity") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(4), "GeneratedStokes") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeumann") != nullptr);
    SFEM_TEST_ASSERT(sfem::Factory::create_op(hex8_space(3), "GeneratedNeumannGeneral") != nullptr);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_generated_wrapper_headers_compile);
    SFEM_RUN_TEST(test_generated_wrapper_factory_registration);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
