#include <stdio.h>

#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <fstream>

#include "sfem_test.hpp"

#include "sfem_Function.hpp"
#include "sfem_StateField.hpp"

#ifdef SFEM_ENABLE_RYAML

int test_dirichlet_conditions_read_yaml() {
    std::string yaml =
            R"(
    dirichlet_conditions:
    - name: right
      type: sideset
      format: expr
      parent: [0]
      lfi: [2]
      value: [-0.6, 0, 0]
      component: [0, 1, 2]
    - name: left
      type: nodeset
      format: expr
      nodes: [0, 1, 2]
      value: 1
      component: 0
    )";

    auto comm = sfem::Communicator::world();
    auto m    = sfem::Mesh::create_hex8_cube(comm);
    auto fs   = sfem::FunctionSpace::create(m, 3);

    // Conditions for standard mesh
    auto conds = sfem::DirichletConditions::create_from_yaml(fs, yaml);

    // Conditions for semi-structured mesh
    auto ssmesh = smesh::to_semistructured(8, m, true, false);
    fs          = sfem::FunctionSpace::create(ssmesh, 3);
    conds       = sfem::DirichletConditions::create_from_yaml(fs, yaml);

    return SFEM_TEST_SUCCESS;
}

int test_dirichlet_file_values_and_profiles() {
    const std::string suffix      = "." + std::to_string(getpid());
    const std::string values_path = "/tmp/sfem_dirichlet_values" + suffix + ".float64.raw";
    const std::string table_path  = "/tmp/sfem_load_profile" + suffix + ".csv";
    {
        const real_t  values[3] = {2, 4, 6};
        std::ofstream stream(values_path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(values), sizeof(values));
    }
    {
        std::ofstream stream(table_path);
        stream << "0,0\n1,2\n";
    }

    std::string yaml =
            "dirichlet_conditions:\n"
            "- type: nodeset\n"
            "  format: expr\n"
            "  nodes: [0, 1, 2]\n"
            "  value: {path: " +
            values_path +
            "}\n"
            "  component: 0\n"
            "  profile: {type: linear_ramp, start_time: 0, end_time: 1, start_value: 0, end_value: 1}\n"
            "- type: nodeset\n"
            "  format: expr\n"
            "  nodes: [3]\n"
            "  value: 2\n"
            "  component: 1\n"
            "  profile: {type: constant, value: 3}\n"
            "- type: nodeset\n"
            "  format: expr\n"
            "  nodes: [4]\n"
            "  value: 2\n"
            "  component: 1\n"
            "  profile: {type: hold, start_time: 0.25, before_value: 0.5, value: 4}\n"
            "- type: nodeset\n"
            "  format: expr\n"
            "  nodes: [5]\n"
            "  value: 2\n"
            "  component: 1\n"
            "  profile: {type: pulse, start_time: 0.25, end_time: 0.75, value: 5, after_value: 0.2}\n"
            "- type: nodeset\n"
            "  format: expr\n"
            "  nodes: [6]\n"
            "  value: 2\n"
            "  component: 1\n"
            "  profile: {type: tabulated, path: " +
            table_path + "}\n";

    auto mesh  = sfem::Mesh::create_hex8_cube(sfem::Communicator::self(), 1, 1, 1);
    auto space = sfem::FunctionSpace::create(mesh, 3);
    auto conds = sfem::DirichletConditions::create_from_yaml(space, yaml);
    SFEM_TEST_ASSERT(conds != nullptr);
    SFEM_TEST_ASSERT(conds->set_time(real_t(0.5)) == SFEM_SUCCESS);

    auto x = sfem::create_host_buffer<real_t>(space->n_dofs());
    std::fill(x->data(), x->data() + space->n_dofs(), real_t(0));
    SFEM_TEST_ASSERT(conds->apply(x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::abs(x->data()[0] - 1) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[3] - 2) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[6] - 3) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[3 * 3 + 1] - 6) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[3 * 4 + 1] - 8) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[3 * 5 + 1] - 10) < 1e-14);
    SFEM_TEST_ASSERT(std::abs(x->data()[3 * 6 + 1] - 2) < 1e-14);

    SFEM_TEST_ASSERT(conds->set_time(real_t(1)) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(conds->apply(x->data()) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::abs(x->data()[3 * 5 + 1] - real_t(0.4)) < 1e-14);

    std::remove(values_path.c_str());
    std::remove(table_path.c_str());
    return SFEM_TEST_SUCCESS;
}

int test_file_backed_initial_state() {
    const std::string suffix    = "." + std::to_string(getpid());
    const std::string full_path = "/tmp/sfem_initial_full" + suffix + ".float64.raw";
    const std::string x_path    = "/tmp/sfem_initial_x" + suffix + ".float64.raw";
    const std::string y_path    = "/tmp/sfem_initial_y" + suffix + ".float64.raw";
    const real_t      full[6]   = {1, 4, 2, 5, 3, 6};
    const real_t      x[3]      = {1, 2, 3};
    const real_t      y[3]      = {4, 5, 6};
    {
        std::ofstream stream(full_path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(full), sizeof(full));
    }
    {
        std::ofstream stream(x_path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(x), sizeof(x));
    }
    {
        std::ofstream stream(y_path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(y), sizeof(y));
    }

    real_t loaded[6] = {};
    SFEM_TEST_ASSERT(sfem::read_state_field(full_path, 6, loaded) == SFEM_SUCCESS);
    SFEM_ASSERT_ARRAY_APPROX_EQ(6, full, loaded, 0);
    std::fill(loaded, loaded + 6, real_t(0));
    SFEM_TEST_ASSERT(sfem::read_state_field_components(x_path + "," + y_path, 3, 2, loaded) == SFEM_SUCCESS);
    SFEM_ASSERT_ARRAY_APPROX_EQ(6, full, loaded, 0);

    std::remove(full_path.c_str());
    std::remove(x_path.c_str());
    std::remove(y_path.c_str());
    return SFEM_TEST_SUCCESS;
}

#endif

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
#ifdef SFEM_ENABLE_RYAML
    SFEM_RUN_TEST(test_dirichlet_conditions_read_yaml);
    SFEM_RUN_TEST(test_dirichlet_file_values_and_profiles);
    SFEM_RUN_TEST(test_file_backed_initial_state);
#endif
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
