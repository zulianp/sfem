// The load profiles the playground scales its loads with.

#include <cmath>

#include "sfem_LoadProfile.hpp"
#include "sfem_test.hpp"

#include <ryml.hpp>
#include <ryml_std.hpp>

namespace {
    int parse(const char *const yaml, sfem::LoadProfile &profile) {
        ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(yaml));
        return sfem::LoadProfile::from_yaml(tree.crootref(), profile, "");
    }
}  // namespace

// The cosine blend from start_value to end_value: the end values outside, half way at the middle,
// and zero slope at both ends.
int test_smooth_ramp() {
    sfem::LoadProfile p;
    SFEM_TEST_ASSERT(parse("{type: smooth_ramp, start_time: 0.2, end_time: 0.6, start_value: 1, end_value: 3}", p) ==
                     SFEM_SUCCESS);
    SFEM_TEST_ASSERT(p.value(0) == 1);
    SFEM_TEST_ASSERT(p.value(1) == 3);
    SFEM_TEST_ASSERT(std::abs(p.value(0.4) - 2) < 1e-12);
    SFEM_TEST_ASSERT(std::abs(p.value(0.3) - (2 - std::sqrt(2.) / 2)) < 1e-12);
    const real_t h = 1e-6;
    SFEM_TEST_ASSERT(std::abs(p.value(0.2 + h) - 1) < 1e-9);
    SFEM_TEST_ASSERT(std::abs(p.value(0.6 - h) - 3) < 1e-9);
    return SFEM_TEST_SUCCESS;
}

// A sine that ends: a half-sine pulse of length 0.05, then nothing.
int test_sine_pulse() {
    sfem::LoadProfile p;
    SFEM_TEST_ASSERT(parse("{type: sine, value: 2, period: 0.1, end_time: 0.05}", p) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::abs(p.value(0.025) - 2) < 1e-12);
    SFEM_TEST_ASSERT(p.value(0.05) >= 0);
    SFEM_TEST_ASSERT(p.value(0.075) == 0);
    sfem::LoadProfile endless;
    SFEM_TEST_ASSERT(parse("{type: sine, period: 0.1}", endless) == SFEM_SUCCESS);
    SFEM_TEST_ASSERT(std::abs(endless.value(0.075) + 1) < 1e-12);
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_smooth_ramp);
    SFEM_RUN_TEST(test_sine_pulse);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
