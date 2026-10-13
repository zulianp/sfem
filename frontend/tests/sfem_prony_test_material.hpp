#pragma once

// The Prony-series Mooney-Rivlin material the validation programs and tests build.
//
// One place that turns the material's description -- C10, C01, K, a Prony series and an
// optional WLF shift -- into the generated operator, so the programs that used to configure
// the hand-written `MooneyRivlinVisco` through its setters do not each spell the generated
// operator's configuration again.  The WLF convention is the one those programs were written
// against and the one `WLFShift` uses: tau_i(T) = a_T(T) tau_i, log10 a_T = -C1 dT / (C2 + dT).
//
// Stepping is the operator's own `HistoryDependent` protocol: `begin_step(t, dt)` before a
// step's solve, `advance(x)` at its converged state.

#include <memory>
#include <string>
#include <vector>

#include "sfem_API.hpp"
#include "sfem_GeneratedMooneyRivlinProny.hpp"

namespace sfem {
    namespace test {

        struct PronyMooneyRivlin {
            real_t              C10{0};
            real_t              C01{0};
            real_t              K{0};
            std::vector<real_t> g;
            std::vector<real_t> tau;
            bool                wlf{false};
            real_t              wlf_C1{0};
            real_t              wlf_C2{0};
            real_t              wlf_T_ref{0};
            real_t              temperature{0};
        };

        inline std::shared_ptr<GeneratedMooneyRivlinProny> make_prony_mooney_rivlin(
                const std::shared_ptr<FunctionSpace> &fs,
                const PronyMooneyRivlin              &material) {
            auto op = std::make_shared<GeneratedMooneyRivlinProny>(fs);
            if (op->initialize() != SFEM_SUCCESS) return nullptr;
            for (const auto &block : fs->mesh_ptr()->blocks()) {
                op->set_value_in_block(block->name(), "C10", material.C10);
                op->set_value_in_block(block->name(), "C01", material.C01);
                op->set_value_in_block(block->name(), "K", material.K);
            }

            PronyRelaxation::Options options;
            if (material.wlf) {
                options.shift       = std::make_shared<WLFShift>(material.wlf_C1, material.wlf_C2, material.wlf_T_ref);
                options.temperature = material.temperature;
            }
            op->relaxation_groups().set_series("", "iso", PronySeries(material.g, material.tau), options);
            return op;
        }

        /// The step weight gamma of the first block's isochoric group, as of the last `begin_step`.
        inline real_t prony_gamma(GeneratedMooneyRivlinProny &op, const std::shared_ptr<FunctionSpace> &fs) {
            return op.relaxation_groups().relaxation(fs->mesh_ptr()->blocks()[0]->name(), "iso").gamma();
        }

    }  // namespace test
}  // namespace sfem
