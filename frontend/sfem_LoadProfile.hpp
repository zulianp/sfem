#pragma once

#include <string>
#include <vector>

#include "sfem_aliases.hpp"
#include "sfem_defs.hpp"

#ifdef SFEM_ENABLE_RYAML
#include <ryml.hpp>
#endif

namespace sfem {

    class LoadProfile {
    public:
        enum class Type { CONSTANT, LINEAR_RAMP, SMOOTH_RAMP, HOLD, PULSE, TABULATED, SINE };

        real_t value(const real_t time) const;

#ifdef SFEM_ENABLE_RYAML
        /// A tabulated profile's relative `path` is read against `base_dir` (as written when empty).
        static int from_yaml(const ryml::ConstNodeRef &node, LoadProfile &profile, const std::string &base_dir);
#endif

    private:
        Type                type_{Type::CONSTANT};
        real_t              value_{1};
        real_t              before_value_{0};
        real_t              after_value_{0};
        real_t              start_time_{0};
        real_t              end_time_{1};
        real_t              start_value_{0};
        real_t              end_value_{1};
        real_t              period_{1};
        std::vector<real_t> times_;
        std::vector<real_t> values_;
    };

}  // namespace sfem
