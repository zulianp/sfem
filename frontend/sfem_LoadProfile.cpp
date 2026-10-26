#include "sfem_LoadProfile.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "sfem_logger.hpp"
#include "smesh_path.hpp"

namespace sfem {

    real_t LoadProfile::value(const real_t time) const {
        switch (type_) {
            case Type::CONSTANT:
                return value_;
            case Type::LINEAR_RAMP: {
                if (time <= start_time_) return start_value_;
                if (time >= end_time_) return end_value_;
                const real_t alpha = (time - start_time_) / (end_time_ - start_time_);
                return start_value_ + alpha * (end_value_ - start_value_);
            }
            case Type::SMOOTH_RAMP: {
                // The cosine blend: zero slope at both ends.
                if (time <= start_time_) return start_value_;
                if (time >= end_time_) return end_value_;
                constexpr real_t pi    = real_t(3.1415926535897932384626433832795);
                const real_t     alpha = (1 - std::cos(pi * (time - start_time_) / (end_time_ - start_time_))) / 2;
                return start_value_ + alpha * (end_value_ - start_value_);
            }
            case Type::HOLD:
                return time < start_time_ ? before_value_ : value_;
            case Type::PULSE:
                return time < start_time_ ? before_value_ : (time <= end_time_ ? value_ : after_value_);
            case Type::TABULATED: {
                if (time <= times_.front()) return values_.front();
                if (time >= times_.back()) return values_.back();
                const auto   upper = std::upper_bound(times_.begin(), times_.end(), time);
                const size_t i1    = static_cast<size_t>(upper - times_.begin());
                const size_t i0    = i1 - 1;
                const real_t a     = (time - times_[i0]) / (times_[i1] - times_[i0]);
                return values_[i0] + a * (values_[i1] - values_[i0]);
            }
            case Type::SINE: {
                if (time > end_time_) return 0;
                constexpr real_t two_pi = real_t(6.283185307179586476925286766559);
                return value_ * std::sin(two_pi * (time - start_time_) / period_);
            }
        }

        return 1;
    }

#ifdef SFEM_ENABLE_RYAML
    namespace {
        std::string yaml_string(const ryml::ConstNodeRef &node) {
            const auto value = node.val();
            return std::string(value.str, value.len);
        }

        template <typename T>
        void read_optional(const ryml::ConstNodeRef &node, const char *key, T &value) {
            if (node.has_child(key)) node[key].load(&value);
        }

        int read_table(const std::string &path, std::vector<real_t> &times, std::vector<real_t> &values) {
            std::ifstream stream(path);
            if (!stream.good()) {
                SFEM_ERROR("Unable to read tabulated load profile %s\n", path.c_str());
                return SFEM_FAILURE;
            }

            std::string line;
            while (std::getline(stream, line)) {
                const auto comment = line.find('#');
                if (comment != std::string::npos) line.resize(comment);
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream row(line);
                real_t             time, value;
                if (!(row >> time >> value)) continue;
                if (!std::isfinite(time) || !std::isfinite(value)) {
                    SFEM_ERROR("Tabulated load profile contains a non-finite value: %s\n", path.c_str());
                    return SFEM_FAILURE;
                }
                times.push_back(time);
                values.push_back(value);
            }

            if (times.empty() || times.size() != values.size()) {
                SFEM_ERROR("Tabulated load profile is empty: %s\n", path.c_str());
                return SFEM_FAILURE;
            }

            for (size_t i = 1; i < times.size(); ++i) {
                if (!(times[i] > times[i - 1])) {
                    SFEM_ERROR("Tabulated load profile times must be strictly increasing: %s\n", path.c_str());
                    return SFEM_FAILURE;
                }
            }

            return SFEM_SUCCESS;
        }
    }  // namespace

    int LoadProfile::from_yaml(const ryml::ConstNodeRef &node, LoadProfile &profile, const std::string &base_dir) {
        if (node.invalid() || !node.is_map() || !node.has_child("type")) {
            SFEM_ERROR("A load profile must be a mapping with a type\n");
            return SFEM_FAILURE;
        }

        const std::string type = yaml_string(node["type"]);
        if (type == "constant") {
            profile.type_ = Type::CONSTANT;
            read_optional(node, "value", profile.value_);
            return SFEM_SUCCESS;
        }

        if (type == "linear_ramp" || type == "ramp" || type == "smooth_ramp") {
            profile.type_ = type == "smooth_ramp" ? Type::SMOOTH_RAMP : Type::LINEAR_RAMP;
            read_optional(node, "start_time", profile.start_time_);
            read_optional(node, "end_time", profile.end_time_);
            read_optional(node, "start_value", profile.start_value_);
            read_optional(node, "end_value", profile.end_value_);
            if (!(profile.end_time_ > profile.start_time_)) {
                SFEM_ERROR("%s end_time must be greater than start_time\n", type.c_str());
                return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        if (type == "hold") {
            profile.type_ = Type::HOLD;
            read_optional(node, "start_time", profile.start_time_);
            read_optional(node, "before_value", profile.before_value_);
            read_optional(node, "value", profile.value_);
            return SFEM_SUCCESS;
        }

        if (type == "pulse") {
            profile.type_ = Type::PULSE;
            read_optional(node, "start_time", profile.start_time_);
            read_optional(node, "end_time", profile.end_time_);
            read_optional(node, "before_value", profile.before_value_);
            read_optional(node, "after_value", profile.after_value_);
            read_optional(node, "value", profile.value_);
            if (profile.end_time_ < profile.start_time_) {
                SFEM_ERROR("pulse end_time must not precede start_time\n");
                return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        if (type == "sine") {
            // value * sin(2 pi (t - start_time) / period), zero after end_time: a half-sine pulse
            // of length p is {period: 2p, end_time: p}.
            profile.type_     = Type::SINE;
            profile.end_time_ = std::numeric_limits<real_t>::infinity();
            read_optional(node, "value", profile.value_);
            read_optional(node, "start_time", profile.start_time_);
            read_optional(node, "period", profile.period_);
            read_optional(node, "end_time", profile.end_time_);
            if (!(profile.period_ > 0)) {
                SFEM_ERROR("sine period must be positive\n");
                return SFEM_FAILURE;
            }
            return SFEM_SUCCESS;
        }

        if (type == "tabulated") {
            if (!node.has_child("path")) {
                SFEM_ERROR("tabulated load profile requires path\n");
                return SFEM_FAILURE;
            }
            profile.type_ = Type::TABULATED;
            std::string path = yaml_string(node["path"]);
            if (!base_dir.empty() && !path.empty() && path[0] != '/') path = (smesh::Path(base_dir) / path).to_string();
            return read_table(path, profile.times_, profile.values_);
        }

        SFEM_ERROR("Unsupported load profile type %s\n", type.c_str());
        return SFEM_FAILURE;
    }
#endif

}  // namespace sfem
