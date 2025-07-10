#include "sfem_Parameters.hpp"

#include <map>

namespace sfem {
    class Parameters::Impl {
    public:
        std::map<std::string, std::shared_ptr<Value>> values;
    };

    Parameters::Parameters() : impl_(std::make_unique<Impl>()) {}

    Parameters::~Parameters() {}

    void Parameters::set_value(const std::string &var_name, const real_t value) {
        impl_->values[var_name] = std::make_shared<ScalarValue>(value);
    }

    void Parameters::set_value(const std::string &var_name, const std::shared_ptr<ScalarValue> &value) {
        impl_->values[var_name] = value;
    }

    std::shared_ptr<Value> Parameters::find_value(const std::string &var_name) const {
        auto it = impl_->values.find(var_name);
        if (it == impl_->values.end()) {
            return nullptr;
        }
        return it->second;
    }

    ScalarValue::ScalarValue(const real_t value) : value_(value) {}

    ScalarValue::~ScalarValue() {}

    real_t ScalarValue::value() const { return value_; }
}  // namespace sfem
