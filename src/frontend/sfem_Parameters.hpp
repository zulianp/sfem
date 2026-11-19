#ifndef SFEM_BLOCK_MATERIAL_HPP
#define SFEM_BLOCK_MATERIAL_HPP

#include "sfem_base.hpp"

#include <memory>
#include <tuple>
#include <string>
#include <memory>

namespace sfem {

    class Value {
    public:
        Value()          = default;
        virtual ~Value() = default;
    };

    class ScalarValue final : public Value {
    public:
        ScalarValue(const real_t value);
        ~ScalarValue();
        real_t value() const;

    private:
        real_t value_;
    };

    class Parameters {
    public:
        Parameters();
        ~Parameters();

        void set_value(const std::string &var_name, const real_t value);
        void set_value(const std::string &var_name, const std::shared_ptr<ScalarValue> &value);
        std::shared_ptr<Value> find_value(const std::string &var_name) const;
        real_t get_real_value(const std::string &var_name, const real_t default_value) const;
        /// @return Stored scalar; aborts via SFEM_ERROR if @a var_name is missing or not a real scalar.
        real_t require_real_value(const std::string &var_name) const;

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    using SharedParameters = std::shared_ptr<Parameters>;
}  // namespace sfem

#endif  // SFEM_BLOCK_MATERIAL_HPP
