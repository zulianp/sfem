#ifndef SFEM_ELASTICITY_PARAMETERS_HPP
#define SFEM_ELASTICITY_PARAMETERS_HPP

#include "sfem_ForwardDeclarations.hpp"
#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include <cmath>

namespace sfem {

    // Forward declarations
    class YoungsModulus;
    class PoissonRatio;
    class BulkModulus;
    class ShearModulus;
    class LameFirstParameter;
    class LameSecondParameter;

    struct YoungsModulus {
        real_t value;

        YoungsModulus() = default;
        explicit YoungsModulus(const real_t value) : value(value) {}
        template <typename A, typename B>
        YoungsModulus(const A &a, const B &b);
    };

    struct PoissonRatio {
        real_t value;

        PoissonRatio() = default;
        explicit PoissonRatio(const real_t value) : value(value) {}
        template <typename A, typename B>
        PoissonRatio(const A &a, const B &b);
    };

    struct BulkModulus {
        real_t value;

        BulkModulus() = default;
        explicit BulkModulus(const real_t value) : value(value) {}
        template <typename A, typename B>
        BulkModulus(const A &a, const B &b);
    };

    struct ShearModulus {
        real_t value;

        ShearModulus() = default;
        explicit ShearModulus(const real_t value) : value(value) {}
        explicit ShearModulus(const LameSecondParameter &mu);
        template <typename A, typename B>
        ShearModulus(const A &a, const B &b);
    };

    struct LameFirstParameter {
        real_t value;

        LameFirstParameter() = default;
        explicit LameFirstParameter(const real_t value) : value(value) {}
        template <typename A, typename B>
        LameFirstParameter(const A &a, const B &b);
    };

    struct LameSecondParameter {
        real_t value;

        LameSecondParameter() = default;
        explicit LameSecondParameter(const real_t value) : value(value) {}
        explicit LameSecondParameter(const ShearModulus &mu);
        template <typename A, typename B>
        LameSecondParameter(const A &a, const B &b);
    };

    namespace detail {
        inline real_t elasticity_sqrt(const real_t value) {
            using std::sqrt;
            return sqrt(value);
        }

        inline real_t youngs_from_lame_first_and_shear(const real_t lambda, const real_t mu) {
            return mu * (3 * lambda + 2 * mu) / (lambda + mu);
        }

        inline real_t shear_from_youngs_and_lame_first(const real_t E, const real_t lambda) {
            return (E - 3 * lambda + elasticity_sqrt(E * E + 2 * E * lambda + 9 * lambda * lambda)) / 4;
        }
    }  // namespace detail

    template <typename A, typename B>
    inline real_t youngs_modulus_value(const A &a, const B &b) {
        return youngs_modulus_value(b, a);
    }

    inline real_t youngs_modulus_value(const PoissonRatio &nu, const BulkModulus &K) {
        return 3 * K.value * (1 - 2 * nu.value);
    }

    inline real_t youngs_modulus_value(const PoissonRatio &nu, const ShearModulus &mu) {
        return 2 * mu.value * (1 + nu.value);
    }

    inline real_t youngs_modulus_value(const PoissonRatio &nu, const LameSecondParameter &mu) {
        return 2 * mu.value * (1 + nu.value);
    }

    inline real_t youngs_modulus_value(const PoissonRatio &nu, const LameFirstParameter &lambda) {
        return lambda.value * (1 + nu.value) * (1 - 2 * nu.value) / nu.value;
    }

    inline real_t youngs_modulus_value(const BulkModulus &K, const ShearModulus &mu) {
        return 9 * K.value * mu.value / (3 * K.value + mu.value);
    }

    inline real_t youngs_modulus_value(const BulkModulus &K, const LameSecondParameter &mu) {
        return 9 * K.value * mu.value / (3 * K.value + mu.value);
    }

    inline real_t youngs_modulus_value(const BulkModulus &K, const LameFirstParameter &lambda) {
        return 9 * K.value * (K.value - lambda.value) / (3 * K.value - lambda.value);
    }

    inline real_t youngs_modulus_value(const LameFirstParameter &lambda, const ShearModulus &mu) {
        return detail::youngs_from_lame_first_and_shear(lambda.value, mu.value);
    }

    inline real_t youngs_modulus_value(const LameFirstParameter &lambda, const LameSecondParameter &mu) {
        return detail::youngs_from_lame_first_and_shear(lambda.value, mu.value);
    }

    template <typename A, typename B>
    inline real_t poisson_ratio_value(const A &a, const B &b) {
        return poisson_ratio_value(b, a);
    }

    inline real_t poisson_ratio_value(const YoungsModulus &E, const BulkModulus &K) {
        return (3 * K.value - E.value) / (6 * K.value);
    }

    inline real_t poisson_ratio_value(const YoungsModulus &E, const ShearModulus &mu) {
        return E.value / (2 * mu.value) - 1;
    }

    inline real_t poisson_ratio_value(const YoungsModulus &E, const LameSecondParameter &mu) {
        return E.value / (2 * mu.value) - 1;
    }

    inline real_t poisson_ratio_value(const YoungsModulus &E, const LameFirstParameter &lambda) {
        return (detail::elasticity_sqrt(E.value * E.value + 2 * E.value * lambda.value + 9 * lambda.value * lambda.value) -
                E.value - lambda.value) /
               (4 * lambda.value);
    }

    inline real_t poisson_ratio_value(const BulkModulus &K, const ShearModulus &mu) {
        return (3 * K.value - 2 * mu.value) / (2 * (3 * K.value + mu.value));
    }

    inline real_t poisson_ratio_value(const BulkModulus &K, const LameSecondParameter &mu) {
        return (3 * K.value - 2 * mu.value) / (2 * (3 * K.value + mu.value));
    }

    inline real_t poisson_ratio_value(const BulkModulus &K, const LameFirstParameter &lambda) {
        return lambda.value / (3 * K.value - lambda.value);
    }

    inline real_t poisson_ratio_value(const LameFirstParameter &lambda, const ShearModulus &mu) {
        return lambda.value / (2 * (lambda.value + mu.value));
    }

    inline real_t poisson_ratio_value(const LameFirstParameter &lambda, const LameSecondParameter &mu) {
        return lambda.value / (2 * (lambda.value + mu.value));
    }

    template <typename A, typename B>
    inline real_t bulk_modulus_value(const A &a, const B &b) {
        return bulk_modulus_value(b, a);
    }

    inline real_t bulk_modulus_value(const YoungsModulus &E, const PoissonRatio &nu) {
        return E.value / (3 * (1 - 2 * nu.value));
    }

    inline real_t bulk_modulus_value(const YoungsModulus &E, const ShearModulus &mu) {
        return E.value * mu.value / (3 * (3 * mu.value - E.value));
    }

    inline real_t bulk_modulus_value(const YoungsModulus &E, const LameSecondParameter &mu) {
        return E.value * mu.value / (3 * (3 * mu.value - E.value));
    }

    inline real_t bulk_modulus_value(const YoungsModulus &E, const LameFirstParameter &lambda) {
        return (E.value + 3 * lambda.value +
                detail::elasticity_sqrt(E.value * E.value + 2 * E.value * lambda.value + 9 * lambda.value * lambda.value)) /
               6;
    }

    inline real_t bulk_modulus_value(const PoissonRatio &nu, const ShearModulus &mu) {
        return 2 * mu.value * (1 + nu.value) / (3 * (1 - 2 * nu.value));
    }

    inline real_t bulk_modulus_value(const PoissonRatio &nu, const LameSecondParameter &mu) {
        return 2 * mu.value * (1 + nu.value) / (3 * (1 - 2 * nu.value));
    }

    inline real_t bulk_modulus_value(const PoissonRatio &nu, const LameFirstParameter &lambda) {
        return lambda.value * (1 + nu.value) / (3 * nu.value);
    }

    inline real_t bulk_modulus_value(const LameFirstParameter &lambda, const ShearModulus &mu) {
        return lambda.value + 2 * mu.value / 3;
    }

    inline real_t bulk_modulus_value(const LameFirstParameter &lambda, const LameSecondParameter &mu) {
        return lambda.value + 2 * mu.value / 3;
    }

    template <typename A, typename B>
    inline real_t shear_modulus_value(const A &a, const B &b) {
        return shear_modulus_value(b, a);
    }

    inline real_t shear_modulus_value(const YoungsModulus &E, const PoissonRatio &nu) {
        return E.value / (2 * (1 + nu.value));
    }

    inline real_t shear_modulus_value(const YoungsModulus &E, const BulkModulus &K) {
        return 3 * K.value * E.value / (9 * K.value - E.value);
    }

    inline real_t shear_modulus_value(const YoungsModulus &E, const LameFirstParameter &lambda) {
        return detail::shear_from_youngs_and_lame_first(E.value, lambda.value);
    }

    inline real_t shear_modulus_value(const PoissonRatio &nu, const BulkModulus &K) {
        return 3 * K.value * (1 - 2 * nu.value) / (2 * (1 + nu.value));
    }

    inline real_t shear_modulus_value(const PoissonRatio &nu, const LameFirstParameter &lambda) {
        return lambda.value * (1 - 2 * nu.value) / (2 * nu.value);
    }

    inline real_t shear_modulus_value(const BulkModulus &K, const LameFirstParameter &lambda) {
        return 3 * (K.value - lambda.value) / 2;
    }

    template <typename A, typename B>
    inline real_t lame_first_parameter_value(const A &a, const B &b) {
        return lame_first_parameter_value(b, a);
    }

    inline real_t lame_first_parameter_value(const YoungsModulus &E, const PoissonRatio &nu) {
        return E.value * nu.value / ((1 + nu.value) * (1 - 2 * nu.value));
    }

    inline real_t lame_first_parameter_value(const YoungsModulus &E, const BulkModulus &K) {
        return 3 * K.value * (3 * K.value - E.value) / (9 * K.value - E.value);
    }

    inline real_t lame_first_parameter_value(const YoungsModulus &E, const ShearModulus &mu) {
        return mu.value * (E.value - 2 * mu.value) / (3 * mu.value - E.value);
    }

    inline real_t lame_first_parameter_value(const YoungsModulus &E, const LameSecondParameter &mu) {
        return mu.value * (E.value - 2 * mu.value) / (3 * mu.value - E.value);
    }

    inline real_t lame_first_parameter_value(const PoissonRatio &nu, const BulkModulus &K) {
        return 3 * K.value * nu.value / (1 + nu.value);
    }

    inline real_t lame_first_parameter_value(const PoissonRatio &nu, const ShearModulus &mu) {
        return 2 * mu.value * nu.value / (1 - 2 * nu.value);
    }

    inline real_t lame_first_parameter_value(const PoissonRatio &nu, const LameSecondParameter &mu) {
        return 2 * mu.value * nu.value / (1 - 2 * nu.value);
    }

    inline real_t lame_first_parameter_value(const BulkModulus &K, const ShearModulus &mu) {
        return K.value - 2 * mu.value / 3;
    }

    inline real_t lame_first_parameter_value(const BulkModulus &K, const LameSecondParameter &mu) {
        return K.value - 2 * mu.value / 3;
    }

    inline ShearModulus::ShearModulus(const LameSecondParameter &mu) : value(mu.value) {}

    inline LameSecondParameter::LameSecondParameter(const ShearModulus &mu) : value(mu.value) {}

    template <typename A, typename B>
    inline YoungsModulus::YoungsModulus(const A &a, const B &b) : value(youngs_modulus_value(a, b)) {}

    template <typename A, typename B>
    inline PoissonRatio::PoissonRatio(const A &a, const B &b) : value(poisson_ratio_value(a, b)) {}

    template <typename A, typename B>
    inline BulkModulus::BulkModulus(const A &a, const B &b) : value(bulk_modulus_value(a, b)) {}

    template <typename A, typename B>
    inline ShearModulus::ShearModulus(const A &a, const B &b) : value(shear_modulus_value(a, b)) {}

    template <typename A, typename B>
    inline LameFirstParameter::LameFirstParameter(const A &a, const B &b) : value(lame_first_parameter_value(a, b)) {}

    template <typename A, typename B>
    inline LameSecondParameter::LameSecondParameter(const A &a, const B &b) : value(shear_modulus_value(a, b)) {}

}  // namespace sfem

#endif  // SFEM_ELASTICITY_PARAMETERS_HPP
