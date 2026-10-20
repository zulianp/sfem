#include "sfem_LineSearch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sfem_aliases.hpp"

namespace sfem {

    const char *to_string(const LineSearchResult::Status status) {
        switch (status) {
            case LineSearchResult::ACCEPTED:
                return "accepted";
            case LineSearchResult::BEST_DECREASE:
                return "best_decrease";
            case LineSearchResult::AT_RESOLUTION:
                return "at_resolution";
            case LineSearchResult::NO_DECREASE:
                return "no_decrease";
            case LineSearchResult::NON_FINITE:
                return "non_finite";
        }
        return "unknown";
    }

    LineSearch::LineSearch(const LineSearchOptions &options) : options_(options) {}

    LineSearchResult LineSearch::search(Merit              &merit,
                                        const real_t *const x,
                                        const real_t *const h,
                                        const real_t        m0,
                                        const real_t        slope) {
        SFEM_TRACE_SCOPE("LineSearch::search");

        LineSearchResult result;
        if (!options_.enabled) {
            result.status = LineSearchResult::ACCEPTED;
            result.alpha  = 1;
            result.merit  = std::numeric_limits<real_t>::quiet_NaN();
            return result;
        }

        const auto &steps = options_.steps;
        const int   n     = (int)steps.size();
        values_.assign(n, std::numeric_limits<real_t>::infinity());

        if (options_.rule == LineSearchOptions::Rule::DAMPED) {
            // The damped step, shortened only where the state cannot be evaluated.
            for (int i = 0; i < n; ++i) {
                const real_t s = options_.damping * steps[i];
                merit.evaluate(x, h, 1, &s, &values_[i]);
                result.evaluations++;
                if (std::isfinite(values_[i])) {
                    result.status = LineSearchResult::ACCEPTED;
                    result.alpha  = s;
                    result.merit  = values_[i];
                    return result;
                }
            }
            result.status = LineSearchResult::NON_FINITE;
            result.alpha  = 0;
            result.merit  = m0;
            return result;
        }

        // Armijo needs a descent direction; along an ascent one only a measured decrease counts.
        auto armijo = [&](const int i) {
            return slope < 0 && values_[i] <= m0 + options_.c1 * steps[i] * slope;
        };

        int accepted = -1;
        if (options_.batched && merit.batched()) {
            merit.evaluate(x, h, n, steps.data(), values_.data());
            result.evaluations = n;
            for (int i = 0; i < n; ++i) {
                if (armijo(i)) {
                    accepted = i;
                    break;
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                merit.evaluate(x, h, 1, &steps[i], &values_[i]);
                result.evaluations++;
                if (armijo(i)) {
                    accepted = i;
                    break;
                }
            }
        }

        if (accepted >= 0) {
            result.status = LineSearchResult::ACCEPTED;
            result.alpha  = steps[accepted];
            result.merit  = values_[accepted];
            return result;
        }

        int best = -1;
        for (int i = 0; i < n; ++i) {
            if (std::isfinite(values_[i]) && (best < 0 || values_[i] < values_[best])) best = i;
        }

        if (best >= 0 && values_[best] < m0) {
            result.status = LineSearchResult::BEST_DECREASE;
            result.alpha  = steps[best];
            result.merit  = values_[best];
            return result;
        }

        result.alpha = 0;
        result.merit = m0;
        if (best < 0) {
            result.status = LineSearchResult::NON_FINITE;
            return result;
        }

        // A merit can only see a decrease while it is larger than the rounding in evaluating it,
        // and near the solution it is not: the decrease falls with the square of the step and the
        // rounding does not move.
        const real_t floor = 64 * std::numeric_limits<real_t>::epsilon() * std::max(std::abs(m0), real_t(1));
        result.status      = std::abs(slope) < floor ? LineSearchResult::AT_RESOLUTION : LineSearchResult::NO_DECREASE;
        return result;
    }

}  // namespace sfem
