// The material-independent half of Prony viscoelasticity, against closed forms.
//
// `PronyRelaxation` is pointwise: every (component, point, element) evolves on
// its own, so a store of many elements driven by a scalar stress history times
// a per-entry amplitude must reproduce the scalar hereditary integral at every
// entry.  For a stress that is linear in time over each step that integral is
//
//     H_i(t_{n+1}) = exp(-dt/tau_i) H_i(t_n) + g_i tau_i/dt (1 - exp(-dt/tau_i)) (S_{n+1} - S_n),
//
// which is the linear-interpolation recursion exactly, so it must match to
// round-off for any sequence of step lengths.  The midpoint recursion is
// second order and is checked by its convergence rate.

#include "sfem_test.hpp"

#include "sfem_HistoryStorage.hpp"
#include "sfem_PronyRelaxation.hpp"
#include "sfem_QuadratureState.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

    constexpr int       N_COMPONENTS = 6;
    constexpr int       N_QP         = 8;
    constexpr ptrdiff_t N_ELEMENTS   = 100;  // one full block and a remainder
    constexpr ptrdiff_t N_ENTRIES    = N_COMPONENTS * N_QP * N_ELEMENTS;

    sfem::PronySeries series() { return sfem::PronySeries({0.3, 0.25, 0.15}, {0.1, 1.0, 10.0}); }

    // A spread of signs and magnitudes, and a per-element range of eight
    // decades, so a codec that scales per element is exercised where it matters.
    std::vector<real_t> amplitudes() {
        std::vector<real_t> a(N_ENTRIES);
        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
            const ptrdiff_t e = k % N_ELEMENTS;
            const real_t    m = std::pow(real_t(10), real_t(e % 9) - 3);
            a[k]              = m * (real_t(1) + real_t((k * 37) % 11) / 7) * ((k % 3) ? 1 : -1);
        }
        return a;
    }

    // A ramp to 1 over [0, t_ramp], then held.  Piecewise linear, so the
    // linear-interpolation recursion is exact on any grid that has t_ramp on it.
    real_t ramp(const real_t t, const real_t t_ramp) { return std::min(t / t_ramp, real_t(1)); }

    // H_i of the scalar ramp history, in closed form.
    real_t exact_H(const sfem::PronySeries &s, const int i, const real_t t, const real_t t_ramp) {
        const real_t tau = s.tau(i);
        const real_t k   = 1 / t_ramp;
        if (t <= t_ramp) return s.g(i) * k * tau * (1 - std::exp(-t / tau));
        return s.g(i) * k * tau * (1 - std::exp(-t_ramp / tau)) * std::exp(-(t - t_ramp) / tau);
    }

    // Total algorithmic stress, g_inf S + sum_i H_i, of the scalar history.
    real_t exact_stress(const sfem::PronySeries &s, const real_t t, const real_t t_ramp) {
        real_t sigma = s.g_inf() * ramp(t, t_ramp);
        for (int i = 0; i < s.n_terms(); i++) sigma += exact_H(s, i, t, t_ramp);
        return sigma;
    }

    // Drive `relaxation` through the ramp on the grid `times` (times[0] = 0) and
    // return, per step, the max over entries of the relative error of the
    // algorithmic stress gamma S + S_hist.
    //
    // The relaxed stress is the caller's to evaluate at both ends of a step, as
    // an operator evaluates it from the displacements; here it is the scalar
    // history times the amplitudes, at the previous and the current time.
    real_t drive(sfem::PronyRelaxation &relaxation, const std::vector<real_t> &times, const real_t t_ramp) {
        const auto          a = amplitudes();
        std::vector<real_t> S(N_ENTRIES), S_n(N_ENTRIES);
        real_t              worst = 0;

        for (size_t n = 1; n < times.size(); n++) {
            const real_t t  = times[n];
            const real_t dt = times[n] - times[n - 1];

            const real_t s_n = ramp(times[n - 1], t_ramp);
            for (ptrdiff_t k = 0; k < N_ENTRIES; k++) S_n[k] = a[k] * s_n;
            relaxation.begin_step(t, dt, S_n.data());

            const real_t s = ramp(t, t_ramp);
            for (ptrdiff_t k = 0; k < N_ENTRIES; k++) S[k] = a[k] * s;

            const real_t *const hist  = relaxation.history();
            const real_t        exact = exact_stress(relaxation.series(), t, t_ramp);
            for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
                const real_t computed = relaxation.gamma() * S[k] + hist[k];
                worst                 = std::max(worst, std::abs(computed - a[k] * exact) / std::abs(a[k]));
            }

            relaxation.advance(S_n.data(), S.data());
        }

        return worst;
    }

    std::vector<real_t> uniform_grid(const real_t t_end, const int n) {
        std::vector<real_t> t(n + 1);
        for (int i = 0; i <= n; i++) t[i] = t_end * i / n;
        return t;
    }

}  // namespace

int test_the_instantaneous_response_before_any_step() {
    sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS);
    SFEM_TEST_ASSERT(relaxation.gamma() == 1);
    for (ptrdiff_t k = 0; k < N_ENTRIES; k++) SFEM_TEST_ASSERT(relaxation.history()[k] == 0);
    return SFEM_TEST_SUCCESS;
}

int test_the_weights_have_their_limits() {
    const auto                           s = series();
    std::vector<real_t>                  alpha(s.n_terms()), b(s.n_terms());
    sfem::LinearInterpolationIntegrator linear;

    // dt -> 0 recovers the instantaneous response, gamma -> 1.
    linear.weights(s, real_t(1e-12), alpha.data(), b.data());
    real_t gamma = s.g_inf();
    for (int i = 0; i < s.n_terms(); i++) gamma += b[i];
    SFEM_TEST_APPROXEQ(gamma, real_t(1), real_t(1e-10));

    // dt -> infinity relaxes every term, gamma -> g_inf, and no weight is
    // dropped to get there.
    linear.weights(s, real_t(1e6), alpha.data(), b.data());
    gamma = s.g_inf();
    for (int i = 0; i < s.n_terms(); i++) gamma += b[i];
    SFEM_TEST_APPROXEQ(gamma, s.g_inf(), real_t(1e-5));
    for (int i = 0; i < s.n_terms(); i++) SFEM_TEST_ASSERT(alpha[i] == 0 || alpha[i] < real_t(1e-30));
    return SFEM_TEST_SUCCESS;
}

int test_linear_interpolation_is_exact_for_a_ramp() {
    sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS);
    const real_t          err = drive(relaxation, uniform_grid(20, 40), 1);
    SFEM_TEST_ASSERT(err < real_t(1e-12));
    return SFEM_TEST_SUCCESS;
}

int test_variable_steps_stay_exact() {
    // Steps that double then shrink: the recursion carries no assumption that
    // dt is fixed, and retrying a step with another dt before committing is
    // allowed.
    std::vector<real_t> t{0, real_t(0.25), real_t(0.5), 1};
    real_t              dt = real_t(0.125);
    while (t.back() < 30) {
        t.push_back(t.back() + dt);
        dt = dt < 4 ? dt * 2 : real_t(0.3);
    }

    sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS);
    const std::vector<real_t> at_rest(N_ENTRIES, 0);
    relaxation.begin_step(0, 5, at_rest.data());  // a trial that is retried below
    const real_t err = drive(relaxation, t, 1);
    SFEM_TEST_ASSERT(err < real_t(1e-12));
    return SFEM_TEST_SUCCESS;
}

int test_midpoint_is_second_order() {
    // On a smooth history (a ramp that never ends) the midpoint recursion
    // converges at second order.
    real_t errors[3];
    for (int r = 0; r < 3; r++) {
        sfem::PronyRelaxation::Options options;
        options.integrator = std::make_shared<sfem::MidpointIntegrator>();
        sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS, options);
        errors[r] = drive(relaxation, uniform_grid(2, 80 << r), 100);
    }

    const real_t rate0 = std::log2(errors[0] / errors[1]);
    const real_t rate1 = std::log2(errors[1] / errors[2]);
    SFEM_TEST_ASSERT(rate0 > real_t(1.9) && rate1 > real_t(1.9));
    return SFEM_TEST_SUCCESS;
}

int test_wlf_shifts_time_and_nothing_else() {
    // At a temperature with shift a_T, a step dt advances the reduced time
    // dt / a_T: the history at time t equals the unshifted history at t / a_T.
    const real_t C1 = real_t(8.86), C2 = real_t(101.6), T_ref = 0, T = 10;
    auto         wlf = std::make_shared<sfem::WLFShift>(C1, C2, T_ref);
    const real_t a_T = wlf->factor(T);
    SFEM_TEST_ASSERT(a_T < 1);  // warmer than the reference relaxes faster
    SFEM_TEST_APPROXEQ(std::log10(a_T), -C1 * (T - T_ref) / (C2 + T - T_ref), real_t(1e-14));

    sfem::PronyRelaxation::Options options;
    options.shift       = wlf;
    options.temperature = T;
    sfem::PronyRelaxation shifted(series(), N_COMPONENTS, N_QP, N_ELEMENTS, options);
    sfem::PronyRelaxation reference(series(), N_COMPONENTS, N_QP, N_ELEMENTS);

    auto grid    = uniform_grid(10, 25);
    auto reduced = grid;
    for (auto &t : reduced) t /= a_T;

    // The ramp lives in reduced time too, so the two runs see one history.
    drive(shifted, grid, a_T * real_t(0.8));
    drive(reference, reduced, real_t(0.8));

    std::vector<real_t> h0(N_ENTRIES), h1(N_ENTRIES);
    for (int i = 0; i < series().n_terms(); i++) {
        shifted.internal_variables().decode(i, h0.data());
        reference.internal_variables().decode(i, h1.data());
        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
            SFEM_TEST_ASSERT(std::abs(h0[k] - h1[k]) <= real_t(1e-12) * (std::abs(h1[k]) + 1e-30));
        }
    }
    return SFEM_TEST_SUCCESS;
}

int test_every_codec_round_trips_within_its_bound() {
    const auto         a = amplitudes();
    const sfem::HistoryStorage formats[] = {sfem::HistoryStorage::FULL,
                                            sfem::HistoryStorage::FLOAT32,
                                            sfem::HistoryStorage::FLOAT16_SCALED,
                                            sfem::HistoryStorage::INT16_SCALED,
                                            sfem::HistoryStorage::INT8_SCALED};
    // Relative to the max-abs of the tensor of an element: what a per-element
    // scale can promise.
    const real_t bounds[] = {0, real_t(6e-8), real_t(1e-3), real_t(2e-5), real_t(4.5e-3)};

    for (int f = 0; f < 5; f++) {
        sfem::QuadratureState state(2, N_COMPONENTS, N_QP, N_ELEMENTS, formats[f]);
        state.encode(1, a.data());

        std::vector<real_t> back(N_ENTRIES);
        state.decode(1, back.data());

        std::vector<real_t> max_abs(N_ELEMENTS, 0);
        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
            max_abs[k % N_ELEMENTS] = std::max(max_abs[k % N_ELEMENTS], std::abs(a[k]));
        }

        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
            const real_t rel = std::abs(back[k] - a[k]) / max_abs[k % N_ELEMENTS];
            if (!(rel <= bounds[f])) {
                printf("%s: entry %ld relative error %g above %g\n", sfem::to_string(formats[f]), (long)k, rel, bounds[f]);
                return SFEM_TEST_FAILURE;
            }
        }

        // Tensor 0 was never written and is still zero.
        state.decode(0, back.data());
        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) SFEM_TEST_ASSERT(back[k] == 0);

        SFEM_TEST_ASSERT(state.bytes() == size_t(2 * N_ENTRIES) * sfem::bytes_per_value(formats[f]) +
                                                  size_t(2 * N_ELEMENTS) * sizeof(scaling_t));
    }
    return SFEM_TEST_SUCCESS;
}

int test_unscaled_float16_round_trips_below_its_range() {
    // No scale, so the values have to sit inside binary16's range on their own;
    // above 65504 it overflows, which is what the scaled format is for.
    auto a = amplitudes();
    for (auto &v : a) v *= real_t(0.1);
    sfem::QuadratureState state(1, N_COMPONENTS, N_QP, N_ELEMENTS, sfem::HistoryStorage::FLOAT16);
    state.encode(0, a.data());
    std::vector<real_t> back(N_ENTRIES);
    state.decode(0, back.data());
    for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
        SFEM_TEST_ASSERT(std::abs(back[k] - a[k]) <= std::ldexp(1.0, -11) * std::abs(a[k]));
    }
    return SFEM_TEST_SUCCESS;
}

int test_reduced_storage_tracks_full_storage() {
    // Only H_i is stored; the relaxed stress is evaluated in real_t at both
    // ends of every step.  The recursion re-encodes H_i every step, so its
    // rounding accumulates: a rounding of eps (relative to the largest entry of
    // an element's tensor) per step is carried forward by term i with weight
    // alpha_i per step, so it sums to at most g_i eps / (1 - alpha_i).  Relative
    // to an entry rather than to the largest, it is amplified by the spread of
    // the entries of an element.  That worst case is the bound; rounding that
    // does not always fall the same way stays well inside it.
    const sfem::HistoryStorage formats[] = {
            sfem::HistoryStorage::FLOAT32, sfem::HistoryStorage::FLOAT16_SCALED, sfem::HistoryStorage::INT16_SCALED, sfem::HistoryStorage::INT8_SCALED};
    const real_t eps[] = {real_t(std::ldexp(1.0, -24)), real_t(std::ldexp(1.0, -11)), real_t(0.5 / 32767), real_t(0.5 / 127)};

    // dt = 0.3 and the ramp ends on the grid, so full storage is exact and
    // what is measured is the storage alone.
    const real_t dt   = real_t(0.3);
    const auto   grid = uniform_grid(30, 100);

    const auto a      = amplitudes();
    real_t     spread = 1;
    {
        std::vector<real_t> lo(N_ELEMENTS, 1e300), hi(N_ELEMENTS, 0);
        for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
            lo[k % N_ELEMENTS] = std::min(lo[k % N_ELEMENTS], std::abs(a[k]));
            hi[k % N_ELEMENTS] = std::max(hi[k % N_ELEMENTS], std::abs(a[k]));
        }
        for (ptrdiff_t e = 0; e < N_ELEMENTS; e++) spread = std::max(spread, hi[e] / lo[e]);
    }

    real_t carried = 0;
    for (int i = 0; i < series().n_terms(); i++) carried += series().g(i) / (1 - std::exp(-dt / series().tau(i)));

    for (int f = 0; f < 4; f++) {
        sfem::PronyRelaxation::Options options;
        options.history_storage = formats[f];
        sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS, options);
        const real_t          err   = drive(relaxation, grid, real_t(0.9));
        const real_t          bound = spread * eps[f] * carried;
        printf("%-15s worst relative stress error over %d steps: %g, bound %g (%zu bytes)\n",
               sfem::to_string(formats[f]),
               (int)grid.size() - 1,
               (double)err,
               (double)bound,
               relaxation.bytes());
        SFEM_TEST_ASSERT(err < bound);
    }
    return SFEM_TEST_SUCCESS;
}

int test_reset_restarts_from_a_prestressed_state() {
    const auto            a = amplitudes();
    sfem::PronyRelaxation relaxation(series(), N_COMPONENTS, N_QP, N_ELEMENTS);
    drive(relaxation, uniform_grid(2, 4), 1);

    // At rest under a held stress a, a fresh step sees only that stress.
    relaxation.reset();
    relaxation.begin_step(0, real_t(0.5), a.data());
    for (ptrdiff_t k = 0; k < N_ENTRIES; k++) {
        real_t b_sum = 0;
        for (int i = 0; i < series().n_terms(); i++) b_sum += relaxation.b(i);
        SFEM_TEST_APPROXEQ(relaxation.history()[k], -b_sum * a[k], real_t(1e-14) * std::abs(a[k]));
    }
    return SFEM_TEST_SUCCESS;
}

int main(int argc, char *argv[]) {
    SFEM_UNIT_TEST_INIT(argc, argv);
    SFEM_RUN_TEST(test_the_instantaneous_response_before_any_step);
    SFEM_RUN_TEST(test_the_weights_have_their_limits);
    SFEM_RUN_TEST(test_linear_interpolation_is_exact_for_a_ramp);
    SFEM_RUN_TEST(test_variable_steps_stay_exact);
    SFEM_RUN_TEST(test_midpoint_is_second_order);
    SFEM_RUN_TEST(test_wlf_shifts_time_and_nothing_else);
    SFEM_RUN_TEST(test_every_codec_round_trips_within_its_bound);
    SFEM_RUN_TEST(test_unscaled_float16_round_trips_below_its_range);
    SFEM_RUN_TEST(test_reduced_storage_tracks_full_storage);
    SFEM_RUN_TEST(test_reset_restarts_from_a_prestressed_state);
    SFEM_UNIT_TEST_FINALIZE();
    return SFEM_UNIT_TEST_ERR();
}
