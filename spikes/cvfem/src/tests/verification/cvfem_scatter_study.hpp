#ifndef CVFEM_SCATTER_STUDY_HPP
#define CVFEM_SCATTER_STUDY_HPP

#include <limits>

template <typename Production>
static int run_scatter_study(MeshData &d, PackedData &p, const bool packed,
                             const scalar_t rho, const scalar_t mu, const scalar_t *ugrad,
                             const int limiter, const scalar_t venkat_c,
                             const int repeat, const int warmup, const int samples,
                             Production production) {
    int team_size = 1;
#pragma omp parallel
    {
#pragma omp single
        team_size = cvfem_n_threads();
    }
    if (team_size != threads_active()) {
        std::fprintf(stderr, "scatter_gate FAIL: requested %d threads, got %d\n",
                     threads_active(), team_size);
        return 1;
    }
    auto probe = [&]() {
        if (packed) {
            if (ugrad) apply_residual_packed_defcor<true>(d, p, rho, mu, ugrad, limiter, venkat_c);
            else apply_residual_packed<true>(d, p, rho, mu, GeomKind::Affine);
        } else {
            apply_residual_atomic_sumfact_simd<true>(
                d.adj_ptr, d.det_ptr, d.elems, d.nelements, d.nnodes, d.p.data(),
                d.pgx.data(), d.pgy.data(), d.pgz.data(), d.points, d.rc.data(), d.rhie_chow_scale,
                d.rx.data(), d.ry.data(), d.rz.data(), d.ux.data(), d.uy.data(), d.uz.data(),
                cvfem_hex8_extras_of(d), rho, mu, ugrad, limiter, venkat_c);
        }
    };
    bool passed = true;
    {
        production();
        std::vector<scalar_t> reference, full, bypass;
        pack_residual(d, reference);
        // Captures cover every active element lane and are freed before timing.
        const size_t count = size_t(d.nelements) * 32;
        std::vector<double> a(count, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> b(count, std::numeric_limits<double>::quiet_NaN());
        g_scatter_capture = a.data();
        g_scatter_accumulate = true;
        probe();
        pack_residual(d, full);
        reset_residual(d.nnodes, d.rx.data(), d.ry.data(), d.rz.data(), d.rc.data());
        g_scatter_capture = b.data();
        g_scatter_accumulate = false;
        probe();
        pack_residual(d, bypass);
        g_scatter_capture = nullptr;
        g_scatter_accumulate = true;

        bool finite = true, nonzero = false, bypass_zero = true;
        for (size_t i = 0; i < count; ++i) {
            finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
            nonzero = nonzero || a[i] != 0;
        }
        const bool identical = std::memcmp(a.data(), b.data(), count * sizeof(double)) == 0;
        for (const auto v : bypass) bypass_zero = bypass_zero && v == scalar_t(0);
        passed = finite && nonzero && identical && bypass_zero;
        std::printf("scatter_capture entries=%zu finite=%d nonzero=%d identical=%d bypass_zero=%d\n",
                    count, int(finite), int(nonzero), int(identical), int(bypass_zero));
        double total_scale = 0;
        for (int f = 0; f < 4; ++f) {
            double scale = 0, error = 0;
            for (size_t i = f; i < reference.size(); i += 4) {
                passed = passed && std::isfinite(reference[i]) && std::isfinite(full[i]);
                scale = std::max(scale, std::abs(double(reference[i])));
                error = std::max(error, std::abs(double(reference[i] - full[i])));
            }
            total_scale = std::max(total_scale, scale);
            const double relative = error / std::max(scale, 1e-30);
            passed = passed && relative <= 1e-11;
            std::printf("scatter_equivalence field=%d relative=%.17g tolerance=1e-11\n", f, relative);
        }
        passed = passed && total_scale > 0;
    }
    if (!passed) {
        std::fprintf(stderr, "scatter_gate FAIL; no timings recorded\n");
        return 1;
    }
    std::puts("scatter_gate PASS");
    std::printf("scatter_config layout=%s elements=%td nodes=%td threads=%d ho=%d limiter=%d "
                "repeat=%d warmup=%d samples=%d\n", packed ? "packed" : "atomic", d.nelements,
                d.nnodes, team_size, int(ugrad != nullptr), limiter, repeat, warmup, samples);
    // All six orders balance drift and the preceding mode's cache state.
    const int orders[6][3] = {{0,1,2}, {2,1,0}, {1,2,0}, {0,2,1}, {2,0,1}, {1,0,2}};
    auto run = [&](const int mode) {
        g_scatter_accumulate = mode != 2;
        if (mode == 0) production();
        else probe();
    };
    for (int sample = 0; sample < samples; ++sample) {
        double seconds[3] = {};
        for (const int mode : orders[sample % 6]) {
            for (int i = 0; i < warmup; ++i) run(mode);
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeat; ++i) run(mode);
            seconds[mode] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repeat;
            if (!(seconds[mode] > 0) || !std::isfinite(seconds[mode])) return 1;
        }
        std::printf("scatter_sample %d %.17g %.17g %.17g\n",
                    sample, seconds[0], seconds[1], seconds[2]);
        std::fflush(stdout);
    }
    g_scatter_accumulate = true;
    std::puts("scatter_study PASS");
    return 0;
}

#endif
