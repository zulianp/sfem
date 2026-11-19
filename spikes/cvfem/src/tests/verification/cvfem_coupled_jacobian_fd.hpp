#ifndef CVFEM_COUPLED_JACOBIAN_FD_HPP
#define CVFEM_COUPLED_JACOBIAN_FD_HPP

#include <array>
#include <limits>

// The caller exits after verification, so the fixture cannot enter a timing run.
static int verify_coupled_jacobian_fd(MeshData &d, PackedData &p, const scalar_t rho,
                                      const scalar_t mu, const bool with_ho,
                                      const int limiter, const scalar_t venkat_c) {
    const ptrdiff_t n = d.nnodes;
    const auto cfg = cvfem_hex8_rc_config_for(d);
    if (cfg.tau.u2_scale != scalar_t(1) || cfg.tau.inv_dt_a0 != scalar_t(0) ||
        cfg.scale != d.rhie_chow_scale) {
        std::fprintf(stderr, "FAIL: coupled verification requires the default steady RC coefficient\n");
        return 2;
    }
    std::vector<scalar_t> dir(size_t(n) * N_FIELDS);
    for (ptrdiff_t i = 0; i < n; ++i) {
        const uint32_t h = uint32_t(i) * 2246822519u + 374761393u;
        const auto noise = [&](const int shift) {
            return scalar_t((h >> shift) & 255u) / scalar_t(255) - scalar_t(0.5);
        };
        // Move the affine initialization off structural limiter and donor ties.
        d.ux[i] += scalar_t(0.03) * noise(0);
        d.uy[i] += scalar_t(0.03) * noise(8);
        d.uz[i] += scalar_t(0.03) * noise(16);
        d.p[i] += scalar_t(0.03) * noise(3);
        for (int c = 0; c < N_FIELDS; ++c) dir[size_t(i) * N_FIELDS + c] = scalar_t(0.25) + noise(c * 5);
    }
    const auto ux = d.ux, uy = d.uy, uz = d.uz, pressure = d.p;
    std::vector<scalar_t> gu, gv, buffer, rp, rm;
    std::vector<scalar_t> exact(dir.size()), no_qgrad(dir.size()), no_coeff_du(dir.size());
    auto set_state = [&](const scalar_t step) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            d.ux[i] = ux[i] + step * dir[size_t(i) * N_FIELDS];
            d.uy[i] = uy[i] + step * dir[size_t(i) * N_FIELDS + 1];
            d.uz[i] = uz[i] + step * dir[size_t(i) * N_FIELDS + 2];
            d.p[i] = pressure[i] + step * dir[size_t(i) * N_FIELDS + 3];
        }
        ++d.state_stamp;
        cvfem_hex8_assemble_nodal_grads_packed(d, p, 0, d.p.data(), 1, d.pgx, d.pgy, d.pgz, buffer);
        if (with_ho) {
            const scalar_t *src[3] = {d.ux.data(), d.uy.data(), d.uz.data()};
            cvfem_hex8_assemble_nodal_grads_packed(d, p, 0, src, 3, gu, buffer);
        }
    };
    auto action = [&](std::vector<scalar_t> &out) {
        if (with_ho)
            apply_jacobian_action_packed(d, p, rho, mu, GeomKind::Affine, dir.data(), out.data(),
                                        gu.data(), gv.data(), limiter, venkat_c);
        else
            apply_jacobian_action_packed(d, p, rho, mu, GeomKind::Affine, dir.data(), out.data());
    };
    set_state(scalar_t(0));
    cvfem_hex8_assemble_nodal_grads_packed(d, p, 0, dir.data() + 3, N_FIELDS,
                                        d.qgx, d.qgy, d.qgz, buffer);
    if (with_ho) {
        const scalar_t *src[3] = {dir.data(), dir.data() + 1, dir.data() + 2};
        const int strides[3] = {N_FIELDS, N_FIELDS, N_FIELDS};
        cvfem_hex8_assemble_nodal_grads_packed(d, p, 0, src, 3, gv, buffer, strides);
    }
    action(exact);

    // Each omitted sensitivity must be visible to the finite-difference oracle.
    auto qgx = d.qgx, qgy = d.qgy, qgz = d.qgz;
    std::fill(d.qgx.begin(), d.qgx.end(), scalar_t(0));
    std::fill(d.qgy.begin(), d.qgy.end(), scalar_t(0));
    std::fill(d.qgz.begin(), d.qgz.end(), scalar_t(0));
    action(no_qgrad);
    d.qgx.swap(qgx); d.qgy.swap(qgy); d.qgz.swap(qgz);
    // Preserve the cache key so the action uses these zero derivative weights.
    auto weights = d.rc_w;
    std::fill(d.rc_w.begin(), d.rc_w.end(), scalar_t(0));
    action(no_coeff_du);
    d.rc_w.swap(weights);

    auto residual = [&](const scalar_t step, std::vector<scalar_t> &out) {
        set_state(step);
        if (with_ho) apply_residual_packed_defcor(d, p, rho, mu, gu.data(), limiter, venkat_c);
        else apply_residual_packed(d, p, rho, mu, GeomKind::Affine);
        pack_residual(d, out);
    };
    const scalar_t tolerance = scalar_t(1e-6);
    bool passed = true;
    int required_steps = 0;
    for (const scalar_t eps : {scalar_t(1e-4), scalar_t(1e-5), scalar_t(1e-6), scalar_t(1e-7)}) {
        residual(eps, rp);
        residual(-eps, rm);
        std::array<scalar_t, N_FIELDS> scale{}, error{};
        scalar_t scale_all = 0, error_all = 0, qgrad_error = 0, coeff_error = 0;
        for (size_t i = 0; i < exact.size(); ++i) {
            const scalar_t fd = (rp[i] - rm[i]) / (scalar_t(2) * eps);
            if (!std::isfinite(fd) || !std::isfinite(exact[i]) || !std::isfinite(no_qgrad[i]) ||
                !std::isfinite(no_coeff_du[i])) {
                std::fprintf(stderr, "FAIL: nonfinite value in coupled derivative check\n");
                return 1;
            }
            const scalar_t delta = std::fabs(fd - exact[i]);
            scale[i % N_FIELDS] = std::max(scale[i % N_FIELDS], std::fabs(fd));
            error[i % N_FIELDS] = std::max(error[i % N_FIELDS], delta);
            scale_all = std::max(scale_all, std::fabs(fd));
            error_all = std::max(error_all, delta);
            qgrad_error = std::max(qgrad_error, std::fabs(fd - no_qgrad[i]));
            coeff_error = std::max(coeff_error, std::fabs(fd - no_coeff_du[i]));
        }
        if (!(scale_all > scalar_t(0))) {
            std::fprintf(stderr, "FAIL: zero coupled finite-difference reference\n");
            return 1;
        }
        bool fields_pass = true;
        for (int c = 0; c < N_FIELDS; ++c) {
            // Check each equation separately so momentum cannot hide continuity errors.
            fields_pass &= scale[c] > scalar_t(0) && error[c] < tolerance * scale[c];
        }
        const bool required = eps <= scalar_t(1e-6);
        const bool controls_fail = qgrad_error > scalar_t(10) * tolerance * scale_all &&
                                   coeff_error > scalar_t(10) * tolerance * scale_all;
        std::printf("verify_jac_coupled: ho %d limiter %d eps %.1e required %d rel %.9e "
                    "fields %.3e %.3e %.3e %.3e no_qgrad %.3e no_coeff_du %.3e\n",
                    int(with_ho), limiter, double(eps), int(required), double(error_all / scale_all),
                    double(error[0] / scale[0]), double(error[1] / scale[1]),
                    double(error[2] / scale[2]), double(error[3] / scale[3]),
                    double(qgrad_error / scale_all), double(coeff_error / scale_all));
        if (required) {
            ++required_steps;
            passed &= fields_pass && controls_fail;
        }
    }
    if (!passed || required_steps != 2) {
        std::fprintf(stderr, "FAIL: coupled derivative or omitted-sensitivity control at a required step\n");
        return 1;
    }
    std::printf("verify_jac_coupled: PASSED (all four fields at both required steps)\n");
    return 0;
}

#endif
