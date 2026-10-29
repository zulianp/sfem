// The problem bench_qp and bench_qp_cuda measure, and what both report against: the mesh, the
// state, the white-noise increment, the exact matrix-free apply and the reference H h; the row
// format and the error measures.  Included after the element's generated kernels are declared.
#ifdef EXACT_ISO
// Isoparametric: the geometry from the node coordinates, at every point of the element's own rule.
extern "C" int EXACT_APPLY(const int, const ptrdiff_t, const ptrdiff_t, idx_t **const, const geom_t *const *const,
                           const double, const double, const ptrdiff_t, const double *const, const double *const,
                           const double *const, const ptrdiff_t, const double *const, const double *const,
                           const double *const, const ptrdiff_t, double *const, double *const, double *const);
#else
extern "C" int EXACT_APPLY(const int, const ptrdiff_t, const ptrdiff_t, idx_t **const, const geom_t *const,
                           const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                           const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                           const geom_t *const, const double, const double, const ptrdiff_t, const double *const,
                           const double *const, const double *const, const ptrdiff_t, const double *const,
                           const double *const, const double *const, const ptrdiff_t, double *const, double *const,
                           double *const);
#endif

#ifdef BSR_AFFINE
extern "C" int MATERIAL_HESSIAN_BSR(const int, const int, const ptrdiff_t, const ptrdiff_t, idx_t **const,
                                    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                                    const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                                    const geom_t *const, const geom_t *const, const real_t, const real_t,
                                    const ptrdiff_t, const void *const, const void *const, const void *const,
                                    const count_t *const, const idx_t *const, void *const);
#else
extern "C" int MATERIAL_HESSIAN_BSR(const int, const int, const ptrdiff_t, const ptrdiff_t, idx_t **const,
                                    const geom_t *const *const, const real_t, const real_t, const ptrdiff_t,
                                    const void *const, const void *const, const void *const, const count_t *const,
                                    const idx_t *const, void *const);
#endif

// The exact matrix-free apply through the packed layout's two-pass scatter, the generated kernel
// the library runs with SFEM_PACKED_TWO_PASS=1.
extern "C" int PACKED_EXACT_APPLY(const int, const ptrdiff_t, const ptrdiff_t, const ptrdiff_t, const ptrdiff_t,
                                  const ptrdiff_t, uint16_t **const, const ptrdiff_t *const, const ptrdiff_t *const,
                                  const ptrdiff_t *const, const idx_t *const, const ptrdiff_t, const ptrdiff_t,
                                  const ptrdiff_t *const, const ptrdiff_t *const, const idx_t *const, void *const,
#ifdef EXACT_ISO
                                  const geom_t *const *const, const double, const double,
#else
                                  const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                                  const geom_t *const, const geom_t *const, const geom_t *const, const geom_t *const,
                                  const geom_t *const, const geom_t *const, const double, const double,
#endif
                                  const ptrdiff_t, const double *const, const double *const, const double *const,
                                  const ptrdiff_t, const double *const, const double *const, const double *const,
                                  const ptrdiff_t, double *const, double *const, double *const);

#ifndef LANE_VS
#define LANE_VS 16
#endif

using El = qp::Element;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Best of three timings of `repeats` calls, after one warm-up: seconds per call.
template <typename F>
static double best_seconds(int repeats, F &&fn) {
    fn();
    double best = 1e300;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const double t0 = now();
        for (int r = 0; r < repeats; ++r) fn();
        best = std::min(best, (now() - t0) / repeats);
    }
    return best;
}

// y = A x with values of type V, components interleaved, or y += A x with ADD (the contract of
// the operators inside Newton).  The harness's BSR product in double, generalised to
// single-precision values with double vectors and accumulation.
template <typename V, bool ADD = false>
static void bsr_apply_typed(const BsrGraph &g, const V *const RSTR values, const double *const RSTR x,
                            double *const RSTR y) {
#pragma omp parallel for
    for (ptrdiff_t row = 0; row < g.n_rows; ++row) {
        double acc[3] = {0, 0, 0};
        for (count_t k = g.rowptr[(size_t)row]; k < g.rowptr[(size_t)row + 1]; ++k) {
            const V *const b = &values[(size_t)k * 9];
            const double *const xv = &x[(size_t)g.colidx[(size_t)k] * 3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) acc[i] += (double)b[i * 3 + j] * xv[j];
        }
        for (int i = 0; i < 3; ++i) y[(size_t)row * 3 + i] = (ADD ? y[(size_t)row * 3 + i] : 0.0) + acc[i];
    }
}

struct Problem {
    Mesh m;
    ptrdiff_t ndof = 0;
    double mu = 2.3333333333333335, lmbda = 2.2;
    std::vector<double> ux, uy, uz, hx, hy, hz;  // state; white-noise increment
    std::vector<double> ex, ey, ez;              // exact H h
    double exact_norm = 0, exact_energy = 0;
    std::vector<geom_t> gpx, gpy, gpz;           // node positions in the geometry's type
};

struct Row {
    std::string name;
    double setup_s = 0, apply_s = 0, bytes = 0, rel_l2 = 0, rel_energy = 0;
};

static void print_row(const Problem &P, const Row &r) {
    const double mdofs = r.apply_s > 0 ? P.ndof / r.apply_s * 1e-6 : 0;
    std::printf("  %-34s %10.3f %10.2f %10.1f %10.2e %10.2e\n", r.name.c_str(), r.setup_s * 1e3, mdofs,
                r.bytes / (double)P.ndof, r.rel_l2, r.rel_energy);
}

static void errors(const Problem &P, const std::vector<double> &x, const std::vector<double> &y,
                   const std::vector<double> &z, Row &row) {
    double num = 0, e = 0;
    for (ptrdiff_t i = 0; i < P.m.nnodes; ++i) {
        const double dx = x[i] - P.ex[i], dy = y[i] - P.ey[i], dz = z[i] - P.ez[i];
        num += dx * dx + dy * dy + dz * dz;
        e += P.hx[i] * dx + P.hy[i] * dy + P.hz[i] * dz;
    }
    row.rel_l2 = std::sqrt(num) / P.exact_norm;
    row.rel_energy = std::fabs(e) / std::fabs(P.exact_energy);
}

// QP_ONLY: substrings separated by '|'; a variant runs when its name contains any of them.
static bool selected(const char *name) {
    const char *f = std::getenv("QP_ONLY");
    if (!f) return true;
    std::string all(f);
    for (size_t a = 0, b; a <= all.size(); a = b + 1) {
        b = all.find('|', a);
        if (b == std::string::npos) b = all.size();
        if (b > a && std::strstr(name, all.substr(a, b - a).c_str())) return true;
    }
    return false;
}

// The exact matrix-free apply at the problem's state, into x, y, z.
static void exact_apply(Problem &P, std::vector<double> &x, std::vector<double> &y, std::vector<double> &z) {
    Mesh &m = P.m;
    const geom_t *const gpoints[3] = {P.gpx.data(), P.gpy.data(), P.gpz.data()};
    (void)gpoints;

#ifdef EXACT_ISO
    EXACT_APPLY(SFEM_CODEGEN_F64, m.nelements, m.nnodes, m.evp.data(), gpoints, P.lmbda, P.mu, 1, P.ux.data(),
                P.uy.data(), P.uz.data(), 1, P.hx.data(), P.hy.data(), P.hz.data(), 1, x.data(), y.data(),
                z.data());
#else
    EXACT_APPLY(SFEM_CODEGEN_F64, m.nelements, m.nnodes, m.evp.data(), m.adj[0].data(), m.adj[1].data(),
                m.adj[2].data(), m.adj[3].data(), m.adj[4].data(), m.adj[5].data(), m.adj[6].data(),
                m.adj[7].data(), m.adj[8].data(), m.det.data(), P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(),
                P.uz.data(), 1, P.hx.data(), P.hy.data(), P.hz.data(), 1, x.data(), y.data(), z.data());
#endif
}

// The problem at n cells per direction, with warp w of the state's nonlinear part.
static void make_problem(Problem &P, const int n, const double warp) {
    if (const char *nu_env = std::getenv("QP_NU")) {
        const double nu = std::atof(nu_env), E = 1.0;
        P.mu = E / (2 * (1 + nu));
        P.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu));
    }
    P.m = build(n);
    Mesh &m = P.m;
    P.ndof = 3 * m.nnodes;

    static const double L[3][3] = {{0.030, 0.020, -0.010}, {-0.015, 0.025, 0.012}, {0.008, -0.018, 0.022}};
    static const double N_PH[3][3] = {{3.0, 1.0, 0.5}, {1.0, 3.0, 1.5}, {0.5, 1.5, 3.0}};
    P.ux.resize(m.nnodes); P.uy.resize(m.nnodes); P.uz.resize(m.nnodes);
    P.hx.resize(m.nnodes); P.hy.resize(m.nnodes); P.hz.resize(m.nnodes);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
        const double p[3] = {m.px[v], m.py[v], m.pz[v]};
        double lin[3] = {0, 0, 0};
        for (int c = 0; c < 3; ++c)
            for (int d = 0; d < 3; ++d) lin[c] += L[c][d] * p[d];
        double *u[3] = {&P.ux[v], &P.uy[v], &P.uz[v]};
        for (int c = 0; c < 3; ++c)
            *u[c] = lin[c] + warp * 0.02 * std::sin(N_PH[c][0] * p[0] + N_PH[c][1] * p[1] + N_PH[c][2] * p[2]);
        P.hx[v] = dist(gen);
        P.hy[v] = dist(gen);
        P.hz[v] = dist(gen);
    }

    // The exact matrix-free apply: the reference for every error below.
    P.ex.assign(m.nnodes, 0); P.ey.assign(m.nnodes, 0); P.ez.assign(m.nnodes, 0);
    P.gpx.resize(m.nnodes);
    P.gpy.resize(m.nnodes);
    P.gpz.resize(m.nnodes);
    for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
        P.gpx[v] = (geom_t)m.px[v];
        P.gpy[v] = (geom_t)m.py[v];
        P.gpz[v] = (geom_t)m.pz[v];
    }
    exact_apply(P, P.ex, P.ey, P.ez);
#ifdef QP_REFERENCE_BSR
    {
        // The store's rule is the BSR assembly's, not the affine exact apply's: BSR f64 is the
        // reference, and the exact matrix-free row reports the difference between the two rules.
        const BsrGraph g = build_bsr_graph(m);
        std::vector<geom_t> gx(m.nnodes), gy(m.nnodes), gz(m.nnodes);
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            gx[v] = (geom_t)m.px[v];
            gy[v] = (geom_t)m.py[v];
            gz[v] = (geom_t)m.pz[v];
        }
        const geom_t *const pts[3] = {gx.data(), gy.data(), gz.data()};
        std::vector<double> values((size_t)g.nnz() * 9, 0.0), xi((size_t)m.nnodes * 3), yi((size_t)m.nnodes * 3);
        MATERIAL_HESSIAN_BSR(SFEM_CODEGEN_F64, 8, m.nelements, m.nnodes, m.evp.data(),
#ifdef BSR_AFFINE
                             m.adj[0].data(), m.adj[1].data(), m.adj[2].data(), m.adj[3].data(), m.adj[4].data(),
                             m.adj[5].data(), m.adj[6].data(), m.adj[7].data(), m.adj[8].data(), m.det.data(),
#else
                             pts,
#endif
                             P.lmbda, P.mu, 1, P.ux.data(), P.uy.data(), P.uz.data(), g.rowptr.data(),
                             g.colidx.data(), values.data());
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            xi[(size_t)v * 3 + 0] = P.hx[v];
            xi[(size_t)v * 3 + 1] = P.hy[v];
            xi[(size_t)v * 3 + 2] = P.hz[v];
        }
        bsr_apply_typed<double>(g, values.data(), xi.data(), yi.data());
        for (ptrdiff_t v = 0; v < m.nnodes; ++v) {
            P.ex[v] = yi[(size_t)v * 3 + 0];
            P.ey[v] = yi[(size_t)v * 3 + 1];
            P.ez[v] = yi[(size_t)v * 3 + 2];
        }
    }
#endif
    for (ptrdiff_t i = 0; i < m.nnodes; ++i) {
        P.exact_norm += P.ex[i] * P.ex[i] + P.ey[i] * P.ey[i] + P.ez[i] * P.ez[i];
        P.exact_energy += P.hx[i] * P.ex[i] + P.hy[i] * P.ey[i] + P.hz[i] * P.ez[i];
    }
    P.exact_norm = std::sqrt(P.exact_norm);
}
