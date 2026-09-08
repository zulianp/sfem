#include "cvfem_hex8_ns_core.hpp"

int main(int argc, char **argv) {
    auto ctx = sfem::initialize(argc, argv);
    SFEM_TRACE_SCOPE("cvfem_hex8_ns_steady");

    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        usage(argv[0]);
        return 0;
    }
    if (argc != 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const smesh::Path output_folder(argv[1]);

    std::string case_name   = smesh::Env::read_string("SFEM_CASE", "");
    int         n           = smesh::Env::read<int>("SFEM_N", 8);
    int         ny          = smesh::Env::read<int>("SFEM_NY", n);
    int         nx          = smesh::Env::read<int>("SFEM_NX", 0);
    int         nz          = smesh::Env::read<int>("SFEM_NZ", 0);
    scalar_t    Lx          = smesh::Env::read<scalar_t>("SFEM_LX", 4);
    scalar_t    Ly          = smesh::Env::read<scalar_t>("SFEM_LY", 1);
    scalar_t    Lz          = smesh::Env::read<scalar_t>("SFEM_LZ", 1);
    scalar_t    rho         = smesh::Env::read<scalar_t>("SFEM_RHO", 1);
    scalar_t    mu          = smesh::Env::read<scalar_t>("SFEM_MU", 0.01);
    scalar_t    U           = smesh::Env::read<scalar_t>("SFEM_U", 1);
    std::string geom_name   = smesh::Env::read_string("SFEM_GEOM", "affine");
    std::string init_name   = smesh::Env::read_string("SFEM_INIT", "zero");
    int         max_newton  = smesh::Env::read<int>("SFEM_NL_MAX_IT", 40);
    scalar_t    newton_rtol = smesh::Env::read<scalar_t>("SFEM_NL_RTOL", 1e-8);
    scalar_t    newton_atol = smesh::Env::read<scalar_t>("SFEM_NL_ATOL", 1e-12);
    scalar_t    lin_rtol    = smesh::Env::read<scalar_t>("SFEM_LSOLVE_RTOL", 1e-8);
    scalar_t    lin_atol    = smesh::Env::read<scalar_t>("SFEM_LSOLVE_ATOL", 1e-14);
    int         lin_max_it  = smesh::Env::read<int>("SFEM_LSOLVE_MAX_IT", 1000);
    scalar_t    verify_tol  = smesh::Env::read<scalar_t>("SFEM_VERIFY_TOL", 1e-2);
    int         verbose     = smesh::Env::read<int>("SFEM_VERBOSE", 0);
    int         use_prec    = smesh::Env::read<int>("SFEM_NO_PREC", 0) ? 0 : 1;
    int         matrix_free = smesh::Env::read<int>("SFEM_MATRIX_FREE", 0);
    int         check_jv    = smesh::Env::read<int>("SFEM_CHECK_JV", 0);
    int         rhie_chow   = smesh::Env::read<int>("SFEM_RHIE_CHOW", 1);
    scalar_t    rc_scale    = smesh::Env::read<scalar_t>("SFEM_RHIE_CHOW_SCALE", 1);
    int         pack_size   = smesh::Env::read<int>("SFEM_PACK_SIZE", 2048);
    // Schur scaling of the pressure block in the block-Jacobi preconditioner.
    // 0 = the original identity-on-pressure behaviour, which is the control.
    scalar_t    pscale      = smesh::Env::read<scalar_t>("SFEM_PC_PSCALE", 0);
    // Damping applied to the plain 1 / A_pp pressure block. 1 = unchanged block-Jacobi.
    scalar_t    pdamp       = smesh::Env::read<scalar_t>("SFEM_PC_PDAMP", 1);
    // SIMPLE Schur approximation; takes precedence over SFEM_PC_PSCALE when set.
    int         pc_simple   = smesh::Env::read<int>("SFEM_PC_SIMPLE", 0);
    int         continuation = smesh::Env::read<int>("SFEM_NL_CONTINUATION", 1);

    if (case_name.empty()) {
        std::fprintf(stderr, "SFEM_CASE is required (poiseuille, couette, or coutte)\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    FlowCase flow;
    if (!parse_case(case_name, flow)) {
        std::fprintf(stderr, "invalid SFEM_CASE '%s' (expected poiseuille, couette, or coutte)\n", case_name.c_str());
        return EXIT_FAILURE;
    }
    if (geom_name != "affine" && geom_name != "isoparam") {
        std::fprintf(stderr, "invalid SFEM_GEOM '%s' (expected affine or isoparam)\n", geom_name.c_str());
        return EXIT_FAILURE;
    }
    if (init_name != "zero" && init_name != "exact") {
        std::fprintf(stderr, "invalid SFEM_INIT '%s' (expected zero or exact)\n", init_name.c_str());
        return EXIT_FAILURE;
    }
    if (n < 1) {
        std::fprintf(stderr, "invalid SFEM_N %d\n", n);
        return EXIT_FAILURE;
    }
    if (ny < 1) {
        std::fprintf(stderr, "invalid SFEM_NY %d\n", ny);
        return EXIT_FAILURE;
    }
    if (Lx <= 0 || Ly <= 0 || Lz <= 0) {
        std::fprintf(stderr, "invalid channel size L=(%g,%g,%g)\n", Lx, Ly, Lz);
        return EXIT_FAILURE;
    }
    if (nx < 1) nx = std::max(1, (int)std::lround((double)ny * (double)Lx / (double)Ly));
    if (nz < 1) nz = std::max(1, (int)std::lround((double)ny * (double)Lz / (double)Ly));

    const GeomKind geom      = parse_geom(geom_name);
    const InitKind init      = (init_name == "exact") ? InitKind::Exact : InitKind::Zero;
    const char    *flow_name = (flow == FlowCase::Poiseuille) ? "poiseuille" : "couette";

    const double tick = smesh::time_seconds();

    std::printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_CASE=%s\n"
            "- SFEM_N=%d\n"
            "- SFEM_NX=%d  SFEM_NY=%d  SFEM_NZ=%d\n"
            "- SFEM_LX=%g  SFEM_LY=%g  SFEM_LZ=%g\n"
            "- SFEM_RHO=%g\n"
            "- SFEM_MU=%g\n"
            "- SFEM_U=%g\n"
            "- SFEM_GEOM=%s\n"
            "- SFEM_INIT=%s\n"
            "- SFEM_NL_MAX_IT=%d\n"
            "- SFEM_NL_RTOL=%g\n"
            "- SFEM_NL_ATOL=%g\n"
            "- SFEM_LSOLVE_RTOL=%g\n"
            "- SFEM_LSOLVE_ATOL=%g\n"
            "- SFEM_LSOLVE_MAX_IT=%d\n"
            "- SFEM_VERIFY_TOL=%g\n"
            "- SFEM_VERBOSE=%d\n"
            "- SFEM_NO_PREC=%d\n"
            "- SFEM_MATRIX_FREE=%d\n"
            "- SFEM_RHIE_CHOW=%d\n"
            "- SFEM_RHIE_CHOW_SCALE=%g\n"
            "- SFEM_PACK_SIZE=%d\n"
            "- SFEM_PC_PSCALE=%g\n"
            "- SFEM_PC_PDAMP=%g\n"
            "- SFEM_PC_SIMPLE=%d\n"
            "----------------------------------------\n",
            flow_name,
            n,
            nx,
            ny,
            nz,
            Lx,
            Ly,
            Lz,
            rho,
            mu,
            U,
            geom_name.c_str(),
            init_name.c_str(),
            max_newton,
            newton_rtol,
            newton_atol,
            lin_rtol,
            lin_atol,
            lin_max_it,
            verify_tol,
            verbose,
            use_prec ? 0 : 1,
            matrix_free,
            rhie_chow,
            (rhie_chow == 0) ? scalar_t(0) : rc_scale,
            pack_size,
            pscale,
            pdamp,
            pc_simple);

    MeshData d;
    d.Lx               = Lx;
    d.Ly               = Ly;
    d.Lz               = Lz;
    d.rhie_chow_scale  = (rhie_chow == 0) ? scalar_t(0) : rc_scale;
    d.mesh   = smesh::Mesh::create_hex8_cube(ctx->communicator(), nx, ny, nz, 0, 0, 0, Lx, Ly, Lz);
    if (!d.mesh || d.mesh->element_type(0) != smesh::HEX8) {
        std::fprintf(stderr, "failed to create HEX8 channel mesh\n");
        return EXIT_FAILURE;
    }

    PackedData   packed_storage;
    PackColoring coloring_storage;
    if (geom == GeomKind::Affine && pack_size > 0) {
        packed_storage = make_packed(d.mesh, pack_size);
        d.packed       = &packed_storage;
        coloring_storage = cvfem_build_pack_coloring(packed_storage.n_packs,
                                                    packed_storage.owned_nodes_ptr,
                                                    packed_storage.ghost_ptr,
                                                    packed_storage.ghost_idx);
        d.coloring = &coloring_storage;
    }

    d.nnodes    = d.mesh->n_nodes();
    d.nelements = d.mesh->n_elements(0);
    d.elems     = d.mesh->elements(0)->data();
    d.points    = d.mesh->points()->data();
    if (geom == GeomKind::Affine) cvfem_hex8_precompute_affine_geometry(d);

    std::vector<uint8_t>  constrained;
    std::vector<scalar_t> bc;
    ptrdiff_t             pin_p = 0;
    mark_constraints(d, flow, mu, U, constrained, bc, pin_p);
    init_fields(d, init, constrained, bc);

    BSR4 bsr;
    const int assemble_jac = (!matrix_free || use_prec) ? 1 : 0;
    if (assemble_jac) {
        bsr = make_bsr4(d.mesh);
        precompute_element_bsr_slots(d, bsr);
    }

    const ptrdiff_t       ndof = d.nnodes * N_FIELDS;
    std::vector<scalar_t> x((size_t)ndof), r((size_t)ndof), dx((size_t)ndof), rhs((size_t)ndof);
    // The lumped pressure mass matrix for the Schur scaling in build_block_jacobi. The
    // geometry it needs is fixed, so this is computed once. Isoparametrically the
    // determinant varies within an element and jacobian_determinant holds the value at
    // the centre; that is accurate enough for a preconditioner scaling.
    std::vector<scalar_t> node_vol;
    build_node_volume(d, node_vol);
    std::vector<scalar_t> inv_diag;
    // Non-empty only when SFEM_PC_SIMPLE is on; build_block_jacobi keys off that.
    std::vector<scalar_t> schur_diag;
    pack_fields(d, x.data());

    const scalar_t Re = rho * U * Ly / mu;
    const scalar_t G  = (flow == FlowCase::Poiseuille) ? (scalar_t(8) * mu * U / (Ly * Ly)) : scalar_t(0);
    std::printf("case: %s\n", flow_name);
    std::printf("geom: %s\n", geom_name.c_str());
    std::printf("channel: L=(%g,%g,%g)  cells=(%d,%d,%d)\n", Lx, Ly, Lz, nx, ny, nz);
    std::printf("nnodes: %ld  nelements: %ld  ndof: %ld\n", (long)d.nnodes, (long)d.nelements, (long)ndof);
    std::printf("rho: %g  mu: %g  U: %g  Re: %g\n", rho, mu, U, Re);
    if (flow == FlowCase::Poiseuille) {
        std::printf("poiseuille: dp/dx=%g  p_in=%g  p_out=%g\n", -G, G * scalar_t(0.5) * Lx, -G * scalar_t(0.5) * Lx);
        std::printf("bc: walls y no-slip; x=0,Lx u=parabola; span z uz=0; pin p at node %ld\n", (long)pin_p);
    } else {
        std::printf("bc: y=0 no-slip; y=Ly lid u=(U,0,0); x=0,Lx u=Uy/H; span z uz=0; pin p at node %ld\n", (long)pin_p);
    }
    std::printf("init: %s\n", init_name.c_str());
    std::printf("hessian: %s\n", matrix_free ? "matrix-free J(u)v" : "assembled BSR");
    if (geom == GeomKind::Affine) {
        std::printf("kernels: residual=%s  assemble=sympy_row  jac-action=%s\n",
                    d.packed ? "packed SIMD sumfact" : "atomic sumfact",
                    d.packed ? "packed SIMD sumfact" : "atomic sumfact");
    }

    auto blas = sfem::make_openmp_blas<scalar_t>();

    scalar_t rho_lin = rho;
    std::shared_ptr<sfem::Operator<scalar_t>> A_bsr;
    if (assemble_jac) {
        A_bsr = sfem::h_bsr_spmv<smesh::count_t, smesh::idx_t, scalar_t>(
                d.nnodes, d.nnodes, 4, bsr.graph->rowptr(), bsr.graph->colidx(), bsr.values, scalar_t(0));
    }

    std::shared_ptr<sfem::Operator<scalar_t>> A;
    if (matrix_free) {
        A = sfem::make_op<scalar_t>(
                ndof,
                ndof,
                [&](const scalar_t *const xx, scalar_t *const yy) {
                    apply_jacobian_action(d, rho_lin, mu, geom, constrained, xx, yy);
                },
                sfem::EXECUTION_SPACE_HOST);
    } else {
        A = A_bsr;
    }

    auto M = sfem::make_op<scalar_t>(
            ndof,
            ndof,
            [&](const scalar_t *const xx, scalar_t *const yy) { apply_block_jacobi(inv_diag, d.nnodes, xx, yy); },
            sfem::EXECUTION_SPACE_HOST);

    auto solver = sfem::h_bcgs<scalar_t>();
    solver->set_n_dofs(ndof);
    solver->set_op(A);
    solver->set_max_it(lin_max_it);
    solver->set_rtol(lin_rtol);
    solver->set_atol(lin_atol);
    solver->verbose = verbose != 0;
    if (use_prec) solver->set_preconditioner_op(M);

    int            newton_it = 0;
    int            converged = 0;
    int            failed    = 0;
    scalar_t       r0        = 0;
    const scalar_t Re_phys   = rho * U * Ly / std::max(mu, scalar_t(1e-30));
    const scalar_t rho_re1   = mu / std::max(U * Ly, scalar_t(1e-30));
    // The re1 stage solves with rho = mu / (U Ly), which makes it the same Re=1 problem
    // whatever mu is -- useful as continuation, but it masks any viscosity dependence in
    // a measurement that sums over both stages. SFEM_NL_CONTINUATION=0 drops it.
    const int      n_stages  = (rho == scalar_t(0) || Re_phys <= scalar_t(1.5) || !continuation) ? 1 : 2;
    for (int stage = 0; stage < n_stages && !failed; ++stage) {
        const scalar_t rho_use = (n_stages == 1 || stage == 1) ? rho : rho_re1;
        rho_lin                = rho_use;
        cvfem_report_schur     = 1;  // one Schur breakdown per stage, not just the first
        std::printf("stage: %s  rho: %g  Re: %g\n",
                    (n_stages == 1 || stage == 1) ? "navier-stokes" : "re1",
                    rho_use,
                    rho_use * U * Ly / std::max(mu, scalar_t(1e-30)));
        converged = 0;
        for (newton_it = 0; newton_it < max_newton; ++newton_it) {
            unpack_fields(d, x.data());
            apply_residual(d, rho_use, mu, geom);
            pack_residual(d, r.data());
            apply_dirichlet_residual(constrained, r.data(), ndof);

            const scalar_t ru = [&]() {
                scalar_t s = 0;
                for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                    s += r[(size_t)i * 4 + 0] * r[(size_t)i * 4 + 0];
                    s += r[(size_t)i * 4 + 1] * r[(size_t)i * 4 + 1];
                    s += r[(size_t)i * 4 + 2] * r[(size_t)i * 4 + 2];
                }
                return std::sqrt(s);
            }();
            const scalar_t rp = [&]() {
                scalar_t s = 0;
                for (ptrdiff_t i = 0; i < d.nnodes; ++i) s += r[(size_t)i * 4 + 3] * r[(size_t)i * 4 + 3];
                return std::sqrt(s);
            }();
            const scalar_t rn = all_finite(r.data(), ndof) ? blas->norm2(ndof, r.data()) : scalar_t(-1);
            if (r0 == scalar_t(0) && rn > 0) r0 = rn;
            const scalar_t rrel = (r0 > 0 && rn >= 0) ? rn / r0 : rn;
            std::printf("newton %d  ||R||: %.6e  rel: %.6e  ||Ru||: %.6e  ||Rp||: %.6e\n", newton_it, rn, rrel, ru, rp);
            if (rn >= 0 && newton_step_converged(rn, r0, newton_atol, newton_rtol)) {
                converged = 1;
                break;
            }
            if (rn < 0) {
                std::fprintf(stderr, "non-finite residual\n");
                failed = 1;
                break;
            }

            if (assemble_jac) {
                assemble_jacobian(d, bsr, rho_use, mu, geom);
                apply_dirichlet_bsr(bsr, constrained, d.nnodes);
                if (use_prec)
                    {
                        if (pc_simple) build_schur_diag(bsr, d.nnodes, schur_diag);
                        build_block_jacobi(bsr, constrained, d.nnodes, node_vol, schur_diag, pscale, pdamp, inv_diag);
                    }
                if (check_jv && matrix_free && newton_it == 0 && stage == 0) {
                    compare_hessian_apply(d, *A_bsr, rho_use, mu, geom, constrained, r.data(), ndof);
                }
            }

#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < ndof; ++i) {
                rhs[i] = -r[i];
                dx[i]  = scalar_t(0);
            }

            const int lin_ok = solver->apply(rhs.data(), dx.data());
            const int lin_it = solver->iterations();
            apply_dirichlet_residual(constrained, dx.data(), ndof);
            const int      dx_ok = all_finite(dx.data(), ndof);
            const scalar_t dxn   = dx_ok ? max_abs(dx.data(), ndof) : scalar_t(-1);
            std::printf("  lin_it: %d  status: %s  |dx|_inf: %.6e\n",
                        lin_it,
                        lin_ok == SFEM_SUCCESS ? "ok" : "fail",
                        dxn);
            if (!dx_ok) {
                std::fprintf(stderr, "non-finite Newton step\n");
                failed = 1;
                break;
            }

            const scalar_t dx_cap = scalar_t(2) * std::max(U, scalar_t(1));
            if (dxn > dx_cap) {
                const scalar_t s = dx_cap / dxn;
#pragma omp parallel for schedule(static)
                for (ptrdiff_t i = 0; i < ndof; ++i) dx[i] *= s;
                std::printf("  damped Newton step by %g (|dx|_inf %g -> %g)\n", s, dxn, dx_cap);
            }

#pragma omp parallel for schedule(static)
            for (ptrdiff_t i = 0; i < ndof; ++i) x[i] += dx[i];
            apply_dirichlet_fields(constrained, bc, x.data(), ndof);
        }
        if (!converged) failed = 1;
    }

    unpack_fields(d, x.data());
    const ErrorNorms err = compute_errors(d, flow, mu, U, constrained);
    std::printf("newton_converged: %d  newton_it: %d\n", converged, newton_it);
    std::printf("u_linf: %.6e  u_l2: %.6e  u_linf_free: %.6e  n_free_u: %ld\n",
                err.u_linf,
                err.u_l2,
                err.u_linf_free,
                (long)err.n_free_u);
    std::printf("p_linf: %.6e  p_l2: %.6e  p_linf_free: %.6e  n_free_p: %ld\n",
                err.p_linf,
                err.p_l2,
                err.p_linf_free,
                (long)err.n_free_p);
    std::printf("p_min: %.6e  p_max: %.6e\n", err.p_min, err.p_max);

    smesh::create_directory(output_folder);
    smesh::create_directory(output_folder / "out");
    if (d.mesh->write(output_folder / "mesh") != SMESH_SUCCESS) {
        std::fprintf(stderr, "failed to write mesh to %s/mesh\n", output_folder.c_str());
        return EXIT_FAILURE;
    }

    auto       out   = smesh::Output::create(d.mesh, output_folder / "out");
    const auto ptype = smesh::TypeToEnum<scalar_t>::value();
    if (out->write_nodal("u.0", ptype, d.ux.data()) != SMESH_SUCCESS ||
        out->write_nodal("u.1", ptype, d.uy.data()) != SMESH_SUCCESS ||
        out->write_nodal("u.2", ptype, d.uz.data()) != SMESH_SUCCESS ||
        out->write_nodal("p", ptype, d.p.data()) != SMESH_SUCCESS) {
        std::fprintf(stderr, "failed to write fields to %s/out\n", output_folder.c_str());
        return EXIT_FAILURE;
    }

    const double tock = smesh::time_seconds();
    std::printf("wrote: %s/mesh  %s/out\n", output_folder.c_str(), output_folder.c_str());
    std::printf("ParaView: create_xdmf.sh %s\n", output_folder.c_str());
    std::printf("cvfem_hex8_ns_steady: %g seconds\n", tock - tick);

    failed = failed || (!converged) || (err.u_linf > verify_tol);
    if (failed) {
        std::fprintf(stderr, "verification failed (converged=%d u_linf=%.6e tol=%.6e)\n", converged, err.u_linf, verify_tol);
    }

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

