#include "cvfem_hex8_ns_op.hpp"

#include "smesh_sideset.hpp"

// The core is included here and nowhere a driver can see it. See the note in the header.
#include "cvfem_hex8_ns_core.hpp"
#include "cvfem_sshex8_ns.hpp"

#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>

namespace sfem {

    static_assert(sizeof(real_t) == sizeof(scalar_t),
                  "CVFEMNavierStokes hands sfem buffers straight to the CVFEM kernels, which are compiled "
                  "for double; a float32 real_t build would need a conversion layer.");

    namespace {
        GeomKind to_geom_kind(const CVFEMGeometry g) {
            return (g == CVFEMGeometry::Isoparam) ? GeomKind::Isoparam : GeomKind::Affine;
        }
    }  // namespace

    class CVFEMNavierStokes::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        MeshData                       d;
        // MeshData holds bare pointers into these, so the Op has to own them.
        PackedData   packed;
        PackColoring coloring;
        BSR4         bsr;  // slot caches only; values come from the caller
        bool         initialized{false};

        // Scratch for the block-diagonal assembly, kept so a smoother rebuilding the
        // preconditioner each Newton step does not reallocate n_nodes * 16 every time.
        std::vector<scalar_t> diag_scratch;

        // See set_option("cache_nodal_pgrad"). pgrad_for is the state the cached gradient
        // belongs to; a different pointer forces a recompute, which makes the common
        // mistake of pointing the operator at a new vector safe. It cannot catch the
        // state changing through the same pointer, which is why this is opt-in.
        bool                cache_pgrad{false};
        const real_t       *pgrad_for{nullptr};

        // Semi-structured path. When the space carries a semi-structured mesh the operator
        // runs the sshex8 kernels over macro-elements instead of the flat ones, and `d`
        // above is left unused. Chosen at initialize() from the space, not configured.
        bool       semi_structured{false};
        SSMeshData ss;

        // See coarser(): the operator derefine_op() built for the next level down.
        std::shared_ptr<CVFEMNavierStokes> coarser;
    };

    CVFEMNavierStokes::CVFEMNavierStokes(const std::shared_ptr<FunctionSpace> &space) : impl_(std::make_unique<Impl>()) {
        impl_->space = space;
    }

    CVFEMNavierStokes::~CVFEMNavierStokes() = default;

    std::unique_ptr<Op> CVFEMNavierStokes::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != N_FIELDS) {
            SFEM_ERROR("cvfem:NavierStokes needs block_size %d (ux, uy, uz, p), got %d\n", N_FIELDS, space->block_size());
            return nullptr;
        }
        return std::make_unique<CVFEMNavierStokes>(space);
    }

    bool CVFEMNavierStokes::is_semi_structured() const { return impl_->semi_structured; }

    const ::SSMeshData *CVFEMNavierStokes::semi_structured_data() const {
        return impl_->semi_structured ? &impl_->ss : nullptr;
    }

    std::shared_ptr<CVFEMNavierStokes> CVFEMNavierStokes::coarser() const { return impl_->coarser; }

    ptrdiff_t CVFEMNavierStokes::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t CVFEMNavierStokes::n_dofs_image() const { return impl_->space->n_dofs(); }

    // Per-element boundary-face bitmask from the mesh skin.
    //
    // smesh::skin_sideset is topological -- it finds exterior faces from element adjacency,
    // not from coordinates -- so it picks up re-entrant faces such as the step of a
    // backward-facing step, which the six-plane coordinate test in hex8_face_on_domain
    // cannot see. Its (parent, lfi) pairs are also invariant under semi-structured level
    // changes, so a mask built on the macro mesh is valid at every multigrid level.
    //
    // smesh numbers HEX8 faces differently from CVFEM_HEX8_BFACE_NODES. The permutation was
    // established by comparing the two node lists as sets, face by face:
    //   smesh 0:{0,1,5,4}=CVFEM 2   1:{1,2,6,5}=1   2:{2,3,7,6}=3
    //   smesh 3:{3,0,4,7}=CVFEM 0   4:{3,2,1,0}=4   5:{4,5,6,7}=5
    static bool build_face_mask(const std::shared_ptr<smesh::Mesh> &mesh, const ptrdiff_t n_elements,
                                std::vector<uint8_t> &mask) {
        static const int lfi_to_cvfem[6] = {2, 1, 3, 0, 4, 5};
        auto             skin            = smesh::skin_sideset(mesh);
        if (!skin) return false;
        auto            par = skin->parent();
        auto            lfi = skin->lfi();
        if (!par || !lfi) return false;
        mask.assign((size_t)n_elements, 0);
        const ptrdiff_t nf = par->size();
        for (ptrdiff_t k = 0; k < nf; ++k) {
            const ptrdiff_t e = (ptrdiff_t)par->data()[k];
            const int       l = (int)lfi->data()[k];
            if (e < 0 || e >= n_elements || l < 0 || l >= 6) return false;
            mask[(size_t)e] |= (uint8_t)(1u << lfi_to_cvfem[l]);
        }
        return true;
    }

    // Compile a Sideset into the per-element bitmask the kernels read.
    //
    // The sideset is the specification and the mask is its compiled form: a sideset is a list
    // of (parent, lfi) pairs, and using it directly inside an element loop would put a search
    // where a bit test belongs.
    static ptrdiff_t compile_sideset_mask(const std::shared_ptr<smesh::Sideset> &ss,
                                          const ptrdiff_t n_elements, std::vector<uint8_t> &mask) {
        static const int lfi_to_cvfem[6] = {2, 1, 3, 0, 4, 5};
        if (mask.size() != (size_t)n_elements) mask.assign((size_t)n_elements, 0);
        auto par = ss->parent();
        auto lfi = ss->lfi();
        if (!par || !lfi) return -1;
        ptrdiff_t n = 0;
        for (ptrdiff_t k = 0; k < par->size(); ++k) {
            const ptrdiff_t e = (ptrdiff_t)par->data()[k];
            const int       l = (int)lfi->data()[k];
            if (e < 0 || e >= n_elements || l < 0 || l >= 6) return -1;
            mask[(size_t)e] |= (uint8_t)(1u << lfi_to_cvfem[l]);
            ++n;
        }
        return n;
    }

    // Refine a boundary mask to the faces lying on one coordinate plane.
    //
    // `corner_off` maps CVFEM local corner index to the element array row holding it: the
    // identity for a plain HEX8, and the lattice extremes for a semi-structured macro
    // element. Getting that wrong is silent -- using a micro cell's corners on a macro
    // element reaches only the first micro cell, so no face matches the plane and the mask
    // comes out empty rather than wrong.
    static ptrdiff_t refine_natural_mask(const std::vector<uint8_t> &face_mask,
                                         const smesh::idx_t *const *elems,
                                         const smesh::geom_t *const *pts, const int *corner_off,
                                         const int axis, const double value,
                                         std::vector<uint8_t> &natural) {
        natural.assign(face_mask.size(), 0);
        ptrdiff_t nfaces = 0;
        for (size_t e = 0; e < face_mask.size(); ++e) {
            const int bm = face_mask[e];
            if (!bm) continue;
            int nm = 0;
            for (int f = 0; f < 6; ++f) {
                if (!((bm >> f) & 1)) continue;
                bool on = true;
                for (int k = 0; k < 4; ++k) {
                    const smesh::idx_t g = elems[corner_off[CVFEM_HEX8_BFACE_NODES[f][k]]][(ptrdiff_t)e];
                    if (std::fabs((double)pts[axis][g] - value) >
                        1e-8 * std::max(1.0, std::fabs(value))) {
                        on = false;
                        break;
                    }
                }
                if (on) { nm |= (1 << f); ++nfaces; }
            }
            natural[e] = (uint8_t)nm;
        }
        return nfaces;
    }

    int CVFEMNavierStokes::initialize(const std::vector<std::string> & /*block_names*/) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::initialize");

        auto &d    = impl_->d;
        auto  mesh = impl_->space->mesh_ptr();

        if (impl_->space->has_semi_structured_mesh()) {
            // Affine macro-elements only; see the note on is_semi_structured(). `geom` is
            // not consulted here, since the macro Jacobian is computed once and reused.
            impl_->semi_structured = true;
            const int level        = smesh::semistructured_level(*mesh);
            sscvfem_init(impl_->ss, mesh, level);

            // SFEM_BOUNDARY_MASK=1 replaces the bounding-box coordinate test with the
            // topological skin. Off by default so every existing box case is untouched;
            // required for a non-box domain, where the coordinate test cannot see a
            // re-entrant face and silently leaves those control volumes unclosed.
            //
            // Built on the MACRO mesh: (parent, lfi) is invariant under semi-structured
            // level changes, so this one mask is correct at every multigrid level and
            // derefine_op -- which re-runs initialize() on the coarse space -- rebuilds an
            // equally valid one rather than needing the array transferred.
            impl_->ss.macro_face_mask.clear();
            if (smesh::Env::read<int>("SFEM_BOUNDARY_MASK", 0)) {
                // A named "skin" sideset, if the mesh carries one, is used in preference to
                // re-deriving the skin: it is the same faces, and rebuilding them here would
                // repeat an element-adjacency pass at every multigrid level.
                auto named = mesh->sidesets("skin");
                if (!named.empty() && named.front()) {
                    if (compile_sideset_mask(named.front(), mesh->n_elements(0),
                                             impl_->ss.macro_face_mask) < 0) {
                        SFEM_ERROR("CVFEMNavierStokes: malformed 'skin' sideset\n");
                        return SFEM_FAILURE;
                    }
                } else if (!build_face_mask(mesh, mesh->n_elements(0), impl_->ss.macro_face_mask)) {
                    SFEM_ERROR("CVFEMNavierStokes: SFEM_BOUNDARY_MASK=1 but skin_sideset failed\n");
                    return SFEM_FAILURE;
                }
            }

            // Natural (do-nothing) outflow faces, taken from a named sideset.
            //
            // This previously refined the boundary mask with a coordinate plane test, which
            // carried the same assumption that makes hex8_face_on_domain wrong on this
            // geometry: it only works where the outlet happens to be an axis-aligned plane.
            // It also had to know how to address a macro element's corners, and getting that
            // wrong produced an empty mask rather than an error.
            //
            // A sideset carries the faces explicitly, is level-invariant -- (parent, lfi)
            // refers to the macro element, which a semi-structured level change leaves alone
            // -- and is the same object the Dirichlet set is derived from, so the two cannot
            // disagree.
            impl_->ss.macro_natural_mask.clear();
            if (!natural_outflow_sideset.empty()) {
                auto named = mesh->sidesets(natural_outflow_sideset);
                if (named.empty() || !named.front()) {
                    SFEM_ERROR("CVFEMNavierStokes: sideset '%s' not found on the mesh\n",
                               natural_outflow_sideset.c_str());
                    return SFEM_FAILURE;
                }
                const ptrdiff_t nf = compile_sideset_mask(named.front(), impl_->ss.nmacro,
                                                          impl_->ss.macro_natural_mask);
                if (nf < 0) {
                    SFEM_ERROR("CVFEMNavierStokes: malformed sideset '%s'\n",
                               natural_outflow_sideset.c_str());
                    return SFEM_FAILURE;
                }
                std::printf("natural outflow: %td macro faces from sideset '%s'\n", nf,
                            natural_outflow_sideset.c_str());
            }

            // Deterministic two-pass scatter, the semi-structured counterpart of the packed
            // HEX8 layout. Off gives the atomic scatter, which is not reproducible across
            // thread counts.
            if (smesh::Env::read<int>("SFEM_SS_SCATTER", 1)) {
                impl_->ss.scatter = std::make_shared<SSScatter>();
                sscvfem_build_scatter(impl_->ss, *impl_->ss.scatter);
            }
            impl_->ss.rhie_chow_scale = rhie_chow_scale;
            impl_->initialized        = true;
            return SFEM_SUCCESS;
        }

        if (!mesh || mesh->element_type(0) != smesh::HEX8) {
            SFEM_ERROR("cvfem:NavierStokes requires a HEX8 mesh\n");
            return SFEM_FAILURE;
        }

        d.mesh = mesh;

        // Packing comes first, and the mesh pointers are read only afterwards.
        //
        // make_packed builds a PackedMesh with modify_mesh = true, and that renumbers the
        // mesh nodes in place (smesh_packed_mesh.cpp: mesh->renumber_nodes(node_map)).
        // The renumbering is load-bearing rather than incidental: the packed kernels index
        // the global arrays as `owned_nodes_ptr[pack] + k`, which is only a valid node id
        // because each pack's owned nodes were made contiguous. Capturing elems/points
        // before this leaves them pointing at the pre-renumbering arrays.
        if (to_geom_kind(geom) == GeomKind::Affine && pack_size > 0) {
            impl_->packed   = make_packed(d.mesh, pack_size);
            d.packed        = &impl_->packed;
            impl_->coloring = cvfem_build_pack_coloring(impl_->packed.n_packs,
                                                       impl_->packed.owned_nodes_ptr,
                                                       impl_->packed.ghost_ptr,
                                                       impl_->packed.ghost_idx);
            d.coloring      = &impl_->coloring;
        }

        d.nnodes    = mesh->n_nodes();
        d.nelements = mesh->n_elements(0);
        d.elems     = mesh->elements(0)->data();
        d.points    = mesh->points()->data();

        // The boundary sub-control-surface treatment closes the control volumes on the
        // domain faces, and identifies those faces by comparing node coordinates against
        // the extents. The driver passes them in from its own channel geometry; here they
        // come from the mesh, so the Op works on any box without being told.
        {
            const auto *const px    = d.points[0];
            const auto *const py    = d.points[1];
            const auto *const pz    = d.points[2];
            scalar_t          hi[3] = {0, 0, 0};
            for (ptrdiff_t i = 0; i < d.nnodes; ++i) {
                hi[0] = std::max(hi[0], (scalar_t)px[i]);
                hi[1] = std::max(hi[1], (scalar_t)py[i]);
                hi[2] = std::max(hi[2], (scalar_t)pz[i]);
            }
            d.Lx = hi[0];
            d.Ly = hi[1];
            d.Lz = hi[2];
        }

        d.rhie_chow_scale = rhie_chow_scale;

        // The coarse levels of the multigrid hierarchy are plain HEX8 meshes and take this
        // path, so they need the same treatment as the fine semi-structured level. Without
        // it a coarse level silently misses the step faces and keeps p_i*a on the outlet
        // with no pin -- a singular coarse operator, whose only visible symptom is that the
        // fine Krylov solve performs zero iterations.
        d.face_mask.clear();
        d.natural_mask.clear();
        if (smesh::Env::read<int>("SFEM_BOUNDARY_MASK", 0)) {
            auto named_skin = mesh->sidesets("skin");
            if (!named_skin.empty() && named_skin.front()) {
                if (compile_sideset_mask(named_skin.front(), d.nelements, d.face_mask) < 0) {
                    SFEM_ERROR("CVFEMNavierStokes: malformed 'skin' sideset\n");
                    return SFEM_FAILURE;
                }
            } else if (!build_face_mask(mesh, d.nelements, d.face_mask)) {
                SFEM_ERROR("CVFEMNavierStokes: SFEM_BOUNDARY_MASK=1 but skin_sideset failed\n");
                return SFEM_FAILURE;
            }
            if (!natural_outflow_sideset.empty()) {
                auto named = mesh->sidesets(natural_outflow_sideset);
                if (named.empty() || !named.front()) {
                    SFEM_ERROR("CVFEMNavierStokes: sideset '%s' not found (coarse level)\n",
                               natural_outflow_sideset.c_str());
                    return SFEM_FAILURE;
                }
                const ptrdiff_t nf =
                        compile_sideset_mask(named.front(), d.nelements, d.natural_mask);
                std::printf("natural outflow (flat): %td faces from sideset '%s'\n", nf,
                            natural_outflow_sideset.c_str());
            }
        }

        // SFEM_BOUNDARY_MASK_CHECK=1 compares the topological mask against the coordinate
        // test, element by element and face by face.
        //
        // This is the gate that makes replacing one with the other safe. On a box both are
        // valid and must agree exactly; if they do, the switch provably cannot change any box
        // result, and a disagreement means the smesh-to-CVFEM face permutation or the corner
        // convention is wrong. Catching that here is the difference between a failed assert
        // and a three-percent error in a step solution several stages later.
        if (smesh::Env::read<int>("SFEM_BOUNDARY_MASK_CHECK", 0)) {
            std::vector<uint8_t> topo;
            if (!build_face_mask(mesh, d.nelements, topo)) {
                std::fprintf(stderr, "boundary_mask_check: skin_sideset failed\n");
            } else {
                ptrdiff_t disagree = 0, n_topo = 0, n_coord = 0;
                for (ptrdiff_t e = 0; e < d.nelements; ++e) {
                    scalar_t ex[8], ey[8], ez[8];
                    for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                        const smesh::idx_t g = d.elems[a][e];
                        ex[a] = d.points[0][g]; ey[a] = d.points[1][g]; ez[a] = d.points[2][g];
                    }
                    for (int f = 0; f < 6; ++f) {
                        const int bt = hex8_face_on_domain(f, ex, ey, ez, d.Lx, d.Ly, d.Lz) ? 1 : 0;
                        const int bm = (topo[(size_t)e] >> f) & 1;
                        n_coord += bt; n_topo += bm;
                        disagree += (bt != bm);
                    }
                }
                std::printf("boundary_mask_check: coord_faces %td  topo_faces %td  disagreements %td  %s\n",
                            n_coord, n_topo, disagree, disagree == 0 ? "MATCH" : "MISMATCH");
            }
        }

        d.ux.assign((size_t)d.nnodes, scalar_t(0));
        d.uy.assign((size_t)d.nnodes, scalar_t(0));
        d.uz.assign((size_t)d.nnodes, scalar_t(0));
        d.p.assign((size_t)d.nnodes, scalar_t(0));
        d.rx.assign((size_t)d.nnodes, scalar_t(0));
        d.ry.assign((size_t)d.nnodes, scalar_t(0));
        d.rz.assign((size_t)d.nnodes, scalar_t(0));
        d.rc.assign((size_t)d.nnodes, scalar_t(0));

        if (to_geom_kind(geom) == GeomKind::Affine) cvfem_hex8_precompute_affine_geometry(d);

        // The sparsity is the mesh's node-to-node graph, which is also what hessian_bsr
        // is handed, so the element-to-slot map can be built once here.
        impl_->bsr.graph  = d.mesh->node_to_node_graph();
        impl_->bsr.rowptr = impl_->bsr.graph->rowptr()->data();
        impl_->bsr.colidx = impl_->bsr.graph->colidx()->data();
        impl_->bsr.nnz    = impl_->bsr.graph->nnz();
        precompute_element_bsr_slots(d, impl_->bsr);

        impl_->initialized = true;
        return SFEM_SUCCESS;
    }

    void CVFEMNavierStokes::set_option(const std::string &name, bool val) {
        if (name == "cache_nodal_pgrad") {
            impl_->cache_pgrad = val;
            impl_->pgrad_for   = nullptr;  // nothing is cached yet
        }
    }

    int CVFEMNavierStokes::update(const real_t *const x) {
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            impl_->ss.rhie_chow_scale = rhie_chow_scale;
            sscvfem_unpack(impl_->ss, x);
            sscvfem_nodal_p_grad(impl_->ss);
            impl_->pgrad_for = x;
            return SFEM_SUCCESS;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);
        assemble_nodal_p_grad(impl_->d, to_geom_kind(geom));
        impl_->pgrad_for = x;
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::gradient");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            impl_->ss.rhie_chow_scale = rhie_chow_scale;
            sscvfem_unpack(impl_->ss, x);
            sscvfem_nodal_p_grad(impl_->ss);
            impl_->pgrad_for = x;
            sscvfem_residual(impl_->ss, rho, mu, out, /*zero_first=*/false);
            return SFEM_SUCCESS;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);
        // apply_residual recomputes the gradient itself, so this leaves it current.
        apply_residual(impl_->d, rho, mu, to_geom_kind(geom));
        impl_->pgrad_for = x;
        // sfem::Op accumulates into out, so add rather than overwrite.
        add_residual(impl_->d, out);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::apply");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            impl_->ss.rhie_chow_scale = rhie_chow_scale;
            sscvfem_unpack(impl_->ss, x);
            if (!(impl_->cache_pgrad && impl_->pgrad_for == x)) {
                sscvfem_nodal_p_grad(impl_->ss);
                impl_->pgrad_for = x;
            }
            sscvfem_apply(impl_->ss, rho, mu, h, out);
            return SFEM_SUCCESS;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);
        if (!(impl_->cache_pgrad && impl_->pgrad_for == x)) {
            assemble_nodal_p_grad(impl_->d, to_geom_kind(geom));
            impl_->pgrad_for = x;
        }
        apply_jacobian_action_accumulate(impl_->d, rho, mu, to_geom_kind(geom), h, out);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::apply_blocks(const real_t *const x, const real_t *const h, real_t *const out,
                                        const int blocks) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::apply_blocks");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (!impl_->semi_structured) {
            SFEM_ERROR("CVFEMNavierStokes::apply_blocks: semi-structured meshes only\n");
            return SFEM_FAILURE;
        }
        impl_->ss.rhie_chow_scale = rhie_chow_scale;
        sscvfem_unpack(impl_->ss, x);
        if (!(impl_->cache_pgrad && impl_->pgrad_for == x)) {
            sscvfem_nodal_p_grad(impl_->ss);
            impl_->pgrad_for = x;
        }
        sscvfem_apply_blocks(impl_->ss, rho, mu, blocks, h, out);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::value(const real_t * /*x*/, real_t *const /*out*/) {
        // Steady Navier-Stokes is not the stationary point of an energy, so there is no
        // value to contribute. Succeeding rather than erroring keeps Function::value
        // usable for the other operators in the same Function.
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::hessian_crs(const real_t *const /*x*/,
                                       const count_t *const /*rowptr*/,
                                       const idx_t *const /*colidx*/,
                                       real_t *const /*values*/) {
        SFEM_ERROR("cvfem:NavierStokes assembles BSR, not CRS; use hessian_bsr\n");
        return SFEM_FAILURE;
    }

    int CVFEMNavierStokes::hessian_bsr(const real_t *const  x,
                                       const count_t *const rowptr,
                                       const idx_t *const   colidx,
                                       real_t *const        values) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::hessian_bsr");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            // No BSR assembly on the semi-structured path, and not an oversight: an
            // assembled matrix per level is the memory the hierarchy exists to avoid, and
            // matrix-free is the default. Refusing beats returning a zero matrix.
            SFEM_ERROR("cvfem:NavierStokes has no BSR assembly on a semi-structured mesh; use matrix-free\n");
            return SFEM_FAILURE;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);

        // The slot caches were built against the mesh graph in initialize(); assembly
        // writes through external_values into the caller's buffer. Accumulate, because
        // Function::hessian_bsr shares one buffer across operators.
        impl_->bsr.rowptr          = rowptr;
        impl_->bsr.colidx          = colidx;
        impl_->bsr.external_values = values;
        assemble_jacobian(impl_->d, impl_->bsr, rho, mu, to_geom_kind(geom), /*zero_first=*/false);
        impl_->bsr.external_values = nullptr;
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::hessian_block_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::hessian_block_diag");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            impl_->ss.rhie_chow_scale = rhie_chow_scale;
            sscvfem_unpack(impl_->ss, x);
            sscvfem_nodal_p_grad(impl_->ss);
            impl_->pgrad_for = x;
            sscvfem_block_diag(impl_->ss, rho, mu, impl_->diag_scratch);
            for (size_t i = 0; i < impl_->diag_scratch.size(); ++i) values[i] += impl_->diag_scratch[i];
            return SFEM_SUCCESS;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);
        assemble_block_diag(impl_->d, rho, mu, to_geom_kind(geom), impl_->diag_scratch);
        const auto &blocks = impl_->diag_scratch;
        for (size_t i = 0; i < blocks.size(); ++i) values[i] += blocks[i];
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::hessian_diag");
        if (!impl_->initialized) return SFEM_FAILURE;
        if (impl_->semi_structured) {
            std::vector<real_t> blocks((size_t)impl_->ss.nnodes * 16, 0);
            hessian_block_diag(x, blocks.data());
            for (ptrdiff_t i = 0; i < impl_->ss.nnodes; ++i)
                for (int c = 0; c < N_FIELDS; ++c)
                    values[(size_t)i * N_FIELDS + c] += blocks[(size_t)i * 16 + c * 4 + c];
            return SFEM_SUCCESS;
        }
        impl_->d.rhie_chow_scale = rhie_chow_scale;
        unpack_fields(impl_->d, x);
        assemble_block_diag(impl_->d, rho, mu, to_geom_kind(geom), impl_->diag_scratch);
        const auto &blocks = impl_->diag_scratch;
        for (ptrdiff_t i = 0; i < impl_->d.nnodes; ++i) {
            const scalar_t *const blk = blocks.data() + (size_t)i * 16;
            for (int c = 0; c < N_FIELDS; ++c) values[(size_t)i * N_FIELDS + c] += blk[c * 4 + c];
        }
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> CVFEMNavierStokes::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::derefine_op");
        // Rediscretisation, not Galerkin coarsening. Not a stylistic choice: the
        // Rhie-Chow coefficient carries h^2/(2 mu) explicitly, so the coarse pressure
        // operator differs from the fine one by roughly 8x per level in 3D. Assembling
        // the CVFEM operator on the coarse mesh gets that right; P^T A P would inherit
        // the fine-grid stabilisation and be inconsistent.
        auto ret = std::static_pointer_cast<CVFEMNavierStokes>(clone_onto(space));
        ret->initialize();
        impl_->coarser = ret;
        return ret;
    }

    std::shared_ptr<Op> CVFEMNavierStokes::clone() const { return clone_onto(impl_->space); }

    void CVFEMNavierStokes::set_body_force(const real_t *fx, const real_t *fy, const real_t *fz) {
        const ptrdiff_t n = impl_->semi_structured ? impl_->ss.nnodes : impl_->d.nnodes;
        auto assign = [&](auto &dst_x, auto &dst_y, auto &dst_z) {
            if (!fx) {
                dst_x.clear(); dst_y.clear(); dst_z.clear();
                return;
            }
            dst_x.assign(fx, fx + n);
            dst_y.assign(fy, fy + n);
            dst_z.assign(fz, fz + n);
        };
        if (impl_->semi_structured) {
            assign(impl_->ss.fx, impl_->ss.fy, impl_->ss.fz);
            impl_->ss.node_vol.clear();  // rebuilt lazily on the next residual
        } else {
            assign(impl_->d.fx, impl_->d.fy, impl_->d.fz);
            impl_->d.node_vol.clear();
        }
    }

    int CVFEMNavierStokes::node_volume(real_t *const out) const {
        if (!impl_->initialized) return SFEM_FAILURE;
        std::vector<scalar_t> v;
        if (impl_->semi_structured) {
            sscvfem_node_volume(impl_->ss, v);
        } else {
            build_node_volume(impl_->d, v);
        }
        for (size_t i = 0; i < v.size(); ++i) out[i] = (real_t)v[i];
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> CVFEMNavierStokes::clone_onto(const std::shared_ptr<FunctionSpace> &space) const {
        auto ret             = std::make_shared<CVFEMNavierStokes>(space);
        ret->rho             = rho;
        ret->mu              = mu;
        ret->rhie_chow_scale = rhie_chow_scale;
        ret->geom            = geom;
        ret->pack_size       = pack_size;
        // The outflow setting must travel to the coarse levels. derefine_op re-runs
        // initialize() on the coarse space, which rebuilds the boundary and natural masks --
        // but only if it knows an outflow plane exists. Without this the coarse operator
        // keeps p_i*a on the outlet and has no pin, so it is singular, its solve returns
        // nothing, and the fine-level Krylov iteration silently does zero work.
        ret->natural_outflow_sideset = natural_outflow_sideset;
        return ret;
    }

    void CVFEMNavierStokes::set_value_in_block(const std::string & /*block_name*/,
                                               const std::string &var_name,
                                               const real_t       value) {
        if (var_name == "rho") {
            rho = value;
        } else if (var_name == "mu") {
            mu = value;
        } else if (var_name == "rhie_chow_scale") {
            rhie_chow_scale = value;
        } else {
            SFEM_ERROR("cvfem:NavierStokes has no parameter '%s'\n", var_name.c_str());
        }
    }

}  // namespace sfem
