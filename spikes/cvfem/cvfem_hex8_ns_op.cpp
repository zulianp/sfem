#include "cvfem_hex8_ns_op.hpp"

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

    std::shared_ptr<CVFEMNavierStokes> CVFEMNavierStokes::coarser() const { return impl_->coarser; }

    ptrdiff_t CVFEMNavierStokes::n_dofs_domain() const { return impl_->space->n_dofs(); }
    ptrdiff_t CVFEMNavierStokes::n_dofs_image() const { return impl_->space->n_dofs(); }

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

    std::shared_ptr<Op> CVFEMNavierStokes::clone_onto(const std::shared_ptr<FunctionSpace> &space) const {
        auto ret             = std::make_shared<CVFEMNavierStokes>(space);
        ret->rho             = rho;
        ret->mu              = mu;
        ret->rhie_chow_scale = rhie_chow_scale;
        ret->geom            = geom;
        ret->pack_size       = pack_size;
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
