#include "cvfem_hex8_ns_op.hpp"

#include "smesh_mesh.hpp"

#include <algorithm>
#include <cmath>

namespace sfem {

    CVFEMNavierStokes::CVFEMNavierStokes(const std::shared_ptr<FunctionSpace> &space) : space_(space) {}

    std::unique_ptr<Op> CVFEMNavierStokes::create(const std::shared_ptr<FunctionSpace> &space) {
        if (space->block_size() != N_FIELDS) {
            SFEM_ERROR("cvfem:NavierStokes needs block_size %d (ux, uy, uz, p), got %d\n", N_FIELDS, space->block_size());
            return nullptr;
        }
        return std::make_unique<CVFEMNavierStokes>(space);
    }

    void CVFEMNavierStokes::sync_scheme_parameters() {
        d_.rhie_chow_scale = rhie_chow_scale;
    }

    int CVFEMNavierStokes::initialize(const std::vector<std::string> & /*block_names*/) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::initialize");

        auto mesh = space_->mesh_ptr();
        if (!mesh || mesh->element_type(0) != smesh::HEX8) {
            SFEM_ERROR("cvfem:NavierStokes requires a HEX8 mesh\n");
            return SFEM_FAILURE;
        }

        d_.mesh      = mesh;
        d_.nnodes    = mesh->n_nodes();
        d_.nelements = mesh->n_elements(0);
        d_.elems     = mesh->elements(0)->data();
        d_.points    = mesh->points()->data();

        // The boundary sub-control-surface treatment closes the control volumes on the
        // domain faces, and it identifies those faces by comparing node coordinates
        // against the extents. The driver passes them in from its own channel geometry;
        // here they come from the mesh, so the Op works on any box without being told.
        {
            const auto *const px = d_.points[0];
            const auto *const py = d_.points[1];
            const auto *const pz = d_.points[2];
            scalar_t          hi[3] = {0, 0, 0};
            for (ptrdiff_t i = 0; i < d_.nnodes; ++i) {
                hi[0] = std::max(hi[0], (scalar_t)px[i]);
                hi[1] = std::max(hi[1], (scalar_t)py[i]);
                hi[2] = std::max(hi[2], (scalar_t)pz[i]);
            }
            d_.Lx = hi[0];
            d_.Ly = hi[1];
            d_.Lz = hi[2];
        }

        sync_scheme_parameters();

        d_.ux.assign((size_t)d_.nnodes, scalar_t(0));
        d_.uy.assign((size_t)d_.nnodes, scalar_t(0));
        d_.uz.assign((size_t)d_.nnodes, scalar_t(0));
        d_.p.assign((size_t)d_.nnodes, scalar_t(0));
        d_.rx.assign((size_t)d_.nnodes, scalar_t(0));
        d_.ry.assign((size_t)d_.nnodes, scalar_t(0));
        d_.rz.assign((size_t)d_.nnodes, scalar_t(0));
        d_.rc.assign((size_t)d_.nnodes, scalar_t(0));

        if (geom == GeomKind::Affine) {
            cvfem_hex8_precompute_affine_geometry(d_);
            if (pack_size > 0) {
                packed_    = make_packed(d_.mesh, pack_size);
                d_.packed  = &packed_;
                coloring_  = cvfem_build_pack_coloring(packed_.n_packs,
                                                      packed_.owned_nodes_ptr,
                                                      packed_.ghost_ptr,
                                                      packed_.ghost_idx);
                d_.coloring = &coloring_;
            }
        }

        // The sparsity is the space's own node-to-node graph, which is also what
        // hessian_bsr is handed, so the element-to-slot map can be built once here.
        bsr_.graph  = d_.mesh->node_to_node_graph();
        bsr_.rowptr = bsr_.graph->rowptr()->data();
        bsr_.colidx = bsr_.graph->colidx()->data();
        bsr_.nnz    = bsr_.graph->nnz();
        precompute_element_bsr_slots(d_, bsr_);

        initialized_ = true;
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::update(const real_t *const x) {
        if (!initialized_) return SFEM_FAILURE;
        sync_scheme_parameters();
        unpack_fields(d_, x);
        assemble_nodal_p_grad(d_, geom);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::gradient");
        if (!initialized_) return SFEM_FAILURE;
        sync_scheme_parameters();
        unpack_fields(d_, x);
        apply_residual(d_, rho, mu, geom);
        // sfem::Op accumulates into out, so add rather than overwrite.
        add_residual(d_, out);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::apply");
        if (!initialized_) return SFEM_FAILURE;
        sync_scheme_parameters();
        unpack_fields(d_, x);
        assemble_nodal_p_grad(d_, geom);
        apply_jacobian_action_accumulate(d_, rho, mu, geom, h, out);
        return SFEM_SUCCESS;
    }

    int CVFEMNavierStokes::value(const real_t * /*x*/, real_t *const out) {
        // Steady Navier-Stokes is not the stationary point of an energy, so there is no
        // value to report. Returning zero rather than erroring keeps Function::value
        // usable for the other operators in the same Function.
        if (out) *out += 0;
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
        if (!initialized_) return SFEM_FAILURE;
        sync_scheme_parameters();
        unpack_fields(d_, x);

        // The slot caches were built against the space's graph in initialize(); assembly
        // writes through external_values into the caller's buffer.
        bsr_.rowptr          = rowptr;
        bsr_.colidx          = colidx;
        bsr_.external_values = values;
        // Accumulate: Function::hessian_bsr shares one buffer across operators.
        assemble_jacobian(d_, bsr_, rho, mu, geom, /*zero_first=*/false);
        bsr_.external_values = nullptr;
        return SFEM_SUCCESS;
    }

    std::shared_ptr<Op> CVFEMNavierStokes::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("CVFEMNavierStokes::derefine_op");
        // Rediscretisation, not Galerkin coarsening. That is not a stylistic choice here:
        // the Rhie-Chow coefficient carries h^2/(2 mu) explicitly, so the coarse pressure
        // operator differs from the fine one by roughly 8x per level in 3D. Assembling
        // the CVFEM operator on the coarse mesh gets that right; P^T A P would inherit
        // the fine-grid stabilisation and be inconsistent.
        auto ret             = std::make_shared<CVFEMNavierStokes>(space);
        ret->rho             = rho;
        ret->mu              = mu;
        ret->rhie_chow_scale = rhie_chow_scale;
        ret->geom            = geom;
        ret->pack_size       = pack_size;
        ret->initialize();
        return ret;
    }

    std::shared_ptr<Op> CVFEMNavierStokes::clone() const {
        auto ret             = std::make_shared<CVFEMNavierStokes>(space_);
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
