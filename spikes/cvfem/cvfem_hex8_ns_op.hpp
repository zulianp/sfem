#pragma once

// sfem::Op over the HEX8 CVFEM Navier-Stokes kernels.
//
// This is the frontend-facing face of cvfem_hex8_ns_core.hpp: it owns a MeshData and
// delegates to the same residual and assembly the standalone driver uses, so the two
// stay numerically identical by construction rather than by parallel maintenance.
//
// The block size is 4 -- (ux, uy, uz, p) per node -- and the element kernels already
// write that interleaved layout (r[i * 4 + c]), which is the layout sfem::Op expects.
// What does not match is MeshData's own state, which is stored field by field; update()
// scatters the incoming vector into it and the residual is gathered back on the way out.
//
// Purpose is the semi-structured GMG executable. This step deliberately keeps the flat
// HEX8 kernels and only puts the Op interface around them, so it can be gated against
// the driver before the semi-structured element handling changes anything underneath.

#include "cvfem_hex8_ns_core.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_Op.hpp"

#include <memory>

namespace sfem {

    class CVFEMNavierStokes final : public Op {
    public:
        static_assert(sizeof(real_t) == sizeof(scalar_t),
                      "CVFEMNavierStokes hands sfem buffers straight to the CVFEM kernels, which are "
                      "compiled for double; a float32 real_t build would need a conversion layer.");

        explicit CVFEMNavierStokes(const std::shared_ptr<FunctionSpace> &space);
        ~CVFEMNavierStokes() override = default;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        const char *name() const override { return "cvfem:NavierStokes"; }
        bool        is_linear() const override { return false; }

        ptrdiff_t n_dofs_domain() const override { return space_->n_dofs(); }
        ptrdiff_t n_dofs_image() const override { return space_->n_dofs(); }

        int initialize(const std::vector<std::string> &block_names = {}) override;

        // Refreshes the nodal pressure gradient that Rhie-Chow interpolation needs. The
        // driver recomputes it inside apply_residual/assemble_jacobian; here it is also
        // exposed on the Op's own update hook so a caller can pay for it once.
        int update(const real_t *const x) override;

        int gradient(const real_t *const x, real_t *const out) override;
        int apply(const real_t *const x, const real_t *const h, real_t *const out) override;
        int value(const real_t *x, real_t *const out) override;

        // Pure virtual on the base, so it has to exist. It refuses: this is a
        // vector-valued problem and BSR is the format for those here, so a scalar CRS
        // expansion of the 4x4 blocks would be a worse matrix nobody wants.
        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override;

        // Scalar diagonal, one value per dof, taken from the diagonal of each 4x4 block.
        int hessian_diag(const real_t *const x, real_t *const values) override;

        // Full 4x4 diagonal block per node, 16 values each, matrix-free. Not a base-class
        // virtual: Op offers hessian_block_diag_sym, whose symmetric packing does not fit
        // a Navier-Stokes block. This is what the block-Jacobi smoother will call.
        int hessian_block_diag(const real_t *const x, real_t *const values);

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> clone() const override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;

        // Physical and scheme parameters. Public so the driver can set them without a
        // parameter-block round trip; set_value_in_block covers the yaml path.
        scalar_t rho{1};
        scalar_t mu{0.01};
        scalar_t rhie_chow_scale{1};
        GeomKind geom{GeomKind::Affine};

        // Affine packing width, mirroring SFEM_PACK_SIZE in the driver. 0 selects the
        // atomic path. Ignored for isoparametric geometry, which has no packed kernel.
        int pack_size{2048};

    private:
        void sync_scheme_parameters();

        std::shared_ptr<FunctionSpace> space_;
        MeshData                       d_;
        // MeshData holds bare pointers into these, so the Op has to own them.
        PackedData                     packed_;
        PackColoring                   coloring_;
        BSR4                           bsr_;  // slot caches only; values come from the caller
        bool                           initialized_{false};
    };

}  // namespace sfem
