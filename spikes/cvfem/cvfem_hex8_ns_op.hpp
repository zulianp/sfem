#pragma once

// sfem::Op over the HEX8 CVFEM Navier-Stokes kernels.
//
// Deliberately opaque. Everything the operator is built from -- MeshData, BSR4, the
// element kernels, and a file-scope `using scalar_t = double` -- sits behind an Impl in
// the .cpp, so a driver that includes this header gets the operator and nothing else.
//
// That matters more here than it usually would. Two families of HEX8 CVFEM headers live
// in this directory: cvfem_hex8_ns_core.hpp behind the solver, and the
// cvfem_hex8_layout_*.hpp family behind the throughput benchmark. They define sixteen of
// the same names and disagree on the physics behind several of them -- the benchmark's
// assembly carries no boundary sub-control-surface or Rhie-Chow terms. A header that
// leaked the core would settle that argument for every driver that included it, and
// would collide outright with any driver that also wanted the benchmark layouts.
//
// Block size is 4 -- (ux, uy, uz, p) per node -- and the element kernels already write
// that interleaved layout, which is what sfem::Op expects.
//
// Parameters are plain public fields rather than a parameter block. They are read at
// initialize() and again on each call, so they may be set in any order beforehand.

#include "sfem_FunctionSpace.hpp"
#include "sfem_Op.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sfem {

    // Mirrors the core's GeomKind, restated so a driver need not include the core to say
    // which geometry treatment it wants.
    enum class CVFEMGeometry { Affine, Isoparam };

    class CVFEMNavierStokes final : public Op {
    public:
        explicit CVFEMNavierStokes(const std::shared_ptr<FunctionSpace> &space);
        ~CVFEMNavierStokes() override;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space);

        const char *name() const override { return "cvfem:NavierStokes"; }
        bool        is_linear() const override { return false; }

        ptrdiff_t n_dofs_domain() const override;
        ptrdiff_t n_dofs_image() const override;

        int initialize(const std::vector<std::string> &block_names = {}) override;

        // Refreshes the nodal pressure gradient Rhie-Chow interpolation needs. The
        // residual and assembly entry points recompute it themselves, so this is only
        // worth calling to pay for it once across several of them.
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
        // a Navier-Stokes block. This is what a block-Jacobi smoother wants. `values`
        // must hold n_nodes * 16 entries and is accumulated into.
        int hessian_block_diag(const real_t *const x, real_t *const values);

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override;
        std::shared_ptr<Op> clone() const override;

        void set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) override;

        real_t        rho{1};
        real_t        mu{0.01};
        real_t        rhie_chow_scale{1};
        CVFEMGeometry geom{CVFEMGeometry::Affine};

        // Affine packing width, mirroring SFEM_PACK_SIZE in the driver. 0 selects the
        // atomic path. Ignored for isoparametric geometry, which has no packed kernel.
        int pack_size{2048};

    private:
        // Shared by clone() and derefine_op(): same parameters, different space.
        std::shared_ptr<Op> clone_onto(const std::shared_ptr<FunctionSpace> &space) const;

        class Impl;
        std::unique_ptr<Impl> impl_;
    };

}  // namespace sfem
