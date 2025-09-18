#include "sfem_SemiStructuredNeoHookeanOgden.hpp"

// C includes
#include "sfem_macros.h"
#include "sshex8_neohookean_ogden.h"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_Mesh.hpp"
#include "sfem_NeoHookeanOgden.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_glob.hpp"

#include <mpi.h>

// FIXME
#include "hex8_neohookean_ogden.h"
#include "hex8_partial_assembly_neohookean_inline.h"

namespace sfem {
    class SemiStructuredNeoHookeanOgden::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;                           ///< Function space for the operator
        enum ElemType                  element_type { INVALID };        ///< Element type
        real_t                         mu{1}, lambda{1};                ///< Lamé parameters
        bool                           use_affine_approximation{true};  ///< Use affine approximation for performance
        SharedBuffer<metric_tensor_t>  partial_assembly;
        Impl(const std::shared_ptr<FunctionSpace> &space) : space(space) {}
    };

    std::unique_ptr<Op> SemiStructuredNeoHookeanOgden::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredNeoHookeanOgden::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredNeoHookeanOgden::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(space->element_type() == SSHEX8);  // REMOVEME once generalized approach
        auto ret = std::make_unique<SemiStructuredNeoHookeanOgden>(space);

        real_t SFEM_SHEAR_MODULUS        = 1;
        real_t SFEM_FIRST_LAME_PARAMETER = 1;

        SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
        SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

        ret->impl_->mu           = SFEM_SHEAR_MODULUS;
        ret->impl_->lambda       = SFEM_FIRST_LAME_PARAMETER;
        ret->impl_->element_type = (enum ElemType)space->element_type();

        int SFEM_HEX8_ASSUME_AFFINE = ret->impl_->use_affine_approximation;
        SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
        ret->impl_->use_affine_approximation = SFEM_HEX8_ASSUME_AFFINE;
        return ret;
    }

    SemiStructuredNeoHookeanOgden::SemiStructuredNeoHookeanOgden(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    int SemiStructuredNeoHookeanOgden::hessian_crs(const real_t *const  x,
                                                   const count_t *const rowptr,
                                                   const idx_t *const   colidx,
                                                   real_t *const        values) {
        assert(false);
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_bsr(const real_t *const  x,
                                                   const count_t *const rowptr,
                                                   const idx_t *const   colidx,
                                                   real_t *const        values) {
        auto &ssm = impl_->space->semi_structured_mesh();
        // SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_bsr", ssm.level());

        // return affine_sshex8_elasticity_bsr(ssm.level(),
        //                                     ssm.n_elements(),
        //                                     ssm.interior_start(),
        //                                     ssm.element_data(),
        //                                     ssm.point_data(),
        //                                     this->mu,
        //                                     this->lambda,
        //                                     rowptr,
        //                                     colidx,
        //                                     values);

        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_diag(const real_t *const x, real_t *const values) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_diag", ssm.level());

        // return affine_sshex8_linear_elasticity_diag(ssm.level(),
        //                                             ssm.n_elements(),
        //                                             ssm.interior_start(),
        //                                             ssm.element_data(),
        //                                             ssm.point_data(),
        //                                             mu,
        //                                             lambda,
        //                                             3,
        //                                             &values[0],
        //                                             &values[1],
        //                                             &values[2]);
        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_block_diag_sym(const real_t *const x, real_t *const values) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::hessian_block_diag_sym", ssm.level());

        //     return affine_sshex8_linear_elasticity_block_diag_sym(ssm.level(),
        //                                                           ssm.n_elements(),
        //                                                           ssm.interior_start(),
        //                                                           ssm.element_data(),
        //                                                           ssm.point_data(),
        //                                                           mu,
        //                                                           lambda,
        //                                                           6,
        //                                                           &values[0],
        //                                                           &values[1],
        //                                                           &values[2],
        //                                                           &values[3],
        //                                                           &values[4],
        //                                                           &values[5]);
        SFEM_IMPLEMENT_ME();
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::gradient(const real_t *const x, real_t *const out) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::gradient", ssm.level());

        return sshex8_neohookean_ogden_gradient(ssm.level(),
                                                ssm.n_elements(),
                                                1,
                                                ssm.n_nodes(),
                                                ssm.element_data(),
                                                ssm.point_data(),
                                                impl_->mu,
                                                impl_->lambda,
                                                3,
                                                &x[0],
                                                &x[1],
                                                &x[2],
                                                3,
                                                &out[0],
                                                &out[1],
                                                &out[2]);
    }

    int SemiStructuredNeoHookeanOgden::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::apply", ssm.level());

        return sshex8_neohookean_ogden_partial_assembly_apply(ssm.level(),
                                                              ssm.n_elements(),
                                                              1,
                                                              ssm.element_data(),
                                                              impl_->partial_assembly->data(),
                                                              3,
                                                              &h[0],
                                                              &h[1],
                                                              &h[2],
                                                              3,
                                                              &out[0],
                                                              &out[1],
                                                              &out[2]);
    }

    int SemiStructuredNeoHookeanOgden::value(const real_t *x, real_t *const out) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::value", ssm.level());

        return sshex8_neohookean_ogden_objective(ssm.level(),
                                                 ssm.n_elements(),
                                                 1,
                                                 ssm.n_nodes(),
                                                 ssm.element_data(),
                                                 ssm.point_data(),
                                                 impl_->mu,
                                                 impl_->lambda,
                                                 3,
                                                 &x[0],
                                                 &x[1],
                                                 &x[2],
                                                 false,
                                                 out);
    }

    int SemiStructuredNeoHookeanOgden::value_steps(const real_t       *x,
                                                   const real_t       *h,
                                                   const int           nsteps,
                                                   const real_t *const steps,
                                                   real_t *const       out) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::value_steps", ssm.level());

        return sshex8_neohookean_ogden_objective_steps(ssm.level(),
                                                       ssm.n_elements(),
                                                       1,
                                                       ssm.n_nodes(),
                                                       ssm.element_data(),
                                                       ssm.point_data(),
                                                       impl_->mu,
                                                       impl_->lambda,
                                                       3,
                                                       &x[0],
                                                       &x[1],
                                                       &x[2],
                                                       3,
                                                       &h[0],
                                                       &h[1],
                                                       &h[2],
                                                       nsteps,
                                                       steps,
                                                       out);
    }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::clone() const {
        SFEM_IMPLEMENT_ME();
        return nullptr;
    }

    SemiStructuredNeoHookeanOgden::~SemiStructuredNeoHookeanOgden() = default;

    int SemiStructuredNeoHookeanOgden::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredNeoHookeanOgden::derefine_op");

        assert(space->has_semi_structured_mesh() || space->element_type() == macro_base_elem(impl_->element_type));

        if (space->has_semi_structured_mesh()) {
            auto ret                             = std::make_shared<SemiStructuredNeoHookeanOgden>(space);
            ret->impl_->element_type             = impl_->element_type;
            ret->impl_->use_affine_approximation = impl_->use_affine_approximation;
            ret->impl_->mu                       = impl_->mu;
            ret->impl_->lambda                   = impl_->lambda;
            // ret->initialize();
            return ret;
        } else {
            assert(space->element_type() == macro_base_elem(impl_->element_type));
            auto ret = std::make_shared<NeoHookeanOgden>(space);
            ret->initialize();
            ret->set_mu(impl_->mu);
            ret->set_lambda(impl_->lambda);
            ret->override_element_types({macro_base_elem(impl_->element_type)});
            return ret;
        }
    }

    std::shared_ptr<Op> SemiStructuredNeoHookeanOgden::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        assert(false);
        fprintf(stderr, "[Error] ss:NeoHookeanOgden::lor_op NOT IMPLEMENTED!\n");
        return nullptr;
    }

    const char *SemiStructuredNeoHookeanOgden::name() const { return "ss:NeoHookeanOgden"; }

    void SemiStructuredNeoHookeanOgden::set_option(const std::string &name, bool val) {
        if (name == "ASSUME_AFFINE") {
            impl_->use_affine_approximation = val;
        }
    }

    int SemiStructuredNeoHookeanOgden::hessian_crs_sym(const real_t *const  x,
                                                       const count_t *const rowptr,
                                                       const idx_t *const   colidx,
                                                       real_t *const        diag_values,
                                                       real_t *const        off_diag_values) {
        SFEM_ERROR("[Error] ss:NeoHookeanOgden::hessian_crs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::hessian_bcrs_sym(const real_t *const  x,
                                                        const count_t *const rowidx,
                                                        const idx_t *const   colidx,
                                                        const ptrdiff_t      block_stride,
                                                        real_t **const       diag_values,
                                                        real_t **const       off_diag_values) {
        SFEM_ERROR("[Error] ss:NeoHookeanOgden::hessian_bcrs_sym NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredNeoHookeanOgden::update(const real_t *const x) {
        auto &ssm = impl_->space->semi_structured_mesh();
        SFEM_TRACE_SCOPE_VARIANT("SemiStructuredNeoHookeanOgden[%d]::update", ssm.level());

        if (!impl_->partial_assembly) {
            impl_->partial_assembly = sfem::create_host_buffer<metric_tensor_t>(ssm.n_elements() * HEX8_S_IKMN_SIZE);
        }

        sshex8_neohookean_ogden_hessian_partial_assembly(ssm.level(),
                                                         ssm.n_elements(),
                                                         1,
                                                         ssm.element_data(),
                                                         ssm.point_data(),
                                                         impl_->mu,
                                                         impl_->lambda,
                                                         3,
                                                         &x[0],
                                                         &x[1],
                                                         &x[2],
                                                         impl_->partial_assembly->data());

        return SFEM_SUCCESS;
    }

    int SemiStructuredNeoHookeanOgden::report(const real_t *const) { return SFEM_SUCCESS; }
}  // namespace sfem
