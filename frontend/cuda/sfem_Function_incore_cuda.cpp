#include "sfem_Function_incore_cuda.hpp"
#include <memory>
#include "boundary_condition.h"

#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "cu_boundary_condition.h"
#include "cu_hex8_adjugate.h"
#include "cu_hex8_fff.h"
#include "cu_laplacian.h"
#include "cu_linear_elasticity.h"
#include "cu_mask.h"
#include "cu_sshex8_laplacian.h"
#include "cu_sshex8_linear_elasticity.h"
#include "cu_tet4_adjugate.h"
#include "cu_tet4_fff.h"

// C++ includes
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sfem_API.hpp"

namespace sfem {

    std::shared_ptr<Buffer<idx_t *>> create_device_elements(const std::shared_ptr<FunctionSpace> &space,
                                                            const enum ElemType                   element_type) {
        if (space->has_semi_structured_mesh()) {
            return to_device(space->semi_structured_mesh().elements());

        } else {
            return to_device(space->mesh().elements());
        }
    }

    std::shared_ptr<Sideset> to_device(const std::shared_ptr<Sideset> &sideset) {
        return std::make_shared<Sideset>(sideset->comm(), to_device(sideset->parent()), to_device(sideset->lfi()));
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> manage_device_buffer(const ptrdiff_t n, T *data) {
        return Buffer<T>::own(n, data, &d_buffer_destroy, MEMORY_SPACE_DEVICE);
    }

    class FFF {
    public:
        FFF(Mesh &mesh, const enum ElemType element_type, const std::shared_ptr<Buffer<idx_t *>> &elements)
            : element_type_(element_type), elements_(elements) {
            void *fff{nullptr};
            if (element_type == HEX8 || element_type == SSHEX8) {
                cu_hex8_fff_allocate(mesh.n_elements(), &fff);
                cu_hex8_fff_fill(mesh.n_elements(), mesh.elements()->data(), mesh.points()->data(), fff);
            } else {
                cu_tet4_fff_allocate(mesh.n_elements(), &fff);
                cu_tet4_fff_fill(mesh.n_elements(), mesh.elements()->data(), mesh.points()->data(), fff);
            }

            // FIXME compute size (currently 6)
            fff_ = manage_device_buffer<void>(mesh.n_elements() * 6, fff);
        }

        FFF(enum ElemType                           element_type,
            const std::shared_ptr<Buffer<idx_t *>> &elements,
            const std::shared_ptr<Buffer<void>>    &fff)
            : element_type_(element_type), elements_(elements), fff_(fff) {}

        ~FFF() {}

        enum ElemType                    element_type() const { return element_type_; }
        ptrdiff_t                        n_elements() const { return elements_->extent(1); }
        std::shared_ptr<Buffer<idx_t *>> elements() const { return elements_; }
        std::shared_ptr<Buffer<void>>    fff() const { return fff_; }

    private:
        enum ElemType                    element_type_;
        std::shared_ptr<Buffer<idx_t *>> elements_;
        std::shared_ptr<Buffer<void>>    fff_;
    };

    class Adjugate {
    public:
        Adjugate(Mesh &mesh, const enum ElemType element_type, const std::shared_ptr<Buffer<idx_t *>> &elements)
            : element_type_(element_type), elements_(elements) {
            void *jacobian_adjugate{nullptr};
            void *jacobian_determinant{nullptr};

            if (element_type == HEX8 || element_type == SSHEX8) {
                cu_hex8_adjugate_allocate(mesh.n_elements(), &jacobian_adjugate, &jacobian_determinant);
                cu_hex8_adjugate_fill(mesh.n_elements(),
                                      mesh.elements()->data(),
                                      mesh.points()->data(),
                                      jacobian_adjugate,
                                      jacobian_determinant);

            } else {
                cu_tet4_adjugate_allocate(mesh.n_elements(), &jacobian_adjugate, &jacobian_determinant);
                cu_tet4_adjugate_fill(mesh.n_elements(),
                                      mesh.elements()->data(),
                                      mesh.points()->data(),
                                      jacobian_adjugate,
                                      jacobian_determinant);
            }

            // FIXME compute size (currently 9)
            jacobian_adjugate_    = manage_device_buffer<void>(mesh.n_elements() * 9, jacobian_adjugate);
            jacobian_determinant_ = manage_device_buffer<void>(mesh.n_elements(), jacobian_determinant);
        }

        Adjugate(enum ElemType                           element_type,
                 const std::shared_ptr<Buffer<idx_t *>> &elements,
                 const std::shared_ptr<Buffer<void>>    &jacobian_adjugate,
                 const std::shared_ptr<Buffer<void>>    &jacobian_determinant)
            : element_type_(element_type),
              elements_(elements),
              jacobian_adjugate_(jacobian_adjugate),
              jacobian_determinant_(jacobian_determinant) {}

        ~Adjugate() {}

        enum ElemType                    element_type() const { return element_type_; }
        ptrdiff_t                        n_elements() const { return elements_->extent(1); }
        std::shared_ptr<Buffer<idx_t *>> elements() const { return elements_; }
        std::shared_ptr<Buffer<void>>    jacobian_determinant() const { return jacobian_determinant_; }
        std::shared_ptr<Buffer<void>>    jacobian_adjugate() const { return jacobian_adjugate_; }

    private:
        enum ElemType                    element_type_;
        std::shared_ptr<Buffer<idx_t *>> elements_;
        std::shared_ptr<Buffer<void>>    jacobian_adjugate_;
        std::shared_ptr<Buffer<void>>    jacobian_determinant_;
    };

    class GPUDirichletConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace>                     space;
        std::shared_ptr<DirichletConditions>               h_dirichlet;
        std::vector<struct DirichletConditions::Condition> conditions;

        GPUDirichletConditions(const std::shared_ptr<DirichletConditions> &dc) : space(dc->space()), h_dirichlet(dc) {
            for (auto &c : dc->conditions()) {
                DirichletConditions::Condition cond{.sideset   = (c.sideset) ? to_device(c.sideset) : nullptr,
                                                    .nodeset   = to_device(c.nodeset),
                                                    .values    = (c.values) ? to_device(c.values) : nullptr,
                                                    .value     = c.value,
                                                    .component = c.component};
                conditions.push_back(cond);
            }
        }

        int apply(real_t *const x) override {
            for (auto &c : conditions) {
                d_constraint_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, c.value, x);
            }

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const g) override {
            for (auto &c : conditions) {
                d_constraint_gradient_nodes_to_value_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, c.value, x, g);
            }

            return SFEM_SUCCESS;
        }

        int apply_value(const real_t value, real_t *const x) override {
            for (auto &c : conditions) {
                d_constraint_nodes_to_value_vec(c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, value, x);
            }

            return SFEM_SUCCESS;
        }

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override {
            for (auto &c : conditions) {
                d_constraint_nodes_copy_vec(c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, src, dest);
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            for (auto &c : conditions) {
                cu_crs_constraint_nodes_to_identity_vec(
                        c.nodeset->size(), c.nodeset->data(), space->block_size(), c.component, 1, rowptr, colidx, values);
            }

            return SFEM_SUCCESS;
        }

        std::shared_ptr<Constraint> lor() const override {
            SFEM_IMPLEMENT_ME();
            return nullptr;
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<sfem::FunctionSpace> &space, bool as_zeros) const override {
            auto h_derefined = std::static_pointer_cast<DirichletConditions>(h_dirichlet->derefine(space, as_zeros));
            return std::make_shared<GPUDirichletConditions>(h_derefined);
        }

        int mask(mask_t *mask) override {
            SFEM_TRACE_SCOPE("GPUDirichletConditions::mask");

            const int block_size = space->block_size();

            int err = SFEM_SUCCESS;
            for (auto &c : conditions) {
                auto nodeset = c.nodeset->data();
                err += cu_mask_nodes(c.nodeset->size(), nodeset, block_size, c.component, mask);
            }

            return err;
        }

        ~GPUDirichletConditions() {}
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc) {
        return std::make_shared<GPUDirichletConditions>(dc);
    }

    class GPUNeumannConditions final : public Op {
    public:
        std::shared_ptr<FunctionSpace>                   space;
        std::shared_ptr<NeumannConditions>               h_neumann;
        std::vector<struct NeumannConditions::Condition> conditions;

        GPUNeumannConditions(const std::shared_ptr<NeumannConditions> &nc) : space(nc->space()), h_neumann(nc) {
            for (auto &c : nc->conditions()) {
                NeumannConditions::Condition cond{.element_type = c.element_type,
                                                  .sideset      = (c.sideset) ? to_device(c.sideset) : nullptr,
                                                  .surface      = to_device(c.surface),
                                                  .values       = (c.values) ? to_device(c.values) : nullptr,
                                                  .value        = c.value,
                                                  .component    = c.component};
                conditions.push_back(cond);
            }
        }

        const char *name() const override { return "gpu:NeumannConditions"; }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override { return SFEM_SUCCESS; }

        int value(const real_t *x, real_t *const out) override { return SFEM_SUCCESS; }

        inline bool is_linear() const override { return true; }

        int n_conditions() const;

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override { return no_op(); }
    };

    std::shared_ptr<Op> to_device(const std::shared_ptr<NeumannConditions> &nc) {
        return std::make_shared<GPUNeumannConditions>(nc);
    }

    class GPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<FFF>           fff;
        enum RealType                  real_type { SFEM_REAL_DEFAULT };
        void                          *stream{SFEM_DEFAULT_STREAM};
        enum ElemType                  element_type { INVALID };

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(1 == space->block_size());
            return std::make_unique<GPULaplacian>(space);
        }

        const char *name() const override { return "gpu:Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto elements = space->device_elements();
            if (!elements) {
                elements = create_device_elements(space, space->element_type());
                space->set_device_elements(elements);
            }

            fff = std::make_shared<FFF>(space->mesh(), space->element_type(), elements);
            return SFEM_SUCCESS;
        }

        GPULaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space), element_type(space->element_type()) {}

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            auto mesh = (mesh_t *)derefined_space->mesh().impl_mesh();

            auto ret = std::make_shared<GPULaplacian>(derefined_space);
            assert(derefined_space->element_type() == macro_base_elem(fff->element_type()));
            assert(ret->element_type == macro_base_elem(fff->element_type()));
            ret->fff = fff;
            return ret;
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            return cu_laplacian_crs(element_type,
                                    fff->n_elements(),
                                    fff->elements()->data(),
                                    fff->n_elements(),  // stride
                                    fff->fff()->data(),
                                    rowptr,
                                    colidx,
                                    real_type,
                                    values,
                                    stream);
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            return cu_laplacian_diag(element_type,
                                     fff->n_elements(),
                                     fff->elements()->data(),
                                     fff->n_elements(),  // stride
                                     fff->fff()->data(),
                                     real_type,
                                     values,
                                     stream);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return cu_laplacian_apply(element_type,
                                      fff->n_elements(),
                                      fff->elements()->data(),
                                      fff->n_elements(),  // stride
                                      fff->fff()->data(),
                                      real_type,
                                      x,
                                      out,
                                      stream);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return cu_laplacian_apply(element_type,
                                      fff->n_elements(),
                                      fff->elements()->data(),
                                      fff->n_elements(),  // stride
                                      fff->fff()->data(),
                                      real_type,
                                      h,
                                      out,
                                      stream);
        }

        int hessian_crs_sym(const real_t *const /*x*/,
                            const count_t *const rowptr,
                            const idx_t *const   colidx,
                            real_t *const        diag_values,
                            real_t *const        off_diag_values) override {
            return cu_laplacian_crs_sym(element_type,
                                        fff->n_elements(),
                                        fff->elements()->data(),
                                        fff->n_elements(),  // stride
                                        fff->fff()->data(),
                                        rowptr,
                                        colidx,
                                        real_type,
                                        diag_values,
                                        off_diag_values,
                                        stream);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class SemiStructuredGPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<FFF>           fff;
        enum RealType                  real_type { SFEM_REAL_DEFAULT };
        void                          *stream{SFEM_DEFAULT_STREAM};
        enum ElemType                  element_type { INVALID };

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(1 == space->block_size());
            return std::make_unique<SemiStructuredGPULaplacian>(space);
        }

        const char *name() const override { return "ss:gpu::Laplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto elements = space->device_elements();

            if (!elements) {
                elements = create_device_elements(space, space->element_type());
                space->set_device_elements(elements);
            }

            fff = std::make_shared<FFF>(space->mesh(), space->element_type(), elements);
            return SFEM_SUCCESS;
        }

        SemiStructuredGPULaplacian(const std::shared_ptr<FunctionSpace> &space)
            : space(space), element_type(space->element_type()) {}

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            if (derefined_space->has_semi_structured_mesh()) {
                auto ret          = std::make_shared<SemiStructuredGPULaplacian>(derefined_space);
                ret->element_type = element_type;
                ret->fff          = std::make_shared<FFF>(
                        element_type,
                        sshex8_derefine_element_connectivity(space->semi_structured_mesh().level(),
                                                             derefined_space->semi_structured_mesh().level(),
                                                             fff->elements()),
                                                             fff->fff());
                ret->real_type = real_type;
                ret->stream = stream;
                return ret;
            } else {
                auto ret = std::make_shared<GPULaplacian>(derefined_space);
                assert(derefined_space->element_type() == macro_base_elem(fff->element_type()));
                assert(ret->element_type == macro_base_elem(fff->element_type()));
                // SFEM_ERROR("AVOID replicating indices create view!\n");  // TODO
                ret->initialize();
                return ret;
            }
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_laplacian_diag[%d]", ssm.level());

            return cu_affine_sshex8_laplacian_diag(ssm.level(),
                                                   fff->n_elements(),
                                                   fff->elements()->data(),
                                                   fff->n_elements(),  // stride
                                                   fff->fff()->data(),
                                                   real_type,
                                                   out,
                                                   stream);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_laplacian_apply[%d]", ssm.level());

            return cu_affine_sshex8_laplacian_apply(ssm.level(),
                                                    fff->n_elements(),
                                                    fff->elements()->data(),
                                                    fff->n_elements(),  // stride
                                                    fff->fff()->data(),
                                                    real_type,
                                                    x,
                                                    out,
                                                    stream);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_laplacian_apply[%d]", ssm.level());

            return cu_affine_sshex8_laplacian_apply(ssm.level(),
                                                    fff->n_elements(),
                                                    fff->elements()->data(),
                                                    fff->n_elements(),  // stride
                                                    fff->fff()->data(),
                                                    real_type,
                                                    h,
                                                    out,
                                                    stream);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class GPULinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Adjugate>      adjugate;
        enum RealType                  real_type { SFEM_REAL_DEFAULT };
        void                          *stream{SFEM_DEFAULT_STREAM};
        enum ElemType                  element_type { INVALID };
        real_t                         mu{1};
        real_t                         lambda{1};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
            return std::make_unique<GPULinearElasticity>(space);
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            auto ret = std::make_shared<GPULinearElasticity>(derefined_space);
            assert(derefined_space->element_type() == macro_base_elem(adjugate->element_type()));
            assert(ret->element_type == macro_base_elem(adjugate->element_type()));
            ret->adjugate = adjugate;
            return ret;
        }

        const char *name() const override { return "gpu:LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
            mu     = SFEM_SHEAR_MODULUS;
            lambda = SFEM_FIRST_LAME_PARAMETER;

            auto elements = space->device_elements();
            if (!elements) {
                elements = create_device_elements(space, space->element_type());
                space->set_device_elements(elements);
            }

            adjugate = std::make_shared<Adjugate>(space->mesh(), space->element_type(), elements);
            return SFEM_SUCCESS;
        }

        GPULinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space), element_type(space->element_type()) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_bsr");

            return cu_linear_elasticity_bsr(element_type,
                                            adjugate->n_elements(),
                                            adjugate->elements()->data(),
                                            adjugate->n_elements(),  // stride
                                            adjugate->jacobian_adjugate()->data(),
                                            adjugate->jacobian_determinant()->data(),
                                            mu,
                                            lambda,
                                            real_type,
                                            rowptr,
                                            colidx,
                                            values,
                                            stream);
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_diag");

            return cu_linear_elasticity_diag(element_type,
                                             adjugate->n_elements(),
                                             adjugate->elements()->data(),
                                             adjugate->n_elements(),  // stride
                                             adjugate->jacobian_adjugate()->data(),
                                             adjugate->jacobian_determinant()->data(),
                                             mu,
                                             lambda,
                                             real_type,
                                             values,
                                             stream);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_apply");

            return cu_linear_elasticity_apply(element_type,
                                              adjugate->n_elements(),
                                              adjugate->elements()->data(),
                                              adjugate->n_elements(),  // stride
                                              adjugate->jacobian_adjugate()->data(),
                                              adjugate->jacobian_determinant()->data(),
                                              mu,
                                              lambda,
                                              real_type,
                                              x,
                                              out,
                                              stream);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("cu_linear_elasticity_apply");

            return cu_linear_elasticity_apply(element_type,
                                              adjugate->n_elements(),
                                              adjugate->elements()->data(),
                                              adjugate->n_elements(),  // stride
                                              adjugate->jacobian_adjugate()->data(),
                                              adjugate->jacobian_determinant()->data(),
                                              mu,
                                              lambda,
                                              real_type,
                                              h,
                                              out,
                                              stream);
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override {
            SFEM_TRACE_SCOPE("LinearElasticity::hessian_block_diag_sym");

            return cu_linear_elasticity_block_diag_sym_aos(element_type,
                                                           adjugate->n_elements(),
                                                           adjugate->elements()->data(),
                                                           adjugate->n_elements(),  // stride
                                                           adjugate->jacobian_adjugate()->data(),
                                                           adjugate->jacobian_determinant()->data(),
                                                           this->mu,
                                                           this->lambda,
                                                           real_type,
                                                           values,
                                                           stream);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class SemiStructuredGPULinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        enum ElemType                  element_type { INVALID };
        std::shared_ptr<Adjugate>      adjugate;

        real_t mu{1}, lambda{1};

        enum RealType real_type { SFEM_REAL_DEFAULT };
        void         *stream{SFEM_DEFAULT_STREAM};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            assert(space->mesh_ptr()->spatial_dimension() == space->block_size());
            return std::make_unique<SemiStructuredGPULinearElasticity>(space);
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &derefined_space) override {
            SFEM_TRACE_SCOPE("SemiStructuredGPULinearElasticity::derefine_op");

            if (derefined_space->has_semi_structured_mesh()) {
                auto ret = std::make_shared<SemiStructuredGPULinearElasticity>(derefined_space);

                ret->adjugate = std::make_shared<Adjugate>(
                        element_type,
                        sshex8_derefine_element_connectivity(space->semi_structured_mesh().level(),
                                                             derefined_space->semi_structured_mesh().level(),
                                                             adjugate->elements()),
                                                             adjugate->jacobian_adjugate(),
                                                             adjugate->jacobian_determinant());

                ret->element_type = element_type;

                ret->mu     = mu;
                ret->lambda = lambda;

                ret->real_type = real_type;
                ret->stream    = stream;
                return ret;
            } else {
                auto ret = std::make_shared<GPULinearElasticity>(derefined_space);
                assert(derefined_space->element_type() == macro_base_elem(adjugate->element_type()));
                assert(ret->element_type == macro_base_elem(adjugate->element_type()));
                // SFEM_ERROR("AVOID replicating indices create view!\n");  // TODO
                ret->initialize();
                return ret;
            }
        }

        const char *name() const override { return "ss::gpu::LinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            real_t SFEM_SHEAR_MODULUS        = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
            mu     = SFEM_SHEAR_MODULUS;
            lambda = SFEM_FIRST_LAME_PARAMETER;

            auto elements = space->device_elements();
            if (!elements) {
                elements = create_device_elements(space, space->element_type());
                space->set_device_elements(elements);
            }

            adjugate = std::make_shared<Adjugate>(space->mesh(), space->element_type(), elements);
            return SFEM_SUCCESS;
        }

        SemiStructuredGPULinearElasticity(const std::shared_ptr<FunctionSpace> &space)
            : space(space), element_type(space->element_type()) {}

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_linear_elasticity_diag[%d]", ssm.level());

            return cu_affine_sshex8_linear_elasticity_diag(ssm.level(),
                                                           adjugate->n_elements(),
                                                           adjugate->elements()->data(),
                                                           adjugate->n_elements(),  // stride
                                                           adjugate->jacobian_adjugate()->data(),
                                                           adjugate->jacobian_determinant()->data(),
                                                           mu,
                                                           lambda,
                                                           real_type,
                                                           3,
                                                           &values[0],
                                                           &values[1],
                                                           &values[2],
                                                           SFEM_DEFAULT_STREAM);
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_linear_elasticity_block_diag_sym_aos[%d]", ssm.level());

            return cu_affine_sshex8_linear_elasticity_block_diag_sym(ssm.level(),
                                                                     adjugate->n_elements(),
                                                                     adjugate->elements()->data(),
                                                                     adjugate->n_elements(),  // stride
                                                                     adjugate->jacobian_adjugate()->data(),
                                                                     adjugate->jacobian_determinant()->data(),
                                                                     this->mu,
                                                                     this->lambda,
                                                                     6,
                                                                     real_type,
                                                                     &out[0],
                                                                     &out[1],
                                                                     &out[2],
                                                                     &out[3],
                                                                     &out[4],
                                                                     &out[5],
                                                                     SFEM_DEFAULT_STREAM);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_linear_elasticity_apply[%d]", ssm.level());

            return cu_affine_sshex8_linear_elasticity_apply(ssm.level(),
                                                            adjugate->n_elements(),
                                                            adjugate->elements()->data(),
                                                            adjugate->n_elements(),  // stride
                                                            adjugate->jacobian_adjugate()->data(),
                                                            adjugate->jacobian_determinant()->data(),
                                                            mu,
                                                            lambda,
                                                            real_type,
                                                            3,
                                                            &x[0],
                                                            &x[1],
                                                            &x[2],
                                                            3,
                                                            &out[0],
                                                            &out[1],
                                                            &out[2],
                                                            SFEM_DEFAULT_STREAM);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            SFEM_TRACE_SCOPE_VARIANT("cu_affine_sshex8_linear_elasticity_apply[%d]", ssm.level());

            return cu_affine_sshex8_linear_elasticity_apply(ssm.level(),
                                                            adjugate->n_elements(),
                                                            adjugate->elements()->data(),
                                                            adjugate->n_elements(),  // stride
                                                            adjugate->jacobian_adjugate()->data(),
                                                            adjugate->jacobian_determinant()->data(),
                                                            mu,
                                                            lambda,
                                                            real_type,
                                                            3,
                                                            &h[0],
                                                            &h[1],
                                                            &h[2],
                                                            3,
                                                            &out[0],
                                                            &out[1],
                                                            &out[2],
                                                            SFEM_DEFAULT_STREAM);
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_IMPLEMENT_ME();
            return SFEM_FAILURE;
        }

        int            report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    void register_device_ops() {
        Factory::register_op("gpu:LinearElasticity", &GPULinearElasticity::create);
        Factory::register_op("gpu:Laplacian", &GPULaplacian::create);
        Factory::register_op("ss:gpu:Laplacian", &SemiStructuredGPULaplacian::create);
        Factory::register_op("ss:gpu:LinearElasticity", &SemiStructuredGPULinearElasticity::create);
    }

}  // namespace sfem
