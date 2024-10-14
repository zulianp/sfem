#include "sfem_Function_incore_cuda.hpp"
#include <memory>
#include "boundary_condition.h"

#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "cu_boundary_condition.h"
#include "cu_laplacian.h"
#include "cu_linear_elasticity.h"
#include "cu_tet4_adjugate.h"
#include "cu_tet4_fff.h"
#include "cu_hex8_fff.h"
#include "cu_proteus_hex8_laplacian.h"

namespace sfem {

    std::shared_ptr<Buffer<idx_t>> create_device_elements(
            const std::shared_ptr<FunctionSpace> &space,
            const enum ElemType element_type) {
        if (space->has_semi_structured_mesh()) {
            auto &ssm = space->semi_structured_mesh();

            int nxe = ssm.n_nodes_per_element();
            idx_t *elements{nullptr};
            elements_to_device(ssm.n_elements(), nxe, ssm.element_data(), &elements);

            printf("create_device_elements %ld %d (ss)\n", ssm.n_elements(), nxe);

            return Buffer<idx_t>::own(
                    ssm.n_elements() * nxe, elements, d_buffer_destroy, MEMORY_SPACE_DEVICE);

        } else {
            auto c_mesh = (mesh_t *)space->mesh().impl_mesh();
            int nxe = elem_num_nodes(element_type);
            idx_t *elements{nullptr};
            elements_to_device(c_mesh->nelements, nxe, c_mesh->elements, &elements);

            printf("create_device_elements %ld %d\n", c_mesh->nelements, nxe);

            return Buffer<idx_t>::own(
                    c_mesh->nelements * nxe, elements, d_buffer_destroy, MEMORY_SPACE_DEVICE);
        }
    }

    class FFF {
    public:
        enum ElemType element_type_;
        ptrdiff_t n_elements_;
        std::shared_ptr<Buffer<idx_t>> elements_;
        void *fff_{nullptr};

        void init(mesh_t *c_mesh) {
            if (c_mesh->element_type == HEX8) {
                // printf("FFF/HEX8\n");
                cu_hex8_fff_allocate(c_mesh->nelements, &fff_);
                cu_hex8_fff_fill(c_mesh->nelements, c_mesh->elements, c_mesh->points, fff_);
            } else {
                // printf("FFF/TET4\n");
                cu_tet4_fff_allocate(c_mesh->nelements, &fff_);
                cu_tet4_fff_fill(c_mesh->nelements, c_mesh->elements, c_mesh->points, fff_);
            }
        }

        FFF(Mesh &mesh,
            const enum ElemType element_type,
            const std::shared_ptr<Buffer<idx_t>> &elements)
            : element_type_(element_type), n_elements_(mesh.n_elements()) {
            auto c_mesh = (mesh_t *)mesh.impl_mesh();
            elements_ = elements;

            init(c_mesh);
        }

        ~FFF() {
            d_buffer_destroy(fff_);
            // d_buffer_destroy(elements_);
        }

        enum ElemType element_type() const { return element_type_; }
        ptrdiff_t n_elements() const { return n_elements_; }
        idx_t *elements() const { return elements_->data(); }
        void *fff() const { return fff_; }
    };

    class Adjugate {
    public:
        enum ElemType element_type_;
        ptrdiff_t n_elements_;
        std::shared_ptr<Buffer<idx_t>> elements_;
        void *jacobian_adjugate_{nullptr};
        void *jacobian_determinant_{nullptr};

        void init(mesh_t *c_mesh) {
            cu_tet4_adjugate_allocate(n_elements_, &jacobian_adjugate_, &jacobian_determinant_);
            cu_tet4_adjugate_fill(n_elements_,
                                  c_mesh->elements,
                                  c_mesh->points,
                                  jacobian_adjugate_,
                                  jacobian_determinant_);
        }

        Adjugate(Mesh &mesh,
                 const enum ElemType element_type,
                 const std::shared_ptr<Buffer<idx_t>> &elements)
            : element_type_(element_type), n_elements_(mesh.n_elements()) {
            auto c_mesh = (mesh_t *)mesh.impl_mesh();
            init(c_mesh);
            elements_ = elements;
        }

        ~Adjugate() {
            d_buffer_destroy(jacobian_adjugate_);
            d_buffer_destroy(jacobian_determinant_);
        }

        enum ElemType element_type() const { return element_type_; }
        ptrdiff_t n_elements() const { return n_elements_; }
        idx_t *elements() const { return elements_->data(); }
        void *jacobian_determinant() const { return jacobian_determinant_; }
        void *jacobian_adjugate() const { return jacobian_adjugate_; }
    };

    class GPUDirichletConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace> space;
        int n_dirichlet_conditions{0};
        boundary_condition_t *dirichlet_conditions{nullptr};
        std::shared_ptr<DirichletConditions> h_dirichlet;

        GPUDirichletConditions(const std::shared_ptr<DirichletConditions> &dc)
            : space(dc->space()), h_dirichlet(dc) {
            n_dirichlet_conditions = dc->n_conditions();
            auto *h_dirichlet_conditions = (boundary_condition_t *)dc->impl_conditions();

            dirichlet_conditions = (boundary_condition_t *)malloc(n_dirichlet_conditions *
                                                                  sizeof(boundary_condition_t));

            for (int d = 0; d < n_dirichlet_conditions; d++) {
                boundary_conditions_host_to_device(&h_dirichlet_conditions[d],
                                                   &dirichlet_conditions[d]);
            }
        }

        int apply(real_t *const x) override {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                dirichlet_conditions[i].idx,
                                                space->block_size(),
                                                dirichlet_conditions[i].component,
                                                dirichlet_conditions[i].value,
                                                x);
            }

            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const g) override {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_gradient_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                         dirichlet_conditions[i].idx,
                                                         space->block_size(),
                                                         dirichlet_conditions[i].component,
                                                         dirichlet_conditions[i].value,
                                                         x,
                                                         g);
            }

            return SFEM_SUCCESS;

            // assert(false);
            // return SFEM_FAILURE;
        }

        int apply_value(const real_t value, real_t *const x) override {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                dirichlet_conditions[i].idx,
                                                space->block_size(),
                                                dirichlet_conditions[i].component,
                                                value,
                                                x);
            }

            return SFEM_SUCCESS;
        }

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) override {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_copy_vec(dirichlet_conditions[i].local_size,
                                            dirichlet_conditions[i].idx,
                                            space->block_size(),
                                            dirichlet_conditions[i].component,
                                            src,
                                            dest);
            }

            return SFEM_SUCCESS;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                cu_crs_constraint_nodes_to_identity_vec(dirichlet_conditions[i].local_size,
                                                        dirichlet_conditions[i].idx,
                                                        space->block_size(),
                                                        dirichlet_conditions[i].component,
                                                        1,
                                                        rowptr,
                                                        colidx,
                                                        values);
            }

            return SFEM_SUCCESS;
        }

        std::shared_ptr<Constraint> lor() const override {
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<sfem::FunctionSpace> &space,
                                             bool as_zeros) const override {
            auto h_derefined = std::static_pointer_cast<DirichletConditions>(
                    h_dirichlet->derefine(space, as_zeros));
            return std::make_shared<GPUDirichletConditions>(h_derefined);
        }

        ~GPUDirichletConditions() {
            d_destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
        }
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc) {
        return std::make_shared<GPUDirichletConditions>(dc);
    }

    // class GPUNeumannConditions final : public Op {
    // public:
    //     GPUNeumannConditions(const std::shared_ptr<NeumannConditions> &dc) {
    //         assert(false && "IMPLEMENT ME!");
    //     }
    // };

    class GPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<FFF> fff;
        enum RealType real_type { SFEM_REAL_DEFAULT };
        void *stream{SFEM_DEFAULT_STREAM};
        enum ElemType element_type { INVALID };

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(1 == space->block_size());
            return std::make_unique<GPULaplacian>(space);
        }

        const char *name() const override { return "GPULaplacian"; }
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

        GPULaplacian(const std::shared_ptr<FunctionSpace> &space)
            : space(space), element_type(space->element_type()) {}

        std::shared_ptr<Op> derefine_op(
                const std::shared_ptr<FunctionSpace> &derefined_space) override {
            auto mesh = (mesh_t *)derefined_space->mesh().impl_mesh();

            auto ret = std::make_shared<GPULaplacian>(derefined_space);
            assert(derefined_space->element_type() == macro_base_elem(fff->element_type()));
            assert(ret->element_type == macro_base_elem(fff->element_type()));
            ret->fff = fff;
            return ret;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            return cu_laplacian_crs(element_type,
                                    fff->n_elements(),
                                    fff->n_elements(),  // stride
                                    fff->elements(),
                                    fff->fff(),
                                    rowptr,
                                    colidx,
                                    real_type,
                                    values,
                                    stream);
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            return cu_laplacian_diag(element_type,
                                     fff->n_elements(),
                                     fff->n_elements(),  // stride
                                     fff->elements(),
                                     fff->fff(),
                                     real_type,
                                     values,
                                     stream);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return cu_laplacian_apply(element_type,
                                      fff->n_elements(),
                                      fff->n_elements(),  // stride
                                      fff->elements(),
                                      fff->fff(),
                                      real_type,
                                      x,
                                      out,
                                      stream);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return cu_laplacian_apply(element_type,
                                      fff->n_elements(),
                                      fff->n_elements(),  // stride
                                      fff->elements(),
                                      fff->fff(),
                                      real_type,
                                      h,
                                      out,
                                      stream);
        }

        int value(const real_t *x, real_t *const out) override {
            std::cerr << "Unimplemented function value in GPULaplacian\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class SemiStructuredGPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<FFF> fff;
        enum RealType real_type { SFEM_REAL_DEFAULT };
        void *stream{SFEM_DEFAULT_STREAM};
        enum ElemType element_type { INVALID };

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
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

        std::shared_ptr<Op> derefine_op(
                const std::shared_ptr<FunctionSpace> &derefined_space) override {
            auto mesh = (mesh_t *)derefined_space->mesh().impl_mesh();

            auto ret = std::make_shared<GPULaplacian>(derefined_space);
            assert(derefined_space->element_type() == macro_base_elem(fff->element_type()));
            assert(ret->element_type == macro_base_elem(fff->element_type()));
            ret->initialize();
            return ret;
        }

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            std::cerr << "Unimplemented function hessian_crs in GPULaplacian\n";
            assert(false);
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
        std::cerr << "Unimplemented function hessian_diag in GPULaplacian\n";
            assert(false);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            return cu_proteus_affine_hex8_laplacian_apply(ssm.level(),
                                                          fff->n_elements(),
                                                          fff->n_elements(),  // stride
                                                          ssm.interior_start(),
                                                          fff->elements(),
                                                          fff->fff(),
                                                          real_type,
                                                          x,
                                                          out,
                                                          stream);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            auto &ssm = space->semi_structured_mesh();
            return cu_proteus_affine_hex8_laplacian_apply(ssm.level(),
                                                          fff->n_elements(),
                                                          fff->n_elements(),  // stride
                                                          ssm.interior_start(),
                                                          fff->elements(),
                                                          fff->fff(),
                                                          real_type,
                                                          h,
                                                          out,
                                                          stream);
        }

        int value(const real_t *x, real_t *const out) override {
            std::cerr << "Unimplemented function value in GPULaplacian\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class GPULinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Adjugate> adjugate;
        enum RealType real_type { SFEM_REAL_DEFAULT };
        void *stream{SFEM_DEFAULT_STREAM};
        enum ElemType element_type { INVALID };
        real_t mu{1};
        real_t lambda{1};

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(mesh->spatial_dim == space->block_size());
            return std::make_unique<GPULinearElasticity>(space);
        }

        std::shared_ptr<Op> derefine_op(
                const std::shared_ptr<FunctionSpace> &derefined_space) override {
            auto mesh = (mesh_t *)derefined_space->mesh().impl_mesh();

            auto ret = std::make_shared<GPULinearElasticity>(derefined_space);
            assert(derefined_space->element_type() == macro_base_elem(adjugate->element_type()));
            assert(ret->element_type == macro_base_elem(adjugate->element_type()));
            ret->adjugate = adjugate;
            return ret;
        }

        const char *name() const override { return "GPULinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);
            mu = SFEM_SHEAR_MODULUS;
            lambda = SFEM_FIRST_LAME_PARAMETER;

            auto elements = space->device_elements();
            if (!elements) {
                elements = create_device_elements(space, space->element_type());
                space->set_device_elements(elements);
            }

            adjugate = std::make_shared<Adjugate>(space->mesh(), space->element_type(), elements);
            return SFEM_SUCCESS;
        }

        GPULinearElasticity(const std::shared_ptr<FunctionSpace> &space)
            : space(space), element_type(space->element_type()) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            std::cerr << "Unimplemented function hessian_crs in GPULinearElasticity\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            return cu_linear_elasticity_diag(element_type,
                                             adjugate->n_elements(),
                                             adjugate->n_elements(),
                                             adjugate->elements(),
                                             adjugate->jacobian_adjugate(),
                                             adjugate->jacobian_determinant(),
                                             mu,
                                             lambda,
                                             real_type,
                                             values,
                                             SFEM_DEFAULT_STREAM);
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return cu_linear_elasticity_apply(element_type,
                                              adjugate->n_elements(),
                                              adjugate->n_elements(),
                                              adjugate->elements(),
                                              adjugate->jacobian_adjugate(),
                                              adjugate->jacobian_determinant(),
                                              mu,
                                              lambda,
                                              real_type,
                                              x,
                                              out,
                                              SFEM_DEFAULT_STREAM);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return cu_linear_elasticity_apply(element_type,
                                              adjugate->n_elements(),
                                              adjugate->n_elements(),
                                              adjugate->elements(),
                                              adjugate->jacobian_adjugate(),
                                              adjugate->jacobian_determinant(),
                                              mu,
                                              lambda,
                                              real_type,
                                              h,
                                              out,
                                              SFEM_DEFAULT_STREAM);
        }

        int value(const real_t *x, real_t *const out) override {
            std::cerr << "Unimplemented function value in GPULinearElasticity\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    void register_device_ops() {
        Factory::register_op("gpu:LinearElasticity", &GPULinearElasticity::create);
        Factory::register_op("gpu:Laplacian", &GPULaplacian::create);
        Factory::register_op("ss:gpu:Laplacian", &SemiStructuredGPULaplacian::create);
    }

}  // namespace sfem
