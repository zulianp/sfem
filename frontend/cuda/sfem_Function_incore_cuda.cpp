#include "sfem_Function_incore_cuda.hpp"
#include <memory>
#include "boundary_condition.h"
#include "isolver_function.h"

#include "linear_elasticity_incore_cuda.h"
#include "laplacian_incore_cuda.h"

#include "sfem_mesh.h"

#include "boundary_condition_incore_cuda.h"

namespace sfem {

    class GPUDirichletConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace> space;
        int n_dirichlet_conditions{0};
        boundary_condition_t *dirichlet_conditions{nullptr};

        GPUDirichletConditions(const std::shared_ptr<DirichletConditions> &dc) 
        : space(dc->space())
        {
            n_dirichlet_conditions = dc->n_conditions();
            auto *h_dirichlet_conditions = (boundary_condition_t *)dc->impl_conditions();

            dirichlet_conditions = (boundary_condition_t *)malloc(n_dirichlet_conditions *
                                                                  sizeof(boundary_condition_t));

            for (int d = 0; d < n_dirichlet_conditions; d++) {
                boundary_conditions_host_to_device(&h_dirichlet_conditions[d],
                                                   &dirichlet_conditions[d]);
            }
        }

        int apply(isolver_scalar_t *const x) {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                dirichlet_conditions[i].idx,
                                                space->block_size(),
                                                dirichlet_conditions[i].component,
                                                dirichlet_conditions[i].value,
                                                x);
            }

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const g) {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_gradient_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                         dirichlet_conditions[i].idx,
                                                         space->block_size(),
                                                         dirichlet_conditions[i].component,
                                                         dirichlet_conditions[i].value,
                                                         x,
                                                         g);
            }

            return ISOLVER_FUNCTION_SUCCESS;

            // assert(false);
            // return ISOLVER_FUNCTION_FAILURE;
        }

        int apply_value(const isolver_scalar_t value, isolver_scalar_t *const x) {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_to_value_vec(dirichlet_conditions[i].local_size,
                                                dirichlet_conditions[i].idx,
                                                space->block_size(),
                                                dirichlet_conditions[i].component,
                                                value,
                                                x);
            }

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int copy_constrained_dofs(const isolver_scalar_t *const src, isolver_scalar_t *const dest) {
            for (int i = 0; i < n_dirichlet_conditions; i++) {
                d_constraint_nodes_copy_vec(dirichlet_conditions[i].local_size,
                                            dirichlet_conditions[i].idx,
                                            space->block_size(),
                                            dirichlet_conditions[i].component,
                                            src,
                                            dest);
            }

            return ISOLVER_FUNCTION_SUCCESS;
        }

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) {
            // for (int i = 0; i < n_dirichlet_conditions; i++) {
            //     d_crs_constraint_nodes_to_identity_vec(dirichlet_conditions[i].local_size,
            //                                            dirichlet_conditions[i].idx,
            //                                            space->block_size(),
            //                                            dirichlet_conditions[i].component,
            //                                            1,
            //                                            rowptr,
            //                                            colidx,
            //                                            values);
            // }

            // return ISOLVER_FUNCTION_SUCCESS;

            assert(false);
            return ISOLVER_FUNCTION_FAILURE;
        }

        ~GPUDirichletConditions() {
            d_destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
        }
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc)
    {
        return std::make_shared<GPUDirichletConditions>(dc);
    }

    class GPUNeumannConditions final : public Op {
    public:
        GPUNeumannConditions(const std::shared_ptr<NeumannConditions> &dc) {
            assert(false && "IMPLEMENT ME!");
        }
    };

    class GPULaplacian final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        cuda_incore_laplacian_t ctx;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(1 == space->block_size());
            return std::make_unique<GPULaplacian>(space);
        }

        const char *name() const override { return "GPULaplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            cuda_incore_laplacian_init((enum ElemType)mesh->element_type,
                                               &ctx,
                                               mesh->nelements,
                                               mesh->elements,
                                               mesh->points);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        GPULaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            cuda_incore_laplacian_apply(&ctx, x, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            cuda_incore_laplacian_apply(&ctx, h, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            assert(0);
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    class GPULinearElasticity final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        cuda_incore_linear_elasticity_t ctx;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(mesh->spatial_dim == space->block_size());
            return std::make_unique<GPULinearElasticity>(space);
        }

        const char *name() const override { return "GPULinearElasticity"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();

            real_t SFEM_SHEAR_MODULUS = 1;
            real_t SFEM_FIRST_LAME_PARAMETER = 1;

            SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
            SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

            cuda_incore_linear_elasticity_init((enum ElemType)mesh->element_type,
                                               &ctx,
                                               SFEM_SHEAR_MODULUS,
                                               SFEM_FIRST_LAME_PARAMETER,
                                               mesh->nelements,
                                               mesh->elements,
                                               mesh->points);

            return ISOLVER_FUNCTION_SUCCESS;
        }

        GPULinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const isolver_scalar_t *const x,
                        const isolver_idx_t *const rowptr,
                        const isolver_idx_t *const colidx,
                        isolver_scalar_t *const values) override {
            return ISOLVER_FUNCTION_FAILURE;
        }

        int hessian_diag(
            const isolver_scalar_t *const /*x*/, 
            isolver_scalar_t *const values) override
        {
            cuda_incore_linear_elasticity_diag(&ctx, values);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int gradient(const isolver_scalar_t *const x, isolver_scalar_t *const out) override {
            cuda_incore_linear_elasticity_apply(&ctx, x, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int apply(const isolver_scalar_t *const x,
                  const isolver_scalar_t *const h,
                  isolver_scalar_t *const out) override {
            cuda_incore_linear_elasticity_apply(&ctx, h, out);
            return ISOLVER_FUNCTION_SUCCESS;
        }

        int value(const isolver_scalar_t *x, isolver_scalar_t *const out) override {
            return ISOLVER_FUNCTION_FAILURE;
        }

        int report(const isolver_scalar_t *const) override { return ISOLVER_FUNCTION_SUCCESS; }
        ExecutionSpace execution_space() const override { return EXECUTION_SPACE_DEVICE; }
    };

    void register_device_ops() {
        Factory::register_op("gpu:LinearElasticity", &GPULinearElasticity::create);
        Factory::register_op("gpu:Laplacian", &GPULaplacian::create);
    }

}  // namespace sfem
