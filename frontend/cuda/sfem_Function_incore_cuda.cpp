#include "sfem_Function_incore_cuda.hpp"
#include <memory>
#include "boundary_condition.h"


#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "cu_laplacian.h"
#include "cu_linear_elasticity.h"
#include "cu_boundary_condition.h"
#include "cu_tet4_fff.h"

namespace sfem {

    class FFF {
    public:
        enum ElemType element_type_;
        ptrdiff_t n_elements_;
        idx_t *elements_;
        void *fff_;

        FFF(Mesh &mesh, const enum ElemType element_type)
            : element_type_(element_type), n_elements_(mesh.n_elements()) {
            auto c_mesh = (mesh_t *)mesh.impl_mesh();

            // FIXME Now harcoded for tets
            cu_tet4_fff_allocate(n_elements_, &fff_);
            cu_tet4_fff_fill(n_elements_, c_mesh->elements, c_mesh->points, fff_);

            elements_to_device(
                    n_elements_, elem_num_nodes(element_type), c_mesh->elements, &elements_);
        }

        ~FFF() {
            d_buffer_destroy(fff_);
            d_buffer_destroy(elements_);
        }

        enum ElemType element_type() const { return element_type_; }
        ptrdiff_t n_elements() const { return n_elements_; }
        idx_t *elements() const { return elements_; }
        void *fff() const { return fff_; }
    };

    class GPUDirichletConditions final : public Constraint {
    public:
        std::shared_ptr<FunctionSpace> space;
        int n_dirichlet_conditions{0};
        boundary_condition_t *dirichlet_conditions{nullptr};

        std::shared_ptr<Constraint> derefine() const {
            assert(false);
            return nullptr;
        }

        GPUDirichletConditions(const std::shared_ptr<DirichletConditions> &dc)
            : space(dc->space()) {
            n_dirichlet_conditions = dc->n_conditions();
            auto *h_dirichlet_conditions = (boundary_condition_t *)dc->impl_conditions();

            dirichlet_conditions = (boundary_condition_t *)malloc(n_dirichlet_conditions *
                                                                  sizeof(boundary_condition_t));

            for (int d = 0; d < n_dirichlet_conditions; d++) {
                boundary_conditions_host_to_device(&h_dirichlet_conditions[d],
                                                   &dirichlet_conditions[d]);
            }
        }

        int apply(real_t *const x) {
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

        int gradient(const real_t *const x, real_t *const g) {
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

        int apply_value(const real_t value, real_t *const x) {
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

        int copy_constrained_dofs(const real_t *const src, real_t *const dest) {
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
                        real_t *const values) {
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

            // return SFEM_SUCCESS;

            assert(false);
            return SFEM_FAILURE;
        }

        std::shared_ptr<Constraint> lor() const override {
            assert(false);
            return nullptr;
        }

        std::shared_ptr<Constraint> derefine(const std::shared_ptr<sfem::FunctionSpace> &,
                                             bool) const override {
            assert(false);
            return nullptr;
        }

        ~GPUDirichletConditions() {
            d_destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
        }
    };

    std::shared_ptr<Constraint> to_device(const std::shared_ptr<DirichletConditions> &dc) {
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
        std::shared_ptr<FFF> fff;
        enum RealType real_type { SFEM_REAL_DEFAULT };

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            auto mesh = (mesh_t *)space->mesh().impl_mesh();
            assert(1 == space->block_size());
            return std::make_unique<GPULaplacian>(space);
        }

        const char *name() const override { return "GPULaplacian"; }
        inline bool is_linear() const override { return true; }

        int initialize() override {
            fff = std::make_shared<FFF>(space->mesh(), space->element_type());
            return SFEM_SUCCESS;
        }

        GPULaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            std::cerr << "Unimplemented function hessian_crs in GPULaplacian\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            return cu_laplacian_apply(fff->element_type(),
                                      fff->n_elements(),
                                      fff->elements(),
                                      fff->fff(),
                                      real_type,
                                      x,
                                      out,
                                      SFEM_DEFAULT_STREAM);
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            return cu_laplacian_apply(fff->element_type(),
                                      fff->n_elements(),
                                      fff->elements(),
                                      fff->fff(),
                                      real_type,
                                      h,
                                      out,
                                      SFEM_DEFAULT_STREAM);
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

            cuda_incore_linear_elasticity_init((enum ElemType)space->element_type(),
                                               &ctx,
                                               SFEM_SHEAR_MODULUS,
                                               SFEM_FIRST_LAME_PARAMETER,
                                               mesh->nelements,
                                               mesh->elements,
                                               mesh->points);

            return SFEM_SUCCESS;
        }

        GPULinearElasticity(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        int hessian_crs(const real_t *const x,
                        const count_t *const rowptr,
                        const idx_t *const colidx,
                        real_t *const values) override {
            std::cerr << "Unimplemented function hessian_crs in GPULinearElasticity\n";
            assert(0);
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const /*x*/, real_t *const values) override {
            cuda_incore_linear_elasticity_diag(&ctx, values);
            return SFEM_SUCCESS;
        }

        int gradient(const real_t *const x, real_t *const out) override {
            cuda_incore_linear_elasticity_apply(&ctx, x, out);
            return SFEM_SUCCESS;
        }

        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            cuda_incore_linear_elasticity_apply(&ctx, h, out);
            return SFEM_SUCCESS;
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
    }

}  // namespace sfem
