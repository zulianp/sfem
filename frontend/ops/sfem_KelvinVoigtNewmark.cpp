#include "sfem_KelvinVoigtNewmark.hpp"

// C includes
#include "kelvin_voigt_newmark.h"

// C++ includes
#include "sfem_Tracer.hpp"

#include "hex8_jacobian.h"
// HAOYU

namespace sfem {

    class KelvinVoigtNewmark final : public Op {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<Buffer<real_t>> vel_[3];
        std::shared_ptr<Buffer<real_t>> acc_[3];
        enum ElemType                  element_type { INVALID };

        real_t k{2.0}, K{5/3}, eta{0.5}, dt{0.2}, gamma{0.5}, beta{0.25}, rho{1.0};

        long   calls{0};
        double total_time{0};

        class Jacobians {
        public:
            std::shared_ptr<Buffer<jacobian_t>> adjugate;
            std::shared_ptr<Buffer<jacobian_t>> determinant;

            Jacobians(const ptrdiff_t n_elements, const int size_adjugate)
                : adjugate(sfem::create_host_buffer<jacobian_t>(n_elements * size_adjugate)),
                  determinant(sfem::create_host_buffer<jacobian_t>(n_elements)) {}
        };

        std::shared_ptr<Jacobians> jacobians;

        static std::unique_ptr<Op> create(const std::shared_ptr<FunctionSpace> &space) {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::create");

            auto mesh = space->mesh_ptr();
            assert(mesh->spatial_dimension() == space->block_size());

            auto ret = std::make_unique<KelvinVoigtNewmark>(space);

            real_t SFEM_YOUNG_MODULUS        = 2.0;
            real_t SFEM_BULK_MODULUS        = 5/3;
            real_t SFEM_DAMPING_RATIO        = 0.5;
            real_t SFEM_DT                = 0.2;
            real_t SFEM_GAMMA            = 0.5;
            real_t SFEM_BETA            = 0.25;
            real_t SFEM_DENSITY        = 1.0;

            SFEM_READ_ENV(SFEM_YOUNG_MODULUS, atof);
            SFEM_READ_ENV(SFEM_BULK_MODULUS, atof);
            SFEM_READ_ENV(SFEM_DAMPING_RATIO, atof);
            SFEM_READ_ENV(SFEM_DT, atof);
            SFEM_READ_ENV(SFEM_GAMMA, atof);
            SFEM_READ_ENV(SFEM_BETA, atof);
            SFEM_READ_ENV(SFEM_DENSITY, atof);

            ret->k           = SFEM_YOUNG_MODULUS;
            ret->K           = SFEM_BULK_MODULUS;
            ret->eta         = SFEM_DAMPING_RATIO;
            ret->dt          = SFEM_DT;
            ret->gamma       = SFEM_GAMMA;
            ret->beta        = SFEM_BETA;
            ret->rho         = SFEM_DENSITY;
            ret->element_type = (enum ElemType)space->element_type();

            return ret;
        }

        std::shared_ptr<Op> lor_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::lor_op");

            auto ret          = std::make_shared<KelvinVoigtNewmark>(space);
            ret->element_type = macro_type_variant(element_type);
            ret->k           = k;
            ret->K           = K;
            ret->eta         = eta;   
            ret->dt          = dt;
            ret->gamma       = gamma;
            ret->beta        = beta;
            ret->rho         = rho;
            return ret;
        }

        std::shared_ptr<Op> derefine_op(const std::shared_ptr<FunctionSpace> &space) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::derefine_op");

            auto ret          = std::make_shared<KelvinVoigtNewmark>(space);
            ret->element_type = macro_base_elem(element_type);
            ret->k           = k;
            ret->K           = K;
            ret->eta         = eta;
            ret->dt          = dt;
            ret->gamma       = gamma;
            ret->beta        = beta;
            ret->rho         = rho;
            return ret;
        }

        const char *name() const override { return "KelvinVoigtNewmark"; }
        inline bool is_linear() const override { return true; }

        void set_field(const char* name, const std::shared_ptr<Buffer<real_t>>& vel, int component) override {
            if (strcmp(name, "velocity") == 0) {
                vel_[component] = vel;
            } else if (strcmp(name, "acceleration") == 0) {
                acc_[component] = vel;
            } else {
                SFEM_ERROR("Invalid field name! Call set_field(\"velocity\", buffer, 0/1/2) or set_field(\"acceleration\", buffer, 0/1/2) first.\n");
            }
        }

        int initialize(const std::vector<std::string> &block_names = {}) override {
            auto mesh = space->mesh_ptr();

            if (element_type == HEX8) {
                jacobians = std::make_shared<Jacobians>(mesh->n_elements(),
                                                        mesh->spatial_dimension() * elem_manifold_dim(element_type));

                hex8_adjugate_and_det_fill(mesh->n_elements(),
                                           mesh->elements()->data(),
                                           mesh->points()->data(),
                                           jacobians->adjugate->data(),
                                           jacobians->determinant->data());
            }

            return SFEM_SUCCESS;
        }

        KelvinVoigtNewmark(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

        ~KelvinVoigtNewmark() {
            if (calls) {
                printf("KelvinVoigtNewmark::apply called %ld times. Total: %g [s], "
                       "Avg: %g [s], TP %g [MDOF/s]\n",
                       calls,
                       total_time,
                       total_time / calls,
                       1e-6 * space->n_dofs() / (total_time / calls));
            }
        }

        int hessian_crs(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_crs");
            SFEM_ERROR("Called unimplemented method!\n");
            // TODO
            return SFEM_FAILURE;
        }

        int hessian_bsr(const real_t *const  x,
                        const count_t *const rowptr,
                        const idx_t *const   colidx,
                        real_t *const        values) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_bsr");
            SFEM_ERROR("Called unimplemented method!\n");

            // auto mesh = (mesh_t *)space->mesh().impl_mesh();

            // auto graph = space->node_to_node_graph();

            // linear_elasticity_bsr(element_type,
            //                       mesh->nelements,
            //                       mesh->nnodes,
            //                       mesh->elements,
            //                       mesh->points,
            //                       this->mu,
            //                       this->lambda,
            //                       graph->rowptr()->data(),
            //                       graph->colidx()->data(),
            //                       values);

            return SFEM_SUCCESS;
        }

        int hessian_bcrs_sym(const real_t *const  x,
                             const count_t *const rowptr,
                             const idx_t *const   colidx,
                             const ptrdiff_t      block_stride,
                             real_t **const       diag_values,
                             real_t **const       off_diag_values) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_bcrs_sym");
            SFEM_ERROR("Called unimplemented method!\n");
            // TODO
            return SFEM_FAILURE;
        }

        int hessian_block_diag_sym(const real_t *const x, real_t *const values) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_block_diag_sym");
            SFEM_ERROR("Called unimplemented method!\n");
            // TODO
            return SFEM_FAILURE;
        }

        int hessian_diag(const real_t *const, real_t *const out) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::hessian_diag");
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }


        int gradient(const real_t *const x, real_t *const out) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::gradient");
            // TODO
            auto mesh = space->mesh_ptr();
            const ptrdiff_t ndofs = mesh->n_nodes() * 3;
            const real_t *u = x;

            // AoS view: pass three component pointers with stride=3
            const real_t* vbase = vel_[0]->data();
            const real_t* vx = &vbase[0];
            const real_t* vy = &vbase[1];
            const real_t* vz = &vbase[2];

            const real_t* abase = acc_[0]->data();
            const real_t* ax = &abase[0];
            const real_t* ay = &abase[1];
            const real_t* az = &abase[2];

            double tick = MPI_Wtime();

            if (jacobians) {
                SFEM_TRACE_SCOPE("kelvin_voigt_newmark_gradient_aos");
                kelvin_voigt_newmark_gradient_aos(mesh->element_type(), mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), 
                                jacobians->adjugate->data(), jacobians->determinant->data(), this->k, this->K, this->eta, this->rho,
                             u, vx, vy, vz, ax, ay, az, out);
            } else {
                SFEM_ERROR("Jacobians not initialized for gradient!\n");
                return SFEM_FAILURE;
            }
        
            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;

            return SFEM_SUCCESS;
        }

        


        int apply(const real_t *const x, const real_t *const h, real_t *const out) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::apply");
            auto mesh = space->mesh_ptr();
            double tick = MPI_Wtime();

            if (jacobians) {
                SFEM_TRACE_SCOPE("kelvin_voigt_newmark_apply_adjugate_soa");
                kelvin_voigt_newmark_apply_adjugate_aos(mesh->element_type(),
                                           mesh->n_elements(),
                                           mesh->n_nodes(),
                                           mesh->elements()->data(),
                                           jacobians->adjugate->data(), jacobians->determinant->data(),
                                           this->dt, this->gamma, this->beta,
                                           this->k, this->K, this->eta, this->rho,
                                           h,
                                           out);
            } else {
                SFEM_ERROR("Jacobians not initialized for apply!\n");
                return SFEM_FAILURE;
            }

            double tock = MPI_Wtime();
            total_time += (tock - tick);
            calls++;

            return SFEM_SUCCESS;
        }

        int value(const real_t *x, real_t *const out) override {
            SFEM_TRACE_SCOPE("KelvinVoigtNewmark::value");
            SFEM_ERROR("Called unimplemented method!\n");
            return SFEM_FAILURE;
        }

        int report(const real_t *const) override { return SFEM_SUCCESS; }
    };

    std::unique_ptr<Op> create_kelvin_voigt_newmark(const std::shared_ptr<FunctionSpace> &space) {
        return KelvinVoigtNewmark::create(space);
    }


}  // namespace sfem
