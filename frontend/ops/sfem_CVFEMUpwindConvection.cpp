#include "sfem_CVFEMUpwindConvection.hpp"
#include "cvfem_operators.hpp"
#include "smesh_mesh.hpp"

#include "smesh_glob.hpp"

namespace sfem {

    std::unique_ptr<Op> CVFEMUpwindConvection::create(const std::shared_ptr<FunctionSpace> &space) {
        auto mesh = space->mesh_ptr();

        assert(1 == space->block_size());

        auto ret = std::make_unique<CVFEMUpwindConvection>(space);

        const char *SFEM_VELX = nullptr;
        const char *SFEM_VELY = nullptr;
        const char *SFEM_VELZ = nullptr;

        SFEM_READ_ENV(SFEM_VELX, );
        SFEM_READ_ENV(SFEM_VELY, );
        SFEM_READ_ENV(SFEM_VELZ, );

        if (!SFEM_VELX || !SFEM_VELY || (!SFEM_VELZ && space->mesh_ptr()->spatial_dimension() == 3)) {
            ret->element_type = (smesh::ElemType)space->element_type();
            return ret;
        }

        auto read_velocity = [](const char *const path) -> std::shared_ptr<Buffer<real_t>> {
            return path ? Buffer<real_t>::from_file(smesh::Path(path)) : nullptr;
        };

        auto vel0 = read_velocity(SFEM_VELX);
        auto vel1 = read_velocity(SFEM_VELY);
        auto vel2 = read_velocity(SFEM_VELZ);
        if (!vel0 || !vel1 || !vel2) {
            fprintf(stderr, "Unable to read input velocity\n");
            assert(0);
            return nullptr;
        }

        ret->vel[0] = vel0;
        ret->vel[1] = vel1;
        ret->vel[2] = vel2;

        return ret;
    }

    CVFEMUpwindConvection::CVFEMUpwindConvection(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    CVFEMUpwindConvection::~CVFEMUpwindConvection() {}

    void CVFEMUpwindConvection::set_field(const char * /* name  = velocity */,
                                          const std::shared_ptr<Buffer<real_t>> &v,
                                          const int                              component) {
        vel[component] = v;
    }

    int CVFEMUpwindConvection::hessian_crs(const real_t *const  x,
                                           const count_t *const rowptr,
                                           const idx_t *const   colidx,
                                           real_t *const        values) {
        // auto mesh = space->mesh_ptr();

        // auto graph = space->dof_to_dof_graph();

        // cvfem_convection_assemble_hessian(element_type,
        //                            mesh->nelements,
        //                            mesh->nnodes,
        //                            mesh->elements,
        //                            mesh->points,
        //                            graph->rowptr()->data(),
        //                            graph->colidx()->data(),
        //                            values);

        // return SFEM_SUCCESS;

        SFEM_ERROR("IMPLEMENT ME");
        return SFEM_FAILURE;
    }

    int CVFEMUpwindConvection::gradient(const real_t *const x, real_t *const out) { return apply(nullptr, x, out); }

    int CVFEMUpwindConvection::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        auto mesh = space->mesh_ptr();

        real_t *vel_[3] = {nullptr, nullptr, nullptr};

        for (int i = 0; i < 3; i++) {
            if (vel[i]) {
                vel_[i] = vel[i]->data();
            }
        }

        cvfem_convection_apply(element_type,
                               mesh->n_elements(),
                               mesh->n_nodes(),
                               mesh->elements(0)->data(),
                               mesh->points()->data(),
                               vel_,
                               h,
                               out);

        return SFEM_SUCCESS;
    }

    int CVFEMUpwindConvection::value(const real_t *x, real_t *const out) {
        // auto mesh = space->mesh_ptr();

        // cvfem_convection_assemble_value(element_type,
        //                          mesh->nelements,
        //                          mesh->nnodes,
        //                          mesh->elements,
        //                          mesh->points,
        //                          x,
        //                          out);

        // return SFEM_SUCCESS;

        SFEM_ERROR("IMPLEMENT ME");
        return SFEM_FAILURE;
    }

    int CVFEMUpwindConvection::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> CVFEMUpwindConvection::clone() const {
        auto ret = std::make_shared<CVFEMUpwindConvection>(space);
        *ret     = *this;
        return ret;
    }

    int CVFEMUpwindConvection::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

}  // namespace sfem
