#include "sfem_SemiStructuredVectorLaplacian.hpp"

// C includes
#include "hex8_fff.hpp"
#include "sshex8_vector_laplacian.hpp"

// C++ includes
#include "sfem_FunctionSpace.hpp"
#include "sfem_LinearElasticity.hpp"
#include "smesh_mesh.hpp"
#include "sfem_SemiStructuredMesh.hpp"

#include "sfem_VectorLaplacian.hpp"
#include "sfem_glob.hpp"

namespace sfem {

    std::unique_ptr<Op> SemiStructuredVectorLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::create");

        assert(space->has_semi_structured_mesh());
        if (!space->has_semi_structured_mesh()) {
            fprintf(stderr,
                    "[Error] SemiStructuredVectorLaplacian::create requires space with "
                    "semi_structured_mesh!\n");
            return nullptr;
        }

        assert(is_semistructured_type(space->element_type()));  // REMOVEME once generalized approach
        auto ret = std::make_unique<SemiStructuredVectorLaplacian>(space);

        ret->element_type = (smesh::ElemType)space->element_type();

        // FIXME
        auto macro_mesh = space->has_semi_structured_mesh() ? sfem::semi_structured_derefine(space->mesh_ptr(), 1) : space->mesh_ptr();
        ret->fff = create_host_buffer<jacobian_t>(macro_mesh->n_elements() * 6);

        if (SFEM_SUCCESS != hex8_fff_fill(macro_mesh->n_elements(),
                                          macro_mesh->elements(0)->data(),
                                          macro_mesh->points()->data(),
                                          ret->fff->data())) {
            SFEM_ERROR("Unable to create fff");
        }

        return ret;
    }

    SemiStructuredVectorLaplacian::SemiStructuredVectorLaplacian(const std::shared_ptr<FunctionSpace> &space) : space(space) {}

    int SemiStructuredVectorLaplacian::hessian_crs(const real_t *const  x,
                                                   const count_t *const rowptr,
                                                   const idx_t *const   colidx,
                                                   real_t *const        values) {
        SFEM_ERROR("[Error] ss:Laplacian::hessian_crs NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredVectorLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_ERROR("[Error] ss:Laplacian::gradient NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredVectorLaplacian::apply(const real_t *const x, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::apply");

        assert(is_semistructured_type(element_type));  // REMOVEME once generalized approach

        auto &ssm = space->mesh();

        double tick = MPI_Wtime();

        const int             block_size = space->block_size();
        std::vector<real_t *> vec_in(block_size), vec_out(block_size);

        // AoS
        for (int d = 0; d < block_size; d++) {
            vec_in[d]  = const_cast<real_t *>(&h[d]);
            vec_out[d] = &out[d];
        }

        int err = affine_sshex8_vector_laplacian_apply_fff(sfem::semi_structured_level(ssm),
                                                           ssm.n_elements(),
                                                           sfem::semi_structured_element_data(ssm),
                                                           this->fff->data(),
                                                           block_size,
                                                           block_size,
                                                           vec_in.data(),
                                                           vec_out.data());

        double tock = MPI_Wtime();
        total_time += (tock - tick);
        calls++;
        return err;
    }

    int SemiStructuredVectorLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_ERROR("[Error] ss:Laplacian::value NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    int SemiStructuredVectorLaplacian::initialize(const std::vector<std::string> &block_names) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> SemiStructuredVectorLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("SemiStructuredVectorLaplacian::derefine_op");

        if (space->has_semi_structured_mesh()) {
            auto ret          = std::make_shared<SemiStructuredVectorLaplacian>(space);
            ret->element_type = element_type;
            return ret;
        } else {
            auto ret          = std::make_shared<VectorLaplacian>(space);
            ret->element_type = space->element_type();
            return ret;
        }
    }

    int SemiStructuredVectorLaplacian::hessian_diag(const real_t *const x, real_t *const values) {
        SFEM_ERROR("[Error] ss:Laplacian::hessian_diag NOT IMPLEMENTED!\n");
        return SFEM_FAILURE;
    }

    std::shared_ptr<Op> SemiStructuredVectorLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        fprintf(stderr, "[Error] ss:Laplacian::lor_op NOT IMPLEMENTED!\n");
        assert(false);
        return nullptr;
    }

    const char *SemiStructuredVectorLaplacian::name() const { return "ss:Laplacian"; }

    int SemiStructuredVectorLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    SemiStructuredVectorLaplacian::~SemiStructuredVectorLaplacian() {
        if (SFEM_PRINT_THROUGHPUT && calls) {
            printf("SemiStructuredVectorLaplacian[%d]::apply called %ld times. "
                   "Total: %g [s], "
                   "Avg: %g [s], TP %g [MDOF/s]\n",
                   sfem::semi_structured_level(space->mesh()),
                   calls,
                   total_time,
                   total_time / calls,
                   1e-6 * space->n_dofs() / (total_time / calls));
        }
    }

    std::shared_ptr<Op> SemiStructuredVectorLaplacian::clone() const {
        auto ret = std::make_shared<SemiStructuredVectorLaplacian>(space);
        *ret     = *this;
        return ret;
    }

}  // namespace sfem
