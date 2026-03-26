#include "sfem_VectorLaplacian.hpp"

#include "sfem_FunctionSpace.hpp"
#include "sfem_MultiDomainOp.hpp"
#include "sfem_OpTracer.hpp"
#include "sfem_defs.hpp"
#include "sfem_logger.hpp"
#include "smesh_glob.hpp"
#include "smesh_kernel_data.hpp"
#include "smesh_mesh.hpp"
#include "vector_laplacian.hpp"

namespace sfem {

    namespace {

        smesh::block_idx_t block_id_for_domain(const smesh::Mesh &mesh, const smesh::Mesh::Block &block) {
            for (size_t i = 0; i < mesh.n_blocks(); i++) {
                if (mesh.block(i).get() == &block) {
                    return static_cast<smesh::block_idx_t>(i);
                }
            }
            SFEM_ERROR("VectorLaplacian: mesh block pointer not found in mesh.blocks()");
            return 0;
        }

        int vector_laplacian_dispatch_apply(const OpDomain &domain,
                                            smesh::Mesh    &mesh,
                                            const int       block_size,
                                            real_t **const SFEM_RESTRICT vec_in,
                                            real_t **const SFEM_RESTRICT vec_out) {
            if (domain.user_data) {
                auto fff = std::static_pointer_cast<smesh::FFF>(domain.user_data);
                return vector_laplacian_apply_opt(domain.element_type,
                                                  domain.block->n_elements(),
                                                  domain.block->elements()->data(),
                                                  fff->fff_AoS()->data(),
                                                  block_size,
                                                  block_size,
                                                  vec_in,
                                                  vec_out);
            }

            return vector_laplacian_apply(domain.element_type,
                                            domain.block->n_elements(),
                                            mesh.n_nodes(),
                                            domain.block->elements()->data(),
                                            mesh.points()->data(),
                                            block_size,
                                            block_size,
                                            vec_in,
                                            vec_out);
        }

    }  // namespace

    class VectorLaplacian::Impl {
    public:
        std::shared_ptr<FunctionSpace> space;
        std::shared_ptr<MultiDomainOp> domains;
#if SFEM_PRINT_THROUGHPUT
        std::unique_ptr<OpTracer> op_profiler;
#endif

        explicit Impl(const std::shared_ptr<FunctionSpace> &sp) : space(sp) {
#if SFEM_PRINT_THROUGHPUT
            const std::string op_name =
                    std::string("VectorLaplacian[") + sfem::type_to_string(sp->element_type()) + "]::apply";
            op_profiler = std::make_unique<OpTracer>(space, op_name);
#endif
        }

        int iterate(const std::function<int(const OpDomain &)> &func) { return domains->iterate(func); }
    };

    inline ptrdiff_t VectorLaplacian::n_dofs_domain() const { return impl_->space->n_dofs(); }

    inline ptrdiff_t VectorLaplacian::n_dofs_image() const { return impl_->space->n_dofs(); }

    int VectorLaplacian::initialize(const std::vector<std::string> &block_names) {
        SFEM_TRACE_SCOPE("VectorLaplacian::initialize");
        impl_->domains = std::make_shared<MultiDomainOp>(impl_->space, block_names);

        auto mesh = impl_->space->mesh_ptr();

        int SFEM_VECTOR_LAPLACIAN_FFF = 1;
        SFEM_READ_ENV(SFEM_VECTOR_LAPLACIAN_FFF, atoi);

        for (auto &n2d : impl_->domains->domains()) {
            OpDomain &domain = n2d.second;

            const bool want_fff =
                    SFEM_VECTOR_LAPLACIAN_FFF || is_semistructured_type(domain.element_type);

            if (want_fff) {
                const smesh::block_idx_t block_id = block_id_for_domain(*mesh, *domain.block);
                auto                     fff      = smesh::FFF::create_AoS(mesh, smesh::MEMORY_SPACE_HOST, block_id);
                if (!fff) {
                    SFEM_ERROR("VectorLaplacian: smesh::FFF::create_AoS failed (block '%s')\n",
                               domain.block->name().c_str());
                    return SFEM_FAILURE;
                }
                domain.user_data = std::static_pointer_cast<void>(fff);
            } else {
                domain.user_data = nullptr;
            }
        }

        return SFEM_SUCCESS;
    }

    std::unique_ptr<Op> VectorLaplacian::create(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::create");

        assert(1 != space->block_size());
        return std::make_unique<VectorLaplacian>(space);
    }

    std::shared_ptr<Op> VectorLaplacian::lor_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::lor_op");

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type())) {
            SMESH_ERROR("VectorLaplacian::lor_op NOT IMPLEMENTED for semi-structured mesh!\n");
            return nullptr;
        }

        auto ret              = std::make_shared<VectorLaplacian>(space);
        ret->impl_->domains = impl_->domains->lor_op(space, {});
        return ret;
    }

    std::shared_ptr<Op> VectorLaplacian::derefine_op(const std::shared_ptr<FunctionSpace> &space) {
        SFEM_TRACE_SCOPE("VectorLaplacian::derefine_op");

        if (space->has_semi_structured_mesh() && is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<VectorLaplacian>(space);
            ret->initialize({});
            return ret;
        }

        if (impl_->space->has_semi_structured_mesh() && is_semistructured_type(impl_->space->element_type()) &&
            !is_semistructured_type(space->element_type())) {
            auto ret = std::make_shared<VectorLaplacian>(space);
            ret->initialize({});
            assert(space->n_blocks() == 1);
            ret->override_element_types({space->element_type()});
            return ret;
        }

        auto ret              = std::make_shared<VectorLaplacian>(space);
        ret->impl_->domains = impl_->domains->derefine_op(space, {});
        return ret;
    }

    VectorLaplacian::VectorLaplacian(const std::shared_ptr<FunctionSpace> &space)
        : impl_(std::make_unique<Impl>(space)) {}

    VectorLaplacian::~VectorLaplacian() = default;

    int VectorLaplacian::hessian_crs(const real_t *const  x,
                                     const count_t *const rowptr,
                                     const idx_t *const   colidx,
                                     real_t *const        values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_crs_sym(const real_t *const  x,
                                        const count_t *const rowptr,
                                        const idx_t *const   colidx,
                                        real_t *const        diag_values,
                                        real_t *const        off_diag_values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_crs_sym");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::hessian_diag(const real_t *const /*x*/, real_t *const values) {
        SFEM_TRACE_SCOPE("VectorLaplacian::hessian_diag");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::gradient(const real_t *const x, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::gradient");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::apply(const real_t *const /*x*/, const real_t *const h, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::apply");
        SFEM_OP_CAPTURE();

        const int             block_size = impl_->space->block_size();
        std::vector<real_t *> vec_in(block_size), vec_out(block_size);

        for (int d = 0; d < block_size; d++) {
            vec_in[d]  = const_cast<real_t *>(&h[d]);
            vec_out[d] = &out[d];
        }

        auto mesh = impl_->space->mesh_ptr();
        return impl_->iterate([&](const OpDomain &domain) {
            return vector_laplacian_dispatch_apply(domain, *mesh, block_size, vec_in.data(), vec_out.data());
        });
    }

    int VectorLaplacian::value(const real_t *x, real_t *const out) {
        SFEM_TRACE_SCOPE("VectorLaplacian::value");
        SFEM_ERROR("IMPLEMENT ME!\n");
        return SFEM_FAILURE;
    }

    int VectorLaplacian::report(const real_t *const) { return SFEM_SUCCESS; }

    std::shared_ptr<Op> VectorLaplacian::clone() const {
        auto ret              = std::make_shared<VectorLaplacian>(impl_->space);
        ret->impl_->domains = impl_->domains;
        return ret;
    }

    void VectorLaplacian::set_value_in_block(const std::string &block_name, const std::string &var_name, const real_t value) {
        impl_->domains->set_value_in_block(block_name, var_name, value);
    }

    void VectorLaplacian::override_element_types(const std::vector<smesh::ElemType> &element_types) {
        impl_->domains->override_element_types(element_types);
    }

}  // namespace sfem
