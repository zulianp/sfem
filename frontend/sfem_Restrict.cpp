#include "sfem_Restrict.hpp"

#include "sfem_API.hpp"
#include "sfem_FunctionSpace.hpp"
#include "sfem_SemiStructuredMesh.hpp"
#include "sfem_Tracer.hpp"
#include "sshex8.h"
#include "ssquad4_interpolate.h"

#ifdef SFEM_ENABLE_CUDA
#include "cu_sshex8_interpolate.h"
#include "cu_ssquad4_interpolate.h"
#endif

namespace sfem {

    template <typename T>
    class Restrict<T>::Impl {
    public:
        std::shared_ptr<FunctionSpace> from_space;
        std::shared_ptr<FunctionSpace> to_space;
        ExecutionSpace                 es;
        SharedBuffer<uint16_t>         element_to_node_incidence_count;
        int                            block_size;

        std::shared_ptr<Operator<T>> actual_op;

        void init() {
            auto from_element = (enum ElemType)from_space->element_type();
            auto to_element   = (enum ElemType)to_space->element_type();

            ptrdiff_t nnodes   = 0;
            idx_t**   elements = nullptr;
            int       nxe;
            if (from_space->has_semi_structured_mesh()) {
                auto& ssmesh = from_space->semi_structured_mesh();
                nxe          = sshex8_nxe(ssmesh.level());
                elements     = ssmesh.element_data();
                nnodes       = ssmesh.n_nodes();
            } else {
                nxe      = elem_num_nodes(from_element);
                elements = from_space->mesh().elements()->data();
                nnodes   = from_space->mesh().n_nodes();
            }

            element_to_node_incidence_count = create_buffer<uint16_t>(nnodes, MEMORY_SPACE_HOST);
            {
                auto buff = element_to_node_incidence_count->data();

                // #pragma omp parallel for // BAD performance with parallel for
                const ptrdiff_t nelements = from_space->mesh().n_elements();
                for (int d = 0; d < nxe; d++) {
                    for (ptrdiff_t i = 0; i < nelements; ++i) {
                        // #pragma omp atomic update
                        buff[elements[d][i]]++;
                    }
                }
            }

#ifdef SFEM_ENABLE_CUDA
            if (EXECUTION_SPACE_DEVICE == es) {
                auto dbuff = to_device(element_to_node_incidence_count);

                auto elements = from_space->device_elements();
                if (!elements) {
                    elements = create_device_elements(from_space, from_space->element_type());
                    from_space->set_device_elements(elements);
                }

                if (from_space->has_semi_structured_mesh()) {
                    if (to_space->has_semi_structured_mesh()) {
                        // FIXME make sure to reuse fine level elements and strides
                        auto to_elements = to_space->device_elements();
                        if (!to_elements) {
                            to_elements = create_device_elements(to_space, to_space->element_type());
                            to_space->set_device_elements(to_elements);
                        }

                        actual_op = make_op<real_t>(
                                to_space->n_dofs(),
                                from_space->n_dofs(),
                                [=](const real_t* const from, real_t* const to) {
                                    SFEM_TRACE_SCOPE("cu_sshex8_restrict");

                                    auto& from_ssm = from_space->semi_structured_mesh();
                                    auto& to_ssm   = to_space->semi_structured_mesh();

                                    cu_sshex8_restrict(from_ssm.n_elements(),
                                                       from_ssm.level(),
                                                       1,
                                                       elements->data(),
                                                       dbuff->data(),
                                                       to_ssm.level(),
                                                       1,
                                                       to_elements->data(),
                                                       block_size,
                                                       SFEM_REAL_DEFAULT,
                                                       1,
                                                       from,
                                                       SFEM_REAL_DEFAULT,
                                                       1,
                                                       to,
                                                       SFEM_DEFAULT_STREAM);
                                },
                                es);
                        return;

                    } else {
                        actual_op = make_op<real_t>(
                                to_space->n_dofs(),
                                from_space->n_dofs(),
                                [=](const real_t* const from, real_t* const to) {
                                    SFEM_TRACE_SCOPE("cu_sshex8_hierarchical_restriction");

                                    auto& ssm = from_space->semi_structured_mesh();
                                    cu_sshex8_hierarchical_restriction(ssm.level(),
                                                                       ssm.n_elements(),
                                                                       elements->data(),
                                                                       dbuff->data(),
                                                                       block_size,
                                                                       SFEM_REAL_DEFAULT,
                                                                       1,
                                                                       from,
                                                                       SFEM_REAL_DEFAULT,
                                                                       1,
                                                                       to,
                                                                       SFEM_DEFAULT_STREAM);
                                },
                                es);
                        return;
                    }

                } else {
                    actual_op = make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t* const from, real_t* const to) {
                                SFEM_TRACE_SCOPE("cu_macrotet4_to_tet4_restriction_element_based");

                                cu_macrotet4_to_tet4_restriction_element_based(from_space->mesh().n_elements(),
                                                                               elements->data(),
                                                                               dbuff->data(),
                                                                               block_size,
                                                                               SFEM_REAL_DEFAULT,
                                                                               1,
                                                                               from,
                                                                               SFEM_REAL_DEFAULT,
                                                                               1,
                                                                               to,
                                                                               SFEM_DEFAULT_STREAM);
                            },
                            es);
                    return;
                }
            } else
#endif
            {
                if (from_space->has_semi_structured_mesh()) {
                    if (!to_space->has_semi_structured_mesh()) {
                        actual_op = make_op<real_t>(
                                to_space->n_dofs(),
                                from_space->n_dofs(),
                                [=](const real_t* const from, real_t* const to) {
                                    SFEM_TRACE_SCOPE("sshex8_hierarchical_restriction");

                                    auto& ssm = from_space->semi_structured_mesh();
                                    sshex8_hierarchical_restriction(ssm.level(),
                                                                    ssm.n_elements(),
                                                                    ssm.element_data(),
                                                                    element_to_node_incidence_count->data(),
                                                                    block_size,
                                                                    from,
                                                                    to);
                                },
                                EXECUTION_SPACE_HOST);
                        return;
                    } else {
                        actual_op = make_op<real_t>(
                                to_space->n_dofs(),
                                from_space->n_dofs(),
                                [=](const real_t* const from, real_t* const to) {
                                    SFEM_TRACE_SCOPE("sshex8_restrict");

                                    auto& from_ssm = from_space->semi_structured_mesh();
                                    auto& to_ssm   = to_space->semi_structured_mesh();

                                    sshex8_restrict(from_ssm.n_elements(),    // nelements,
                                                    from_ssm.level(),         // from_level
                                                    1,                        // from_level_stride
                                                    from_ssm.element_data(),  // from_elements
                                                    element_to_node_incidence_count->data(),
                                                    to_ssm.level(),         // to_level
                                                    1,                      // to_level_stride
                                                    to_ssm.element_data(),  // to_elements
                                                    block_size,             // vec_size
                                                    from,
                                                    to);
                                },
                                EXECUTION_SPACE_HOST);
                        return;
                    }
                } else {
                    actual_op = make_op<real_t>(
                            to_space->n_dofs(),
                            from_space->n_dofs(),
                            [=](const real_t* const from, real_t* const to) {
                                SFEM_TRACE_SCOPE("hierarchical_restriction_with_counting");

                                hierarchical_restriction_with_counting(from_element,
                                                                       to_element,
                                                                       from_space->mesh().n_elements(),
                                                                       from_space->mesh().elements()->data(),
                                                                       element_to_node_incidence_count->data(),
                                                                       block_size,
                                                                       from,
                                                                       to);
                            },
                            EXECUTION_SPACE_HOST);
                    return;
                }
            }
        }

        Impl(const std::shared_ptr<FunctionSpace>& from,
             const std::shared_ptr<FunctionSpace>& to,
             const ExecutionSpace                  es,
             const int                             block_size)
            : from_space(from), to_space(to), es(es), block_size(block_size) {
            init();
        }

        int apply(const T* const x, T* const y) { return actual_op->apply(x, y); }
    };

    template <typename T>
    Restrict<T>::Restrict(const std::shared_ptr<FunctionSpace>& from,
                          const std::shared_ptr<FunctionSpace>& to,
                          const ExecutionSpace                  es)
        : impl_(std::make_unique<Impl>(from, to, es, from->block_size())) {}

    template <typename T>
    Restrict<T>::Restrict(const std::shared_ptr<FunctionSpace>& from,
                          const std::shared_ptr<FunctionSpace>& to,
                          const ExecutionSpace                  es,
                          const int                             block_size)
        : impl_(std::make_unique<Impl>(from, to, es, block_size)) {}

    template <typename T>
    std::shared_ptr<Restrict<T>> Restrict<T>::create(const std::shared_ptr<FunctionSpace>& from,
                                                     const std::shared_ptr<FunctionSpace>& to,
                                                     const ExecutionSpace                  es,
                                                     const int                             block_size) {
        return std::make_shared<Restrict<T>>(from, to, es, block_size);
    }

    template <typename T>
    Restrict<T>::~Restrict() = default;

    template <typename T>
    int Restrict<T>::apply(const T* const x, T* const y) {
        return impl_->apply(x, y);
    }

    template <typename T>
    ptrdiff_t Restrict<T>::rows() const {
        return impl_->from_space->n_dofs();
    }

    template <typename T>
    ptrdiff_t Restrict<T>::cols() const {
        return impl_->to_space->n_dofs();
    }

    template <typename T>
    ExecutionSpace Restrict<T>::execution_space() const {
        return impl_->es;
    }

    template <typename T>
    const SharedBuffer<uint16_t>& Restrict<T>::element_to_node_incidence_count() const {
        return impl_->element_to_node_incidence_count;
    }

    template <typename T>
    class SurfaceRestrict<T>::Impl {
    public:
        int                    from_level;
        enum ElemType          from_elem_type;
        ptrdiff_t              from_n_nodes;
        SharedBuffer<idx_t*>   from_sides;
        SharedBuffer<uint16_t> from_count;

        int                  to_level;
        enum ElemType        to_elem_type;
        ptrdiff_t            to_n_nodes;
        SharedBuffer<idx_t*> to_sides;

        ExecutionSpace es;
        int            block_size;

        std::shared_ptr<Operator<T>> actual_op;

        Impl(const int                     from_level,
             const enum ElemType           from_elem_type,
             const ptrdiff_t               from_n_nodes,
             const SharedBuffer<idx_t*>&   from_sides,
             const SharedBuffer<uint16_t>& from_count,
             const int                     to_level,
             const enum ElemType           to_elem_type,
             const ptrdiff_t               to_n_nodes,
             const SharedBuffer<idx_t*>&   to_sides,
             const ExecutionSpace          es,
             const int                     block_size)
            : from_level(from_level),
              from_elem_type(from_elem_type),
              from_n_nodes(from_n_nodes),
              from_sides(from_sides),
              from_count(from_count),
              to_level(to_level),
              to_elem_type(to_elem_type),
              to_n_nodes(to_n_nodes),
              to_sides(to_sides),
              es(es),
              block_size(block_size) {}

        int apply(const T* const x, T* const y) {
#ifdef SFEM_ENABLE_CUDA
            if (es == EXECUTION_SPACE_DEVICE) {
                cu_ssquad4_restrict(from_sides->extent(1),
                                    from_level,
                                    1,
                                    from_sides->data(),
                                    from_count->data(),
                                    to_level,
                                    1,
                                    to_sides->data(),
                                    block_size,
                                    SFEM_REAL_DEFAULT,
                                    1,
                                    x,
                                    SFEM_REAL_DEFAULT,
                                    1,
                                    y,
                                    SFEM_DEFAULT_STREAM);
                return SFEM_SUCCESS;
            }
#endif
            ssquad4_restrict(from_sides->extent(1),
                             from_level,
                             1,
                             from_sides->data(),
                             from_count->data(),
                             to_level,
                             1,
                             to_sides->data(),
                             block_size,
                             x,
                             y);

            return SFEM_SUCCESS;
        }
    };

    template <typename T>
    SurfaceRestrict<T>::SurfaceRestrict(const int                     from_level,
                                        const enum ElemType           from_elem_type,
                                        const ptrdiff_t               from_n_nodes,
                                        const SharedBuffer<idx_t*>&   from_sides,
                                        const SharedBuffer<uint16_t>& from_count,
                                        const int                     to_level,
                                        const enum ElemType           to_elem_type,
                                        const ptrdiff_t               to_n_nodes,
                                        const SharedBuffer<idx_t*>&   to_sides,
                                        const ExecutionSpace          es,
                                        const int                     block_size)
        : impl_(std::make_unique<Impl>(from_level,
                                       from_elem_type,
                                       from_n_nodes,
                                       from_sides,
                                       from_count,
                                       to_level,
                                       to_elem_type,
                                       to_n_nodes,
                                       to_sides,
                                       es,
                                       block_size)) {}

    template <typename T>
    std::shared_ptr<SurfaceRestrict<T>> SurfaceRestrict<T>::create(const int                     from_level,
                                                                   const enum ElemType           from_elem_type,
                                                                   const ptrdiff_t               from_n_nodes,
                                                                   const SharedBuffer<idx_t*>&   from_sides,
                                                                   const SharedBuffer<uint16_t>& from_count,
                                                                   const int                     to_level,
                                                                   const enum ElemType           to_elem_type,
                                                                   const ptrdiff_t               to_n_nodes,
                                                                   const SharedBuffer<idx_t*>&   to_sides,
                                                                   const ExecutionSpace          es,
                                                                   const int                     block_size) {
        return std::make_shared<SurfaceRestrict<T>>(from_level,
                                                    from_elem_type,
                                                    from_n_nodes,
                                                    from_sides,
                                                    from_count,
                                                    to_level,
                                                    to_elem_type,
                                                    to_n_nodes,
                                                    to_sides,
                                                    es,
                                                    block_size);
    }

    template <typename T>
    SurfaceRestrict<T>::~SurfaceRestrict() = default;

    template <typename T>
    int SurfaceRestrict<T>::apply(const T* const x, T* const y) {
        return impl_->apply(x, y);
    }

    template <typename T>
    ptrdiff_t SurfaceRestrict<T>::rows() const {
        return impl_->to_n_nodes * impl_->block_size;
    }

    template <typename T>
    ptrdiff_t SurfaceRestrict<T>::cols() const {
        return impl_->from_n_nodes * impl_->block_size;
    }

    template <typename T>
    ExecutionSpace SurfaceRestrict<T>::execution_space() const {
        return impl_->es;
    }

    template class Restrict<real_t>;
    template class SurfaceRestrict<real_t>;

}  // namespace sfem
