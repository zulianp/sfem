#ifndef SFEM_SELF_CONTACT_HPP
#define SFEM_SELF_CONTACT_HPP

#include "sfem_FunctionSpace.hpp"
#include "smesh_crs_graph.hpp"

#include <memory>

#ifdef SFEM_ENABLE_YAML
#include "smesh_forward_declarations.hpp"
#endif

namespace sfem {

    class Contact {
    public:
        virtual ~Contact() = default;

        virtual void recompute(const std::shared_ptr<Buffer<real_t>>& displacement) = 0;

        virtual const std::shared_ptr<smesh::CRSGraph<count_t, idx_t>>& graph() const           = 0;
        virtual smesh::SharedBuffer<real_t>&                            values()                = 0;
        virtual smesh::SharedBuffer<real_t>&                            mass_vector()           = 0;
        virtual smesh::SharedBuffer<real_t*>&                           normals()               = 0;
        virtual smesh::SharedBuffer<real_t>&                            distances()             = 0;
        virtual smesh::SharedBuffer<real_t>&                            frozen_displacement()   = 0;
        virtual const smesh::SharedBuffer<real_t>&                      distances_whole() const = 0;
        virtual const smesh::SharedBuffer<real_t>&                      directors() const       = 0;
    };

    std::shared_ptr<Contact> create_contact(const std::shared_ptr<FunctionSpace>& space,
                                            const std::shared_ptr<smesh::Mesh>&   surface,
                                            real_t                                margin,
                                            real_t                                search_radius_sqr,
                                            ExecutionSpace                        es);

#ifdef SFEM_ENABLE_YAML
    std::shared_ptr<Contact> create_contact(const std::shared_ptr<FunctionSpace>& space,
                                            const std::shared_ptr<smesh::Mesh>&   surface,
                                            const ryml::ConstNodeRef&             node,
                                            ExecutionSpace                        es);
#endif

    std::shared_ptr<Contact> create_mulitbody_contact(const std::shared_ptr<FunctionSpace>& space,
                                                      const std::shared_ptr<smesh::Mesh>&   surface,
                                                      real_t                                margin,
                                                      real_t                                search_radius_sqr,
                                                      ExecutionSpace                        es);

}  // namespace sfem

#endif  // SFEM_SELF_CONTACT_HPP
