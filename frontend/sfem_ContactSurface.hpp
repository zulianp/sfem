#ifndef SFEM_CONTACT_SURFACE_HPP
#define SFEM_CONTACT_SURFACE_HPP

#include "sfem_defs.h"

#include "sfem_Buffer.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "sfem_Grid.hpp"

#include <mpi.h>
#include <memory>

namespace sfem {
    class ContactSurface {
    public:
        virtual ~ContactSurface()                                      = default;
        virtual std::shared_ptr<Buffer<geom_t *>> points()             = 0;
        virtual std::shared_ptr<Buffer<idx_t *>>  elements()           = 0;
        virtual std::shared_ptr<Buffer<idx_t>>    node_mapping()       = 0;
        virtual enum ElemType                     element_type() const = 0;
    };

    class MeshContactSurface final : public ContactSurface {
    public:
        MeshContactSurface();
        ~MeshContactSurface();

        static std::unique_ptr<MeshContactSurface> create(const std::shared_ptr<FunctionSpace> &space,
                                                          const std::shared_ptr<Sideset>       &sideset,
                                                          const enum ExecutionSpace             es);

        std::shared_ptr<Buffer<geom_t *>> points() override;
        std::shared_ptr<Buffer<idx_t *>>  elements() override;
        std::shared_ptr<Buffer<idx_t>>    node_mapping() override;
        enum ElemType                     element_type() const override;

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class SSMeshContactSurface final : public ContactSurface {
    public:
        SSMeshContactSurface();
        ~SSMeshContactSurface();

        std::shared_ptr<Buffer<geom_t *>> points() override;
        std::shared_ptr<Buffer<idx_t *>>  elements() override;
        std::shared_ptr<Buffer<idx_t>>    node_mapping() override;
        enum ElemType                     element_type() const override;

        static std::unique_ptr<SSMeshContactSurface> create(const std::shared_ptr<FunctionSpace> &space,
                                                            const std::shared_ptr<Sideset>       &sideset,
                                                            const enum ExecutionSpace             es);

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<ContactSurface> create_contact_surface(const std::shared_ptr<FunctionSpace> &space,
                                                           const std::shared_ptr<Sideset>       &sideset,
                                                           const enum ExecutionSpace             es);
}  // namespace sfem

#endif  // SFEM_CONTACT_SURFACE_HPP
