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
        virtual ~ContactSurface()                           = default;
        virtual SharedBuffer<geom_t *> points()             = 0;
        virtual SharedBuffer<idx_t *>  elements()           = 0;
        virtual SharedBuffer<idx_t>    node_mapping()       = 0;
        virtual enum ElemType          element_type() const = 0;

#ifdef SFEM_ENABLE_CUDA
        virtual SharedBuffer<geom_t *> points_device()       = 0;
        virtual SharedBuffer<idx_t *>  elements_device()     = 0;
        virtual SharedBuffer<idx_t>    node_mapping_device() = 0;
#endif

        virtual std::shared_ptr<Buffer<idx_t *>> semi_structured_elements() { return nullptr; }

        virtual void displace_points(const real_t *disp) = 0;
        virtual void reset_points()                      = 0;
    };

    class MeshContactSurface final : public ContactSurface {
    public:
        MeshContactSurface();
        ~MeshContactSurface();

        static std::unique_ptr<MeshContactSurface> create(const std::shared_ptr<FunctionSpace>        &space,
                                                          const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                          const enum ExecutionSpace                    es);

        static std::unique_ptr<MeshContactSurface> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                    const std::string                    &path,
                                                                    const enum ExecutionSpace             es);

        SharedBuffer<geom_t *> points() override;
        SharedBuffer<idx_t *>  elements() override;
        SharedBuffer<idx_t>    node_mapping() override;

#ifdef SFEM_ENABLE_CUDA
        SharedBuffer<geom_t *> points_device() override;
        SharedBuffer<idx_t *>  elements_device() override;
        SharedBuffer<idx_t>    node_mapping_device() override;
#endif
        enum ElemType          element_type() const override;

        void displace_points(const real_t *disp) override;
        void reset_points() override;

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    class SSMeshContactSurface final : public ContactSurface {
    public:
        SSMeshContactSurface();
        ~SSMeshContactSurface();

        SharedBuffer<geom_t *> points() override;
        SharedBuffer<idx_t *>  elements() override;
        SharedBuffer<idx_t>    node_mapping() override;

#ifdef SFEM_ENABLE_CUDA
        SharedBuffer<geom_t *> points_device() override;
        SharedBuffer<idx_t *>  elements_device() override;
        SharedBuffer<idx_t>    node_mapping_device() override;
#endif
        enum ElemType          element_type() const override;

        static std::unique_ptr<SSMeshContactSurface> create(const std::shared_ptr<FunctionSpace>        &space,
                                                            const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                            const enum ExecutionSpace                    es);

        static std::unique_ptr<SSMeshContactSurface> create_from_file(const std::shared_ptr<FunctionSpace> &space,
                                                                      const std::string                    &path);

        std::shared_ptr<Buffer<idx_t *>> semi_structured_elements() override;
        void                             displace_points(const real_t *disp) override;
        void                             reset_points() override;

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

    std::shared_ptr<ContactSurface> create_contact_surface(const std::shared_ptr<FunctionSpace>        &space,
                                                           const std::vector<std::shared_ptr<Sideset>> &sidesets,
                                                           const enum ExecutionSpace                    es);
}  // namespace sfem

#endif  // SFEM_CONTACT_SURFACE_HPP
