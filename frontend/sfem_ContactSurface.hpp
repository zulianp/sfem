#ifndef SFEM_CONTACT_SURFACE_HPP
#define SFEM_CONTACT_SURFACE_HPP

#include "sfem_defs.hpp"

#include "sfem_aliases.hpp"
#include "sfem_ForwardDeclarations.hpp"
#include "smesh_grid.hpp"

#include <mpi.h>
#include <memory>

namespace sfem {
    class ContactSurface {
    public:
        virtual ~ContactSurface()                           = default;
        virtual SharedBuffer<geom_t *> points()             = 0;
        virtual SharedBuffer<idx_t *>  elements()           = 0;
        virtual SharedBuffer<idx_t>    node_mapping()       = 0;
        virtual SharedBuffer<idx_t>    geometry_node_mapping() { return node_mapping(); }
        virtual smesh::ElemType          element_type() const = 0;

#ifdef SFEM_ENABLE_CUDA
        virtual SharedBuffer<geom_t *> points_device()       = 0;
        virtual SharedBuffer<idx_t *>  elements_device()     = 0;
        virtual SharedBuffer<idx_t>    node_mapping_device() = 0;
#endif

        /// The rest positions of points() (host), and the displacement displace_points last added, in
        /// real_t (3 components per point, AoS, zero beyond the mesh's dimension; zero after
        /// reset_points; in the execution space): points() rounds their sum to geom_t. A sampler
        /// that adds them keeps the displacement's precision.
        virtual SharedBuffer<geom_t *> rest_points()  = 0;
        virtual SharedBuffer<real_t>   displacement() = 0;
#ifdef SFEM_ENABLE_CUDA
        virtual SharedBuffer<geom_t *> rest_points_device() = 0;
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
        SharedBuffer<geom_t *> rest_points() override;
        SharedBuffer<real_t>   displacement() override;

#ifdef SFEM_ENABLE_CUDA
        SharedBuffer<geom_t *> points_device() override;
        SharedBuffer<geom_t *> rest_points_device() override;
        SharedBuffer<idx_t *>  elements_device() override;
        SharedBuffer<idx_t>    node_mapping_device() override;
#endif
        smesh::ElemType          element_type() const override;

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
        SharedBuffer<geom_t *> rest_points() override;
        SharedBuffer<real_t>   displacement() override;
        SharedBuffer<idx_t>    geometry_node_mapping() override;

#ifdef SFEM_ENABLE_CUDA
        SharedBuffer<geom_t *> points_device() override;
        SharedBuffer<geom_t *> rest_points_device() override;
        SharedBuffer<idx_t *>  elements_device() override;
        SharedBuffer<idx_t>    node_mapping_device() override;
#endif
        smesh::ElemType          element_type() const override;

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
