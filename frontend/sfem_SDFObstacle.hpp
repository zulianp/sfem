#ifndef SFEM_SDFOBSTACLE_HPP
#define SFEM_SDFOBSTACLE_HPP

#include "sfem_defs.h"

#include "sfem_Buffer.hpp"
#include "sfem_Communicator.hpp"
#include "sfem_Grid.hpp"

#include <mpi.h>
#include <memory>

namespace sfem {
    class Obstacle {
    public:
        virtual ~Obstacle()                                 = default;
        virtual int sample(enum ElemType                element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t **const SFEM_RESTRICT normals,
                           real_t *const SFEM_RESTRICT  gap) = 0;

        virtual int sample_normals(enum ElemType                element_type,
                                   const ptrdiff_t              nelements,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   real_t **const SFEM_RESTRICT normals) = 0;

        virtual int sample_value(enum ElemType                element_type,
                                 const ptrdiff_t              nelements,
                                 const ptrdiff_t              nnodes,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 real_t *const SFEM_RESTRICT  gap) = 0;
    };

    class SDFObstacle final : public Obstacle {
    public:
        SDFObstacle();
        ~SDFObstacle();

        void set_variational(const bool enabled);
        int  sample(enum ElemType                element_type,
                    const ptrdiff_t              nelements,
                    const ptrdiff_t              nnodes,
                    idx_t **const SFEM_RESTRICT  elements,
                    geom_t **const SFEM_RESTRICT points,
                    real_t **const SFEM_RESTRICT normals,
                    real_t *const SFEM_RESTRICT  gap) override;

        int sample_normals(enum ElemType                element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t **const SFEM_RESTRICT normals) override;

        int sample_value(enum ElemType                element_type,
                         const ptrdiff_t              nelements,
                         const ptrdiff_t              nnodes,
                         idx_t **const SFEM_RESTRICT  elements,
                         geom_t **const SFEM_RESTRICT points,
                         real_t *const SFEM_RESTRICT  gap) override;

        static std::shared_ptr<SDFObstacle> create_from_file(const std::shared_ptr<Communicator>& comm,
                                                             const std::string        &path,
                                                             const enum ExecutionSpace es);
        static std::shared_ptr<SDFObstacle> create(const std::shared_ptr<Grid<geom_t>> &sdf, const enum ExecutionSpace es);

    public:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };
}  // namespace sfem

#endif  // SFEM_SDFOBSTACLE_HPP
