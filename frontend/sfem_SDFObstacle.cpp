#include "sfem_SDFObstacle.hpp"

#include "sfem_resample_gap.h"

#ifdef SFEM_ENABLE_CUDA
#include "cu_resample_gap.h"
#endif

namespace sfem {

    class SDFObstacle::Impl {
    public:
        std::shared_ptr<Grid<geom_t>> sdf;
        bool                          variational{true};
        enum ExecutionSpace           execution_space { EXECUTION_SPACE_HOST };
    };

    std::shared_ptr<SDFObstacle> SDFObstacle::create_from_file(const std::shared_ptr<Communicator> &comm,
                                                               const std::string                   &path,
                                                               const enum ExecutionSpace            es) {
        return create(Grid<geom_t>::create_from_file(comm, path.c_str()), es);
    }

    std::shared_ptr<SDFObstacle> SDFObstacle::create(const std::shared_ptr<Grid<geom_t>> &sdf,
                                                     const enum ExecutionSpace            execution_space) {
        auto obs = std::make_shared<SDFObstacle>();
#ifdef SFEM_ENABLE_CUDA
        if (execution_space == EXECUTION_SPACE_DEVICE) {
            obs->impl_->sdf = to_device(sdf);
        } else
#endif
        {
            obs->impl_->sdf = sdf;
        }
        obs->impl_->execution_space = execution_space;
        return obs;
    }

    SDFObstacle::SDFObstacle() : impl_(std::make_unique<Impl>()) {}
    SDFObstacle::~SDFObstacle() {}

    void SDFObstacle::set_variational(const bool enabled) { impl_->variational = enabled; }

    int SDFObstacle::sample(enum ElemType                element_type,
                            const ptrdiff_t              nelements,
                            const ptrdiff_t              nnodes,
                            idx_t **const SFEM_RESTRICT  elements,
                            geom_t **const SFEM_RESTRICT points,
                            real_t **const SFEM_RESTRICT normals,
                            real_t *const SFEM_RESTRICT  gap) {
        auto sdf = impl_->sdf;

#ifdef SFEM_ENABLE_CUDA
        if (impl_->execution_space == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    // Output
                    gap,
                    normals);
        }
#endif
        return resample_gap(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                sdf->nlocal(),
                sdf->stride(),
                sdf->origin(),
                sdf->delta(),
                sdf->data(),
                // Output
                gap,
                normals[0],
                normals[1],
                normals[2]);
    }

    int SDFObstacle::sample_normals(enum ElemType                element_type,
                                    const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const SFEM_RESTRICT  elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    real_t **const SFEM_RESTRICT normals) {
        auto sdf = impl_->sdf;

#ifdef SFEM_ENABLE_CUDA
        if (impl_->execution_space == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap_normals(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    // Output
                    normals);
        }
#endif
        return resample_gap_normals(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                sdf->nlocal(),
                sdf->stride(),
                sdf->origin(),
                sdf->delta(),
                sdf->data(),
                // Output
                normals[0],
                normals[1],
                normals[2]);
    }

    int SDFObstacle::sample_value(enum ElemType                element_type,
                                  const ptrdiff_t              nelements,
                                  const ptrdiff_t              nnodes,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  real_t *const SFEM_RESTRICT  gap) {
        auto sdf = impl_->sdf;
#ifdef SFEM_ENABLE_CUDA
        if (impl_->execution_space == EXECUTION_SPACE_DEVICE) {
            return cu_resample_gap_value(
                    // Mesh
                    element_type,
                    nelements,
                    nnodes,
                    elements,
                    points,
                    // SDF
                    sdf->nlocal(),
                    sdf->stride(),
                    sdf->origin(),
                    sdf->delta(),
                    sdf->data(),
                    // Output
                    gap);
        }
#endif
        return resample_gap_value(
                // Mesh
                element_type,
                nelements,
                nnodes,
                elements,
                points,
                // SDF
                sdf->nlocal(),
                sdf->stride(),
                sdf->origin(),
                sdf->delta(),
                sdf->data(),
                // Output
                gap);
    }
}  // namespace sfem
