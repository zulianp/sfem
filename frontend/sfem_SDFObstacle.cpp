#include "sfem_SDFObstacle.hpp"

#include "sfem_resample_gap.h"

namespace sfem {

    class SDFObstacle::Impl {
    public:
        std::shared_ptr<Grid<geom_t>> sdf;
        bool                          variational{true};
    };

    std::shared_ptr<SDFObstacle> SDFObstacle::create_from_file(MPI_Comm                  comm,
                                                               const std::string        &path,
                                                               const enum ExecutionSpace es) {
        return create(Grid<geom_t>::create_from_file(comm, path.c_str()), es);
    }

    std::shared_ptr<SDFObstacle> SDFObstacle::create(const std::shared_ptr<Grid<geom_t>> &sdf, const enum ExecutionSpace es) {
        assert(es == sfem::EXECUTION_SPACE_HOST);
        auto obs        = std::make_shared<SDFObstacle>();
        obs->impl_->sdf = sdf;
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
