#ifndef CVFEM_DARWISH_CACHE_STORAGE_HPP
#define CVFEM_DARWISH_CACHE_STORAGE_HPP
#include <limits>
#include <stdexcept>
#include <vector>

#include "kernels/microkernels/hex8/cvfem_darwish_cache_experiment.hpp"

// The Darwish coefficient cache's storage. The sweeps take the DarwishCacheExperiment base, which
// only points at these vectors (DESIGN.md: a buffer a caller owns arrives in src/kernels/ as a
// pointer). allocate() re-points it, so the pointers stay valid until the next allocate().
template <typename T>
struct DarwishCacheStorage : DarwishCacheExperiment<T> {
    using Base = DarwishCacheExperiment<T>;
    std::vector<typename Base::Tile> tile_storage;
    std::vector<unsigned char> built_storage;
    void allocate(ptrdiff_t elements, uint64_t state_revision) {
        this->invalidate();
        if (elements <= 0) throw std::runtime_error("empty Darwish cache");
        const size_t count = (size_t(elements) + Base::width - 1) / Base::width;
        if (count > std::numeric_limits<size_t>::max() / sizeof(typename Base::Tile))
            throw std::runtime_error("Darwish cache size overflow");
        tile_storage.resize(count);
        built_storage.assign(count, 0);
        this->tiles = tile_storage.data();
        this->built = built_storage.data();
        this->count = count;
        this->revision = state_revision;
    }
};
#endif
