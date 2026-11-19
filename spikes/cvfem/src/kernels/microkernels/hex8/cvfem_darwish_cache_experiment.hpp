#ifndef CVFEM_DARWISH_CACHE_EXPERIMENT_HPP
#define CVFEM_DARWISH_CACHE_EXPERIMENT_HPP
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cmath>

enum class DarwishCacheMode { Off, Build, Read };

// What the sweeps read and write: the coefficient tiles and their built flags, as pointers. The
// storage belongs to the caller (DarwishCacheStorage, frontend/staging/cvfem_darwish_cache_storage.hpp):
// nothing under src/kernels/ owns a buffer.
//
// The owner must invalidate before changing geometry, packing or physical parameters.
// A state revision also prevents reading coefficients from an earlier Newton state.
template <typename T>
struct DarwishCacheExperiment {
    static constexpr size_t width = cvfem_hex8_vec_size<T>;
    static constexpr size_t entries = 12 * 3 * 4 * width;
    struct alignas(ALIGN_BYTES) Tile { T coefficients[entries]; };
    Tile *tiles = nullptr;
    unsigned char *built = nullptr;
    size_t count = 0;
    uint64_t revision = 0;
    bool ready = false;
    void invalidate() { ready = false; }
    void require_ready(uint64_t state_revision) const {
        if (!ready || revision != state_revision)
            throw std::runtime_error("stale or incomplete Darwish cache");
    }
    size_t bytes() const { return count * sizeof(Tile) + count; }
    const T *read(ptrdiff_t begin) const { return tiles[size_t(begin) / width].coefficients; }
    T *build(ptrdiff_t begin) { return tiles[size_t(begin) / width].coefficients; }
    void mark_built(ptrdiff_t begin) { built[size_t(begin) / width] = 1; }
    void finish_build() {
        if (!tiles || !built) throw std::runtime_error("unallocated Darwish cache");
        for (size_t i = 0; i < count; ++i)
            if (!built[i]) throw std::runtime_error("incomplete Darwish cache");
        for (size_t i = 0; i < count; ++i)
            for (T value : tiles[i].coefficients)
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite Darwish coefficient");
        ready = true;
    }
};
#endif
