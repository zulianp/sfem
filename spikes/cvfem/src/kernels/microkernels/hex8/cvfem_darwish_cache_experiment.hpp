#ifndef CVFEM_DARWISH_CACHE_EXPERIMENT_HPP
#define CVFEM_DARWISH_CACHE_EXPERIMENT_HPP
#include <vector>
#include <stdexcept>
#include <limits>
#include <cstdint>

enum class DarwishCacheMode { Off, Build, Read };

// The owner must invalidate before changing geometry, packing or physical parameters.
// A state revision also prevents reading coefficients from an earlier Newton state.
template <typename T>
struct DarwishCacheExperiment {
    static constexpr size_t width = cvfem_hex8_vec_size<T>;
    static constexpr size_t entries = 12 * 3 * 4 * width;
    struct alignas(ALIGN_BYTES) Tile { T coefficients[entries]; };
    std::vector<Tile> tiles;
    std::vector<unsigned char> built;
    uint64_t revision = 0;
    bool ready = false;
    void invalidate() { ready = false; }
    void allocate(ptrdiff_t elements, uint64_t state_revision) {
        invalidate();
        if (elements <= 0) throw std::runtime_error("empty Darwish cache");
        const size_t count = (size_t(elements) + width - 1) / width;
        if (count > std::numeric_limits<size_t>::max() / sizeof(Tile))
            throw std::runtime_error("Darwish cache size overflow");
        tiles.resize(count);
        built.assign(count, 0);
        revision = state_revision;
    }
    void require_ready(uint64_t state_revision) const {
        if (!ready || revision != state_revision)
            throw std::runtime_error("stale or incomplete Darwish cache");
    }
    size_t bytes() const { return tiles.capacity() * sizeof(Tile) + built.capacity(); }
    const T *read(ptrdiff_t begin) const { return tiles[size_t(begin) / width].coefficients; }
    T *build(ptrdiff_t begin) { return tiles[size_t(begin) / width].coefficients; }
    void mark_built(ptrdiff_t begin) { built[size_t(begin) / width] = 1; }
    void finish_build() {
        for (auto value : built)
            if (!value) throw std::runtime_error("incomplete Darwish cache");
        for (const auto &tile : tiles)
            for (T value : tile.coefficients)
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite Darwish coefficient");
        ready = true;
    }
};
#endif
