#include "sfem_StateFiles.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef SFEM_ENABLE_CUDA
#include "smesh_device_arrays.hpp"
#endif

namespace sfem {

    namespace {
        // A host copy of device bytes, or the host bytes themselves.
        const char *host_bytes(const void *const data, const size_t bytes, const MemorySpace space, std::vector<char> &copy) {
#ifdef SFEM_ENABLE_CUDA
            if (space == MEMORY_SPACE_DEVICE) {
                copy.resize(bytes);
                if (bytes) smesh::device::device_to_host<char>(bytes, static_cast<const char *>(data), copy.data());
                return copy.data();
            }
#else
            (void)space;
            (void)copy;
            (void)bytes;
#endif
            return static_cast<const char *>(data);
        }
    }  // namespace

    int write_state_bytes(const smesh::Path &path, const void *const data, const size_t bytes, const MemorySpace space) {
        std::vector<char> copy;
        const char *const h = host_bytes(data, bytes, space, copy);
        FILE             *f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "state: cannot write %s\n", path.c_str());
            return SFEM_FAILURE;
        }
        const bool ok = bytes == 0 || fwrite(h, 1, bytes, f) == bytes;
        return (fclose(f) == 0 && ok) ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int read_state_bytes(const smesh::Path &path, void *const data, const size_t bytes, const MemorySpace space) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "state: cannot read %s\n", path.c_str());
            return SFEM_FAILURE;
        }
        fseek(f, 0, SEEK_END);
        const long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size < 0 || (size_t)size != bytes) {
            fprintf(stderr, "state: %s holds %ld bytes, %zu expected\n", path.c_str(), size, bytes);
            fclose(f);
            return SFEM_FAILURE;
        }
        std::vector<char> h(bytes);
        const bool        ok = bytes == 0 || fread(h.data(), 1, bytes, f) == bytes;
        fclose(f);
        if (!ok) return SFEM_FAILURE;
#ifdef SFEM_ENABLE_CUDA
        if (space == MEMORY_SPACE_DEVICE) {
            if (bytes) smesh::device::host_to_device<char>(bytes, h.data(), static_cast<char *>(data));
            return SFEM_SUCCESS;
        }
#else
        (void)space;
#endif
        if (bytes) memcpy(data, h.data(), bytes);
        return SFEM_SUCCESS;
    }

    int write_state_numbers(const smesh::Path &path, const std::vector<std::pair<std::string, double>> &numbers) {
        FILE *f = fopen(path.c_str(), "w");
        if (!f) {
            fprintf(stderr, "state: cannot write %s\n", path.c_str());
            return SFEM_FAILURE;
        }
        // %a: the double exactly; strtod reads it back.
        for (const auto &n : numbers) fprintf(f, "%s %a\n", n.first.c_str(), n.second);
        return fclose(f) == 0 ? SFEM_SUCCESS : SFEM_FAILURE;
    }

    int read_state_numbers(const smesh::Path &path, std::map<std::string, double> &numbers) {
        FILE *f = fopen(path.c_str(), "r");
        if (!f) {
            fprintf(stderr, "state: cannot read %s\n", path.c_str());
            return SFEM_FAILURE;
        }
        char name[256], value[128];
        while (fscanf(f, "%255s %127s", name, value) == 2) numbers[name] = strtod(value, nullptr);
        fclose(f);
        return SFEM_SUCCESS;
    }

    int state_number(const std::map<std::string, double> &numbers, const char *const name, double *const value) {
        auto it = numbers.find(name);
        if (it == numbers.end()) {
            fprintf(stderr, "state: no \"%s\" in the saved state\n", name);
            return SFEM_FAILURE;
        }
        *value = it->second;
        return SFEM_SUCCESS;
    }

}  // namespace sfem
