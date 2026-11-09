#ifndef SFEM_STATE_FILES_HPP
#define SFEM_STATE_FILES_HPP

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "sfem_aliases.hpp"
#include "sfem_base.hpp"
#include "smesh_path.hpp"

namespace sfem {

    // The files of a run's saved state (Playground::write_state, a restart): what an object carries
    // from one step to the next, so that a new run can take the next step as this one would.

    /// `bytes` bytes at `data`, in host or device memory, into the file `path` as they are.
    int write_state_bytes(const smesh::Path &path, const void *const data, const size_t bytes, const MemorySpace space);

    /// The file `path` into `bytes` bytes at `data` (host or device memory); fails unless the file
    /// holds exactly that many bytes.
    int read_state_bytes(const smesh::Path &path, void *const data, const size_t bytes, const MemorySpace space);

    template <typename T>
    int write_state_buffer(const smesh::Path &path, const SharedBuffer<T> &b) {
        return write_state_bytes(path, b->data(), b->size() * sizeof(T), b->mem_space());
    }

    /// Into the buffer as it is: its size is the file's, its pointer stays.
    template <typename T>
    int read_state_buffer(const smesh::Path &path, const SharedBuffer<T> &b) {
        return read_state_bytes(path, b->data(), b->size() * sizeof(T), b->mem_space());
    }

    /// Named numbers, one "name value" per line, the values exact (hexadecimal floating point).
    int write_state_numbers(const smesh::Path &path, const std::vector<std::pair<std::string, double>> &numbers);

    /// The numbers of a file write_state_numbers wrote; fails if the file cannot be read.
    int read_state_numbers(const smesh::Path &path, std::map<std::string, double> &numbers);

    /// numbers[name], or fails (with a message naming the file's key) when it is missing.
    int state_number(const std::map<std::string, double> &numbers, const char *const name, double *const value);

}  // namespace sfem

#endif  // SFEM_STATE_FILES_HPP
