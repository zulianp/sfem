#ifndef SFEM_BUFFER_HPP
#define SFEM_BUFFER_HPP

#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>

#include "sfem_base.h"

#include "matrixio_array.h"
#include "sfem_MPIType.hpp"

namespace sfem {

    enum ExecutionSpace { EXECUTION_SPACE_HOST = 0, EXECUTION_SPACE_DEVICE = 1, EXECUTION_SPACE_INVALID = -1 };

    static ExecutionSpace execution_space_from_string(const std::string &str) {
        if (str == "host") {
            return EXECUTION_SPACE_HOST;
        }

        if (str == "device") {
            return EXECUTION_SPACE_DEVICE;
        }

        SFEM_ERROR("Invalid ExecutionSpace: %s\n", str.c_str());
        return EXECUTION_SPACE_INVALID;
    }

    enum MemorySpace {
        MEMORY_SPACE_HOST    = EXECUTION_SPACE_HOST,
        MEMORY_SPACE_DEVICE  = EXECUTION_SPACE_DEVICE,
        MEMORY_SPACE_INVALID = EXECUTION_SPACE_INVALID
    };

    template <typename T>
    class Buffer {
    public:
        Buffer(const size_t n, T *const ptr, std::function<void(void *)> destroy, MemorySpace mem_space)
            : n_(n), ptr_(ptr), destroy_(destroy), mem_space_(mem_space) {}

        Buffer() : n_(0), ptr_(nullptr), destroy_(nullptr), mem_space_(MEMORY_SPACE_INVALID) {}

        ~Buffer() {
            if (destroy_) {
                destroy_((void *)ptr_);
            }
        }

        inline T *const       data() { return ptr_; }
        inline const T *const data() const { return ptr_; }
        inline size_t         size() const { return n_; }
        inline MemorySpace    mem_space() const { return mem_space_; }

        void print(std::ostream &os) {
            if (mem_space_ == MEMORY_SPACE_DEVICE) {
                os << "On the device!\n";
                return;
            } else {
                os << "Buffer size " << n_ << "\n";
                for (ptrdiff_t i = 0; i < n_; i++) {
                    os << ptr_[i] << " ";
                }
                os << "\n";
            }
        }

        static std::shared_ptr<Buffer<T>> wrap(const ptrdiff_t n, T *x, enum MemorySpace mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T>>(n, x, nullptr, mem_space);
        }

        static std::shared_ptr<Buffer<T>> own(const ptrdiff_t             n,
                                              T                          *x,
                                              std::function<void(void *)> destroy,
                                              enum MemorySpace            mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T>>(n, x, destroy, mem_space);
        }

        static std::shared_ptr<Buffer<T>> make_empty() { return std::make_shared<Buffer<T>>(); }

        int to_file(const char *path) {
            FILE *file = fopen(path, "wb");
            if (!file) return SFEM_FAILURE;

            const size_t written = fwrite(ptr_, sizeof(T), n_, file);
            assert(written == n_);

            fclose(file);
            return written == n_ ? SFEM_SUCCESS : SFEM_FAILURE;
        }

    private:
        size_t                      n_{0};
        T                          *ptr_{nullptr};
        std::function<void(void *)> destroy_;
        MemorySpace                 mem_space_;
    };

    template <typename T>
    class Buffer<T *> {
    public:
        Buffer(const size_t                        n0,
               const size_t                        n1,
               T **const                           ptr,
               std::function<void(int n, void **)> destroy,
               MemorySpace                         mem_space)
            : extent_{n0, n1}, ptr_(ptr), destroy_(destroy), mem_space_(mem_space) {}

        ~Buffer() {
            if (destroy_) {
                destroy_(extent_[0], (void **)ptr_);
            }
        }

        inline T **const       data() { return ptr_; }
        inline const T **const data() const { return ptr_; }
        inline size_t          extent(int i) const { return extent_[i]; }
        inline MemorySpace     mem_space() const { return mem_space_; }

        void print(std::ostream &os) {
            if (mem_space_ == MEMORY_SPACE_DEVICE) {
                os << "On the device!\n";
                return;
            }

            os << "Buffer size " << extent_[0] << ", " << extent_[1] << "\n";
            for (ptrdiff_t i = 0; i < extent_[0]; i++) {
                for (ptrdiff_t j = 0; j < extent_[1]; j++) {
                    os << ptr_[i][j] << " ";
                }
                os << "\n";
            }
            os << "\n";
        }

        static std::shared_ptr<Buffer<T *>> wrap(const ptrdiff_t  n0,
                                                 const ptrdiff_t  n1,
                                                 T              **x,
                                                 enum MemorySpace mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T *>>(n0, n1, x, nullptr, mem_space);
        }

        static std::shared_ptr<Buffer<T *>> own(const ptrdiff_t                     n0,
                                                const ptrdiff_t                     n1,
                                                T                                 **x,
                                                std::function<void(int n, void **)> destroy,
                                                enum MemorySpace                    mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T *>>(n0, n1, x, destroy, mem_space);
        }

        int to_files(const char *format) {
            char path[2048];
            for (int i = 0; i < extent_[0]; i++) {
                int nchars = snprintf(path, sizeof(path), format, i);
                assert(nchars < sizeof(path));

                if (nchars >= sizeof(path)) {
                    SFEM_ERROR("Path is too long!\n");
                }

                FILE *file = fopen(path, "wb");
                if (!file) return SFEM_FAILURE;

                const size_t written = fwrite(ptr_[i], sizeof(T), extent_[1], file);
                assert(written == extent_[1]);

                fclose(file);
            }

            return SFEM_SUCCESS;
        }

    private:
        size_t                              extent_[2];
        T                                 **ptr_{nullptr};
        std::function<void(int n, void **)> destroy_;
        MemorySpace                         mem_space_;
    };

    template <typename T>
    std::shared_ptr<Buffer<T>> create_host_buffer(const ptrdiff_t n) {
        auto ret = std::make_shared<Buffer<T>>(n, (T *)calloc(n, sizeof(T)), &free, MEMORY_SPACE_HOST);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> create_host_buffer(const ptrdiff_t n0, const ptrdiff_t n1) {
        T **data = (T **)malloc(n0 * sizeof(T *));
        for (int i = 0; i < n0; ++i) {
            data[i] = (T *)calloc(n1, sizeof(T));
        }

        auto ret = std::make_shared<Buffer<T *>>(
                n0,
                n1,
                data,
                [=](int n, void **x) {
                    for (int i = 0; i < n; ++i) {
                        free(x[i]);
                    }
                    free(x);
                },
                MEMORY_SPACE_HOST);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> create_host_buffer_fake_SoA(const ptrdiff_t n0, const ptrdiff_t n1) {
        T *allocated = (T *)calloc(n0 * n1, sizeof(T));

        T **data = (T **)malloc(n0 * sizeof(T *));
        for (int i = 0; i < n0; ++i) {
            data[i] = &allocated[i];
        }

        auto ret = std::make_shared<Buffer<T *>>(
                n0,
                n1,
                data,
                [=](int n, void **x) {
                    free(x[0]);
                    free(x);
                },
                MEMORY_SPACE_HOST);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> manage_host_buffer(const ptrdiff_t n, T *data) {
        return Buffer<T>::own(n, data, &free, MEMORY_SPACE_HOST);
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> manage_host_buffer(const ptrdiff_t n0, const ptrdiff_t n1, T **data) {
        auto ret = std::make_shared<Buffer<T *>>(
                n0,
                n1,
                data,
                [=](int n, void **x) {
                    for (int i = 0; i < n; ++i) {
                        free(x[i]);
                    }
                    free(x);
                },
                MEMORY_SPACE_HOST);
        return ret;
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> view(const std::shared_ptr<Buffer<T>> &buffer, const ptrdiff_t begin, const ptrdiff_t end) {
        return std::make_shared<Buffer<T>>(
                end - begin, &buffer->data()[begin], [keep_alive = buffer](void *) { (void)keep_alive; }, buffer->mem_space());
    }

    template <typename T>
    std::shared_ptr<Buffer<T *>> view(const std::shared_ptr<Buffer<T *>> &buffer,
                                      const ptrdiff_t                     begin0,
                                      const ptrdiff_t                     end0,
                                      const ptrdiff_t                     begin1,
                                      const ptrdiff_t                     end1) {
        const ptrdiff_t extent0 = end0 - begin0;
        const ptrdiff_t extent1 = end1 - begin1;

        T **new_buffer = (T **)malloc(extent0 * sizeof(T *));
        for (ptrdiff_t i0 = 0; i0 < extent0; i0++) {
            new_buffer[i0] = &(buffer->data()[begin0 + i0][begin1]);
        }

        return std::make_shared<Buffer<T *>>(
                extent0, extent1, new_buffer, [keep_alive = buffer](int, void *buff) { free(buff); }, buffer->mem_space());
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> sub(const std::shared_ptr<Buffer<T *>> &buffer, const ptrdiff_t i0) {
        return std::make_shared<Buffer<T>>(
                buffer->extent(1), buffer->data()[i0], [keep_alive = buffer](void *) {}, buffer->mem_space());
    }

    template <typename T>
    std::shared_ptr<Buffer<T>> create_buffer_from_file(MPI_Comm comm, const char *path) {
        T        *data{nullptr};
        ptrdiff_t local_size{0}, global_size{0};
        array_create_from_file(comm, path, sfem::MPIType<T>::value(), (void **)&data, &local_size, &global_size);
        return manage_host_buffer<T>(local_size, data);
    }

    template<typename O, typename I>
    std::shared_ptr<Buffer<O>> astype(const std::shared_ptr<Buffer<I>> &in) 
    {
        if constexpr(std::is_same<O, I>::value) {
            return in;
        }

        const ptrdiff_t size = in->size();
        auto out = create_host_buffer<O>(size);

        auto din = in->data();
        auto dout = out->data();

        for(ptrdiff_t i = 0; i < size; i++) {
            dout[i] = din[i];
        }

        return out;
    }

}  // namespace sfem

#endif  // SFEM_BUFFER_HPP
