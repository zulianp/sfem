#ifndef SFEM_BUFFER_HPP
#define SFEM_BUFFER_HPP

#include <iostream>
#include <memory>
#include <functional>

namespace sfem {

    enum ExecutionSpace {
        EXECUTION_SPACE_HOST = 0,
        EXECUTION_SPACE_DEVICE = 1,
        EXECUTION_SPACE_INVALID = -1
    };

    enum MemorySpace {
        MEMORY_SPACE_HOST = EXECUTION_SPACE_HOST,
        MEMORY_SPACE_DEVICE = EXECUTION_SPACE_DEVICE,
        MEMORY_SPACE_INVALID = EXECUTION_SPACE_INVALID
    };

    template <typename T>
    class Buffer {
    public:
        Buffer(const size_t n,
               T *const ptr,
               std::function<void(void *)> destroy,
               MemorySpace mem_space)
            : n_(n), ptr_(ptr), destroy_(destroy), mem_space_(mem_space) {}

        ~Buffer() {
            if (destroy_) {
                destroy_((void *)ptr_);
            }
        }

        inline T *const data() { return ptr_; }
        inline const T *const data() const { return ptr_; }
        inline size_t size() const { return n_; }
        inline MemorySpace mem_space() const { return mem_space_; }

        void print(std::ostream &os) {
            if (mem_space_ == MEMORY_SPACE_DEVICE) {
                os << "On the device!\n";
                return;
            } else {
                os << "Buffer size " << n_ << "\n";    
                for (std::ptrdiff_t i = 0; i < n_; i++) {
                    os << ptr_[i] << " ";
                }
                os << "\n";
            }
        }
        
        static std::shared_ptr<Buffer<T>> wrap(const std::ptrdiff_t n,
                                        T *x,
                                        enum MemorySpace mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T>>(n, x, nullptr, mem_space);
        }

        static std::shared_ptr<Buffer<T>> own(const std::ptrdiff_t n,
                                        T *x,
                                        std::function<void(void *)> destroy,
                                        enum MemorySpace mem_space = MEMORY_SPACE_INVALID) {
            return std::make_shared<Buffer<T>>(n, x, destroy, mem_space);
        }

    private:
        size_t n_{0};
        T *ptr_{nullptr};
        std::function<void(void *)> destroy_;
        MemorySpace mem_space_;
    };

    template <typename T>
    std::shared_ptr<Buffer<T>> h_buffer(const std::ptrdiff_t n) {
        auto ret =
            std::make_shared<Buffer<T>>(n, (T *)calloc(n, sizeof(T)), &free, MEMORY_SPACE_HOST);
        return ret;
    }

 

}  // namespace sfem

#endif  // SFEM_BUFFER_HPP
