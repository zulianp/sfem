#ifndef SFEM_CODEGEN_REFERENCE_TRI3_Q6_HPP
#define SFEM_CODEGEN_REFERENCE_TRI3_Q6_HPP

#include "../../../reference/cuda/quad_tri_q6.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_tri3_q6 {
  static __host__ __device__ __forceinline__ const s_t *shape() {
    static const s_t data[18] = {s_t(0.10810301816807022), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.44594849091596489), s_t(0.10810301816807021), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.091576213509770743), s_t(0.81684757298045851)};
    return data;
  }
  static __host__ __device__ __forceinline__ const s_t *grad_ref_x() {
    static const s_t data[18] = {s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static __host__ __device__ __forceinline__ const s_t *grad_ref_y() {
    static const s_t data[18] = {s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
