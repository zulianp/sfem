#ifndef SFEM_CODEGEN_REFERENCE_QUAD_TET_Q1_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_TET_Q1_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_tet_q1 {
  static __host__ __device__ __forceinline__ const s_t *q_weight() {
    static const s_t data[1] = {s_t(0.16666666666666666)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
