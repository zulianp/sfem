#ifndef SFEM_CODEGEN_REFERENCE_QUAD_TET_Q4_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_TET_Q4_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_tet_q4 {
  static __host__ __device__ __forceinline__ const s_t *q_weight() {
    static const s_t data[4] = {s_t(0.041666666666666664), s_t(0.041666666666666664), s_t(0.041666666666666664), s_t(0.041666666666666664)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
