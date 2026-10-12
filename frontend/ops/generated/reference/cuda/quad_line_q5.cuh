#ifndef SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q5_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_LINE_Q5_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_line_q5 {
  static __host__ __device__ __forceinline__ const s_t *q_weight_1d() {
    static const s_t data[5] = {s_t(0.11846344252809454), s_t(0.23931433524968312), s_t(0.2844444444444445), s_t(0.23931433524968312), s_t(0.11846344252809454)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
