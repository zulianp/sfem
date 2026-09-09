#ifndef SFEM_CODEGEN_REFERENCE_QUAD_TRI_Q6_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_TRI_Q6_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_tri_q6 {
  static const s_t *q_weight() {
    static const s_t data[6] = {s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.11169079483900569), s_t(0.054975871827660998), s_t(0.054975871827660998), s_t(0.054975871827660998)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
