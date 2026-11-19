#ifndef SFEM_CODEGEN_REFERENCE_QUAD_TRI_Q3_HPP
#define SFEM_CODEGEN_REFERENCE_QUAD_TRI_Q3_HPP

namespace sfem {
namespace codegen {

template <typename s_t>
struct quad_tri_q3 {
  static const s_t *q_weight() {
    static const s_t data[3] = {s_t(0.16666666666666666), s_t(0.16666666666666666), s_t(0.16666666666666666)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
