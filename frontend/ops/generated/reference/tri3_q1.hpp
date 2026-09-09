#ifndef SFEM_CODEGEN_REFERENCE_TRI3_Q1_HPP
#define SFEM_CODEGEN_REFERENCE_TRI3_Q1_HPP

#include "../../reference/quad_tri_q1.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_tri3_q1 {
  static const s_t *shape() {
    static const s_t data[3] = {s_t(0.33333333333333343), s_t(0.33333333333333331), s_t(0.33333333333333331)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[3] = {s_t(-1), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[3] = {s_t(-1), s_t(0), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
