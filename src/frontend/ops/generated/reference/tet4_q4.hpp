#ifndef SFEM_CODEGEN_REFERENCE_TET4_Q4_HPP
#define SFEM_CODEGEN_REFERENCE_TET4_Q4_HPP

#include "quad_tet_q4.hpp"

namespace sfem {
namespace codegen {

template <typename s_t>
struct ref_tet4_q4 {
  static const s_t *shape() {
    static const s_t data[16] = {s_t(0.5854101966249684), s_t(0.1381966011250105), s_t(0.1381966011250105), s_t(0.1381966011250105), s_t(0.13819660112501045), s_t(0.58541019662496852), s_t(0.1381966011250105), s_t(0.1381966011250105), s_t(0.13819660112501045), s_t(0.1381966011250105), s_t(0.58541019662496852), s_t(0.1381966011250105), s_t(0.13819660112501042), s_t(0.1381966011250105), s_t(0.1381966011250105), s_t(0.58541019662496852)};
    return data;
  }
  static const s_t *grad_ref_x() {
    static const s_t data[16] = {s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0), s_t(-1), s_t(1), s_t(0), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_y() {
    static const s_t data[16] = {s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0), s_t(-1), s_t(0), s_t(1), s_t(0)};
    return data;
  }
  static const s_t *grad_ref_z() {
    static const s_t data[16] = {s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1), s_t(-1), s_t(0), s_t(0), s_t(1)};
    return data;
  }
};

}  // namespace codegen
}  // namespace sfem

#endif
