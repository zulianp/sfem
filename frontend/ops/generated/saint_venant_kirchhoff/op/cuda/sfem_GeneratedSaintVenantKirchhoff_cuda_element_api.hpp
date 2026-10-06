#pragma once

#include <cstddef>

#include "../../d2/proteus_quad4/cuda/saint_venant_kirchhoff_proteus_quad4_element.hpp"
#include "../../d2/quad4/cuda/saint_venant_kirchhoff_quad4_element.hpp"
#include "../../d2/tri3/cuda/saint_venant_kirchhoff_tri3_element.hpp"
#include "../../d2/tri6/cuda/saint_venant_kirchhoff_tri6_element.hpp"
#include "../../d3/hex27/cuda/saint_venant_kirchhoff_hex27_element.hpp"
#include "../../d3/hex8/cuda/saint_venant_kirchhoff_hex8_element.hpp"
#include "../../d3/proteus_hex27/cuda/saint_venant_kirchhoff_proteus_hex27_element.hpp"
#include "../../d3/proteus_hex64/cuda/saint_venant_kirchhoff_proteus_hex64_element.hpp"
#include "../../d3/proteus_hex8/cuda/saint_venant_kirchhoff_proteus_hex8_element.hpp"
#include "../../d3/tet10/cuda/saint_venant_kirchhoff_tet10_element.hpp"
#include "../../d3/tet4/cuda/saint_venant_kirchhoff_tet4_element.hpp"

namespace sfem {
namespace codegen {

} // namespace codegen
} // namespace sfem
