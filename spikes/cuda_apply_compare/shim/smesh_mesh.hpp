#pragma once
// The generated `op/*_c_abi.hpp` names `smesh::ElemType` and
// `enum smesh::PrimitiveType` in declarations only, and the operator
// translation units this spike links never take their values.  Pulling in the
// real smesh headers would drag MPI and the whole mesh frontend into a
// comparison that is about arithmetic.
namespace smesh {
enum ElemType { SMESH_SPIKE_ELEM_TYPE_STUB = 0 };
enum PrimitiveType { SMESH_SPIKE_PRIMITIVE_TYPE_STUB = 0 };
}  // namespace smesh
