#pragma once

// The colored layout's isoparametric sweeps.
//
// Separated by geometry as DESIGN.md asks, by MOVING whole functions: these two sets were always
// distinct sweeps, differing in where the Jacobian comes from -- one adjugate per element read
// from a table, against one derived per sub-control volume from the node coordinates. Nothing is
// duplicated to achieve the split.
//
// Where a format's two geometries are ONE sweep templated on `bool ISO` -- the packed and store
// layouts -- they stay that way. DESIGN.md's clause is "logically separated (now they are mixed
// in with enum and booleans)"; the enum and the booleans are gone and the choice is made at
// compile time, which is the separation it asks for. Splitting those physically would mean two
// copies of the pack staging, the drain and the ghost reduction.

#include "kernels/colored/cvfem_hex8_best_ecolored.hpp"

