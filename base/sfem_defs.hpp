#ifndef SFEM_DEFS_H
#define SFEM_DEFS_H

#include "sfem_base.hpp"
#include "smesh_adjacency.hpp"
#include "smesh_elem_type.hpp"
#include "smesh_graph.hpp"

#include <assert.h>


typedef const char* OperatorType;
static OperatorType MATRIX_FREE = "MF";
static OperatorType CRS = "CRS";
static OperatorType SPLITCRS = "SPLITCRS";
static OperatorType ALIGNEDCRS = "ALIGNEDCRS";
static OperatorType CRS_SYM = "CRS_SYM";
static OperatorType BSR = "BSR";
static OperatorType BSR_SYM = "BSR_SYM";
static OperatorType COO_SYM = "COO_SYM";
static OperatorType SPLITDACRS = "SPLITDACRS";
static OperatorType SELL = "SELL";

#define SFEM_UNSUPPORTED_ELEMENT_ERROR(element_type) SFEM_ERROR("Unsupported element type %d\n", element_type);

typedef enum {
    SFEM_ACCELERATOR_TYPE_CPU = 0,
    SFEM_ACCELERATOR_TYPE_CUDA = 1,
    SFEM_ACCELERATOR_TYPE_OPENCL = 2,
    SFEM_ACCELERATOR_TYPE_OPENACC = 3,
    SFEM_ACCELERATOR_TYPE_HIP = 4
} AcceleratorsType;

typedef enum {
    ADJOINT_BASE = 0,
    ADJOINT_REFINE_ONE_STEP,
    ADJOINT_REFINE_ITERATIVE,
    ADJOINT_REFINE_HYTEG_REFINEMENT
} AdjointRefineType;

static void* SFEM_DEFAULT_STREAM = 0;

namespace sfem {
using smesh::crs_to_coo;
using smesh::elem_higher_order;
using smesh::elem_lower_order;
using smesh::elem_manifold_dim;
using smesh::elem_num_nodes;
using smesh::elem_num_sides;
using smesh::extract_skin_sideset;
using smesh::is_second_order_lagrange;
using smesh::is_semistructured_type;
using smesh::macro_base_elem;
using smesh::macro_type_variant;
using smesh::proteus_hex_micro_elements_per_dim;
using smesh::proteus_hex_type;
using smesh::semistructured_type;
using smesh::shell_type;
using smesh::side_type;
using smesh::type_from_string;
using smesh::type_to_string;
}

#endif  // SFEM_DEFS_H
