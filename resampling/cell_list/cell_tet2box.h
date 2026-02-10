#ifndef __CELL_TET2BOX_H__
#define __CELL_TET2BOX_H__

#include <stdbool.h>
#include <stddef.h>

#include "sfem_base.h"

#include "cell_list_3d_map.h"
#include "precision_types.h"

//////////////////////////////////////////////////////////
// print_bounding_box_statistics
//////////////////////////////////////////////////////////
void print_bounding_box_statistics(const boxes_t *boxes);

int                                                                     //
make_mesh_tets_boxes(const ptrdiff_t                    start_element,  // Mesh
                     const ptrdiff_t                    end_element,    //
                     const ptrdiff_t                    nnodes,         //
                     const idx_t** const SFEM_RESTRICT  elems,          //
                     const geom_t** const SFEM_RESTRICT xyz,            //
                     boxes_t**                          boxes);

#endif  // __CELL_TET2BOX_H__