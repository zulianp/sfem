#ifndef SFEM_EXTRACT_SURFACE_GRAPH
#define SFEM_EXTRACT_SURFACE_GRAPH

#include "sfem_base.h"
#include <stddef.h>

void extract_surface_connectivity(const ptrdiff_t n_elements,
                                  idx_t** const elems,
                                  ptrdiff_t* n_surf_elements,
                                  idx_t** surf_elems,
                                  idx_t** parent_element);

#endif //SFEM_EXTRACT_SURFACE_GRAPH
