#ifndef SFEM_EXTRACT_SURFACE_GRAPH
#define SFEM_EXTRACT_SURFACE_GRAPH

#include "sfem_base.h"
#include <stddef.h>


/**
 * @brief Extracts surface connectivity from the given element connectivity.
 *
 * This function takes the element connectivity represented by `elems` and extracts the surface connectivity
 * by identifying the surface faces. The surface connectivity is stored in `surf_elems` and the corresponding
 * parent element IDs are stored in `parent_element`.
 *
 * @param n_elements The number of elements in the connectivity.
 * @param elems A pointer to an array of pointers representing the element connectivity.
 * @param n_surf_elements Pointer to store the number of surface elements.
 * @param surf_elems Array of pointers to store the surface element connectivity.
 * @param parent_element Pointer to store the parent element IDs corresponding to the surface elements.
 */
void extract_surface_connectivity(const ptrdiff_t n_elements,
                                  idx_t** const elems,
                                  ptrdiff_t* n_surf_elements,
                                  idx_t** surf_elems,
                                  idx_t** parent_element);

#endif //SFEM_EXTRACT_SURFACE_GRAPH
