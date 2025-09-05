#pragma once

#include "sfem_base.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs element clustering based on dual graph connectivity
 * 
 * This function implements a greedy clustering algorithm that groups elements
 * into clusters of approximately equal size while maximizing intra-cluster
 * connectivity.
 * 
 * @param n_elements Number of elements to cluster
 * @param cluster_size Target size for each cluster
 * @param adj_ptr Adjacency pointer array (CSR format) - size n_elements + 1
 * @param adj_idx Adjacency index array (CSR format) - size nnz
 * @param elem2cluster Output array for element-to-cluster mapping - size n_elements
 * @param cluster_sizes Output array for actual cluster sizes - size n_clusters
 * @param n_clusters Output parameter for number of clusters created
 * 
 * @return 0 on success, non-zero on error
 */
int sfem_element_clustering(
    const ptrdiff_t n_elements,
    const ptrdiff_t cluster_size,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    idx_t *const SFEM_RESTRICT elem2cluster,
    ptrdiff_t *const SFEM_RESTRICT cluster_sizes,
    ptrdiff_t *n_clusters
);

/**
 * @brief Calculates connectivity metrics for clustered elements
 * 
 * @param n_elements Number of elements
 * @param adj_ptr Adjacency pointer array (CSR format)
 * @param adj_idx Adjacency index array (CSR format)
 * @param elem2cluster Element-to-cluster mapping
 * @param connectivity Output array for element connectivity - size n_elements
 * @param total_connectivity Output parameter for total connectivity
 * @param max_connectivity Output parameter for maximum connectivity
 * @param min_connectivity Output parameter for minimum connectivity
 * 
 * @return 0 on success, non-zero on error
 */
int sfem_element_clustering_openmp(
    const ptrdiff_t n_elements,
    const ptrdiff_t cluster_size,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    idx_t *const SFEM_RESTRICT elem2cluster,
    ptrdiff_t *const SFEM_RESTRICT cluster_sizes,
    ptrdiff_t *n_clusters
);

int sfem_calculate_cluster_connectivity(
    const ptrdiff_t n_elements,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    const idx_t *const SFEM_RESTRICT elem2cluster,
    int *const SFEM_RESTRICT connectivity,
    idx_t *total_connectivity,
    idx_t *max_connectivity,
    idx_t *min_connectivity
);

#ifdef __cplusplus
}
#endif
