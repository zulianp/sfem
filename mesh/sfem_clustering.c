#include "sfem_clustering.h"
#include "sfem_macros.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to calculate connectivity to current cluster
static idx_t calculate_connectivity(
    ptrdiff_t elem, 
    ptrdiff_t cluster_id,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    const idx_t *const SFEM_RESTRICT elem2cluster)
{
    idx_t connectivity = 0;
    for (ptrdiff_t j = adj_ptr[elem]; j < adj_ptr[elem + 1]; j++) {
        const ptrdiff_t neighbor = adj_idx[j];
        if (neighbor != elem && elem2cluster[neighbor] == cluster_id) {
            connectivity++;
        }
    }
    return connectivity;
}

int sfem_element_clustering(
    const ptrdiff_t n_elements,
    const ptrdiff_t cluster_size,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    idx_t *const SFEM_RESTRICT elem2cluster,
    ptrdiff_t *const SFEM_RESTRICT cluster_sizes,
    ptrdiff_t *n_clusters)
{
    if (!n_elements || !cluster_size || !adj_ptr || !adj_idx || !elem2cluster || !cluster_sizes || !n_clusters) {
        return 1;
    }

    // Calculate number of clusters needed
    *n_clusters = (n_elements + cluster_size - 1) / cluster_size;

    // Initialize all elements as unassigned
    for (ptrdiff_t i = 0; i < n_elements; i++) {
        elem2cluster[i] = SFEM_IDX_INVALID;
    }

    // Initialize cluster sizes
    for (ptrdiff_t i = 0; i < *n_clusters; i++) {
        cluster_sizes[i] = 0;
    }

    // Use a simple approach: maintain a list of unassigned elements
    ptrdiff_t *unassigned_elements = malloc(n_elements * sizeof(ptrdiff_t));
    if (!unassigned_elements) {
        return 1;
    }

    ptrdiff_t unassigned_count = 1;
    unassigned_elements[0] = 0;

    // Simple greedy clustering algorithm
    ptrdiff_t current_cluster = 0;
    ptrdiff_t elements_assigned = 0;

    while (elements_assigned < n_elements) {
        // If current cluster is full, start new cluster
        if (cluster_sizes[current_cluster] >= cluster_size) {
            current_cluster++;
            if (current_cluster >= *n_clusters) break;
        }

        // Find the best unassigned element for current cluster
        ptrdiff_t best_elem = -1;
        idx_t best_connectivity = -1;
        ptrdiff_t best_elem_idx = -1;

        ptrdiff_t start = unassigned_count - 1;
        ptrdiff_t end = -1;
        ptrdiff_t inc = -1;

        if (!cluster_sizes[current_cluster]) {
            // Do not take it from the last neighbors when creating a new cluster
            start = 0;
            end = unassigned_count;
            inc = 1;
        }

        for (ptrdiff_t i = start; i != end; i += inc) {
            ptrdiff_t elem = unassigned_elements[i];
            if (elem2cluster[elem] == SFEM_IDX_INVALID) {
                idx_t connectivity = calculate_connectivity(elem, current_cluster, adj_ptr, adj_idx, elem2cluster);

                // Prefer elements with higher connectivity to current cluster
                if (connectivity > best_connectivity) {
                    best_elem = elem;
                    best_connectivity = connectivity;
                    best_elem_idx = i;
                }
            }
        }

        // If no element found, take the first unassigned one
        if (best_elem == -1) {
            for (ptrdiff_t i = 0; i < unassigned_count; i++) {
                ptrdiff_t elem = unassigned_elements[i];
                if (elem2cluster[elem] == SFEM_IDX_INVALID) {
                    best_elem = elem;
                    best_elem_idx = i;
                    break;
                }
            }
        }

        if (best_elem == -1) break;  // No more unassigned elements

        // Assign to current cluster
        elem2cluster[best_elem] = current_cluster;
        cluster_sizes[current_cluster]++;
        elements_assigned++;

        // Remove from unassigned list (swap with last element and pop)
        assert(unassigned_elements[best_elem_idx] == best_elem);
        unassigned_elements[best_elem_idx] = unassigned_elements[unassigned_count - 1];
        unassigned_count--;

        // Add neighbors to unassigned list
        for (ptrdiff_t k = adj_ptr[best_elem]; k < adj_ptr[best_elem + 1]; k++) {
            ptrdiff_t neigh = adj_idx[k];
            if (elem2cluster[neigh] == SFEM_IDX_INVALID) {
                unassigned_elements[unassigned_count] = neigh;
                unassigned_count++;
            }
        }
    }

    assert(elements_assigned == n_elements);

    free(unassigned_elements);
    return 0;
}

int sfem_calculate_cluster_connectivity(
    const ptrdiff_t n_elements,
    const count_t *const SFEM_RESTRICT adj_ptr,
    const element_idx_t *const SFEM_RESTRICT adj_idx,
    const idx_t *const SFEM_RESTRICT elem2cluster,
    int *const SFEM_RESTRICT connectivity,
    idx_t *total_connectivity,
    idx_t *max_connectivity,
    idx_t *min_connectivity)
{
    if (!n_elements || !adj_ptr || !adj_idx || !elem2cluster || !connectivity || 
        !total_connectivity || !max_connectivity || !min_connectivity) {
        return 1;
    }

    *total_connectivity = 0;
    *max_connectivity = 0;
    *min_connectivity = n_elements;

    for (ptrdiff_t i = 0; i < n_elements; i++) {
        idx_t elem_connectivity = 0;
        for (ptrdiff_t j = adj_ptr[i]; j < adj_ptr[i + 1]; j++) {
            const ptrdiff_t neighbor = adj_idx[j];
            if (neighbor != i && elem2cluster[neighbor] == elem2cluster[i]) {
                elem_connectivity++;
            }
        }
        
        connectivity[i] = elem_connectivity;
        *total_connectivity += elem_connectivity;
        
        if (elem_connectivity > *max_connectivity) {
            *max_connectivity = elem_connectivity;
        }
        if (elem_connectivity < *min_connectivity) {
            *min_connectivity = elem_connectivity;
        }
    }

    return 0;
}
