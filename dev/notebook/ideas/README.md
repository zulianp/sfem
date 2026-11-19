
# Ideas

## Mesh footprint reduction

### Idea 1

1) Sort mesh based on node connectivity (e.g., graph-based partitioning)
2) Split node index into offset + int16 (max 65'536 local idx) for indexing subset of nodes in local subdomains. For tet10 we would reduce footprint to `100/((32 * 10)/(16 * 10 + 32)) = 60%` of the original.
3) Identify element sets based on incidence. Elements may be incident to multiple partitions.	
Options:

	- Assign element to the one of the partitions -> requires halo handling
	- Assign element to all the incident partitions (duplicate elements) -> requires node skipping
	- Create a separate partition, which we call ``elemental skeleton'', for elements incident to more than 1 node-partition -> Heavily rely on the previous decomposition step. might loose memory locality in the skeleton. Requires aggregation from skeleton to contiguous dof vector (scatter), similar to spMV (consider the surface to volume ratio for cost), skeleton could be evaluated on the CPU while the rest on the GPU?
	- Skeleton based on separating nodes elements connected to such nodes are put in their own set (difference is that there is a complete separation between subsets of nodes)

### Idea 2

1) Macro elements and boundary operator
	
	- Outer layer is explicitly indexed
	- Inner layer is implicitly indexed (global_offset + element_idx * n_inner_nodes + lid)
	- Boundary elements have a polynomial map or they are piecewise polynomial for describing more complex structures
	- Inner elements are sub-parametric

This idea requires integration with a mesh generator in order to generate appropriate representations

## High-order, contact and multigrid

### Idea 1

For multibody contact we need to construct a system of the form

```
(I + C)^T A (I + C) = (C^T A + A) (I + C) = (C^T A) + (C^T A C) + (A C) + A = B + A = L
```
- A is a matrix-free operator
- B is a sparse block matrix with the coupled degrees of freedom, which we construct performing a partial assembly of A exclusively on the element incident to contact-boundary nodes. Due to a small surface to volume ratio B memory footprint should be rather small and will allow us to perform special operations and apply it separtely from A.

In case of (monotone-)mg the coarse operator is constructed as 

```
L_c = P^T L P = P^T (B + A) P  = (P^T B + P^T A) P = P^T B P + P^T A P = P^T ((C^T A) + (C^T A C) + (A C)) P + (P^T A P) =  P^T ((C^T A) + (C^T A C) + (A C)) P + A_c = B_c + A_c
```
where `A_c` is the full coarse matrix-free operator and B_c is the contact-based extension created with the Galerkin projection (in case of truncated basis we can also include the diff matrix to subtract the unwanted contributions arising from A_c) 

### Idea 2

FAS? nonliner multilevel contact

The contact-boundary elements are assembled as they are in the fine level in the coarse and the continuity with the adjacent coarse elements is ensured by a variational restriction operator.
Instead of restricting the residual we restrict the displacement and have a nonlinear iteration on the coarse level

