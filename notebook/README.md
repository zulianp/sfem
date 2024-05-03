# Performance

You can find folders with dates (Year Month day) with the performance runs from the `benchmarks` folder.


## 2024/05/03 (SFEM comparision A100 vs P100)

 for largest mesh. Significant speed up is observed by using A100 on large meshes **(167.7 million tet4 elements)**. For small meshes, however, the we have the opposite picture.

The peak throughput reached on the A100 is **5.3 GDOF/s**

### Linear elasticity (Throughput [MDOF/s]) 

| Geometry   | Element   |   A100 |   P100 |
|:-----------|:----------|-------:|-------:|
| cylinder   | tet10     | 4226.2 | 2312.4 |
| cylinder   | tet4      | 2016.7 |  909.6 |
| cylinder   | macrotet4 | 5330.6 | 2632   |
| sphere     | tet10     | 4369.9 | 2311.8 |
| sphere     | tet4      | 1989.7 |  909.7 |
| sphere     | macrotet4 | 4493.1 | 2623.6 |


### Laplacian (Throughput [MDOF/s]) 

| Geometry | Element     |   A100 |   P100 |
|:-----------|:----------|-------:|-------:|
| cylinder   | tet10     | 4788.2 | 2628.9 |
| cylinder   | tet4      | 1809.3 |  888.3 |
| cylinder   | macrotet4 | 4945.9 | 2720.6 |
| sphere     | tet10     | 4695.7 | 2611.8 |
| sphere     | tet4      | 1759.3 |  887.5 |
| sphere     | macrotet4 | 4856.4 | 2719.5 |


# Comparision with other Utopia software

## Plain CG on Piz Daint (NVIDIA P100)

Apple and oranges comparision for ballpark numbers

## Matrix-free CG TET10 FE (SFEM)

```
#elements 3'690'496 
#nodes 	  5'031'697
#dofs 	 15'095'091

Iterations: 		7238 
Residual: 			9.8621e-11

Total solve time:  	75.254  [s]
Iteration time: 	0.010 	[s]
all  			   	76.9185 [s]
```

## Matrix-based HEX8 FE (Belos/Tpetra/Utopia/MARS)

```
#elements 5'000'211
#nodes 	  5'088'448
#dofs 	  15265344

Iterations  		1145 
Residual: 			9.817345e-10

Total solve time:   42.64 	[s]
Total Op time:		38.8    [s]
Op time:			0.0338	[s]
Iteration time:		0.037 	[s]
all:				- 		[s]
```
**Op time ratio 0.0338/0.010 approx 3x**


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


