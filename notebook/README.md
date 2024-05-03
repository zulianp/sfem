# Performance

You can find folders with dates (Year Month day) with the performance runs from the `benchmarks` folder.


## 2024/05/03 (SFEM comparision A100 vs P100)

Throughput [MDOF/s] for largest mesh. Significant speed up is observed by using A100 on large meshes. For small meshes, however, the we have the opposite picture.

The peak throughput reached on the A100 is **5.3 GDOF/s**

### Linear elasticity 

| Geometry   | Element   |   A100 |   P100 |
|:-----------|:----------|-------:|-------:|
| cylinder   | tet10     | 4226.2 | 2312.4 |
| cylinder   | tet4      | 2016.7 |  909.6 |
| cylinder   | macrotet4 | 5330.6 | 2632   |
| sphere     | tet10     | 4369.9 | 2311.8 |
| sphere     | tet4      | 1989.7 |  909.7 |
| sphere     | macrotet4 | 4493.1 | 2623.6 |


### Laplacian

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

#elements 3'690'496 
#nodes 	  5'031'697
#dofs 	 15'095'091

Iterations: 		7238 
Residual: 			9.8621e-11

Total solve time:  	75.254  [s]
Iteration time: 	0.010 	[s]
all  			   	76.9185 [s]

## Matrix-based HEX8 FE (Belos/Tpetra/Utopia/MARS)

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

**Op time ratio 0.0338/0.010 approx 3x**
