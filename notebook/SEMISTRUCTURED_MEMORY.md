# Memory footprint of semi-structured discretizations for basic Conjugate Gradient solve

This does not account for pre-computed assembly data. 
Hence, it is a lower bound that favors more unstructured over the other two variants.

## Problem size

#elements 16000000 	
#nodes    16200801

mesh indices and points are 32 bits (int32, float32)

## Fully unstructured
mem_hex8_mesh:   0.512 		[GB]
mem_points:      0.19441 	[GB]
field float64:   0.13		[GB]
field float32:   0.065		[GB]

CG (float64): 0.512  + 0.19441 +  5 * 0.13 = 1.35 [GB]
CG (float32): 0.512  + 0.19441 +  5 * 0.065 = 1 [GB]

*Fully structured*
connectivity: 		 0 		[GB]

CG (float64): 5 * 0.13 = 0.65 [GB]
mem reduction: 2x

CG (float32): 5 * 0.065 = 0.325 [GB]
mem reduction: 3x

## Semi-structured (store all points and indices)

Base: 	20 	
R: 		10
mem_sshex8_mesh: 0.085184 	[GB]
mem_connectivity_reduction:	6x

CG (float64): 0.085184 + 0.19441 + 5 * 0.13
0.92	    				[GB]

Base: 	10 	
R: 		20
mem_sshex8_mesh: 0.074088 	[GB]
mem_connectivity_reduction:	6.9x

Base: 	5 	
R: 		40
mem_sshex8_mesh: 0.068921 	[GB]
mem_connectivity_reduction: 7.42x

Base: 	4 	
R: 		50
mem_sshex8_mesh: 0.0679173 	[GB]
mem_connectivity_reduction: 7.5x

CG (float64): 0.0679173 + 0.19441 + 5 * 0.13 = 0.91 [GB]
mem reduction: 1.48x

CG (float32): 0.0679173 + 0.19441 + 5 * 0.065 = 0.58 [GB]
mem reduction: 1.7x

## Semi-structured (store only corner nodes)

Base: 	4 	
R: 		50
mem_sshex8_mesh: 0.0679173 	[GB]
mem_connectivity_reduction: 7.5x
mem_corner_points: 2.7e-06 	[GB]

CG (float64): 0.0679173 + 2.7e-06 + 5 * 0.13 = 0.718	[GB]
mem reduction: 1.88x (vs. 2x of fully-structured)

CG (float32): 0.0679173 + 2.7e-06 + 5 * 0.065 = 0.39 [GB]
mem reduction: 2.32x (vs. 3x of fully-structured)
