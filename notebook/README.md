# Performance Diary

You can find folders with dates (Year Month day) with the performance runs from the `benchmarks` folder.


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
