<!-- ELEMENT_MATRIX.md -->

# Single element

## Thread-based

*MEDIUM*
`1e-9*135005697/0.0216876 = 6.2250178443 [GDOF/s]`

test_poisson_and_boundary_selector #dofs 135005697 (20.1368 seconds)

*SMALL*

`1e-9*2146689/0.000343582 = 6.247 [GDOF/s]`

`200: residual abs: 2.20431e-09, rel: 9.10976e-10 (rtol = 1e-09, atol = 1e-16, alpha = 117.629)`
test_poisson_and_boundary_selector #dofs 2146689 (0.158578 seconds)


## Warp-based

*MEDIUM*
`1e-9*135005697/0.160673 =  0.84 [GDOF/s]`

test_poisson_and_boundary_selector #dofs 135005697 (126.108 seconds)

*SMALL*

`1e-9*2146689/0.00260489 = 0.824 [GDOF/s]`

`200: residual abs: 2.20431e-09, rel: 9.10976e-10 (rtol = 1e-09, atol = 1e-16, alpha = 117.629)`
test_poisson_and_boundary_selector #dofs 2146689 (0.607479 seconds)


# Semi-structured (SoA data-structures)


`2146689/0.000196291*1e-9 = 10.9362579028 [GDOF/s]`


test_poisson_and_boundary_selector #dofs 135005697 (10.8988 seconds)


## Thread-based

`1e-9*135005697/0.0110029 = 12.27 [GDOF/s]`

## Warp-based

`1e-9*135005697/0.0097146 = 13.897  [GDOF/s]`


# Semi-structured (AoS data-structures)

level=4 

`1e-9*135005697/0.00548833 = 24.598 [GDOF/s] (double precision)`
`1e-9*135005697/0.00285115 = 47.35 [GDOF/s] (single precision)`

vs Matrix-free (SoA memory efficient hierarchical layout)

`1e-9*135005697/0.00729184 = 18.514626898 [GDOF/s] (double precision)`


## Tensor core vs Cuda core

## Tensor core
`1e-9*135005697/0.00580673 = 23.2498 [GDOF/s] (double precision)`
`1e-9*135005697/0.00511098 = 26.4148 [GDOF/s] (mixed precision)`

## Cuda core
`1e-9*135005697/0.00537848 = 25.101  [GDOF/s] (double precision)`
