# HEX8 semistructured discretization

Basic implementation of laplace operator using semistructured HEX8 meshes

Performance on ARM M1 CPU with 8 threads

```c
SFEM_HEX8_ASSUME_AFFINE=1 
SFEM_ELEMENT_REFINE_LEVEL=8
SFEM_REPEAT=666

#elements 59319 #microelements 30371328 #nodes 30664297
#nodexelement 729 #microelementsxelement 512
Operator TTS:			0.0721					[s]
Operator throughput:	0.8						[ME/s]
Operator throughput:	421.0					[MmicroE/s]
Operator throughput:	425.1					[MDOF/s]
Operator memory 0.490629 (2 x coeffs) + 0.00142366 (FFFs) + 0.172974 (index) = 0.665027 [GB]
Total:			48.3685	[s]
```