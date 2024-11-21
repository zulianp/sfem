# HEX8 quadrature in linear elasticity operator


CPU
```bash
#elements 30280 #ndofs fine 2599413 coarse 107325

FULL QUADRATURE (TENSOR PRODUCT)
coarse_op 0.000216007 [s], 496.858 [MDOF/s]

PARTIAL QUADRATURE (TENSOR PRODUCT)
coarse_op 0.000141859 [s], 756.561 [MDOF/s]

1ST ORDER QUADRATURE (SYMBOL)
coarse_op 0.000200033 [s], 536.536 [MDOF/s]
```

GPU
```bash

FULL QUADRATURE (TENSOR PRODUCT)
fine_op 0.093761 [s], 986.167 [MDOF/s]

FULL QUADRATURE (SYMBOL)
fine_op 0.0883501 [s], 1046.56 [MDOF/s]

```