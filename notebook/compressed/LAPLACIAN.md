# Scalar packed laplacian double precision

1e-9 * (340 + 8) * 64000000 / 2.689e-02 = 828.2632949052 [GFLOP/s]

6 * 4 READS (float)
8 * 8 READS (double)
8 * 8 WRITE (double, atomicAdd)
8 * 2 READS (int16)

AI = `(340 + 8) / (6 * 4 + 8 * 8 + 8 * 8 + 8 * 2) = 2.07142857143` FLOPS/Byte (memory bound)
Grace Ridge Point: 6.93 FLOPS/Byte

Achievable peak: `2.07142857143 * 512 = 1'060.5714285722 GFLOP/s`

828.2632949052 / 3960 = 0.2091573977 = 20% of nominal peak
828.2632949052 / 1060.5714285722  = 0.7809594645 = 78% of achievable peak


2398.150 [MDOF/s]  (64481201, 64481201) Double Precision


Original: 
AI = `(340 + 8) / (6 * 4 + 8 * 8 + 8 * 8 + 8 * 8) = 1.61` FLOPS/Byte (memory bound)
Achievable peak: `1.61 * 512 = 824.32 GFLOP/s`
828.2632949052/824.32 = 1.004 = 100.4783694324% of originally achievable peak  

```
Peak: 3688.00 GFLOP/s, BW: 500.00 GB/s, ridge OI: 7.376 FLOP/B

case             FLOP/elem  B/elem  OI [F/B]  roof [GF/s]  ach [GF/s]  %roof  %peak
---------------  ---------  ------  --------  -----------  ----------  -----  -----
New (fp64)             348     168    2.0714       1035.7       828.3  80.0%  22.5%
Original (fp64)        348     216    1.6111        805.6           -      -      -
```


## Single Precision

AI = `(340 + 8) / (6 * 4 + 8 * 4 + 8 * 4 + 8 * 2) = 3.346` FLOPS/Byte (memory bound)
Achievable peak: `3.346 * 512 = 1713.152 GFLOP/s`
3465.576 [MDDF/s]  (64481201, 64481201) Single Precision (1196.776 GFLOP/s)
70% of achievable peak
Speed-up over double precision: 1.45x


# Vector 

6 * 4     READS (float)
3 * 8 * 8 READS (double)
3 * 8 * 8 WRITE (double, atomicAdd)
8 * 2     READS (int16)

AI = `3 * (340 + 8) / (6 * 4 + 3 * 8 * 8 + 3 * 8 * 8 + 8 * 2) = 2.4622641509` FLOPS/Byte (memory bound)
Achievable peak: `2.4622641509 * 512 = 1'260.68 GFLOP/s`



6 * 4     READS (float)
3 * 8 * 4 READS (float)
3 * 8 * 4 WRITE (float, atomicAdd)
8 * 2     READS (int16)

AI = `3 * (340 + 8) / (6 * 4 + 3 * 8 * 4 + 3 * 8 * 4 + 8 * 2) = 4.5` FLOPS/Byte (memory bound)
Achievable peak: `4.5 * 512 = 2'304 GFLOP/s`