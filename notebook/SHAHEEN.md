<!-- SHAHEEN.md -->
# SHAHEEN Cluster

# Specs

```yaml
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        52 bits physical, 57 bits virtual
Byte Order:                           Little Endian
CPU(s):                               384
On-line CPU(s) list:                  0-383
Vendor ID:                            AuthenticAMD
Model name:                           AMD EPYC 9654 96-Core Processor
CPU family:                           25
Model:                                17
Thread(s) per core:                   2
Core(s) per socket:                   96
Socket(s):                            2
Stepping:                             1
Frequency boost:                      enabled
CPU max MHz:                          3707.8120
CPU min MHz:                          1500.0000
BogoMIPS:                             4793.21
Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d
Virtualization:                       AMD-V
L1d cache:                            6 MiB (192 instances)
L1i cache:                            6 MiB (192 instances)
L2 cache:                             192 MiB (192 instances)
L3 cache:                             768 MiB (24 instances)
NUMA node(s):                         8
```

# Runs

`#microelements 65536000, #micronodes 66049281`

sfem_PoissonTest 

```bash
OMP_NUM_THREADS=192
SemiStructuredLaplacian[16]::apply(affine) called 326 times. Total: 6.03169 [s], Avg: 0.0185021 [s], TP 3569.82 [MDOF/s]
OMP_NUM_THREADS=384 # (Hyper-threading)
SemiStructuredLaplacian[16]::apply(affine) called 326 times. Total: 5.96174 [s], Avg: 0.0182875 [s], TP 3611.71 [MDOF/s]
```

# Resampling

Vectorization check

```bash
objdump -d  ./CMakeFiles/sfem.dir/resampling/sfem_resample_field_V.c.o  | grep zmm
objdump -d  ./CMakeFiles/sfem.dir/resampling/tet10/tet10_resample_field_V2.c.o  | grep zmm
```


SHEX8 -> TET4 
1 node 192 cores
```cpp
real_t bits    32
ptrdiff_t bits 64
Nr of elements  167772160
Nr of nodes     28292577
Nr of point_struc 729000000
Resample        0.00494997 (seconds)
Throughput      3.389354e+10 (elements/second)
Throughput      5.715702e+09 (nodes/second)
Throughput      1.472735e+11 (point_struc/second)
Throughput      1.626890e+11 (quadrature points/second)
FLOPS           1.405902e+19 (FLOP/S)
<BenchH> mpi_rank, mpi_size, tot_nelements, tot_nnodes, npoint_struc, clock, elements_second, nodes_second, nodes_struc_second, quadrature_points_second
<BenchR> 0,   192,   167772160,   28292577,   729000000,   0.00494997,   3.38935e+10,   5.7157e+09,   1.47274e+11,  1.62689e+11
```

2 nodes 
```
Rank: [0]  real_t bits    32
Rank: [0]  ptrdiff_t bits 64
Rank: [0]  Nr of elements  167772160
Rank: [0]  Nr of nodes     28292577
Rank: [0]  Nr of point_struc 729000000
Rank: [0]  Resample        0.00269158 (seconds)
Rank: [0]  Throughput      6.233231e+10 (elements/second)
Rank: [0]  Throughput      1.051153e+10 (nodes/second)
Rank: [0]  Throughput      2.708450e+11 (point_struc/second)
Rank: [0]  Throughput      2.991951e+11 (quadrature points/second)
Rank: [0]  FLOPS           2.591709e+19 (FLOP/S)
<BenchH> mpi_rank, mpi_size, tot_nelements, tot_nnodes, npoint_struc, clock, elements_second, nodes_second, nodes_struc_second, quadrature_points_second
<BenchR> 0,   384,   167772160,   28292577,   729000000,   0.00269158,   6.23323e+10,   1.05115e+10,   2.70845e+11,  2.99195e+11
```

4 nodes 
```
Rank: [0]  real_t bits    32
Rank: [0]  ptrdiff_t bits 64
Rank: [0]  Nr of elements  167772160
Rank: [0]  Nr of nodes     28292577
Rank: [0]  Nr of point_struc 729000000
Rank: [0]  Resample        0.00126249 (seconds)
Rank: [0]  Throughput      1.328904e+11 (elements/second)
Rank: [0]  Throughput      2.241023e+10 (nodes/second)
Rank: [0]  Throughput      5.774326e+11 (point_struc/second)
Rank: [0]  Throughput      6.378740e+11 (quadrature points/second)
Rank: [0]  FLOPS           5.510323e+19 (FLOP/S)
<BenchH> mpi_rank, mpi_size, tot_nelements, tot_nnodes, npoint_struc, clock, elements_second, nodes_second, nodes_struc_second, quadrature_points_second
<BenchR> 0,   768,   167772160,   28292577,   729000000,   0.00126249,   1.3289e+11,   2.24102e+10,   5.77433e+11,  6.37874e+11
```

