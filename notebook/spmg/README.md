<!-- README.md -->


# M1 Max Pro 8 threads

## Small example 6M nodes

```c++
3 x 3 spheres 
#microelements 6553600, #micronodes 6697665
n_dofs: 20092995

TTS: 217.314 [s] (i.e., 3 [m] and 36 [s])
10|33|990) [lagr++ 10] norm_pen 7.052373e-12, norm_rpen 5.608164e-13

LinearElasticity::apply called 279 times. Total: 0.012408 [s], Avg: 4.44731e-05 [s], TP 4.85687 [MDOF/s]
SemiStructuredLinearElasticity[2]::apply(affine) called 693 times. Total: 0.058612 [s], Avg: 8.45772e-05 [s], TP 12.8758 [MDOF/s]
SemiStructuredLinearElasticity[4]::apply(affine) called 693 times. Total: 0.088383 [s], Avg: 0.000127537 [s], TP 51.8674 [MDOF/s]
SemiStructuredLinearElasticity[8]::apply(affine) called 693 times. Total: 0.240119 [s], Avg: 0.000346492 [s], TP 130.99 [MDOF/s]
SemiStructuredLinearElasticity[16]::apply(affine) called 693 times. Total: 1.6883 [s], Avg: 0.00243622 [s], TP 137.348 [MDOF/s]
SemiStructuredLinearElasticity[32]::apply(affine) called 693 times. Total: 12.3578 [s], Avg: 0.0178323 [s], TP 143.906 [MDOF/s]
SemiStructuredLinearElasticity[64]::apply(affine) called 1298 times. Total: 178.365 [s], Avg: 0.137415 [s], TP 146.221 [MDOF/s]
```

```csv
count_iter,count_mg_cycles,count_nl_smooth,count_smooth,norm_penetration,norm_residual,energy_norm_correction,penalty_param,omega,rate
1,2,6,60,7.80912,0.00016983,3.09875,10,0.1,1
2,4,12,120,1.84516,0.000224483,0.0958868,100,0.01,0.0309438
3,6,18,180,0.0266599,4.89927e-05,0.0307723,1000,0.001,0.320923
4,9,27,270,0,5.11038e-07,0.00279591,1000,1e-06,0.090858
5,15,45,450,0.000149453,6.25471e-10,0.000268467,1000,1e-09,0.0960214
6,17,51,510,2.06026e-05,2.78835e-08,8.13816e-05,10000,0.0001,0.303134
7,19,57,570,1.82375e-06,4.67829e-09,4.33828e-06,10000,1e-08,0.0533078
8,29,87,870,1.42201e-07,7.32334e-12,5.16329e-07,10000,1e-11,0.119017
9,31,93,930,7.91497e-10,3.56753e-12,1.331e-08,100000,1e-05,0.0257782
10,33,99,990,7.05237e-12,5.60816e-13,2.26525e-10,100000,1e-10,0.0170191
```

## Medium size example 53M nodes

```c++
3 x 3 spheres 
#microelements 52428800, #micronodes 53003649
n_dofs: 159010947

TBA
11|36|1080) [lagr++ 11] norm_pen 7.218486e-12, norm_rpen 9.550587e-13, penetration_tol 1.000000e-14, penalty_param 1.000000e+05


LinearElasticity::apply called 306 times. Total: 0.016008 [s], Avg: 5.23137e-05 [s], TP 4.12894 [MDOF/s]
SemiStructuredLinearElasticity[2]::apply(affine) called 756 times. Total: 0.063469 [s], Avg: 8.39537e-05 [s], TP 12.9714 [MDOF/s]
SemiStructuredLinearElasticity[4]::apply(affine) called 756 times. Total: 0.092006 [s], Avg: 0.000121701 [s], TP 54.3545 [MDOF/s]
SemiStructuredLinearElasticity[8]::apply(affine) called 756 times. Total: 0.263747 [s], Avg: 0.000348872 [s], TP 130.097 [MDOF/s]
SemiStructuredLinearElasticity[16]::apply(affine) called 756 times. Total: 1.9344 [s], Avg: 0.00255873 [s], TP 130.772 [MDOF/s]
SemiStructuredLinearElasticity[32]::apply(affine) called 756 times. Total: 122.984 [s], Avg: 0.162678 [s], TP 15.7746 [MDOF/s]
SemiStructuredLinearElasticity[64]::apply(affine) called 756 times. Total: 160.593 [s], Avg: 0.212424 [s], TP 94.5889 [MDOF/s]
SemiStructuredLinearElasticity[128]::apply(affine) called 1416 times. Total: 3510.72 [s], Avg: 2.47932 [s], TP 64.1349 [MDOF/s]

```

```csv
count_iter,count_mg_cycles,count_nl_smooth,count_smooth,norm_penetration,norm_residual,energy_norm_correction,penalty_param,omega,rate
1,2,6,60,15.8636,0.000123298,4.38407,10,0.1,1
2,4,12,120,3.90934,0.0001353,0.0963879,100,0.01,0.0219859
3,6,18,180,0.0822152,9.17072e-05,0.0330794,1000,0.001,0.343191
4,9,27,270,0,8.99437e-07,0.00491116,1000,1e-06,0.148466
5,16,48,480,2.27162e-05,5.8364e-10,0.000577702,1000,1e-09,0.11763
6,18,54,540,5.22362e-05,6.36316e-08,0.000115943,10000,0.0001,0.200697
7,20,60,600,9.52962e-06,6.18319e-09,9.82044e-06,10000,1e-08,0.0847006
8,30,90,900,1.17158e-06,9.28873e-12,1.30131e-06,10000,1e-11,0.13251
9,32,96,960,1.36286e-08,1.04206e-11,8.67241e-08,100000,1e-05,0.066644
10,34,102,1020,2.14521e-10,2.21501e-12,1.69421e-09,100000,1e-10,0.0195356
11,36,108,1080,7.21849e-12,9.55059e-13,2.60857e-10,100000,1e-11,0.15397
```

# MULTISPHERE

```bash
#microelements 102400000, #micronodes 103297761
create_device_elements 200000 729 (ss)
AREA: 1
Writing mesh in test_contact/coarse_mesh
Derefine 8 -> 4
create_device_elements 200000 125 (ss)
Derefine 4 -> 2
create_device_elements 200000 27 (ss)
Derefine 2 -> 1
create_device_elements 200000 8
0) 	L=8
n_dofs: 309893283
n_ops: 1
n_constraints: 1
1) 	L=4
n_dofs: 39074643
n_ops: 1
n_constraints: 1
2) 	L=2
n_dofs: 4969323
n_ops: 1
n_constraints: 1
3) 	L=1
n_dofs: 642663
n_ops: 1
n_constraints: 1
0) r_norm=0.00184811 (<0.01)
1) r_norm=0.00039883 (<0.01)
lagr_ub: 5.224278e+02
1|2|24) [lagr++ 1] norm_pen 0.000000e+00, norm_rpen 3.988296e-04, penetration_tol 1.584893e-03, penalty_param 1.000000e+02
0) r_norm=0.000537988 (<0.0001)
1) r_norm=0.000343337 (<0.0001)
2) r_norm=0.000220416 (<0.0001)
3) r_norm=0.000141657 (<0.0001)
4) r_norm=9.1031e-05 (<0.0001)
lagr_ub: 3.851690e+02
2|7|84) [lagr++ 2] norm_pen 0.000000e+00, norm_rpen 9.103099e-05, penetration_tol 2.511886e-05, penalty_param 1.000000e+02
0) r_norm=1.67194e-05 (<1e-06)
1) r_norm=1.01319e-05 (<1e-06)
2) r_norm=6.43469e-06 (<1e-06)
3) r_norm=4.10117e-06 (<1e-06)
4) r_norm=2.61619e-06 (<1e-06)
5) r_norm=1.66945e-06 (<1e-06)
6) r_norm=1.06559e-06 (<1e-06)
7) r_norm=6.80242e-07 (<1e-06)
lagr_ub: 2.758045e+02
3|15|180) [lagr++ 3] norm_pen 0.000000e+00, norm_rpen 6.802422e-07, penetration_tol 3.981072e-07, penalty_param 1.000000e+02
0) r_norm=6.01142e-05 (<1e-08)
1) r_norm=3.78785e-05 (<1e-08)
2) r_norm=2.40967e-05 (<1e-08)
3) r_norm=1.53514e-05 (<1e-08)
4) r_norm=9.78513e-06 (<1e-08)
5) r_norm=6.23924e-06 (<1e-08)
6) r_norm=3.9792e-06 (<1e-08)
7) r_norm=2.5383e-06 (<1e-08)
8) r_norm=1.6193e-06 (<1e-08)
9) r_norm=1.03307e-06 (<1e-08)
10) r_norm=6.59086e-07 (<1e-08)
11) r_norm=4.20512e-07 (<1e-08)
12) r_norm=2.68307e-07 (<1e-08)
13) r_norm=1.71193e-07 (<1e-08)
14) r_norm=1.09231e-07 (<1e-08)
15) r_norm=6.96964e-08 (<1e-08)
16) r_norm=4.4471e-08 (<1e-08)
17) r_norm=2.83755e-08 (<1e-08)
18) r_norm=1.81054e-08 (<1e-08)
19) r_norm=1.15525e-08 (<1e-08)
20) r_norm=7.37123e-09 (<1e-08)
lagr_ub: 2.506130e+02
4|36|432) [lagr++ 4] norm_pen 0.000000e+00, norm_rpen 7.371232e-09, penetration_tol 6.309573e-09, penalty_param 1.000000e+02
0) r_norm=1.39767e-05 (<1e-10)
1) r_norm=8.82687e-06 (<1e-10)
2) r_norm=5.62349e-06 (<1e-10)
3) r_norm=3.5855e-06 (<1e-10)
4) r_norm=2.28633e-06 (<1e-10)
5) r_norm=1.45801e-06 (<1e-10)
6) r_norm=9.29864e-07 (<1e-10)
7) r_norm=5.93049e-07 (<1e-10)
8) r_norm=3.78251e-07 (<1e-10)
9) r_norm=2.41253e-07 (<1e-10)
10) r_norm=1.53873e-07 (<1e-10)
11) r_norm=9.81424e-08 (<1e-10)
12) r_norm=6.25968e-08 (<1e-10)
13) r_norm=3.99252e-08 (<1e-10)
14) r_norm=2.54649e-08 (<1e-10)
15) r_norm=1.62419e-08 (<1e-10)
16) r_norm=1.03593e-08 (<1e-10)
17) r_norm=6.60732e-09 (<1e-10)
18) r_norm=4.21424e-09 (<1e-10)
19) r_norm=2.68789e-09 (<1e-10)
20) r_norm=1.71437e-09 (<1e-10)
21) r_norm=1.09345e-09 (<1e-10)
22) r_norm=6.97412e-10 (<1e-10)
23) r_norm=4.44817e-10 (<1e-10)
24) r_norm=2.83709e-10 (<1e-10)
25) r_norm=1.80953e-10 (<1e-10)
26) r_norm=1.15414e-10 (<1e-10)
27) r_norm=7.3612e-11 (<1e-10)
lagr_ub: 2.446918e+02
5|64|768) [lagr++ 5] norm_pen 0.000000e+00, norm_rpen 7.361200e-11, penetration_tol 1.000000e-10, penalty_param 1.000000e+02
0) r_norm=3.37149e-06 (<1e-11)
1) r_norm=2.11674e-06 (<1e-11)
2) r_norm=1.34635e-06 (<1e-11)
3) r_norm=8.57828e-07 (<1e-11)
4) r_norm=5.46852e-07 (<1e-11)
5) r_norm=3.48679e-07 (<1e-11)
6) r_norm=2.22337e-07 (<1e-11)
7) r_norm=1.41781e-07 (<1e-11)
8) r_norm=9.04123e-08 (<1e-11)
9) r_norm=5.76561e-08 (<1e-11)
10) r_norm=3.67676e-08 (<1e-11)
11) r_norm=2.34469e-08 (<1e-11)
12) r_norm=1.49522e-08 (<1e-11)
13) r_norm=9.53503e-09 (<1e-11)
14) r_norm=6.08048e-09 (<1e-11)
15) r_norm=3.87752e-09 (<1e-11)
16) r_norm=2.4727e-09 (<1e-11)
17) r_norm=1.57684e-09 (<1e-11)
18) r_norm=1.00555e-09 (<1e-11)
19) r_norm=6.41233e-10 (<1e-11)
20) r_norm=4.08912e-10 (<1e-11)
21) r_norm=2.60761e-10 (<1e-11)
22) r_norm=1.66286e-10 (<1e-11)
23) r_norm=1.06039e-10 (<1e-11)
24) r_norm=6.76203e-11 (<1e-11)
25) r_norm=4.31209e-11 (<1e-11)
26) r_norm=2.74978e-11 (<1e-11)
27) r_norm=1.75351e-11 (<1e-11)
28) r_norm=1.11819e-11 (<1e-11)
29) r_norm=7.1306e-12 (<1e-11)
lagr_ub: 2.433506e+02
6|94|1128) [lagr++ 6] norm_pen 0.000000e+00, norm_rpen 7.130599e-12, penetration_tol 1.584893e-12, penalty_param 1.000000e+02
```

```c++
name,calls,total,avg
ConjugateGradient::apply,94,3.37316,0.0358847
ContactConditions::assemble_mass_vector,1,5.04977,5.04977
ContactConditions::distribute_contact_forces,752,0.00177386,2.35886e-06
ContactConditions::hessian_block_diag_sym,1,0.000328245,0.000328245
ContactConditions::init,2,0.0215931,0.0107966
ContactConditions::normal_project,758,0.00285458,3.76593e-06
ContactConditions::signed_distance,1,0.00551548,0.00551548
DirichletConditions::derefine,3,0.0120329,0.00401096
Function::apply,25391,8.698,0.000342562
Function::apply_constraints,2,0.000883456,0.000441728
Function::apply_zero_constraints,564,0.00511146,9.06288e-06
Function::constaints_mask,4,0.000726054,0.000181514
Function::copy_constrained_dofs,25391,0.155237,6.11384e-06
Function::derefine,3,0.0290091,0.0096697
Function::hessian_block_diag_sym,4,0.0174643,0.00436608
GPUDirichletConditions::mask,4,0.000716294,0.000179074
LinearElasticity::hessian_block_diag_sym,1,0.0162065,0.0162065
Mass::create,1,1.6e-06,1.6e-06
Mesh::write,1,0.729635,0.729635
Output::write,2,4.28756,2.14378
SSMeshContactSurface::collect_points,1,5.04443,5.04443
ScaledBlockVectorMult::apply,24633,0.087787,3.5638e-06
SemiStructuredGPULinearElasticity::derefine_op,3,0.016177,0.00539233
SemiStructuredMesh::export_as_standard,1,5.29403,5.29403
SemiStructuredMesh::init,1,0.194271,0.194271
ShiftableBlockSymJacobi::apply,24351,0.0517909,2.12685e-06
ShiftableBlockSymJacobi::set_diag,4,0.135575,0.0338936
ShiftableBlockSymJacobi::shift,846,0.00935098,1.10532e-05
ShiftedPenaltyMultigrid::apply,1,141.941,141.941
ShiftedPenaltyMultigrid::cycle(1),94,15.0346,0.159943
ShiftedPenaltyMultigrid::cycle(2),94,7.33414,0.0780228
ShiftedPenaltyMultigrid::cycle(3),94,3.37492,0.0359034
ShiftedPenaltyMultigrid::eval_residual_and_jacobian,658,0.0902757,0.000137197
ShiftedPenaltyMultigrid::nonlinear_cycle,94,76.1171,0.809756
ShiftedPenaltyMultigrid::nonlinear_smooth,188,0.156422,0.000832032
ShiftedPenaltyMultigrid::penalty_pseudo_galerkin_assembly,94,0.0202744,0.000215686
Sideset::create_from_selector,2,0.0304142,0.0152071
collect_energy_norm_correction,6,0.30126,0.05021
create_ssgmg::construct,1,0.409778,0.409778
cu_affine_sshex8_linear_elasticity_apply[2],1974,3.91385,0.0019827
cu_affine_sshex8_linear_elasticity_apply[4],1974,4.55856,0.0023093
cu_affine_sshex8_linear_elasticity_apply[8],1886,0.017077,9.05462e-06
cu_affine_sshex8_linear_elasticity_block_diag_sym_aos[2],1,2.751e-06,2.751e-06
cu_affine_sshex8_linear_elasticity_block_diag_sym_aos[4],1,2.72e-06,2.72e-06
cu_affine_sshex8_linear_elasticity_block_diag_sym_aos[8],1,0.00123624,0.00123624
cu_linear_elasticity_apply,19557,0.0418666,2.14075e-06
cu_sshex8_hierarchical_prolongation,94,0.000376884,4.0094e-06
cu_sshex8_hierarchical_restriction,94,0.000292597,3.11273e-06
cu_sshex8_prolongate,188,0.000513969,2.73388e-06
cu_sshex8_restrict,188,63.9816,0.340328
sshex8_extract_nodeset_from_sideset,5,0.0829495,0.0165899
sshex8_fill_points,1,5.03159,5.03159
ssquad4_restrict,282,0.0189899,6.734e-05
test_contact,1,176.592,176.592
```
