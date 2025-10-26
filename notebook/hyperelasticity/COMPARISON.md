# Axis aligned elements M1 Max Pro

## MF plus Jacobi Preconditioner

SFEM_OPERATOR=NeoHookeanOgden SFEM_USE_PACKED_MESH=1 SFEM_USE_PRECONDITIONER=1 SFEM_ELEMENTS_PER_PACK=2048 ./hyperelasticity_hex8.sh 
Total linear iterations: 27421

```c++
name,calls,total,avg
ConjugateGradient::apply,218,31.482,0.144413
DirichletConditions::apply_value,516,0.063862,0.000123764
DirichletConditions::copy_constrained_dofs,56586,6.9749,0.000123262
DirichletConditions::create_from_yaml,1,0.003582,0.003582
DirichletConditions::gradient,516,0.064339,0.000124688
DirichletConditions::value,2,0.000592,0.000296
DirichletConditions::value_steps,436,0.061508,0.000141073
Function::apply,28075,17.5006,0.000623351
Function::constraints_gradient,258,0.064583,0.000250322
Function::copy_constrained_dofs,28293,6.99885,0.00024737
Function::gradient,258,0.416771,0.00161539
Function::hessian_diag,258,0.562015,0.00217835
Function::update,258,0.078569,0.000304531
Function::value,1,0.001881,0.001881
Function::value_steps,218,1.49588,0.00686182
Mesh::read,1,0.003297,0.003297
Mesh::write,1,0.002129,0.002129
NeoHookeanOgden::apply,28075,10.5427,0.00037552
NeoHookeanOgden::create,1,0,0
NeoHookeanOgden::gradient,258,0.351929,0.00136407
NeoHookeanOgden::hessian_diag,258,0.497688,0.00192902
NeoHookeanOgden::update,258,0.078389,0.000303833
NeoHookeanOgden::value,1,0.001282,0.001282
NeoHookeanOgden::value_steps,218,1.43387,0.0065774
Output::write_time_step,82,0.156051,0.00190306
ShiftableJacobi::apply,27857,1.39201,4.99697e-05
Sideset::read,3,0.003139,0.00104633
solve_hyperelasticity,1,34.342,34.342
```

## MF NO Jacobi Preconditioner

SFEM_OPERATOR=NeoHookeanOgden SFEM_USE_PACKED_MESH=1 SFEM_USE_PRECONDITIONER=0 SFEM_ELEMENTS_PER_PACK=2048 SFEM_OP_TYPE=MF ./hyperelasticity_hex8.sh 
Total linear iterations: 33057

```c++
name,calls,total,avg
ConjugateGradient::apply,221,33.9549,0.153642
DirichletConditions::copy_constrained_dofs,67440,8.13909,0.000120686
DirichletConditions::create_from_yaml,1,0.003058,0.003058
DirichletConditions::gradient,522,0.063856,0.00012233
DirichletConditions::value,2,0.000275,0.0001375
DirichletConditions::value_steps,442,0.063315,0.000143247
Function::apply,33499,20.6954,0.000617792
Function::constraints_gradient,261,0.064118,0.000245663
Function::copy_constrained_dofs,33720,8.16619,0.000242176
Function::gradient,261,0.422659,0.00161938
Function::update,261,0.079566,0.000304851
Function::value,1,0.001356,0.001356
Function::value_steps,221,1.48782,0.00673223
Mesh::read,1,0.003147,0.003147
Mesh::write,1,0.00161,0.00161
NeoHookeanOgden::apply,33499,12.567,0.000375146
NeoHookeanOgden::create,1,1e-06,1e-06
NeoHookeanOgden::gradient,261,0.358273,0.00137269
NeoHookeanOgden::update,261,0.079388,0.000304169
NeoHookeanOgden::value,1,0.00108,0.00108
NeoHookeanOgden::value_steps,221,1.424,0.00644346
Output::write_time_step,82,0.149782,0.00182661
Sideset::read,3,0.002879,0.000959667
solve_hyperelasticity,1,36.2269,36.2269
```

Total linear iterations: 65231

## BSR plus Jacobi Preconditioner

SFEM_OPERATOR=NeoHookeanOgden SFEM_USE_PACKED_MESH=1 SFEM_USE_PRECONDITIONER=1 SFEM_ELEMENTS_PER_PACK=2048 SFEM_OP_TYPE=BSR ./hyperelasticity_hex8.sh
Total linear iterations: 22208

```c++
name,calls,total,avg
BSRSpMV::apply,22688,7.02717,0.000309731
ConjugateGradient::apply,160,24.8244,0.155153
DirichletConditions::apply_value,400,0.054882,0.000137205
DirichletConditions::copy_constrained_dofs,45696,6.12706,0.000134083
DirichletConditions::create_from_yaml,1,0.002999,0.002999
DirichletConditions::gradient,400,0.054382,0.000135955
DirichletConditions::hessian_bsr,322,0.053919,0.00016745
DirichletConditions::value,2,0.000476,0.000238
DirichletConditions::value_steps,320,0.05193,0.000162281
Function::constraints_gradient,200,0.054616,0.00027308
Function::copy_constrained_dofs,22848,6.17384,0.000270214
Function::gradient,200,0.35259,0.00176295
Function::hessian_bsr,161,2.05207,0.0127458
Function::hessian_diag,200,0.47623,0.00238115
Function::update,200,0.065344,0.00032672
Function::value,1,0.001629,0.001629
Function::value_steps,160,1.15716,0.00723226
Mesh::initialize_node_to_node_graph,1,0.001668,0.001668
Mesh::read,1,0.000422,0.000422
Mesh::write,1,0.001835,0.001835
NeoHookeanOgden::create,1,1e-06,1e-06
NeoHookeanOgden::gradient,200,0.297751,0.00148876
NeoHookeanOgden::hessian_bsr,161,1.99749,0.0124068
NeoHookeanOgden::hessian_diag,200,0.420911,0.00210456
NeoHookeanOgden::update,200,0.065125,0.000325625
NeoHookeanOgden::value,1,0.001151,0.001151
NeoHookeanOgden::value_steps,160,1.10464,0.00690401
Output::write_time_step,82,0.168335,0.00205287
ShiftableJacobi::apply,22528,1.23733,5.4924e-05
Sideset::read,3,0.002611,0.000870333
solve_hyperelasticity,1,29.222,29.222
```


## BSR NO Jacobi Preconditioner

SFEM_OPERATOR=NeoHookeanOgden SFEM_USE_PACKED_MESH=1 SFEM_USE_PRECONDITIONER=0 SFEM_ELEMENTS_PER_PACK=2048 SFEM_OP_TYPE=BSR ./hyperelasticity_hex8.sh
Total linear iterations: 24695


```c++
name,calls,total,avg
BSRSpMV::apply,25015,7.59715,0.000303704
ConjugateGradient::apply,160,23.5613,0.147258
DirichletConditions::copy_constrained_dofs,50350,6.39795,0.00012707
DirichletConditions::create_from_yaml,1,0.003241,0.003241
DirichletConditions::gradient,400,0.047956,0.00011989
DirichletConditions::hessian_bsr,322,0.049092,0.00015246
DirichletConditions::value,2,0.000475,0.0002375
DirichletConditions::value_steps,320,0.044284,0.000138387
Function::constraints_gradient,200,0.048162,0.00024081
Function::copy_constrained_dofs,25175,6.44686,0.000256082
Function::gradient,200,0.322215,0.00161107
Function::hessian_bsr,161,1.98427,0.0123247
Function::update,200,0.062091,0.000310455
Function::value,1,0.001569,0.001569
Function::value_steps,160,1.11137,0.00694606
Mesh::initialize_node_to_node_graph,1,0.001514,0.001514
Mesh::read,1,0.003334,0.003334
Mesh::write,1,0.001481,0.001481
NeoHookeanOgden::create,1,0,0
NeoHookeanOgden::gradient,200,0.273835,0.00136917
NeoHookeanOgden::hessian_bsr,161,1.93462,0.0120163
NeoHookeanOgden::update,200,0.06189,0.00030945
NeoHookeanOgden::value,1,0.001092,0.001092
NeoHookeanOgden::value_steps,160,1.06658,0.00666614
Output::write_time_step,82,0.137887,0.00168155
Sideset::read,3,0.002938,0.000979333
solve_hyperelasticity,1,27.2844,27.2844
```