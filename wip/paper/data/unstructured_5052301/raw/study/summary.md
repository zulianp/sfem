Paired normal wall times; each arm includes its pressure-gradient reconstruction.
Jacobian state gradients are cached; direction gradients are rebuilt per apply.
Ranges show all sample variation, not confidence intervals. No samples are removed.
Storage is an explicit array ledger, not peak process memory or a solver footprint.

| Mesh | Threads | Pack | Operator | Atomic ms range | Packed ms range | Mean time ratio | Paired ratio range |
|---|---:|---:|---|---|---|---:|---|
| medium | 1 | 256 | residual | 79.825007 [79.662502, 80.727681] | 50.877852 [50.833779, 50.957071] | 1.568954 | [1.565127, 1.586938] |
| medium | 1 | 256 | jacobian | 90.926680 [90.799164, 91.054074] | 63.665860 [63.615227, 63.725005] | 1.428186 | [1.426050, 1.430079] |
| medium | 1 | 1024 | residual | 79.762231 [79.603693, 80.699610] | 50.583587 [50.461457, 51.394098] | 1.576840 | [1.548888, 1.596899] |
| medium | 1 | 1024 | jacobian | 90.939805 [90.815539, 91.070299] | 63.227395 [62.965813, 65.084492] | 1.438298 | [1.398423, 1.444810] |
| medium | 72 | 1024 | residual | 1.497343 [1.465540, 1.713775] | 1.456472 [1.448175, 1.468937] | 1.028062 | [0.997698, 1.174989] |
| medium | 72 | 1024 | jacobian | 1.596722 [1.590384, 1.604005] | 1.728789 [1.718793, 1.742451] | 0.923607 | [0.917348, 0.928502] |
| medium | 72 | 256 | residual | 1.469433 [1.460935, 1.482030] | 0.956006 [0.945198, 0.977984] | 1.537053 | [1.499003, 1.567957] |
| medium | 72 | 256 | jacobian | 1.599309 [1.585939, 1.622538] | 1.148623 [1.110492, 1.371785] | 1.392370 | [1.162522, 1.445146] |
| large | 1 | 256 | residual | 283.886616 [283.267144, 286.390047] | 182.469996 [182.154238, 183.222868] | 1.555799 | [1.552099, 1.569724] |
| large | 1 | 256 | jacobian | 323.619154 [323.066914, 325.154472] | 230.120900 [229.732915, 231.743148] | 1.406301 | [1.395664, 1.414179] |
| large | 1 | 1024 | residual | 283.466208 [282.958855, 285.975558] | 181.268576 [180.983515, 182.073130] | 1.563791 | [1.555096, 1.578956] |
| large | 1 | 1024 | jacobian | 325.703890 [325.092841, 327.736903] | 226.974503 [226.624330, 227.873652] | 1.434980 | [1.427115, 1.444847] |
| large | 72 | 1024 | residual | 4.638191 [4.612149, 4.652220] | 2.916652 [2.867266, 3.055663] | 1.590245 | [1.517069, 1.620763] |
| large | 72 | 1024 | jacobian | 5.147461 [5.127960, 5.168647] | 3.559037 [3.544347, 3.577631] | 1.446307 | [1.440681, 1.454080] |
| large | 72 | 256 | residual | 4.685733 [4.630685, 4.928457] | 2.732797 [2.721318, 2.752779] | 1.714629 | [1.682957, 1.799933] |
| large | 72 | 256 | jacobian | 5.121171 [5.090703, 5.155728] | 3.391855 [3.353219, 3.638925] | 1.509844 | [1.408553, 1.537546] |

All setup times, array capacities, mesh identities, commands and samples are retained in records.json.
This tests a procedural irregular mesh and interior operators; it does not establish PDE accuracy or solver speedup.
