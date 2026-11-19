# Testing

## Unit tests

The unit tests are in the `tests/` folders of `src/` (for example `src/frontend/tests`,
`src/algebra/bsr/tests`) and run with CTest:

```sh
cd build
ctest -j 8                      # everything
ctest -R Playground             # by name
ctest -R Parallel               # the MPI tests (4 ranks each)
ctest --output-on-failure -R MGSDF
```

A CUDA build adds the device tests (`src/frontend/tests/cuda`); on Alps run them on a compute
node. `sfem_StreamGate` checks that every kernel launch names its stream, and
`sfem_DocsCasesTest` that the Playground cases in these docs pass the schema.

The libraries in `external/` (smesh, sresample) add their own tests to the same CTest run.

## Verification and validation

`verification_and_validation/` runs simulations against analytic and reference solutions
(manufactured solutions, convergence rates, viscoelastic creep and relaxation):

```sh
verification_and_validation/run_all.py --build-dir build        # the whole suite
verification_and_validation/run_all.py --list
verification_and_validation/run_all.py --tier fast
```

See `verification_and_validation/README.md` for the cases and options.

## Build configurations

`scripts/testing/test_CPU_configs.sh` and `scripts/testing/test_CUDA_configs.sh` build SFEM
under many option combinations (`--pairwise`, `--all`; `RUN_TESTS=1` runs CTest after each).
[Build matrix](build-matrix.md) records the last results.
