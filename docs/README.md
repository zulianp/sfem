# SFEM documentation

## Using SFEM

- [Building](building.md): requirements, options, Python bindings, Alps.
- [Testing](testing.md): unit tests, verification and validation, build configurations.
- [Build matrix](build-matrix.md): the option combinations tested on macOS and Alps.

## The Playground

The Playground runs solid mechanics simulations described by a YAML file.

- [Overview](playground/overview.md): what it does, the command line, a first case.
- [Case reference](playground/case-reference.md): every key of a case.
- [Examples](playground/examples.md): the cases in `workflows/`.
- [Outputs](playground/outputs.md): fields, history, probes, and viewing results in ParaView.
- [Parallel and GPU runs](playground/parallel-and-gpu.md): threads, MPI, GPUs, Alps.

## Technical notes

- [Elemental assembly](ELEMENTAL_ASSEMBLE.md): calling a generated operator's element kernels
  directly.
- `merit.tex`: sampling the residual-based merit in one pass.
- `partial_assembly.tex`: partial assembly by L2 projection.
- The code generator: `python/codegen/framework/README.md`.
