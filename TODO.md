# TODO

Most urgent SFEM/SMESH improvements. Details live in `CRITIQUE.md` and `docs/ARCHITECTURE.md`.

| Pri | Area | Task | Done When |
| --- | --- | --- | --- |
| P0 | Boundary | Enforce SFEM/SMESH ownership rules. | New mesh-derived work lands in SMESH; physics/solver work lands in SFEM. |
| P0 | Kernel data | Finish FFF/Jacobian single source of truth. | Standard, packed, generated, and CUDA paths share SMESH geometry data or document differences. |
| P0 | Blocks | Add stable block-id lookup/storage. | Operators stop resolving blocks by pointer search. |
| P0 | Sidesets | Finalize multiblock sideset semantics. | Sidesets carry block identity and reject ambiguous mixed-block inputs. |
| P0 | Hot paths | Remove per-apply heap allocations. | Repeated matrix-free apply allocates nothing after initialization. |
| P0 | Tests | Add operator parity matrix and harness. | CPU/packed/generated/CUDA/AoS/SoA parity is tested with stated tolerances. |
| P0 | Build | Add CMake presets for core configs. | Debug, Release, ASAN, MPI, no-MPI, Python, CUDA, AVX512 configs are reproducible. |
| P0 | Runtime | Move env parsing to drivers. | Operators consume typed parameters and tests avoid process-global env state. |
| P0 | Errors | Normalize recoverable error handling. | Invalid input and unsupported elements do not abort from library APIs. |
| P1 | P2 workflow | Convert pressure-projection/P2 TODOs to tests. | P2 divergence, projection, lumped mass, and full workflow are tested or marked unsupported. |
| P1 | Codegen | Make regeneration deterministic. | Clean checkout regenerates generated operators with an empty diff. |
| P1 | Repo hygiene | Clean ignores and generated artifacts. | Clean checkout status is meaningful; YAML metadata is not broadly ignored. |
