# Verification and Validation

This directory contains self-contained numerical cases that exercise SFEM drivers against independent oracles. A case passes only when every declared comparison reproduces an analytical solution, manufactured solution, or measurement within its stated tolerance. Solver completion by itself is never a pass condition.

Run the complete suite from the repository root:

```bash
verification_and_validation/run_all.py
```

Useful options:

```bash
verification_and_validation/run_all.py --list
verification_and_validation/run_all.py --case cylindrical_pressure_vessel
verification_and_validation/run_all.py --build-dir build64 --output-dir /tmp/sfem-vv
verification_and_validation/run_all.py --verbose
```

The runner discovers one `case.yaml` in each immediate child directory. Each case declares:

- a command that generates its mesh from source;
- checked-in YAML input templates for the driver;
- the SFEM driver and environment;
- a postprocessor that writes the common `verification.json` schema;
- oracle provenance and numerical tolerances.

The suite writes one isolated output directory and log per case plus `report.json` at the output root. It returns a non-zero status when setup fails, a driver fails, convergence warnings are found, an oracle report is missing, or any oracle comparison exceeds tolerance.

## Adding a case

Create `<case_id>/case.yaml`, a deterministic mesh generator, the required YAML templates, and a verifier. The verifier report must contain a non-empty `checks` array. Every check must identify its oracle and include `observed`, `expected`, `error`, `tolerance`, and `passed` fields. The top-level runner deliberately rejects empty reports so smoke tests cannot be mistaken for validation.
