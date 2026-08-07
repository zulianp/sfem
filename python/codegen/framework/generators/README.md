# Generators

This package contains executable source-generation entry points.

Use `regenerate_all.sh` to run the standard generated-operator set:

```bash
python/codegen/framework/generators/regenerate_all.sh
```

Optional generators:

- `SFEM_GENERATE_CUDA=1` runs the CUDA generator.
- `SFEM_CUDA_ARGS="--material neohookean_ogden"` passes CUDA generator arguments.
- `SFEM_GENERATOR_MANIFESTS="path/to/manifest.json ..."` enables aggregate op registration generation.

Matrix-format generation options are documented in
[`../docs/matrix_formats.md`](../docs/matrix_formats.md).
