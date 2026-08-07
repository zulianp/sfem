# SFEM Hardware Tools

Standalone utilities for querying compute hardware. These tools are **not** part of the main SFEM CMake build.

## opencl_devices

Lists all OpenCL platforms and devices with their queryable properties.

### Build

```bash
cd hardware
cmake -S . -B build
cmake --build build
```

### Run

```bash
./build/opencl_devices
```

### Requirements

- C99 compiler
- OpenCL headers and runtime (system OpenCL framework on macOS, `ocl-icd-opencl-dev` on Linux)
