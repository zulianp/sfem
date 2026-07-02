#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
exec "$ROOT_DIR/python/codegen/framework/mlir/scripts/bench_tet4_cube.sh" "$@"
