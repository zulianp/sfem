"""Typed raw-array I/O used by self-contained validation cases."""

from pathlib import Path

import numpy as np


_DTYPES = {
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
}


def canonical_dtype(dtype):
    result = np.dtype(dtype).newbyteorder("=")
    if result not in _DTYPES.values():
        supported = ", ".join(sorted(_DTYPES))
        raise ValueError(f"unsupported raw dtype {result}; expected one of {supported}")
    return result


def dtype_name(dtype):
    canonical = canonical_dtype(dtype)
    for name, candidate in _DTYPES.items():
        if canonical == candidate:
            return name
    raise AssertionError("unreachable dtype mapping")


def dtype_from_path(path, default=None):
    path = Path(path)
    for token in reversed(path.name.split(".")):
        if token in _DTYPES:
            return _DTYPES[token]
    if default is None:
        raise ValueError(f"cannot infer raw dtype from filename: {path.name}")
    return canonical_dtype(default)


def typed_raw_name(stem, dtype):
    return f"{stem}.{dtype_name(dtype)}.raw"


def write_raw(path, values, dtype=None, require_finite=False):
    path = Path(path)
    source = np.asarray(values)
    target_dtype = canonical_dtype(dtype or source.dtype)
    if target_dtype.kind in "iu":
        if source.dtype.kind not in "iuf" or not np.all(np.isfinite(source)):
            raise ValueError(f"cannot convert non-numeric values to {target_dtype}")
        limits = np.iinfo(target_dtype)
        if np.any(source < limits.min) or np.any(source > limits.max) or np.any(source != np.floor(source)):
            raise ValueError(f"values cannot be represented exactly as {target_dtype}")
    array = np.asarray(source, dtype=target_dtype)
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"cannot write non-finite values to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(path)
    return path


def read_raw(path, dtype=None, count=-1, require_finite=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.fromfile(path, dtype=canonical_dtype(dtype) if dtype is not None else dtype_from_path(path), count=count)
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"raw array contains non-finite values: {path}")
    return array
