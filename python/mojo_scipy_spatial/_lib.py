"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.path.join(ROOT, "dist", "libmojo-scipy-spatial.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "msp_kdtree_query": ([I] * 14 + [F, F], None),
    "msp_kdtree_query_radius": ([I] * 16, None),
    "msp_convex_hull_2d": ([I, I, I, I, F], I),
    "msp_delaunay_2d": ([I] * 12, I),
    "msp_circumcenters_2d": ([I, I, I, I], None),
    "msp_delaunay_transforms_2d": ([I, I, I, I], None),
}

_library = None


def build() -> str:
    subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        check=True,
    )
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIBRARY):
            build()
        library = ctypes.CDLL(LIBRARY)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(library, name)
            function.argtypes = argtypes
            function.restype = restype
        _library = library
    return _library


def f64(value) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError("complex-valued arrays are not supported")
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if value.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must have dtype float64 or int64")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    if value.size == 0 or value.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty")
    return int(value.ctypes.data)
