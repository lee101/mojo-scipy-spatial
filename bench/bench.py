"""Benchmarks against SciPy on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from scipy import spatial as scipy_spatial

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_scipy_spatial as mojo_spatial  # noqa: E402


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def kd_query_case():
    rng = np.random.default_rng(10)
    points = rng.random((100_000, 3))
    queries = rng.random((20_000, 3))
    ours = mojo_spatial.cKDTree(points)
    scipy = scipy_spatial.cKDTree(points)
    actual = ours.query(queries, k=4)
    expected = scipy.query(queries, k=4)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12)
    return lambda: ours.query(queries, k=4), lambda: scipy.query(queries, k=4)


def kd_radius_case():
    rng = np.random.default_rng(11)
    points = rng.random((50_000, 3))
    queries = rng.random((2_000, 3))
    ours = mojo_spatial.cKDTree(points)
    scipy = scipy_spatial.cKDTree(points)
    actual = ours.query_ball_point(queries, 0.035, return_length=True)
    expected = scipy.query_ball_point(queries, 0.035, return_length=True)
    np.testing.assert_array_equal(actual, expected)
    return (
        lambda: ours.query_ball_point(queries, 0.035, return_length=True),
        lambda: scipy.query_ball_point(queries, 0.035, return_length=True),
    )


def hull_case():
    points = np.random.default_rng(12).normal(size=(250_000, 2))
    actual = mojo_spatial.ConvexHull(points)
    expected = scipy_spatial.ConvexHull(points)
    np.testing.assert_allclose(actual.volume, expected.volume, rtol=1e-13)
    return (
        lambda: mojo_spatial.ConvexHull(points),
        lambda: scipy_spatial.ConvexHull(points),
    )


def delaunay_case():
    points = np.random.default_rng(13).random((2_000, 2))
    actual = mojo_spatial.Delaunay(points)
    expected = scipy_spatial.Delaunay(points)
    assert len(actual.simplices) == len(expected.simplices)
    return (
        lambda: mojo_spatial.Delaunay(points),
        lambda: scipy_spatial.Delaunay(points),
    )


def voronoi_case():
    points = np.random.default_rng(14).random((1_500, 2))
    actual = mojo_spatial.Voronoi(points)
    expected = scipy_spatial.Voronoi(points)
    assert len(actual.vertices) == len(expected.vertices)
    return (
        lambda: mojo_spatial.Voronoi(points),
        lambda: scipy_spatial.Voronoi(points),
    )


CASES = [
    ("KDTree.query k=4 (100k data, 20k query, 3D)", kd_query_case),
    ("KDTree.query_ball_point (50k data, 2k query, 3D)", kd_radius_case),
    ("ConvexHull construction (250k points, 2D)", hull_case),
    ("Delaunay construction (2k points, 2D)", delaunay_case),
    ("Voronoi construction (1.5k points, 2D)", voronoi_case),
]


def main():
    print(f"Machine: {cpu_name()} ({platform.machine()}), Python {platform.python_version()}")
    print()
    print("| Case | mojo-scipy-spatial | SciPy | Relative |")
    print("|---|---:|---:|---:|")
    for name, prepare in CASES:
        ours, scipy = prepare()
        ours()
        scipy()
        ours_time = best_time(ours)
        scipy_time = best_time(scipy)
        ratio = scipy_time / ours_time
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {format_time(ours_time)} | "
            f"{format_time(scipy_time)} | {label} |"
        )


if __name__ == "__main__":
    main()
