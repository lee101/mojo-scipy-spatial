# mojo-scipy-spatial

`mojo-scipy-spatial` is a standalone Mojo implementation of a compute-focused
subset of `scipy.spatial`, exposed to Python through a small ctypes layer. It
implements real spatial data structures and planar geometry algorithms; it does
not import SciPy at runtime or delegate covered operations to SciPy.

There is no separately installable upstream package named `scipy-spatial`.
`scipy.spatial` is a submodule of the `scipy` distribution. The test and
benchmark environments therefore install `scipy` and compare this project
directly with `scipy.spatial`.

## Coverage

The Python package exports the familiar SciPy names:

- `KDTree` and `cKDTree`: balanced tree construction, exact Euclidean `query`,
  `query_ball_point`, `query_ball_tree`, `query_pairs`, `count_neighbors`, and
  `sparse_distance_matrix`. Ranked `k` sequences, distance upper bounds, leading
  query dimensions, and SciPy's missing-neighbor sentinels are supported.
- `ConvexHull`: two-dimensional vertices, edges (`simplices`), facet equations,
  neighbors, perimeter (`area`), and enclosed area (`volume`).
- `Delaunay`: two-dimensional simplices, neighbors, boundary edges,
  barycentric transforms, vertex lookup, and `find_simplex`. This is also the
  triangulation engine used by `Voronoi`.
- `Voronoi`: two-dimensional vertices, ridge points, ridge vertices, regions,
  point-to-region mapping, and bounds, including unbounded ridges and
  cocircular sites.
- `distance_matrix`, `minkowski_distance`, and `minkowski_distance_p`.

Euclidean KD-tree traversals, monotone-chain hull construction, Bowyer-Watson
triangulation, and circumcenter calculation run in Mojo. Non-Euclidean
Minkowski queries use a NumPy compatibility path. `cKDTree` is an alias of
`KDTree`, not a separate implementation.

Not covered are higher-dimensional Qhull operations, periodic KD-trees,
furthest-site diagrams, incremental updates, weighted pair counting, Qhull
option strings, approximate `query_ball_point` searches (`eps != 0`), and
SciPy's plotting helpers. The accepted KD-tree construction tuning arguments
do not currently change the one-point-per-node layout. The planar triangulator
uses floating-point predicates rather than Qhull's adaptive robustness
machinery; duplicate or collinear sites raise `QhullError`.

## Install

The repository pins the tested Mojo nightly in `pixi.toml`.

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-scipy-spatial.so`. For packaging outside Pixi,
the Python project metadata is in `pyproject.toml`; a compatible Mojo compiler
is still needed to build the shared library.

## Usage

```python
import numpy as np
from mojo_scipy_spatial import ConvexHull, KDTree, Voronoi

points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [0.4, 0.6],
])

tree = KDTree(points)
distances, indices = tree.query([[0.1, 0.2]], k=2)
assert indices.shape == (1, 2)

hull = ConvexHull(points)
assert hull.volume == 1.0

diagram = Voronoi(points)
print(diagram.vertices)
```

Run this example in the project environment with:

```bash
pixi run python examples/quickstart.py
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz
(`x86_64`), Python 3.13.14. Each row uses identical pre-generated input,
validates results before timing, warms both implementations, and reports the
best of five runs.

| Case | mojo-scipy-spatial | SciPy | Relative |
|---|---:|---:|---:|
| KDTree.query k=4 (100k data, 20k query, 3D) | 11.6 ms | 50.2 ms | 4.31x faster |
| KDTree.query_ball_point (50k data, 2k query, 3D) | 994.3 us | 7.5 ms | 7.55x faster |
| ConvexHull construction (250k points, 2D) | 110.4 ms | 42.4 ms | 2.60x slower |
| Delaunay construction (2k points, 2D) | 27.9 ms | 12.8 ms | 2.19x slower |
| Voronoi construction (1.5k points, 2D) | 73.9 ms | 11.1 ms | 6.64x slower |

The KD-tree query kernels are faster than SciPy for these large batches.
SciPy's Qhull-backed planar geometry remains faster than this port's
floating-point monotone-chain and Bowyer-Watson implementations.

No GPU path is provided. The measured hot kernels are branch-heavy tree
traversals or scans that perform well below two floating-point operations per
byte moved, so host/device transfer and launch overhead cannot be justified.

## How it works

All Mojo kernels live in one compilation unit to keep build overhead fixed.
Exported functions use the C ABI and receive NumPy buffer addresses as 64-bit
integers. Each Mojo entry point reconstructs mutable typed pointers with
`AnyOrigin[mut=True]`; Python retains ownership of every input, output, and
scratch allocation.

Points use C-contiguous row-major `float64` arrays. Tree indices, child links,
triangle topology, and output indices use contiguous `int64` arrays. KD-tree
nodes are stored in flat parallel arrays and searched with caller-owned stack
space. Delaunay triangles and temporary cavity edges likewise live in
preallocated parallel buffers, so no allocation or object ownership crosses
the FFI boundary. The Python bridge rejects empty, misaligned, non-contiguous,
or incorrectly typed buffers before taking an address, and keeps every NumPy
array referenced for the full synchronous call.
