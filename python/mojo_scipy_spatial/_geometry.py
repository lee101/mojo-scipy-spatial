"""Planar convex hull, Delaunay, and Voronoi APIs."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib


class QhullError(RuntimeError):
    pass


def _planar_points(points, minimum=3):
    values = f64(points)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("this port currently supports points of shape (n, 2)")
    if len(values) < minimum:
        raise QhullError(f"at least {minimum} planar points are required")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    return values


class ConvexHull:
    """Two-dimensional convex hull with SciPy-compatible core attributes."""

    def __init__(self, points, incremental=False, qhull_options=None):
        if incremental:
            raise NotImplementedError("incremental hull updates are not covered")
        if qhull_options not in (None, ""):
            raise NotImplementedError("Qhull option strings are not covered")
        self.points = _planar_points(points)
        self.ndim = 2
        self.npoints = len(self.points)
        self.min_bound = np.min(self.points, axis=0)
        self.max_bound = np.max(self.points, axis=0)
        order = np.lexsort((self.points[:, 1], self.points[:, 0]))
        unique = np.ones(len(order), dtype=bool)
        previous = order[:-1]
        current = order[1:]
        unique[1:] = (
            (self.points[current, 0] != self.points[previous, 0])
            | (self.points[current, 1] != self.points[previous, 1])
        )
        order = np.ascontiguousarray(order[unique], dtype=np.int64)
        if len(order) < 3:
            raise QhullError("not enough unique points for a two-dimensional hull")
        work = np.empty(2 * len(order), dtype=np.int64)
        scale = max(float((self.max_bound - self.min_bound).max()), 1.0)
        count = lib().msp_convex_hull_2d(
            addr(self.points),
            addr(order),
            addr(work),
            len(order),
            np.finfo(float).eps * scale * scale * 8.0,
        )
        if count < 3:
            raise QhullError("input points are collinear")
        self.vertices = work[:count].copy()
        self.simplices = np.column_stack(
            (self.vertices, np.roll(self.vertices, -1))
        ).astype(np.int64)
        self.nsimplex = len(self.simplices)

        polygon = self.points[self.vertices]
        shifted = np.roll(polygon, -1, axis=0)
        edge = shifted - polygon
        lengths = np.linalg.norm(edge, axis=1)
        self.area = float(lengths.sum())
        self.volume = float(
            0.5
            * abs(
                np.dot(polygon[:, 0], shifted[:, 1])
                - np.dot(polygon[:, 1], shifted[:, 0])
            )
        )
        normals = np.column_stack((edge[:, 1], -edge[:, 0])) / lengths[:, None]
        offsets = -np.einsum("ij,ij->i", normals, polygon)
        self.equations = np.column_stack((normals, offsets))
        edge_ids = np.arange(self.nsimplex)
        self.neighbors = np.column_stack(
            (np.roll(edge_ids, 1), np.roll(edge_ids, -1))
        ).astype(np.int64)
        self.coplanar = np.empty((0, 3), dtype=np.int64)
        self.good = None

    def add_points(self, points, restart=False):
        raise NotImplementedError("incremental hull updates are not covered")

    def close(self):
        return None


def _triangle_neighbors(simplices):
    neighbors = np.full((len(simplices), 3), -1, dtype=np.int64)
    edges = np.stack(
        (
            simplices[:, [1, 2]],
            simplices[:, [2, 0]],
            simplices[:, [0, 1]],
        ),
        axis=1,
    ).reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    paired = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    first = order[:-1][paired]
    second = order[1:][paired]
    neighbors[first // 3, first % 3] = second // 3
    neighbors[second // 3, second % 3] = first // 3
    return neighbors


class Delaunay:
    """Planar Delaunay triangulation used by :class:`Voronoi`."""

    def __init__(
        self, points, furthest_site=False, incremental=False, qhull_options=None
    ):
        if furthest_site or incremental:
            raise NotImplementedError(
                "furthest-site and incremental triangulations are not covered"
            )
        if qhull_options not in (None, ""):
            raise NotImplementedError("Qhull option strings are not covered")
        self.points = _planar_points(points)
        order = np.lexsort((self.points[:, 1], self.points[:, 0]))
        ordered = self.points[order]
        if np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
            raise QhullError("duplicate points are not supported")
        centered = self.points - self.points[0]
        if np.linalg.matrix_rank(centered, tol=np.finfo(float).eps * 32) < 2:
            raise QhullError("input points are collinear")
        self.ndim = 2
        self.npoints = len(self.points)
        minimum = self.points.min(axis=0)
        maximum = self.points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        span = max(float((maximum - minimum).max()), 1.0)
        supertriangle = np.array(
            [
                [center[0] - 1024 * span, center[1] - 1024 * span],
                [center[0] + 1024 * span, center[1] - 1024 * span],
                [center[0], center[1] + 1024 * span],
            ]
        )
        augmented = np.ascontiguousarray(
            np.vstack((self.points, supertriangle)), dtype=np.float64
        )
        capacity = max(64, 16 * self.npoints + 32)
        triangle_a = np.empty(capacity, dtype=np.int64)
        triangle_b = np.empty(capacity, dtype=np.int64)
        triangle_c = np.empty(capacity, dtype=np.int64)
        bad = np.empty(capacity, dtype=np.int64)
        edge_u = np.empty(3 * capacity, dtype=np.int64)
        edge_v = np.empty(3 * capacity, dtype=np.int64)
        center_x = np.empty(capacity, dtype=np.float64)
        center_y = np.empty(capacity, dtype=np.float64)
        radius2 = np.empty(capacity, dtype=np.float64)
        count = lib().msp_delaunay_2d(
            addr(augmented),
            addr(triangle_a),
            addr(triangle_b),
            addr(triangle_c),
            addr(bad),
            addr(edge_u),
            addr(edge_v),
            addr(center_x),
            addr(center_y),
            addr(radius2),
            self.npoints,
            capacity,
        )
        if count < 0:
            raise RuntimeError("Delaunay work capacity exhausted")
        if count == 0:
            raise QhullError("could not form a planar triangulation")
        self.simplices = np.ascontiguousarray(
            np.column_stack(
                (triangle_a[:count], triangle_b[:count], triangle_c[:count])
            ),
            dtype=np.int64,
        )
        self.nsimplex = count
        self.neighbors = _triangle_neighbors(self.simplices)
        self.paraboloid_scale = 1.0
        self.paraboloid_shift = 0.0
        self.equations = np.full((count, 4), np.nan, dtype=float)
        self.transform = np.empty((count, 3, 2), dtype=float)
        lib().msp_delaunay_transforms_2d(
            addr(self.points),
            addr(self.simplices),
            addr(self.transform),
            count,
        )
        self.coplanar = np.empty((0, 3), dtype=np.int64)
        self.good = np.ones(count, dtype=bool)
        self.vertex_to_simplex = np.full(self.npoints, -1, dtype=np.int64)
        self.vertex_to_simplex[self.simplices.ravel()] = np.repeat(
            np.arange(count, dtype=np.int64), 3
        )
        opposite_edges = np.stack(
            (
                self.simplices[:, [1, 2]],
                self.simplices[:, [2, 0]],
                self.simplices[:, [0, 1]],
            ),
            axis=1,
        )
        self.convex_hull = np.ascontiguousarray(
            opposite_edges[self.neighbors < 0], dtype=np.int64
        )
        self.max_bound = self.points.max(axis=0)
        self.min_bound = self.points.min(axis=0)

    def find_simplex(self, xi, bruteforce=False, tol=None):
        query = np.asarray(xi, dtype=float)
        if query.ndim == 0 or query.shape[-1] != 2:
            raise ValueError("xi must contain planar points")
        flat = query.reshape(-1, 2)
        result = np.full(len(flat), -1, dtype=np.int64)
        tolerance = 100 * np.finfo(float).eps if tol is None else tol
        for index, point in enumerate(flat):
            for triangle_index, triangle in enumerate(self.simplices):
                vertices = self.points[triangle]
                matrix = np.column_stack(
                    (vertices[1] - vertices[0], vertices[2] - vertices[0])
                )
                coordinates = np.linalg.solve(matrix, point - vertices[0])
                barycentric = np.array(
                    [1.0 - coordinates.sum(), coordinates[0], coordinates[1]]
                )
                if np.all(barycentric >= -tolerance):
                    result[index] = triangle_index
                    break
        return result.reshape(query.shape[:-1])

    def add_points(self, points, restart=False):
        raise NotImplementedError("incremental triangulation is not covered")

    def close(self):
        return None


class Voronoi:
    """Two-dimensional ordinary Voronoi diagram."""

    def __init__(
        self,
        points,
        furthest_site=False,
        incremental=False,
        qhull_options=None,
    ):
        triangulation = Delaunay(
            points,
            furthest_site=furthest_site,
            incremental=incremental,
            qhull_options=qhull_options,
        )
        self.points = triangulation.points
        self.ndim = 2
        self.npoints = len(self.points)
        self.furthest_site = False
        self.max_bound = triangulation.max_bound
        self.min_bound = triangulation.min_bound
        self._triangulation = triangulation

        raw_vertices = np.empty((triangulation.nsimplex, 2), dtype=np.float64)
        lib().msp_circumcenters_2d(
            addr(self.points),
            addr(triangulation.simplices),
            addr(raw_vertices),
            triangulation.nsimplex,
        )
        scale = max(float((self.max_bound - self.min_bound).max()), 1.0)
        vertex_map = np.empty(triangulation.nsimplex, dtype=np.int64)
        unique_vertices = []
        tolerance = np.finfo(float).eps * scale * 64
        cells: dict[tuple[int, int], list[int]] = {}
        for raw_index, center in enumerate(raw_vertices):
            mapped = -1
            cell = (
                int(math.floor(center[0] / tolerance)),
                int(math.floor(center[1] / tolerance)),
            )
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for unique_index in cells.get(
                        (cell[0] + dx, cell[1] + dy), ()
                    ):
                        if (
                            np.linalg.norm(center - unique_vertices[unique_index])
                            <= tolerance
                        ):
                            mapped = unique_index
                            break
                    if mapped >= 0:
                        break
                if mapped >= 0:
                    break
            if mapped < 0:
                mapped = len(unique_vertices)
                unique_vertices.append(center)
                cells.setdefault(cell, []).append(mapped)
            vertex_map[raw_index] = mapped
        self.vertices = np.asarray(unique_vertices, dtype=np.float64)

        opposite_edges = np.stack(
            (
                triangulation.simplices[:, [1, 2]],
                triangulation.simplices[:, [2, 0]],
                triangulation.simplices[:, [0, 1]],
            ),
            axis=1,
        )
        triangle_indices = np.broadcast_to(
            np.arange(triangulation.nsimplex, dtype=np.int64)[:, None],
            triangulation.neighbors.shape,
        )
        keep = (triangulation.neighbors < 0) | (
            triangle_indices < triangulation.neighbors
        )
        ridge_points = np.sort(opposite_edges[keep], axis=1)
        first_triangles = triangle_indices[keep]
        second_triangles = triangulation.neighbors[keep]
        ridge_order = np.lexsort((ridge_points[:, 1], ridge_points[:, 0]))
        ridge_points = ridge_points[ridge_order]
        first_triangles = first_triangles[ridge_order]
        second_triangles = second_triangles[ridge_order]
        first_vertices = vertex_map[first_triangles]
        second_vertices = np.full(len(first_vertices), -1, dtype=np.int64)
        interior = second_triangles >= 0
        second_vertices[interior] = vertex_map[second_triangles[interior]]
        distinct = ~interior | (first_vertices != second_vertices)
        ridge_points = ridge_points[distinct]
        first_vertices = first_vertices[distinct]
        second_vertices = second_vertices[distinct]

        ridge_vertices = []
        for first, second in zip(first_vertices, second_vertices):
            if second < 0:
                ridge_vertices.append([int(first), -1])
            else:
                ridge_vertices.append(sorted((int(first), int(second))))
        self.ridge_points = np.ascontiguousarray(ridge_points, dtype=np.int64)
        self.ridge_vertices = ridge_vertices
        self.ridge_dict = dict(zip(map(tuple, self.ridge_points), ridge_vertices))

        incident_vertices = [[] for _ in range(self.npoints)]
        boundary = np.zeros(self.npoints, dtype=bool)
        for triangle_index, triangle in enumerate(triangulation.simplices):
            for point in triangle:
                incident_vertices[int(point)].append(int(vertex_map[triangle_index]))
        boundary[triangulation.convex_hull.ravel()] = True

        self.regions = []
        for point_index, vertex_indices in enumerate(incident_vertices):
            px, py = self.points[point_index]
            region = sorted(
                set(vertex_indices),
                key=lambda vertex: math.atan2(
                    self.vertices[vertex, 1] - py,
                    self.vertices[vertex, 0] - px,
                ),
            )
            if boundary[point_index]:
                region.insert(0, -1)
            self.regions.append(region)
        self.point_region = np.arange(self.npoints, dtype=np.int64)

    def add_points(self, points, restart=False):
        raise NotImplementedError("incremental Voronoi updates are not covered")

    def close(self):
        return None
