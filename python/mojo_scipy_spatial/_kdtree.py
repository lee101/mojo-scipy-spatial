"""SciPy-compatible KD-tree classes backed by Mojo traversal kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib


def minkowski_distance_p(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)
    difference = np.abs(x - y)
    if p == np.inf:
        return np.amax(difference, axis=-1)
    if p == 1:
        return np.sum(difference, axis=-1)
    return np.sum(difference**p, axis=-1)


def minkowski_distance(x, y, p=2):
    powered = minkowski_distance_p(x, y, p)
    return powered if p in (1, np.inf) else powered ** (1.0 / p)


def distance_matrix(x, y, p=2, threshold=1_000_000):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be two-dimensional arrays with matching columns")
    return minkowski_distance(x[:, None, :], y[None, :, :], p)


class KDTree:
    """Balanced exact KD-tree with SciPy's principal query methods.

    Euclidean nearest-neighbor and radius traversal runs in Mojo. Other
    Minkowski metrics retain compatible behavior through a NumPy fallback.
    """

    def __init__(
        self,
        data,
        leafsize=10,
        compact_nodes=True,
        copy_data=False,
        balanced_tree=True,
        boxsize=None,
    ):
        values = f64(data)
        if values.ndim != 2:
            raise ValueError("data must be of shape (n, m)")
        if len(values) == 0:
            raise ValueError("data must contain at least one point")
        if not np.isfinite(values).all():
            raise ValueError("data must be finite")
        if leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        if boxsize is not None:
            raise NotImplementedError("periodic boxsize is not covered")

        self.data = values.copy() if copy_data else values
        self.n, self.m = self.data.shape
        self.leafsize = int(leafsize)
        self.boxsize = None
        point_indices: list[int] = []
        axes: list[int] = []
        left: list[int] = []
        right: list[int] = []

        def make_node(indices: np.ndarray) -> int:
            node = len(point_indices)
            point_indices.append(-1)
            axes.append(0)
            left.append(-1)
            right.append(-1)
            spread = np.ptp(self.data[indices], axis=0)
            axis = int(np.argmax(spread))
            ordered = indices[
                np.argsort(self.data[indices, axis], kind="stable")
            ]
            middle = len(ordered) // 2
            point_indices[node] = int(ordered[middle])
            axes[node] = axis
            if middle:
                left[node] = make_node(ordered[:middle])
            if middle + 1 < len(ordered):
                right[node] = make_node(ordered[middle + 1 :])
            return node

        make_node(np.arange(self.n, dtype=np.int64))
        self._point_indices = i64(point_indices)
        self._axes = i64(axes)
        self._left = i64(left)
        self._right = i64(right)
        self._stack_stride = self.n.bit_length() + 1
        self.size = self.n

    def _validate_queries(self, x):
        queries = f64(x)
        if queries.ndim == 0 or queries.shape[-1] != self.m:
            raise ValueError(
                f"x must consist of vectors of length {self.m}, got {queries.shape}"
            )
        if not np.isfinite(queries).all():
            raise ValueError("x must be finite")
        leading_shape = queries.shape[:-1]
        return np.ascontiguousarray(queries.reshape(-1, self.m)), leading_shape

    def _brute_query(self, queries, k, p, distance_upper_bound):
        distances = minkowski_distance(
            queries[:, None, :], self.data[None, :, :], p
        )
        order = np.argsort(distances, axis=1, kind="stable")[:, :k]
        selected = np.take_along_axis(distances, order, axis=1)
        invalid = selected > distance_upper_bound
        selected[invalid] = np.inf
        order[invalid] = self.n
        if k > self.n:
            padding = k - self.n
            selected = np.pad(selected, ((0, 0), (0, padding)), constant_values=np.inf)
            order = np.pad(order, ((0, 0), (0, padding)), constant_values=self.n)
        return selected, order

    def query(
        self,
        x,
        k=1,
        eps=0,
        p=2,
        distance_upper_bound=np.inf,
        workers=1,
    ):
        queries, leading_shape = self._validate_queries(x)
        if p < 1:
            raise ValueError("p must be at least 1")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if workers == 0:
            raise ValueError("workers must not be zero")

        if np.isscalar(k):
            requested = None
            if not isinstance(k, (int, np.integer)):
                raise TypeError("k must be an integer or a sequence of integer ranks")
            max_k = int(k)
            if max_k < 1:
                raise ValueError("k must be positive")
        else:
            raw_requested = np.asarray(k)
            if not np.issubdtype(raw_requested.dtype, np.integer):
                raise TypeError("k ranks must be integers")
            requested = np.asarray(raw_requested, dtype=np.int64)
            if requested.ndim != 1 or len(requested) == 0 or np.any(requested < 1):
                raise ValueError("k must contain positive ranks")
            max_k = int(requested.max())

        if p == 2:
            computed_k = min(max_k, self.n)
            if len(queries) == 0:
                distances = np.empty((0, max_k), dtype=np.float64)
                indices = np.empty((0, max_k), dtype=np.int64)
            elif distance_upper_bound < 0:
                distances = np.full((len(queries), max_k), np.inf)
                indices = np.full((len(queries), max_k), self.n, dtype=np.int64)
            else:
                distances = np.empty((len(queries), computed_k), dtype=np.float64)
                indices = np.empty((len(queries), computed_k), dtype=np.int64)
                stack = np.empty(
                    (len(queries), self._stack_stride), dtype=np.int64
                )
                lib().msp_kdtree_query(
                    addr(self.data),
                    addr(queries),
                    addr(self._point_indices),
                    addr(self._axes),
                    addr(self._left),
                    addr(self._right),
                    addr(stack),
                    addr(distances),
                    addr(indices),
                    self.n,
                    self.m,
                    len(queries),
                    computed_k,
                    self._stack_stride,
                    float(distance_upper_bound),
                    float(eps),
                )
                distances[indices == self.n] = np.inf
                np.sqrt(distances, out=distances)
                if max_k > computed_k:
                    padding = max_k - computed_k
                    distances = np.pad(
                        distances, ((0, 0), (0, padding)), constant_values=np.inf
                    )
                    indices = np.pad(
                        indices, ((0, 0), (0, padding)), constant_values=self.n
                    )
        else:
            distances, indices = self._brute_query(
                queries, max_k, p, distance_upper_bound
            )

        if requested is not None:
            distances = distances[:, requested - 1]
            indices = indices[:, requested - 1]
            result_shape = leading_shape + (len(requested),)
        elif max_k == 1:
            distances = distances[:, 0]
            indices = indices[:, 0]
            result_shape = leading_shape
        else:
            result_shape = leading_shape + (max_k,)
        return distances.reshape(result_shape), indices.reshape(result_shape)

    def query_ball_point(
        self,
        x,
        r,
        p=2.0,
        eps=0,
        workers=1,
        return_sorted=None,
        return_length=False,
    ):
        queries, leading_shape = self._validate_queries(x)
        if p < 1:
            raise ValueError("p must be at least 1")
        if eps != 0:
            raise NotImplementedError(
                "approximate query_ball_point searches are not covered"
            )
        if workers == 0:
            raise ValueError("workers must not be zero")
        if np.iscomplexobj(r):
            raise TypeError("complex radii are not supported")
        radii = np.ascontiguousarray(
            np.broadcast_to(np.asarray(r, dtype=float), leading_shape).reshape(-1)
        )
        if len(queries) == 0:
            if return_length:
                return np.empty(leading_shape, dtype=np.int64)
            return np.empty(leading_shape, dtype=object)
        if p == 2:
            counts = np.empty(len(queries), dtype=np.int64)
            stack = np.empty(
                (len(queries), self._stack_stride), dtype=np.int64
            )
            lib().msp_kdtree_query_radius(
                addr(self.data),
                addr(queries),
                addr(self._point_indices),
                addr(self._axes),
                addr(self._left),
                addr(self._right),
                addr(stack),
                addr(counts),
                addr(radii),
                addr(counts),
                addr(counts),
                self.n,
                self.m,
                len(queries),
                self._stack_stride,
                0,
            )
            if return_length:
                return counts.reshape(leading_shape)
            if not counts.any():
                results = [[] for _ in counts]
                if leading_shape == ():
                    return results[0]
                output = np.empty(len(results), dtype=object)
                output[:] = results
                return output.reshape(leading_shape)
            offsets = np.empty(len(queries), dtype=np.int64)
            offsets[0] = 0
            if len(offsets) > 1:
                np.cumsum(counts[:-1], out=offsets[1:])
            indices = np.empty(int(counts.sum()), dtype=np.int64)
            lib().msp_kdtree_query_radius(
                addr(self.data),
                addr(queries),
                addr(self._point_indices),
                addr(self._axes),
                addr(self._left),
                addr(self._right),
                addr(stack),
                addr(counts),
                addr(radii),
                addr(offsets),
                addr(indices),
                self.n,
                self.m,
                len(queries),
                self._stack_stride,
                1,
            )
            results = [
                indices[offset : offset + count].tolist()
                for offset, count in zip(offsets, counts)
            ]
        else:
            results = []
            for query, radius in zip(queries, radii):
                if radius < 0:
                    found = []
                else:
                    all_distances = minkowski_distance(self.data, query, p)
                    found = np.flatnonzero(all_distances <= radius).tolist()
                results.append(found)

        if return_sorted is not False:
            for found in results:
                found.sort()

        if return_length:
            lengths = np.asarray([len(found) for found in results], dtype=np.int64)
            return lengths.reshape(leading_shape)
        if leading_shape == ():
            return results[0]
        output = np.empty(len(results), dtype=object)
        output[:] = results
        return output.reshape(leading_shape)

    def query_ball_tree(self, other, r, p=2.0, eps=0):
        if not isinstance(other, KDTree) or other.m != self.m:
            raise ValueError("other must be a KDTree with matching dimensionality")
        return other.query_ball_point(
            self.data, r, p=p, eps=eps, return_sorted=True
        ).tolist()

    def query_pairs(self, r, p=2.0, eps=0, output_type="set"):
        neighborhoods = self.query_ball_point(
            self.data, r, p=p, eps=eps, return_sorted=True
        )
        pairs = [(i, j) for i, row in enumerate(neighborhoods) for j in row if i < j]
        if output_type == "set":
            return set(pairs)
        if output_type == "ndarray":
            return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        raise ValueError("output_type must be 'set' or 'ndarray'")

    def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
        if weights is not None:
            raise NotImplementedError("weighted pair counting is not covered")
        radii = np.asarray(r)
        scalar = radii.ndim == 0
        radii = np.atleast_1d(radii)
        counts = np.asarray(
            [
                sum(len(row) for row in self.query_ball_tree(other, float(radius), p))
                for radius in radii
            ],
            dtype=np.int64,
        )
        if not cumulative:
            counts[1:] -= counts[:-1].copy()
        return int(counts[0]) if scalar else counts

    def sparse_distance_matrix(
        self, other, max_distance, p=2.0, output_type="dok_matrix"
    ):
        neighborhoods = self.query_ball_tree(other, max_distance, p=p)
        rows, columns, values = [], [], []
        for row, neighbors in enumerate(neighborhoods):
            for column in neighbors:
                rows.append(row)
                columns.append(column)
                values.append(
                    float(minkowski_distance(self.data[row], other.data[column], p))
                )
        if output_type == "dict":
            return dict(zip(zip(rows, columns), values))
        if output_type == "ndarray":
            dtype = [("i", np.intp), ("j", np.intp), ("v", float)]
            return np.array(list(zip(rows, columns, values)), dtype=dtype)
        from scipy.sparse import coo_matrix

        matrix = coo_matrix(
            (values, (rows, columns)), shape=(self.n, other.n), dtype=float
        )
        if output_type == "coo_matrix":
            return matrix
        if output_type == "dok_matrix":
            return matrix.todok()
        raise ValueError("unsupported output_type")


cKDTree = KDTree
