import numpy as np
import pytest
from scipy import spatial as scipy_spatial

import mojo_scipy_spatial as spatial


def canonical_rows(values, decimals=11):
    rounded = np.round(np.asarray(values), decimals)
    return sorted(map(tuple, rounded.tolist()))


def canonical_simplices(values):
    return {tuple(sorted(map(int, row))) for row in values}


def test_convex_hull_known_vector():
    points = np.array(
        [[0, 0], [2, 0], [2, 1], [1, 0.25], [0, 1], [0, 0]], dtype=float
    )
    hull = spatial.ConvexHull(points)
    assert hull.vertices.tolist() == [0, 1, 2, 4]
    assert hull.volume == pytest.approx(2.0)
    assert hull.area == pytest.approx(6.0)
    assert np.all(hull.equations[:, :2] @ points[3] + hull.equations[:, 2] <= 1e-14)
    assert hull.neighbors.shape == hull.simplices.shape
    assert hull.min_bound.tolist() == [0.0, 0.0]
    assert hull.max_bound.tolist() == [2.0, 1.0]


@pytest.mark.parametrize("seed,n", [(1, 25), (19, 100), (44, 300)])
def test_convex_hull_matches_scipy(seed, n):
    points = np.random.default_rng(seed).normal(size=(n, 2))
    actual = spatial.ConvexHull(points)
    expected = scipy_spatial.ConvexHull(points)
    assert set(actual.vertices) == set(expected.vertices)
    assert canonical_simplices(actual.simplices) == canonical_simplices(
        expected.simplices
    )
    assert actual.volume == pytest.approx(expected.volume, rel=2e-14)
    assert actual.area == pytest.approx(expected.area, rel=2e-14)


def test_hull_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        spatial.ConvexHull(np.ones((4, 3)))
    with pytest.raises(spatial.QhullError):
        spatial.ConvexHull([[0, 0], [1, 1], [2, 2]])


@pytest.mark.parametrize("seed,n", [(2, 10), (11, 40), (101, 120)])
def test_delaunay_matches_scipy(seed, n):
    points = np.random.default_rng(seed).random((n, 2))
    actual = spatial.Delaunay(points)
    expected = scipy_spatial.Delaunay(points)
    assert canonical_simplices(actual.simplices) == canonical_simplices(
        expected.simplices
    )
    assert canonical_simplices(actual.convex_hull) == canonical_simplices(
        expected.convex_hull
    )
    probes = np.vstack((points[:8], [[-1, -1], [2, 2], [0.5, 0.5]]))
    np.testing.assert_array_equal(
        actual.find_simplex(probes) >= 0, expected.find_simplex(probes) >= 0
    )


def test_delaunay_transform_metadata_matches_scipy():
    points = np.random.default_rng(332).random((600, 2))
    actual = spatial.Delaunay(points)
    expected = scipy_spatial.Delaunay(points)
    assert canonical_simplices(actual.simplices) == canonical_simplices(
        expected.simplices
    )
    for triangle_index, triangle in enumerate(actual.simplices):
        transform = actual.transform[triangle_index]
        transformed = (
            transform[:2] @ (points[triangle] - transform[2]).T
        ).T
        barycentric = np.column_stack(
            (transformed, 1.0 - transformed.sum(axis=1))
        )
        np.testing.assert_allclose(barycentric, np.eye(3), atol=2e-13)


def test_delaunay_transform_is_barycentric():
    points = np.random.default_rng(7).random((20, 2))
    triangulation = spatial.Delaunay(points)
    for triangle_index, triangle in enumerate(triangulation.simplices):
        transform = triangulation.transform[triangle_index]
        for local_index, point_index in enumerate(triangle):
            first = transform[:2] @ (points[point_index] - transform[2])
            barycentric = np.r_[first, 1 - first.sum()]
            np.testing.assert_allclose(
                barycentric, np.eye(3)[local_index], atol=2e-13
            )
    assert triangulation.neighbors.shape == triangulation.simplices.shape
    assert np.all(triangulation.vertex_to_simplex >= 0)


@pytest.mark.parametrize("seed,n", [(9, 12), (77, 50), (123, 100)])
def test_voronoi_matches_scipy(seed, n):
    points = np.random.default_rng(seed).random((n, 2))
    actual = spatial.Voronoi(points)
    expected = scipy_spatial.Voronoi(points)
    assert canonical_rows(actual.vertices) == canonical_rows(expected.vertices)
    assert canonical_simplices(actual.ridge_points) == canonical_simplices(
        expected.ridge_points
    )
    actual_unbounded = {
        tuple(sorted(map(int, edge)))
        for edge, vertices in zip(actual.ridge_points, actual.ridge_vertices)
        if -1 in vertices
    }
    expected_unbounded = {
        tuple(sorted(map(int, edge)))
        for edge, vertices in zip(expected.ridge_points, expected.ridge_vertices)
        if -1 in vertices
    }
    assert actual_unbounded == expected_unbounded


def test_voronoi_triangle_and_cocircular_square_vectors():
    triangle = np.array([[0.0, 0.0], [2, 0], [0, 2]])
    diagram = spatial.Voronoi(triangle)
    np.testing.assert_allclose(diagram.vertices, [[1, 1]])
    assert len(diagram.ridge_points) == 3
    assert all(-1 in ridge for ridge in diagram.ridge_vertices)

    square = np.array([[0.0, 0.0], [1, 0], [1, 1], [0, 1]])
    diagram = spatial.Voronoi(square)
    np.testing.assert_allclose(diagram.vertices, [[0.5, 0.5]])
    assert canonical_simplices(diagram.ridge_points) == {
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
    }
    assert len(diagram.regions) == len(square)
    np.testing.assert_array_equal(diagram.point_region, np.arange(len(square)))
    np.testing.assert_array_equal(diagram.min_bound, [0.0, 0.0])
    np.testing.assert_array_equal(diagram.max_bound, [1.0, 1.0])


def test_geometry_rejects_unsupported_modes():
    points = np.array([[0.0, 0.0], [1, 0], [0, 1]])
    with pytest.raises(NotImplementedError):
        spatial.ConvexHull(points, incremental=True)
    with pytest.raises(NotImplementedError):
        spatial.Voronoi(points, furthest_site=True)
