import numpy as np
import pytest
from scipy import spatial as scipy_spatial

import mojo_scipy_spatial as spatial


@pytest.fixture
def point_cloud():
    rng = np.random.default_rng(421)
    return rng.normal(size=(127, 4)), rng.normal(size=(23, 4))


@pytest.mark.parametrize("k", [1, 5, 140, [1, 3, 9]])
def test_query_matches_scipy(point_cloud, k):
    points, queries = point_cloud
    actual = spatial.KDTree(points).query(queries, k=k)
    expected = scipy_spatial.KDTree(points).query(queries, k=k)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_query_preserves_leading_shape_and_scalar(point_cloud):
    points, queries = point_cloud
    tree = spatial.cKDTree(points)
    distances, indices = tree.query(queries[:6].reshape(2, 3, 4), k=2)
    assert distances.shape == (2, 3, 2)
    assert indices.shape == (2, 3, 2)
    scalar_distance, scalar_index = tree.query(queries[0])
    assert np.ndim(scalar_distance) == 0
    assert np.ndim(scalar_index) == 0


def test_query_empty_batch_and_negative_bound_do_not_cross_null_buffers():
    tree = spatial.KDTree(np.eye(3))
    distances, indices = tree.query(np.empty((0, 3)), k=4)
    assert distances.shape == (0, 4)
    assert indices.shape == (0, 4)
    distance, index = tree.query([0, 0, 0], distance_upper_bound=-1)
    assert np.isinf(distance)
    assert index == tree.n
    distance, index = spatial.KDTree([[1.0, 0.0]]).query(
        [0.0, 0.0], distance_upper_bound=1.0
    )
    assert np.isinf(distance)
    assert index == 1


def test_query_simd_tail_and_large_batch():
    rng = np.random.default_rng(330)
    points = rng.random((600, 11))
    queries = rng.random((300, 11))
    actual_tree = spatial.cKDTree(points)
    expected_tree = scipy_spatial.cKDTree(points)

    tail = actual_tree.query(queries[:31], k=4)
    expected_tail = expected_tree.query(queries[:31], k=4)
    np.testing.assert_allclose(tail[0], expected_tail[0], rtol=2e-14)
    np.testing.assert_array_equal(tail[1], expected_tail[1])

    batch = actual_tree.query(queries, k=4)
    expected_batch = expected_tree.query(queries, k=4)
    np.testing.assert_allclose(batch[0], expected_batch[0], rtol=2e-14)
    np.testing.assert_array_equal(batch[1], expected_batch[1])


def test_distance_upper_bound_and_non_euclidean(point_cloud):
    points, queries = point_cloud
    for p in (1, np.inf, 3):
        actual = spatial.KDTree(points).query(
            queries, k=7, p=p, distance_upper_bound=1.8
        )
        expected = scipy_spatial.KDTree(points).query(
            queries, k=7, p=p, distance_upper_bound=1.8
        )
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-13)
        np.testing.assert_array_equal(actual[1], expected[1])


def test_query_ball_point_matches_scipy(point_cloud):
    points, queries = point_cloud
    actual_tree = spatial.KDTree(points)
    expected_tree = scipy_spatial.KDTree(points)
    radii = np.linspace(0.5, 2.0, len(queries))
    actual = actual_tree.query_ball_point(queries, radii)
    expected = expected_tree.query_ball_point(queries, radii)
    assert [sorted(row) for row in actual] == [sorted(row) for row in expected]
    np.testing.assert_array_equal(
        actual_tree.query_ball_point(queries, radii, return_length=True),
        expected_tree.query_ball_point(queries, radii, return_length=True),
    )


def test_query_ball_point_large_counts_and_compact_results():
    rng = np.random.default_rng(331)
    points = rng.random((700, 3))
    queries = rng.random((300, 3))
    radii = np.linspace(0.01, 0.3, len(queries))
    actual_tree = spatial.cKDTree(points)
    expected_tree = scipy_spatial.cKDTree(points)
    actual = actual_tree.query_ball_point(queries, radii)
    expected = expected_tree.query_ball_point(queries, radii)
    assert [sorted(row) for row in actual] == [sorted(row) for row in expected]
    np.testing.assert_array_equal(
        actual_tree.query_ball_point(queries, radii, return_length=True),
        expected_tree.query_ball_point(queries, radii, return_length=True),
    )


def test_pair_and_tree_queries_match_scipy():
    rng = np.random.default_rng(12)
    first = rng.random((40, 3))
    second = rng.random((31, 3))
    actual_first = spatial.KDTree(first)
    actual_second = spatial.KDTree(second)
    expected_first = scipy_spatial.KDTree(first)
    expected_second = scipy_spatial.KDTree(second)
    assert actual_first.query_pairs(0.25) == expected_first.query_pairs(0.25)
    assert actual_first.query_ball_tree(actual_second, 0.3) == (
        expected_first.query_ball_tree(expected_second, 0.3)
    )
    assert actual_first.count_neighbors(actual_second, [0.1, 0.2, 0.4]) == pytest.approx(
        expected_first.count_neighbors(expected_second, [0.1, 0.2, 0.4])
    )
    for output_type in ("dict", "ndarray", "coo_matrix", "dok_matrix"):
        actual = actual_first.sparse_distance_matrix(
            actual_second, 0.3, output_type=output_type
        )
        expected = expected_first.sparse_distance_matrix(
            expected_second, 0.3, output_type=output_type
        )
        if output_type == "dict":
            assert actual == pytest.approx(expected)
        elif output_type == "ndarray":
            assert {
                (int(row["i"]), int(row["j"])): float(row["v"]) for row in actual
            } == pytest.approx(
                {
                    (int(row["i"]), int(row["j"])): float(row["v"])
                    for row in expected
                }
            )
        else:
            np.testing.assert_allclose(actual.toarray(), expected.toarray())


def test_distance_helpers_match_scipy(point_cloud):
    points, queries = point_cloud
    for p in (1, 2, 3, np.inf):
        np.testing.assert_allclose(
            spatial.distance_matrix(points[:9], queries[:7], p=p),
            scipy_spatial.distance_matrix(points[:9], queries[:7], p=p),
        )
        np.testing.assert_allclose(
            spatial.minkowski_distance(points[:7], queries[:7], p=p),
            scipy_spatial.minkowski_distance(points[:7], queries[:7], p=p),
        )
        np.testing.assert_allclose(
            spatial.minkowski_distance_p(points[:7], queries[:7], p=p),
            scipy_spatial.minkowski_distance_p(points[:7], queries[:7], p=p),
        )


def test_validation_errors():
    with pytest.raises(ValueError):
        spatial.KDTree([])
    with pytest.raises(NotImplementedError):
        spatial.KDTree(np.eye(2), boxsize=1)
    tree = spatial.KDTree(np.eye(2))
    with pytest.raises(ValueError):
        tree.query([1, 2, 3])
    with pytest.raises(ValueError):
        tree.query([1, 2], k=0)
    with pytest.raises(TypeError):
        tree.query([1, 2], k=1.5)
    with pytest.raises(TypeError):
        tree.query([1, 2], k=[1.0, 2.0])
    with pytest.raises(NotImplementedError):
        tree.query_ball_point([1, 2], 1, eps=0.1)
