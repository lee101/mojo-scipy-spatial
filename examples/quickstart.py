import numpy as np

from mojo_scipy_spatial import ConvexHull, KDTree, Voronoi

points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.4, 0.6],
    ]
)

tree = KDTree(points)
distances, indices = tree.query([[0.1, 0.2]], k=2)
assert indices.shape == (1, 2)

hull = ConvexHull(points)
assert hull.volume == 1.0

diagram = Voronoi(points)
print(diagram.vertices)
