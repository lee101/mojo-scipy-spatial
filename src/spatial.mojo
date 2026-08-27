"""Compute kernels for KD-tree search and planar computational geometry."""

from std.math import abs
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def squared_distance(points: FPtr, point_index: Int, query: FPtr, d: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var j = 0
    var base = point_index * d
    while j + W <= d:
        var delta = points.load[width=W](base + j) - query.load[width=W](j)
        acc += delta * delta
        j += W
    var result = acc.reduce_add()
    while j < d:
        var delta = points[base + j] - query[j]
        result += delta * delta
        j += 1
    return result


def insert_neighbor(
    distances: FPtr, indices: IPtr, k: Int, value: Float64, index: Int, n: Int
):
    if value > distances[k - 1]:
        return
    if value == distances[k - 1]:
        # A sentinel index marks the strict distance_upper_bound itself, while
        # a real index participates in deterministic tie-breaking.
        if Int(indices[k - 1]) == n or Int(indices[k - 1]) <= index:
            return
    var pos = k - 1
    while pos > 0:
        var previous = distances[pos - 1]
        var previous_index = Int(indices[pos - 1])
        if previous < value or (previous == value and previous_index <= index):
            break
        distances[pos] = previous
        indices[pos] = indices[pos - 1]
        pos -= 1
    distances[pos] = value
    indices[pos] = Int64(index)


@export("msp_kdtree_query")
def msp_kdtree_query(
    points_addr: Int,
    queries_addr: Int,
    point_index_addr: Int,
    axis_addr: Int,
    left_addr: Int,
    right_addr: Int,
    stack_addr: Int,
    distances_addr: Int,
    indices_addr: Int,
    n: Int,
    d: Int,
    m: Int,
    k: Int,
    stack_stride: Int,
    upper_bound: Float64,
    eps: Float64,
) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var queries = FPtr(unsafe_from_address=queries_addr)
    var point_index = IPtr(unsafe_from_address=point_index_addr)
    var axes = IPtr(unsafe_from_address=axis_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)
    var distances = FPtr(unsafe_from_address=distances_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)
    var upper2 = upper_bound * upper_bound
    var approximation = (1.0 + eps) * (1.0 + eps)

    @parameter
    def process_query(q: Int):
        var result_distances = distances + q * k
        var result_indices = indices + q * k
        for rank in range(k):
            result_distances[rank] = upper2
            result_indices[rank] = Int64(n)

        var query_stack = stack + q * stack_stride
        var stack_size = 1
        query_stack[0] = Int64(0)
        var query = queries + q * d
        while stack_size > 0:
            stack_size -= 1
            var node = Int(query_stack[stack_size])
            if node < 0:
                continue
            var data_index = Int(point_index[node])
            var distance = squared_distance(points, data_index, query, d)
            insert_neighbor(
                result_distances, result_indices, k, distance, data_index, n
            )

            var axis = Int(axes[node])
            var delta = query[axis] - points[data_index * d + axis]
            var near = Int(left[node])
            var far = Int(right[node])
            if delta > 0.0:
                near = Int(right[node])
                far = Int(left[node])
            if far >= 0 and delta * delta <= result_distances[k - 1] / approximation:
                query_stack[stack_size] = Int64(far)
                stack_size += 1
            if near >= 0:
                query_stack[stack_size] = Int64(near)
                stack_size += 1

    for q in range(m):
        process_query(q)


@export("msp_kdtree_query_radius")
def msp_kdtree_query_radius(
    points_addr: Int,
    queries_addr: Int,
    point_index_addr: Int,
    axis_addr: Int,
    left_addr: Int,
    right_addr: Int,
    stack_addr: Int,
    counts_addr: Int,
    radii_addr: Int,
    offsets_addr: Int,
    indices_addr: Int,
    n: Int,
    d: Int,
    m: Int,
    stack_stride: Int,
    write_indices: Int,
) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var queries = FPtr(unsafe_from_address=queries_addr)
    var point_index = IPtr(unsafe_from_address=point_index_addr)
    var axes = IPtr(unsafe_from_address=axis_addr)
    var left = IPtr(unsafe_from_address=left_addr)
    var right = IPtr(unsafe_from_address=right_addr)
    var stack = IPtr(unsafe_from_address=stack_addr)
    var counts = IPtr(unsafe_from_address=counts_addr)
    var radii = FPtr(unsafe_from_address=radii_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var indices = IPtr(unsafe_from_address=indices_addr)

    @parameter
    def process_query(q: Int):
        var count = 0
        var radius = radii[q]
        if radius < 0.0:
            counts[q] = 0
            return
        var radius2 = radius * radius
        var query_stack = stack + q * stack_stride
        var stack_size = 1
        query_stack[0] = Int64(0)
        var query = queries + q * d
        while stack_size > 0:
            stack_size -= 1
            var node = Int(query_stack[stack_size])
            if node < 0:
                continue
            var data_index = Int(point_index[node])
            if squared_distance(points, data_index, query, d) <= radius2:
                if write_indices != 0:
                    indices[Int(offsets[q]) + count] = Int64(data_index)
                count += 1

            var axis = Int(axes[node])
            var delta = query[axis] - points[data_index * d + axis]
            var near = Int(left[node])
            var far = Int(right[node])
            if delta > 0.0:
                near = Int(right[node])
                far = Int(left[node])
            if far >= 0 and delta * delta <= radius2:
                query_stack[stack_size] = Int64(far)
                stack_size += 1
            if near >= 0:
                query_stack[stack_size] = Int64(near)
                stack_size += 1
        counts[q] = Int64(count)

    for q in range(m):
        process_query(q)


def cross(points: FPtr, a: Int, b: Int, c: Int) -> Float64:
    return (
        (points[b * 2] - points[a * 2])
        * (points[c * 2 + 1] - points[a * 2 + 1])
        - (points[b * 2 + 1] - points[a * 2 + 1])
        * (points[c * 2] - points[a * 2])
    )


@export("msp_convex_hull_2d")
def msp_convex_hull_2d(
    points_addr: Int,
    sorted_indices_addr: Int,
    hull_addr: Int,
    n: Int,
    tolerance: Float64,
) abi("C") -> Int:
    var points = FPtr(unsafe_from_address=points_addr)
    var order = IPtr(unsafe_from_address=sorted_indices_addr)
    var hull = IPtr(unsafe_from_address=hull_addr)
    if n <= 1:
        if n == 1:
            hull[0] = order[0]
        return n

    var size = 0
    for i in range(n):
        var index = Int(order[i])
        while size >= 2 and cross(points, Int(hull[size - 2]), Int(hull[size - 1]), index) <= tolerance:
            size -= 1
        hull[size] = Int64(index)
        size += 1
    var lower_size = size
    for reverse_i in range(n - 1):
        var index = Int(order[n - 2 - reverse_i])
        while size > lower_size and cross(points, Int(hull[size - 2]), Int(hull[size - 1]), index) <= tolerance:
            size -= 1
        hull[size] = Int64(index)
        size += 1
    return size - 1


def set_circumcircle(
    points: FPtr,
    center_x: FPtr,
    center_y: FPtr,
    radius2: FPtr,
    t: Int,
    a: Int,
    b: Int,
    c: Int,
):
    var ax = points[a * 2]
    var ay = points[a * 2 + 1]
    var bx = points[b * 2]
    var by = points[b * 2 + 1]
    var cx = points[c * 2]
    var cy = points[c * 2 + 1]
    var denominator = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(denominator) < 1.0e-30:
        center_x[t] = 0.0
        center_y[t] = 0.0
        radius2[t] = -1.0
        return
    var aa = ax * ax + ay * ay
    var bb = bx * bx + by * by
    var cc = cx * cx + cy * cy
    var ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / denominator
    var uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / denominator
    var dx = ux - ax
    var dy = uy - ay
    center_x[t] = ux
    center_y[t] = uy
    radius2[t] = dx * dx + dy * dy


def add_boundary_edge(edge_u: IPtr, edge_v: IPtr, count: Int, u: Int, v: Int) -> Int:
    for i in range(count):
        if (
            (Int(edge_u[i]) == u and Int(edge_v[i]) == v)
            or (Int(edge_u[i]) == v and Int(edge_v[i]) == u)
        ):
            for j in range(i, count - 1):
                edge_u[j] = edge_u[j + 1]
                edge_v[j] = edge_v[j + 1]
            return count - 1
    edge_u[count] = Int64(u)
    edge_v[count] = Int64(v)
    return count + 1


@export("msp_delaunay_2d")
def msp_delaunay_2d(
    points_addr: Int,
    triangle_a_addr: Int,
    triangle_b_addr: Int,
    triangle_c_addr: Int,
    bad_addr: Int,
    edge_u_addr: Int,
    edge_v_addr: Int,
    center_x_addr: Int,
    center_y_addr: Int,
    radius2_addr: Int,
    n: Int,
    capacity: Int,
) abi("C") -> Int:
    var points = FPtr(unsafe_from_address=points_addr)
    var triangle_a = IPtr(unsafe_from_address=triangle_a_addr)
    var triangle_b = IPtr(unsafe_from_address=triangle_b_addr)
    var triangle_c = IPtr(unsafe_from_address=triangle_c_addr)
    var bad = IPtr(unsafe_from_address=bad_addr)
    var edge_u = IPtr(unsafe_from_address=edge_u_addr)
    var edge_v = IPtr(unsafe_from_address=edge_v_addr)
    var center_x = FPtr(unsafe_from_address=center_x_addr)
    var center_y = FPtr(unsafe_from_address=center_y_addr)
    var radius2 = FPtr(unsafe_from_address=radius2_addr)

    triangle_a[0] = Int64(n)
    triangle_b[0] = Int64(n + 1)
    triangle_c[0] = Int64(n + 2)
    set_circumcircle(points, center_x, center_y, radius2, 0, n, n + 1, n + 2)
    var triangle_count = 1

    for p in range(n):
        var edge_count = 0
        var px = points[p * 2]
        var py = points[p * 2 + 1]
        var t = 0
        while t + W <= triangle_count:
            var dx = center_x.load[width=W](t) - px
            var dy = center_y.load[width=W](t) - py
            var distances2 = dx * dx + dy * dy
            var limits = (
                radius2.load[width=W](t) * (1.0 + 1.0e-12) + 1.0e-24
            )
            comptime for lane in range(W):
                var triangle = t + lane
                var is_bad = distances2[lane] <= limits[lane]
                bad[triangle] = Int64(1 if is_bad else 0)
                if is_bad:
                    edge_count = add_boundary_edge(
                        edge_u,
                        edge_v,
                        edge_count,
                        Int(triangle_a[triangle]),
                        Int(triangle_b[triangle]),
                    )
                    edge_count = add_boundary_edge(
                        edge_u,
                        edge_v,
                        edge_count,
                        Int(triangle_b[triangle]),
                        Int(triangle_c[triangle]),
                    )
                    edge_count = add_boundary_edge(
                        edge_u,
                        edge_v,
                        edge_count,
                        Int(triangle_c[triangle]),
                        Int(triangle_a[triangle]),
                    )
            t += W
        while t < triangle_count:
            var dx = center_x[t] - px
            var dy = center_y[t] - py
            var is_bad = (
                dx * dx + dy * dy
                <= radius2[t] * (1.0 + 1.0e-12) + 1.0e-24
            )
            bad[t] = Int64(1 if is_bad else 0)
            if is_bad:
                edge_count = add_boundary_edge(
                    edge_u, edge_v, edge_count, Int(triangle_a[t]), Int(triangle_b[t])
                )
                edge_count = add_boundary_edge(
                    edge_u, edge_v, edge_count, Int(triangle_b[t]), Int(triangle_c[t])
                )
                edge_count = add_boundary_edge(
                    edge_u, edge_v, edge_count, Int(triangle_c[t]), Int(triangle_a[t])
                )
            t += 1

        var write = 0
        for t in range(triangle_count):
            if bad[t] == 0:
                triangle_a[write] = triangle_a[t]
                triangle_b[write] = triangle_b[t]
                triangle_c[write] = triangle_c[t]
                center_x[write] = center_x[t]
                center_y[write] = center_y[t]
                radius2[write] = radius2[t]
                write += 1
        triangle_count = write

        if triangle_count + edge_count > capacity:
            return -1
        for e in range(edge_count):
            var u = Int(edge_u[e])
            var v = Int(edge_v[e])
            if cross(points, u, v, p) < 0.0:
                var temporary = u
                u = v
                v = temporary
            triangle_a[triangle_count] = Int64(u)
            triangle_b[triangle_count] = Int64(v)
            triangle_c[triangle_count] = Int64(p)
            set_circumcircle(
                points, center_x, center_y, radius2, triangle_count, u, v, p
            )
            triangle_count += 1

    var write = 0
    for t in range(triangle_count):
        if (
            Int(triangle_a[t]) < n
            and Int(triangle_b[t]) < n
            and Int(triangle_c[t]) < n
        ):
            triangle_a[write] = triangle_a[t]
            triangle_b[write] = triangle_b[t]
            triangle_c[write] = triangle_c[t]
            write += 1
    return write


@export("msp_circumcenters_2d")
def msp_circumcenters_2d(
    points_addr: Int,
    triangles_addr: Int,
    centers_addr: Int,
    count: Int,
) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var triangles = IPtr(unsafe_from_address=triangles_addr)
    var centers = FPtr(unsafe_from_address=centers_addr)

    @parameter
    def process_triangle(t: Int):
        var a = Int(triangles[t * 3])
        var b = Int(triangles[t * 3 + 1])
        var c = Int(triangles[t * 3 + 2])
        var ax = points[a * 2]
        var ay = points[a * 2 + 1]
        var bx = points[b * 2]
        var by = points[b * 2 + 1]
        var cx = points[c * 2]
        var cy = points[c * 2 + 1]
        var denominator = 2.0 * (
            ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
        )
        var aa = ax * ax + ay * ay
        var bb = bx * bx + by * by
        var cc = cx * cx + cy * cy
        centers[t * 2] = (
            aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)
        ) / denominator
        centers[t * 2 + 1] = (
            aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)
        ) / denominator

    for t in range(count):
        process_triangle(t)


@export("msp_delaunay_transforms_2d")
def msp_delaunay_transforms_2d(
    points_addr: Int,
    triangles_addr: Int,
    transforms_addr: Int,
    count: Int,
) abi("C"):
    var points = FPtr(unsafe_from_address=points_addr)
    var triangles = IPtr(unsafe_from_address=triangles_addr)
    var transforms = FPtr(unsafe_from_address=transforms_addr)

    @parameter
    def process_triangle(t: Int):
        var a = Int(triangles[t * 3])
        var b = Int(triangles[t * 3 + 1])
        var c = Int(triangles[t * 3 + 2])
        var ax = points[a * 2]
        var ay = points[a * 2 + 1]
        var bx = points[b * 2]
        var by = points[b * 2 + 1]
        var cx = points[c * 2]
        var cy = points[c * 2 + 1]
        var acx = ax - cx
        var acy = ay - cy
        var bcx = bx - cx
        var bcy = by - cy
        var inverse_determinant = 1.0 / (acx * bcy - bcx * acy)
        var base = t * 6
        transforms[base] = bcy * inverse_determinant
        transforms[base + 1] = -bcx * inverse_determinant
        transforms[base + 2] = -acy * inverse_determinant
        transforms[base + 3] = acx * inverse_determinant
        transforms[base + 4] = cx
        transforms[base + 5] = cy

    for t in range(count):
        process_triangle(t)
