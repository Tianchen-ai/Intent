import intent
import intent.language as I


POINTS = 65536
CLUSTERS = 64
FEATURES = 64


@intent.kernel
def kmeans_assign(
    points: I.In[I.f16, (POINTS, FEATURES)],
    centroids: I.In[I.f16, (CLUSTERS, FEATURES)],
    assignments: I.Out[I.i32, (POINTS,)],
    distances: I.Out[I.f32, (POINTS,)],
):
    features = I.domain(0, FEATURES)
    for point in I.parallel(I.domain(0, POINTS)):
        best_distance = I.cast(I.inf, I.f32)
        best_cluster = I.cast(0, I.i32)
        point_values = I.cast(points[point, features], I.f32)
        for cluster in range(CLUSTERS):
            difference = point_values - I.cast(
                centroids[cluster, features],
                I.f32,
            )
            distance = I.reduce.sum(
                difference * difference,
                axis=0,
            )
            if distance < best_distance:
                best_distance = distance
                best_cluster = I.cast(cluster, I.i32)
        assignments[point] = best_cluster
        distances[point] = best_distance
