def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """

    sums = [[0.0] * len(points[0]) for _ in range(k)]
    counts = [0] * k

    for i, point in enumerate(points):
        cluster_idx = assignments[i]
        counts[cluster_idx] += 1
        for d in range(len(point)):
            sums[cluster_idx][d] += point[d]

    centroids = []
    for i in range(k):
        if counts[i] > 0:     
            mean = [s / counts[i] for s in sums[i]]
            centroids.append(mean)
        else:
            centroids.append([0.0] * len(points[0]))

    return centroids
