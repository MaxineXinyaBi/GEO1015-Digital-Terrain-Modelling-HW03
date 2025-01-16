import numpy as np
from scipy.spatial import KDTree
import rerun as rr


def simple_dbscan(points, eps, min_samples):
    """
    Simple DBSCAN implementation

    Args:
        points: np.array of shape (N, D) containing N D-dimensional points
        eps: The maximum distance between two samples for them to be considered neighbors
        min_samples: The number of samples in a neighborhood for a point to be considered a core point

    Returns:
        np.array of shape (N,) containing cluster labels (-1 for noise)
    """
    N = len(points)
    labels = np.full(N, -1)
    cluster_id = 0

    # Create KDTree for efficient neighbor searches
    tree = KDTree(points)

    def expand_cluster(point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if labels[point] == -1:
                labels[point] = cluster_id
                new_neighbors = tree.query_ball_point(points[point], eps)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend([n for n in new_neighbors if labels[n] == -1])
            i += 1

    # Find core points and expand clusters
    for i in range(N):
        if labels[i] != -1:
            continue

        neighbors = tree.query_ball_point(points[i], eps)

        if len(neighbors) >= min_samples:
            expand_cluster(i, neighbors, cluster_id)
            cluster_id += 1

    return labels


def rht_detect_planes(points, params):
    # Parameters
    alpha = params['alpha']
    epsilon = params['epsilon']
    neighborhood_radius = params['neighborhood_radius_1stplane']

    # Create KDTree for efficient point queries
    pt_in_kdtree = KDTree(points)

    # Initialize
    N = len(points)
    segment_ids = np.zeros(N, dtype=int)
    current_segment = 1
    remaining = np.ones(N, dtype=bool)

    # Accumulator settings
    n_theta = params['n_theta']  # -π to π
    n_phi = params['n_phi']  # 0 to π
    n_rho = params['n_rho']  # Distance bins

    # Calculate the range for rho
    max_dist = 2.0 * np.max(np.linalg.norm(points, axis=1))
    min_dist = -max_dist

    while np.sum(remaining) > alpha:
        # Initialize accumulator
        accumulator = np.zeros((n_theta, n_phi, n_rho), dtype=int)
        best_planes = {}

        # Sampling phase
        n_samples = min(4000, np.sum(remaining))
        indices = np.random.choice(np.where(remaining)[0], n_samples, replace=False)

        for index in indices:
            p1 = points[index]

            # Find neighbors within the radius
            neighbor_indices = pt_in_kdtree.query_ball_point(p1, neighborhood_radius)
            neighbor_indices = [i for i in neighbor_indices if remaining[i] and i != index]

            if len(neighbor_indices) < 2:
                continue

            # Select 2 additional random neighbors
            idx2, idx3 = np.random.choice(neighbor_indices, 2, replace=False)
            p2, p3 = points[idx2], points[idx3]

            # Calculate the plane's params
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                continue

            normal /= norm
            d = np.dot(normal, p1)

            # Calculate spherical coordinates
            phi = np.arccos(np.clip(normal[2], -1.0, 1.0))
            theta = np.arctan2(normal[1], normal[0])

            # Calculate accumulator index
            theta_idx = int((theta + np.pi) * (n_theta - 1) / (2 * np.pi))
            phi_idx = int(phi * (n_phi - 1) / np.pi)
            rho_idx = int((d - min_dist) * (n_rho - 1) / (max_dist - min_dist))

            # Vote in accumulator
            if (0 <= theta_idx < n_theta and
                    0 <= phi_idx < n_phi and
                    0 <= rho_idx < n_rho):
                accumulator[theta_idx, phi_idx, rho_idx] += 1
                key = (theta_idx, phi_idx, rho_idx)
                best_planes[key] = (normal, d)

        # Find the maximum vote
        max_votes = np.max(accumulator)
        if max_votes < alpha:
            break

        # Get the best plane parameters
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        normal, d = best_planes[max_idx]

        # Find the points that belong to this plane
        remaining_indices = np.where(remaining)[0]
        remaining_points = points[remaining]
        distances = np.abs(np.dot(remaining_points, normal) - d)
        inliers = distances < epsilon

        # Apply clustering to filter inliers using our simple_dbscan
        inlier_points = remaining_points[inliers]
        clusters = simple_dbscan(inlier_points,
                                 eps=params['max_inlier_distance'],
                                 min_samples=3)
        largest_cluster = max(
            set(clusters), key=lambda c: np.sum(clusters == c) if c != -1 else 0
        )
        filtered_inliers = (clusters == largest_cluster)

        # Check and update remaining points
        if np.sum(filtered_inliers) >= alpha:
            segment_ids[remaining_indices[inliers][filtered_inliers]] = current_segment
            remaining[remaining_indices[inliers][filtered_inliers]] = False
            current_segment += 1

    return segment_ids


def detect(lazfile, params, viz=False):
    """
    Detect planes in a LAZ file and assign segment IDs to points.

    Inputs:
      lazfile: a laspy input file
      params: a dictionary with all the parameters necessary for the algorithm
      viz: whether the visualizer (rerun) should display results

    Output:
      - a NumPy array Nx4; each point has x-y-z-segmentid
    """
    # Extract points
    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).T

    # Center points
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # Detect planes
    segment_ids = rht_detect_planes(points_centered, params)

    # Create output array
    points_with_segments = np.column_stack((points, segment_ids))

    # Visualize if requested
    if viz:
        rr.init("Hough Transform Plane Detection", spawn=True)
        rr.log("allpts", rr.Points3D(points, colors=[78, 205, 189], radii=0.1))

        unique_segments = np.unique(segment_ids)
        for seg_id in unique_segments:
            if seg_id == 0:
                continue
            subset = points[segment_ids == seg_id]
            rr.log(
                f"subset_{seg_id}",
                rr.Points3D(
                    subset,
                    colors=[np.random.randint(0, 255) for _ in range(3)],
                    radii=0.1,
                ),
            )

    return points_with_segments
