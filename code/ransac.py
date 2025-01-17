import time
import rerun as rr
import laspy
import planedetection as pd
from scipy.spatial import KDTree, distance
import numpy as np


def best_fitting_plane(points):
    """
    Calculate the best-fitting plane for a given set of points and its normal vector.

    Parameters:
        points (np.ndarray): A Nx3 array of points (x, y, z).

    Returns:
        plane_normal (np.ndarray): A unit vector normal to the best-fitting plane.
        plane_point (np.ndarray): A point on the best-fitting plane (mean of the points).
        plane_equation (tuple): The coefficients (a, b, c, d) of the plane equation ax + by + cz + d = 0.
    """
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points around the centroid
    centered_points = points - centroid

    # Perform Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(centered_points)

    # The normal vector is the last row of vh (or vh.T[:, -1])
    plane_normal = vh[-1, :]

    # Plane equation coefficients (a, b, c)
    a, b, c = plane_normal
    d = -np.dot(plane_normal, centroid)  # Calculate the plane constant

    return plane_normal, centroid, (a, b, c, d)


def is_point_on_plane(point_1stplane, points_2ndplane, normal, threshold=0.5):
    """
        Check if a point lies on a plane within a given threshold.

        Parameters:
            point_1stplane (np.ndarray): The point to check (x, y, z).
            points_2ndplane (np.ndarray): Points defining the second plane.
            normal (np.ndarray): Normal vector of the plane.
            threshold (float): Distance threshold to consider the point on the plane.

        Returns:
            bool: True if the point is within the threshold distance from the plane, False otherwise.
        """
    random_idx_p1 = np.random.choice(len(points_2ndplane))
    p1 = points_2ndplane[random_idx_p1]

    # Calculate the signed distance of the point from the plane
    distance = np.dot(normal, point_1stplane - p1)

    # Check if the distance is within the threshold
    return abs(distance) <= threshold


# for future need
def get_bounding_box(points):
    """Calculate the axis-aligned bounding box (AABB) for the given points."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords


def merge_similar_planes(plane_segments, p, angular_tolerance=20):
    """
    Merges plane segments that are spatially close (based on bounding box overlap) and have similar normal vectors with an angular tolerance.

    :param plane_segments: List of tuples (plane_id, points_in_plane, plane_normal, centroid)
    :param p: Original point cloud
    :param distance_threshold: Maximum allowed bounding box overlap distance for merging
    :param angular_tolerance: Maximum angular difference (in degrees) for merging plane normals
    :return: Merged point cloud with plane segment IDs
    """
    segment_clusters = []

    # Use centroids and normals for clustering
    centroids = [seg[3] for seg in plane_segments]

    points = []
    for seg in plane_segments:
        pts_indexes = seg[1]
        points.append(np.array([p[idx] for idx in pts_indexes]))

    normals =  []

    for pts in points:
        normals.append(np.array(best_fitting_plane(pts)[0]))

    # Manual clustering based on spatial proximity (bounding box) and normal alignment
    cluster_labels = -np.ones(len(centroids), dtype=int)
    current_cluster_id = 0

    # Convert angular tolerance to cosine similarity threshold
    angular_tolerance_radians = np.radians(angular_tolerance)
    normal_similarity_threshold = np.cos(angular_tolerance_radians)


    for i in range(len(centroids)):
        if cluster_labels[i] != -1:
            continue

        cluster_labels[i] = current_cluster_id
        for j in range(i + 1, len(centroids)):
            if cluster_labels[j] != -1:
                continue

            # Get the bounding box for both plane segments
            plane_i_points = p[plane_segments[i][1]]
            plane_j_points = p[plane_segments[j][1]]

            # Get the correct normal for plane_i (from normals list)
            normal_j = normals[j]

            mean_pt_i = np.mean(plane_i_points, axis=0)

            pt_in_planefit = is_point_on_plane(mean_pt_i, plane_j_points, normal_j)

            # Check normal similarity using cosine of the angle between normals
            normal_sim = np.dot(normals[i], normals[j]) / (np.linalg.norm(normals[i]) * np.linalg.norm(normals[j]))

            if pt_in_planefit:
                print("point in plane")
            if np.abs(normal_sim) >= normal_similarity_threshold:
                print("normal condition")
            # Only merge if bounding boxes overlap and normals are similar enough
            if pt_in_planefit and np.abs(normal_sim) >= normal_similarity_threshold:
                print("MERGE!")
                cluster_labels[j] = current_cluster_id

        current_cluster_id += 1

    # Merge plane points based on cluster assignments
    merged_points = []

    for cluster_id in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster_id)[0]
        print(f"Cluster {cluster_id}: Segments = {indices}")
        merged_cluster_points = []

        for idx in indices:
            plane_id, points_in_plane, plane_normal, centroid = plane_segments[idx]
            merged_cluster_points.extend(points_in_plane)

        # Assign a new segment ID
        merged_segment_id = cluster_id + 1
        merged_points.extend(
            np.column_stack((p[merged_cluster_points], np.full(len(merged_cluster_points), merged_segment_id))))

    return np.array(merged_points)


def detect(lazfile, params, viz=False):
    p = lazfile.xyz
    k = params.get("k")
    min_score = params.get("min_score")
    epsilon = params.get("epsilon")
    neighborhood_radius_allplanes = params.get("neighborhood_radius_allplanes") # Neighborhood radius for limiting points that are too far to be added together
    neighborhood_radius_1stplane = params.get("neighborhood_radius_1stplane") # Neighborhood radius for selecting the second and third point

    segment_id = 1
    plane_segments = []

    # Create KDTree for efficient point queries
    pt_in_kdtree = KDTree(p)
    point_mask = np.ones(p.shape[0], dtype=bool) # Boolean mask for point availability

    while point_mask.sum() >= 3: # Continue while enough points are available
        sbest, best_plane_points = 0, None

        # k iterations
        for _ in range(k):
            available_indices = np.where(point_mask)[0]

            if len(available_indices) < 3:
                break

            # Step 1.1: Select the first random point
            first_index = np.random.choice(available_indices)
            first_point = p[first_index]

            # Step 1.2: Find neighborhood points within the specified radius
            neighbor_indices = pt_in_kdtree.query_ball_point(first_point, neighborhood_radius_1stplane)
            neighbor_indices = np.intersect1d(neighbor_indices, available_indices)
            neighbor_indices = np.delete(neighbor_indices, np.where(neighbor_indices == first_index))


            if len(neighbor_indices) < 2:
                continue

            other_indices = np.random.choice(neighbor_indices, size=2, replace=False)

            random_indices = np.array([first_index, other_indices[0], other_indices[1]])
            M_randompoints = p[random_indices]

            normal, normal_magnitude, collinear = pd.points_collinear(M_randompoints)

            if not collinear:
                A, B, C, D = pd.constructplane(M_randompoints)
                centroid = np.mean(M_randompoints, axis=0)
                distances = np.abs(A * p[:, 0] + B * p[:, 1] + C * p[:, 2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)

                neighbor_indices_1 = pt_in_kdtree.query_ball_point(centroid, neighborhood_radius_allplanes)
                valid_indices = np.where((distances < epsilon) & point_mask)[0]

                valid_in_neighbors = valid_indices[np.isin(valid_indices, neighbor_indices_1)]

                s = len(valid_in_neighbors)
                if s > sbest:
                    sbest = s
                    best_plane_points = valid_in_neighbors

        if sbest >= min_score:
            # segmented_points = np.column_stack((p[best_plane_points], np.full(sbest, segment_id)))
            centroid = np.mean(p[best_plane_points], axis=0)
            plane_segments.append((segment_id, best_plane_points, normal, centroid))
            point_mask[best_plane_points] = False
            segment_id += 1
        else:
            break

    pts = merge_similar_planes(plane_segments, p)

    if viz:
        rr.init("myview", spawn=True)
        rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
        unique_segments = np.unique(pts[:, 3])
        for i2 in unique_segments:
            subset = pts[pts[:, 3] == i2][:, :3]
            rr.log(
                f"subset_{i2}",
                rr.Points3D(
                    subset,
                    colors=[np.random.randint(0, 255) for _ in range(3)],
                    radii=0.1,
                ),
            )

    return pts