import time
import rerun as rr
import laspy
import planedetection as pd
from scipy.spatial import KDTree
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


def is_point_on_plane(point_1stplane, points_2ndplane, normal, threshold=0.4):
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



def get_bounding_box(points):# for future need
    """Calculate the axis-aligned bounding box (AABB) for the given points."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords


def merge_similar_planes(plane_segments, p, angular_tolerance=20):
    """
    Merges plane segments that  have similar normal vectors within an angular tolerance and the mean point of one plane lies on the other plane within a given threshold.

    :param plane_segments: List of tuples (plane_id, points_in_plane, plane_normal, centroid)
    :param p: Original point cloud
    :param angular_tolerance: Maximum angular difference (in degrees) for merging plane normals
    :return: Merged point cloud with plane segment IDs
    """

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

            # Get plane points
            plane_i_points = p[plane_segments[i][1]]
            plane_j_points = p[plane_segments[j][1]]

            # Get the correct normal for plane_i (from normals list)
            normal_j = normals[j]

            # Check mean point of plane i
            mean_pt_i = np.mean(plane_i_points, axis=0)

            # Check if mean point of plane i fit into plane j
            pt_in_planefit = is_point_on_plane(mean_pt_i, plane_j_points, normal_j)

            # Check normal similarity using cosine of the angle between normals
            normal_sim = np.dot(normals[i], normals[j]) / (np.linalg.norm(normals[i]) * np.linalg.norm(normals[j]))

            if pt_in_planefit:
                print("bounding_box_overlap")
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
    neighborhood_radius_1stplane = params.get("neighborhood_radius_1stplane")  # Neighborhood radius for selecting the second and third point

    segment_id = 1
    
    plane_segments = []

    # Create KDTree for efficient point queries
    pt_in_kdtree = KDTree(p)
    point_mask = np.ones(p.shape[0], dtype=bool)  # Boolean mask for point availability


    while point_mask.sum() >= 3:  # Continue while enough points are available
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

            # tep 1.3: Ensure at least 2 points for plane selection
            if len(neighbor_indices) < 2:
                continue

            # Step 1.4: Select two more points randomly from the neighborhood
            other_indices = np.random.choice(neighbor_indices, size=2, replace=False)

            # Step 1.5: Form the set of points for plane fitting
            random_indices = np.array([first_index, other_indices[0], other_indices[1]])
            M_randompoints = p[random_indices]

            # Step 2.1: Check if points are collinear
            normal, normal_magnitude, collinear = pd.points_collinear(M_randompoints)

            if not collinear:

                # Step 2.2: Construct plane
                A, B, C, D = pd.constructplane(M_randompoints)
                
                # Step 3.1: Calculate the distance of all points to plane
                distances = np.abs(A * p[:, 0] + B * p[:, 1] + C * p[:, 2] + D) / np.sqrt(A ** 2 + B ** 2 + C ** 2)
                
                # Step 3.2: Calculate indices for points that are closer than epsilon
                valid_indices = np.where((distances < epsilon) & point_mask)[0]

                # Step 4.1: Calculate centroid of created plane
                centroid = np.mean(M_randompoints, axis=0)

                # Step 4.2: Calculate neighbours of centroid
                neighbor_indices = pt_in_kdtree.query_ball_point(centroid, neighborhood_radius_allplanes)
             
                # Step 4.3: Remove from valid_indices the ones that are not in neighbor_indices
                inlier_indices = valid_indices[np.isin(valid_indices, neighbor_indices)]

                # Step 5: Replace the newest plane, if new plane contains more points
                s = len(inlier_indices)
                if s > sbest : 
                    sbest = s
                    best_plane_points = inlier_indices

        # Step 6: Ensure that plane has a minimal number of points (min_score)
        if sbest >= min_score:

            # Step 7: Assign segment ID and mask points
            segmented_points = np.column_stack((p[best_plane_points], np.full(sbest, segment_id)))
            centroid = np.mean(p[best_plane_points], axis=0)
            plane_segments.append((segment_id, best_plane_points, normal, centroid))
            point_mask[best_plane_points] = False
            segment_id += 1
        
        else:
            break
    
    # Step 8: Combine all segmented points
    pts = merge_similar_planes(plane_segments, p)

    # pts = np.vstack(all_segmented_points) if all_segmented_points else np.empty((0, 4))

    # calculate unique segments for visualization
    unique_segments = np.unique(pts[:, 3])

    if viz:
        rr.init("myview", spawn=True)
        rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
        for i2 in unique_segments:
            subset = pts[pts[:, 3] == i2][:, :3]
            rr.log(
                "subset_{}".format(i2),
                rr.Points3D(
                    subset[:],
                    colors=[
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ],
                    radii=0.1,
                ),
            )
            rr.log(
                "logs_{}".format(i2),
                rr.TextLog(
                    "size subset_{}=={}".format(i2, subset.shape[0]),
                    level=rr.TextLogLevel.TRACE,
                ),
            )
            time.sleep(0.5)

    return pts
