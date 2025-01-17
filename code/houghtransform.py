import numpy as np
import rerun as rr
from scipy.spatial import KDTree, ConvexHull
from sklearn.cluster import DBSCAN




def rht_detect_planes(points, params):
    """
    Randomized Hough Transform for plane detection.
    """
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

    # calculate the range if rho
    max_dist = 2.0 * np.max(np.linalg.norm(points, axis=1))
    min_dist = -max_dist
    ins_points_in_plane_count=0

    while np.sum(remaining) > alpha:
        print(f'Remaining points: {np.sum(remaining)}')
        # Initialize accumulator
        accumulator = np.zeros((n_theta, n_phi, n_rho))
        best_planes = {}  # store each bin's best plane param

        # Sampling phase
        n_samples = min(params['n_samples'], N)

        for _ in range(n_samples):
            if np.sum(remaining) < 3:
                break

            # Choose 1 random point
            idx1 = np.random.choice(np.where(remaining)[0], 1)[0]
            p1 = points[idx1]

            # Find neighbors within the radius
            neighbor_indices = pt_in_kdtree.query_ball_point(p1, neighborhood_radius)
            neighbor_indices = [i for i in neighbor_indices if remaining[i] and i != idx1]

            if len(neighbor_indices) < 2:
                continue
            # Select 2 additional random neighbors
            idx2, idx3 = np.random.choice(neighbor_indices, 2, replace=False)
            p2, p3 = points[idx2], points[idx3]

            # calculate the plane's params计算平面参数
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                continue

            normal = normal / norm
            d = np.dot(normal, p1)

            # calculate the coordinate on the sphere
            phi = np.arccos(np.clip(normal[2], -1.0, 1.0))
            theta = np.arctan2(normal[1], normal[0])

            # Calculate accumulator index
            theta_idx = int((theta + np.pi) * (n_theta - 1) / (2 * np.pi))
            phi_idx = int(phi * (n_phi - 1) / np.pi)
            rho_idx = int((d - min_dist) * (n_rho - 1) / (max_dist - min_dist))

            # Check index range and vote
            if (0 <= theta_idx < n_theta and
                    0 <= phi_idx < n_phi and
                    0 <= rho_idx < n_rho):
                accumulator[theta_idx, phi_idx, rho_idx] += 1
                # Storage plane parameters
                key = (theta_idx, phi_idx, rho_idx)
                best_planes[key] = (normal, d)

        # Find the maximum vote
        max_votes = np.max(accumulator)

        # Get the best plane parameters
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        if max_idx not in best_planes:
            continue

        normal, d = best_planes[max_idx]

        # Find the points that belong to this plane
        remaining_points = points[remaining]
        distances = np.abs(np.dot(remaining_points, normal) - d)
        inliers = distances < epsilon

        # Apply neighbor distance filtering
        inlier_points = remaining_points[inliers]

        # Use clustering to filter out overextended regions
        dbscan = DBSCAN(eps=params['max_inlier_distance'], min_samples=3)
        clusters = dbscan.fit_predict(inlier_points)
        largest_cluster = max(
            set(clusters), key=lambda c: np.sum(clusters == c) if c != -1 else 0
        )
        filtered_inliers = (clusters == largest_cluster)

        # Geometric Filtering: Use Convex Hull

        print('test convex hull')
        if np.sum(filtered_inliers) >= 4: # Ensure enough points for ConvexHull
            hull = ConvexHull(inlier_points[filtered_inliers])
            hull_dimensions = np.ptp(hull.points, axis=0)  # Peak-to-peak
            sorted_dimensions = np.sort(hull_dimensions)# distances
            aspect_ratio = sorted_dimensions[2] / sorted_dimensions[1]
            # Check thresholds for validity

            max_hull_aspect_ratio = params.get('max_hull_aspect_ratio')

            if aspect_ratio > max_hull_aspect_ratio:
                print('rejected')
                continue  # Reject this plane


        remaining_idx = np.where(remaining)[0]
        segment_ids[remaining_idx[inliers][filtered_inliers]] = current_segment
        num_points_in_plane = len(remaining_idx[inliers][filtered_inliers])
        if num_points_in_plane<alpha:
            print('not enough points in plane')
            ins_points_in_plane_count+=1
            if ins_points_in_plane_count>=params['max_ins_points_in_plane_count']:
                break
            else:
                continue

        remaining[remaining_idx[inliers][filtered_inliers]] = False
        current_segment += 1
        print('plane detected')



    return segment_ids


def detect(lazfile, params, viz=False):
    """
    !!! TO BE COMPLETED !!!
    !!! You are free to subdivide the functionality of this function into several functions !!!

    Function that detects all the planes in the input LAZ file.

    Inputs:
      lazfile: a laspy input file
      params: a dictionary with all the parameters necessary for the algorithm
      viz: whether the visualiser (rerun, or polyscope) should be displaying results or not

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
