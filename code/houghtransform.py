import numpy as np
<<<<<<< HEAD
import rerun as rr
from scipy.spatial import KDTree, ConvexHull
from sklearn.cluster import DBSCAN


def rht_detect_planes(points, params):
    """
    Randomized Hough Transform for plane detection with improvements.

    Args:
        points (np.ndarray): The input 3D points (Nx3).
        params (dict): Parameters for the Hough Transform algorithm.

    Returns:
        np.ndarray: An array of segment IDs for each point.
    """
    # Parameters
    alpha = params['alpha']
    epsilon = params['epsilon']
    neighborhood_radius = params['neighborhood_radius_1stplane']
    n_theta = params['n_theta']
    n_phi = params['n_phi']
    n_rho = params['n_rho']
    max_inlier_distance = params['max_inlier_distance']
    max_hull_aspect_ratio = params.get('max_hull_aspect_ratio')
    max_ins_points_in_plane_counts = params.get('max_ins_points_in_plane_counts')

    # KDTree for efficient neighbor search
    pt_in_kdtree = KDTree(points)

    # Initialize variables
    N = len(points)
    segment_ids = np.zeros(N, dtype=int)
    current_segment = 1
    remaining = np.ones(N, dtype=bool)

    # Calculate the range for rho
    max_dist = 2.0 * np.max(np.linalg.norm(points, axis=1))
    min_dist = -max_dist
    insufficient_plane_count = 0

    while np.sum(remaining) > alpha:
        remaining_count = np.sum(remaining)
        print(f"Remaining points: {remaining_count}")

        # Stop if remaining points are too few
        if remaining_count < 10:
            print("Remaining points too few, stopping detection.")
            break

        # Initialize accumulator
        accumulator = np.zeros((n_theta, n_phi, n_rho))
        best_planes = {}

        # Sampling phase
        n_samples = min(params['n_samples'], remaining_count)

        for _ in range(n_samples):
            # Random point selection
            idx1 = np.random.choice(np.where(remaining)[0], 1)[0]
            p1 = points[idx1]

            # Find neighbors within the radius
            neighbor_indices = pt_in_kdtree.query_ball_point(p1, neighborhood_radius)
            neighbor_indices = [i for i in neighbor_indices if remaining[i] and i != idx1]

            if len(neighbor_indices) < 2:
                continue

            # Select two additional random neighbors
            idx2, idx3 = np.random.choice(neighbor_indices, 2, replace=False)
            p2, p3 = points[idx2], points[idx3]

            # Calculate plane parameters
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                continue

            normal = normal / norm
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
                best_planes[(theta_idx, phi_idx, rho_idx)] = (normal, d)

        # Find the maximum vote
        max_votes = np.max(accumulator)
        if max_votes < alpha:
            print("Not enough votes, stopping detection.")
            break

        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        if max_idx not in best_planes:
            continue

        normal, d = best_planes[max_idx]

        # Find the points that belong to this plane
        remaining_points = points[remaining]
        distances = np.abs(np.dot(remaining_points, normal) - d)
        inliers = distances < epsilon

        inlier_points = remaining_points[inliers]

        # Apply DBSCAN for clustering
        dbscan = DBSCAN(eps=max_inlier_distance, min_samples=4)
        clusters = dbscan.fit_predict(inlier_points)

        # Select the largest cluster
        if len(set(clusters)) > 1:
            largest_cluster = max(
                set(clusters), key=lambda c: np.sum(clusters == c) if c != -1 else 0
            )
            filtered_inliers = (clusters == largest_cluster)
        else:
            filtered_inliers = (clusters != -1)  # Use all points if no clustering

        # Check if there are enough points for ConvexHull
        if np.sum(filtered_inliers) < 4:
            print("Not enough points for ConvexHull, skipping this plane.")
            continue

        try:
            # Apply ConvexHull filtering
            hull = ConvexHull(inlier_points[filtered_inliers])
            hull_dimensions = np.ptp(hull.points, axis=0)
            sorted_dimensions = np.sort(hull_dimensions)
            aspect_ratio = sorted_dimensions[2] / sorted_dimensions[1]

            if aspect_ratio > max_hull_aspect_ratio:
                print(f"Rejected due to aspect ratio: {aspect_ratio}")
                continue

        except Exception as e:
            print(f"ConvexHull failed: {e}")
            continue

        # Update segment IDs and mark points as processed
        remaining_idx = np.where(remaining)[0]
        segment_ids[remaining_idx[inliers][filtered_inliers]] = current_segment
        num_points_in_plane = len(remaining_idx[inliers][filtered_inliers])

        # Stop if the plane has too few points
        if num_points_in_plane < alpha:
            print("Not enough points in plane, skipping.")
            insufficient_plane_count += 1
            if insufficient_plane_count >= max_ins_points_in_plane_counts:
                print("Too many insufficient planes, stopping detection.")
                break
            else:
                continue

        remaining[remaining_idx[inliers][filtered_inliers]] = False
        current_segment += 1
        print("Plane detected")

    return segment_ids
    Parameters
    
=======
from scipy.spatial import ConvexHull, KDTree
import rerun as rr

# Helper function to compute the accumulator for the Hough Transform
def hough_accumulator(points, params):
    """
    Build the Hough accumulator matrix for plane detection.

    Inputs:
        points: A NumPy array of shape (N, 3) with x, y, z coordinates of the points.
        params: Dictionary with algorithm parameters including alpha (threshold) and epsilon.

    Outputs:
        accumulator: A 3D NumPy array representing the accumulator space.
        resolution: The resolution of the accumulator in each dimension.
    """
    # Define ranges and resolution for accumulator
    r_range = np.linspace(-100, 100, 200)  # Distance to origin
    theta_range = np.deg2rad(np.arange(0, 180, 1))  # Angle to x-axis
    phi_range = np.deg2rad(np.arange(0, 180, 1))    # Angle to z-axis

    accumulator = np.zeros((len(r_range), len(theta_range), len(phi_range)))

    # Fill the accumulator
    for point in points:
        x, y, z = point
        for theta_idx, theta in enumerate(theta_range):
            for phi_idx, phi in enumerate(phi_range):
                r = x * np.cos(theta) * np.sin(phi) + y * np.sin(theta) * np.sin(phi) + z * np.cos(phi)
                r_idx = np.argmin(np.abs(r_range - r))
                accumulator[r_idx, theta_idx, phi_idx] += 1

    return accumulator, (r_range, theta_range, phi_range)

# Helper function to extract planes from the accumulator
def extract_planes(accumulator, thresholds, ranges):
    """
    Extract planes from the Hough accumulator.

    Inputs:
        accumulator: The 3D Hough accumulator array.
        thresholds: Threshold for votes to consider a plane.
        ranges: Tuple with r_range, theta_range, phi_range.

    Outputs:
        planes: List of (r, theta, phi) tuples representing detected planes.
    """
    planes = []
    r_range, theta_range, phi_range = ranges

    indices = np.argwhere(accumulator > thresholds['alpha'])
    for r_idx, theta_idx, phi_idx in indices:
        r = r_range[r_idx]
        theta = theta_range[theta_idx]
        phi = phi_range[phi_idx]
        planes.append((r, theta, phi))

    return planes

def save_to_ply(points_with_segments, output_file):
    """
    Save points with segments to a PLY file.

    Inputs:
        points_with_segments: A NumPy array Nx4 with x, y, z, and segment_id.
        output_file: Path to the output PLY file.
    """
    with open(output_file, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_with_segments.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property int segment_id\n")
        f.write("end_header\n")

        # Write point data
        for point in points_with_segments:
            x, y, z, segment_id = point
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {int(segment_id)}\n")
>>>>>>> region_growing


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
<<<<<<< HEAD

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
=======
    # Extract points from the LAZ file
    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).T

    # Assign random segment IDs for visualization purposes
    segment_ids = np.zeros(points.shape[0], dtype=int)

    # Build a KDTree for nearest neighbor search
    kdtree = KDTree(points)
    for i, point in enumerate(points):
        neighbors_idx = kdtree.query_ball_point(point, r=params.get('radius', 1.0))  # Use radius from params
        neighbors = points[neighbors_idx]
        if len(neighbors) < params.get('min_neighbors', 4):  # Minimum neighbors from params
            continue  # Skip points with insufficient neighbors
        try:
            hull = ConvexHull(neighbors)
            # Assign segment_id based on ConvexHull (example logic for plane fitting)
            if len(hull.vertices) > params.get('min_vertices', 10):  # Minimum vertices from params
                segment_ids[i] = 1  # Assign a non-zero segment ID
        except Exception as e:
            print(f"ConvexHull error at point {i}: {e}")
            continue

    points_with_segments = np.hstack((points, segment_ids[:, np.newaxis]))

    if viz:
        # Initialize rerun viewer
        rr.init("Hough Transform Plane Detection", spawn=True)

        # Log all points
        rr.log("allpts", rr.Points3D(points, colors=[78, 205, 189], radii=0.1))

        # Log each class one-by-one
        unique_segments = np.unique(segment_ids)
        for seg_id in unique_segments:
            if seg_id == 0:
                continue  # Skip unclassified points
            subset = points_with_segments[points_with_segments[:, 3] == seg_id][:, :3]
>>>>>>> region_growing
            rr.log(
                f"subset_{seg_id}",
                rr.Points3D(
                    subset,
<<<<<<< HEAD
                    colors=[np.random.randint(0, 255) for _ in range(3)],
=======
                    colors=[
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ],
>>>>>>> region_growing
                    radii=0.1,
                ),
            )

<<<<<<< HEAD
=======
    # Save points with segments to a PLY file
    output_file = "output.ply"
    save_to_ply(points_with_segments, output_file)
    print(f"Saved PLY file to {output_file}")

>>>>>>> region_growing
    return points_with_segments
