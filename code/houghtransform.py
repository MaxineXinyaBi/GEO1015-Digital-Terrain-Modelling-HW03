import numpy as np
from scipy.spatial import ConvexHull, KDTree
import rerun as rr


# Helper function to extract planes from the accumulator
def extract_planes(accumulator, alpha, ranges):
    """
    Extract planes from the Hough accumulator based on thresholds and ranges.

    Inputs:
        accumulator: The 3D Hough accumulator array.
        alpha: Threshold for votes to consider a plane.
        ranges: Tuple with r_range, theta_range, phi_range.

    Outputs:
        planes: List of (r, theta, phi) tuples representing detected planes.
    """
    planes = []
    r_range, theta_range, phi_range = ranges

    indices = np.argwhere(accumulator > alpha)  # Use alpha threshold
    for r_idx, theta_idx, phi_idx in indices:
        r = r_range[r_idx]
        theta = theta_range[theta_idx]
        phi = phi_range[phi_idx]
        planes.append((r, theta, phi))

    return planes


# Main plane detection function
def detect(lazfile, params, viz=False):
    """
    Detect planes in the input LAZ file using Hough Transform.

    Inputs:
      lazfile: a laspy input file.
      params: a dictionary with all the parameters necessary for the algorithm.
      viz: whether the visualiser (rerun, or polyscope) should display results or not.

    Output:
      - a NumPy array Nx4; each point has x-y-z-segmentid.
    """
    # Extract points from the LAZ file
    points = np.vstack((lazfile.x, lazfile.y, lazfile.z)).T

    # Initialize segment IDs
    segment_ids = np.zeros(points.shape[0], dtype=int)

    # KDTree for nearest neighbor search
    kdtree = KDTree(points)
    radius = params.get('radius')
    alpha = params.get('alpha')  # Voting threshold
    epsilon = params.get('epsilon')  # Plane fitting tolerance

    for i, point in enumerate(points):
        # Find neighbors within the radius
        neighbors_idx = kdtree.query_ball_point(point, r=radius)
        neighbors = points[neighbors_idx]

        if len(neighbors) < params.get('min_neighbors'):
            continue

        try:
            hull = ConvexHull(neighbors)
            if len(hull.vertices) > params.get('min_vertices'):
                # Further refine based on epsilon
                distances = np.abs(np.dot(neighbors - point, hull.equations[:, :-1].T))
                if np.all(distances < epsilon):
                    segment_ids[i] = 1  # Example: Assign a single segment ID
        except Exception as e:
            print(f"ConvexHull error at point {i}: {e}")
            continue

    points_with_segments = np.hstack((points, segment_ids[:, np.newaxis]))

    if viz:
        visualize_with_rerun(points_with_segments, segment_ids)

    return points_with_segments


def visualize_with_rerun(points_with_segments, segment_ids):
    """Visualize points and segments using Rerun."""
    rr.init("Hough Transform Plane Detection", spawn=True)
    rr.log("allpts", rr.Points3D(points_with_segments[:, :3], colors=[78, 205, 189], radii=0.1))

    unique_segments = np.unique(segment_ids)
    for seg_id in unique_segments:
        if seg_id == 0:
            continue
        subset = points_with_segments[points_with_segments[:, 3] == seg_id][:, :3]
        rr.log(
            f"subset_{seg_id}",
            rr.Points3D(
                subset,
                colors=[
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                ],
                radii=0.1,
            ),
        )
