import numpy as np
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

    # Save points with segments to a PLY file
    output_file = "output.ply"
    save_to_ply(points_with_segments, output_file)
    print(f"Saved PLY file to {output_file}")

    return points_with_segments
