from scipy.spatial import KDTree
import time
import rerun as rr
from planedetection import *
import numpy as np

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
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z)).transpose()

    k = params["k"]
    max_angle = np.radians(params["max_angle"])
    min_planarity = params["minimum_planarity"]
    min_region_size = params["minimum_region_size"]
    # distance_threshold = params["distance_threshold"]

    # Step 1: Compute geometric features
    normals, linearity, planarity, sphericity = compute_normals_and_geometry_features(pts, k)

    # Step 2: Select seed points
    seed_indices = select_seed_pts(planarity, linearity, sphericity, min_planarity)

    # Step 3: Grow regions from seeds
    regions = region_growing(seed_indices, pts, k, max_angle, normals, min_region_size)

    # Step 4: Initialize segmentation
    segment_ids = np.zeros(len(pts), dtype=int)
    for i, region in enumerate(regions, start=1):
        segment_ids[region] = i

    # Step 5: Compute plane equations
    # plane_equations = region_equation(pts, regions)

    # Step 6: Assign remaining points
    # segment_ids = assign_pts_to_planes(pts, plane_equations, segment_ids, distance_threshold)

    result = np.column_stack((pts, segment_ids))

    # rerun visualize
    if viz:
        # Initialize Rerun viewer
        rr.init("plane_detection", spawn=True)

        # Visualize all points initially
        rr.log("all_points", rr.Points3D(pts, colors=[100, 100, 100], radii=0.1))


        # Visualize segmented planes
        num_segments = len(regions)
        for i in range(1, num_segments + 1):
            # Get points belonging to current segment
            segment_points = pts[segment_ids == i]

            # Generate random color for this segment
            segment_color = [
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ]

            # Log segment points
            rr.log(
                f"plane_segment_{i}",
                rr.Points3D(
                    segment_points,
                    colors=segment_color,
                    radii=0.1
                )
            )

            # Log segment information
            rr.log(
                f"segment_{i}_info",
                rr.TextLog(
                    f"Segment {i} size: {len(segment_points)}",
                    level=rr.TextLogLevel.TRACE
                )
            )


            time.sleep(0.1)

    return result



def get_eigenvalues_eigenvectors(points):
    """get eigenvalues and eigenvectors of a plane"""
    centroid = np.mean(points, axis=0)
    centered_pts = points - centroid
    cov_matrix = np.dot(centered_pts.T, centered_pts)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

def compute_normals_and_geometry_features(pts, k):
    """
       Compute normal vectors and geometric features (linearity, planarity, sphericity)
       for each point using its k nearest neighbors.

       Returns normals and geometric features arrays for all points.
       """
    kdtree = KDTree(pts)
    normals = np.zeros_like(pts)
    linearity = np.zeros(len(pts))
    planarity = np.zeros(len(pts))
    sphericity = np.zeros(len(pts))


    for i in range(len(pts)):
        # get k neighbours of each point
        _, indices = kdtree.query(pts[i], k=k)
        neighbour_pts = pts[indices]

        eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(neighbour_pts)
        lambda1, lambda2, lambda3 = eigenvalues

        if lambda1 != 0:
            linearity[i] = (lambda1 - lambda2) / lambda1
            planarity[i] = (lambda2 - lambda3) / lambda1
            sphericity[i] = lambda3 / lambda1


        normal = eigenvectors[:, 2]

        # flip the normal vector if it is pointing inwards
        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals, linearity, planarity, sphericity


def select_seed_pts(planarity, linearity, sphericity, min_planarity):
    """choose seed points"""
    # good seed point should have least plane fitting error
    mask = (
            (planarity >= min_planarity) &
            (planarity >  linearity) &
            (planarity >  sphericity)

    )
    final_seeds = np.where(mask)[0]

    return final_seeds


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def normal_vector_angle(p1, p2, normals):
    """calculate the angle between normal vectors
    p1: index of the first pt,
    p2: index of the second pt,
    normals: the list to store the pt's normal"""
    p1_normal = normals[p1]
    p2_normal = normals[p2]
    p1_unit = unit_vector(p1_normal)
    p2_unit = unit_vector(p2_normal)
    angle = np.arccos(np.clip(np.abs(np.dot(p1_unit, p2_unit)), -1.0, 1.0))
    return angle


def region_growing(seed_idx, pts, k, max_angle, normals, min_region_size):
    """the main region growing function"""
    print("\n=== Region Growing Debug Info ===")
    print(f"Total points: {len(pts)}")
    print(f"Number of seeds: {len(seed_idx)}")
    print(
        f"Parameters: k={k}, max_angle={np.degrees(max_angle):.2f}°, min_region_size={min_region_size}")
    kdtree = KDTree(pts)
    # initialise the region list
    regions = []
    # record pt that have been processed, the initial status is false
    processed_pt = np.zeros(len(pts), dtype=bool)

    # looping all the seed
    for seed_id in seed_idx:
        # if the seed is already processed, jumping it
        if processed_pt[seed_id]:
            continue
        # add the seed_id into s and r
        # s: the pt whose neighbors still needs to be checked
        # r: pt in current region
        s = {seed_id}
        r = {seed_id}
        while len(s):
            current_pt = s.pop()
            # searching the 20 neighbors of current pt
            distance, neighbor_indices = kdtree.query(pts[current_pt], k)
            # looping every neighbor, if not in other regions, check if it belongs to this region
            for neighbor_id in neighbor_indices:
                if not processed_pt[neighbor_id]:
                    # calculate the angle between normal vectors
                    # with the region's seed
                    # with current pt
                    # distance = np.linalg.norm(pts[neighbor_id] - pts[current_pt])
                    seed_angle = normal_vector_angle(seed_id, neighbor_id, normals)
                    current_angle = normal_vector_angle(current_pt, neighbor_id, normals)
                    # if it is within threshold, update s and r and processing status
                    if seed_angle <= max_angle and current_angle <= max_angle:
                        s.add(neighbor_id)
                        r.add(neighbor_id)
                        processed_pt[neighbor_id] = True

        # Only append regions that meet the minimum size requirement
        if len(r) >= min_region_size:
            regions.append(list(r))

    return regions

# ====== The optional unassigned points reclassification =====
# def select_plane_points(pts,plane_searching_range=5):
#     """select points to construct plane equation"""
#     # the centre of the point cloud
#     kdtree = KDTree(pts)
#     centroid = np.mean(pts, axis=0)
#
#     # the nearest point from centre
#     distances = np.linalg.norm(pts - centroid, axis=1)
#     first_idx = np.argmin(distances)
#     first_pt = pts[first_idx]
#
#     # find all the pts within 5 m distance from the first one
#     neighbour_indices = kdtree.query_ball_point(first_pt, plane_searching_range)
#     neighbour_pts =  pts[neighbour_indices]
#
#     # locate second pt, within 0.5 -2 m to 1st pt
#     distances_to_first = np.linalg.norm(neighbour_pts - first_pt, axis=1)
#     mask_2nd_pt =  (distances_to_first > 0.5) & (distances_to_first < 2)
#     valid_indices = np.where(mask_2nd_pt)[0]
#
#     if len(valid_indices) > 0:
#         second_idx = np.random.choice(valid_indices)
#         second_pt = neighbour_pts[second_idx]
#     else:
#         valid_indices = np.where(distances_to_first > 0.1)[0]
#         second_idx = valid_indices[0]
#         second_pt = neighbour_pts[second_idx]
#
#     # find the 3rd pt 0.5 - 2 m from the pt1 and pt2 line
#     line_vector = second_pt - first_pt
#     line_vector = line_vector / np.linalg.norm(line_vector)
#
#     # calculate the distance from the rest of the pts to line, choose the longest one
#     point_to_line_dist = []
#     for pt in neighbour_pts:
#         point_vector = pt - first_pt
#         dist = np.linalg.norm(np.cross(point_vector, line_vector))
#         if dist < 0.5 or dist > 2.0:
#             dist = 0
#         point_to_line_dist.append(dist)
#
#     third_idx = np.argmax(point_to_line_dist)
#     third_pt = neighbour_pts[third_idx]
#
#     return np.array([first_pt, second_pt, third_pt])
#
#
# def region_equation(pts, regions):
#     """calculate region equation"""
#     plane_equations = {}
#     for i, region in enumerate(regions, start=1):
#         region_pts = pts[region]
#
#         three_points = select_plane_points(region_pts)
#
#         if not points_collinear(three_points):
#             A, B, C, D = constructplane(three_points)
#             plane_equations[i] = (A, B, C, D)
#
#     return plane_equations
#
#
# def assign_pts_to_planes(pts, plane_equations, segment_ids, distance_threshold):
#     """
#     Assign remaining unclassified points to the nearest plane if within threshold.
#
#     Process:
#     1. Find unclassified points (segment_id = 0)
#     2. For each point, calculate distance to all planes
#     3. Assign to nearest plane if distance < threshold
#
#     Returns updated segment_ids array.
#     """
#     unclassified = np.where(segment_ids == 0)[0]
#
#     for pt_idx in unclassified:
#         point = pts[pt_idx]
#         min_dist = float('inf')
#         best_region = None
#
#         for region_id, (A, B, C, D) in plane_equations.items():
#             dist = distance_pt_to_plane(A, B, C, D, point)
#             if dist < distance_threshold and dist < min_dist:
#                 min_dist = dist
#                 best_region = region_id
#
#         if best_region is not None:
#             segment_ids[pt_idx] = best_region
#
#     return segment_ids

