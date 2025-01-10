import numpy as np
from scipy.spatial import KDTree
import time
import rerun as rr


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
    min_planarity = 0.8

    normals, linearity, planarity, sphericity = compute_normals_and_geometry_features(pts, k)

    seed_indices = select_seed_pts(pts, planarity, linearity, sphericity, min_planarity)

    regions = region_growing(seed_indices, pts, k, max_angle, normals)

    segment_ids = np.zeros(len(pts), dtype=int)
    for i, region in enumerate(regions, start=1):
        segment_ids[region] = i

    segment_ids = assign_uncategorized_points(pts, regions, segment_ids, normals, max_angle, k)

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

            # Optional: visualize normals for this segment
            segment_normals = normals[segment_ids == i]
            rr.log(
                f"normals_segment_{i}",
                rr.Arrows3D(
                    vectors=segment_normals * 0.5,
                    origins=segment_points,
                    colors=segment_color
                )
            )

            time.sleep(0.1)

    return result



def get_eigenvalues_eigenvectors(points):
    centroid = np.mean(points, axis=0)
    centered_pts = points - centroid
    cov_matrix = np.dot(centered_pts.T, centered_pts)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

def compute_normals_and_geometry_features(pts, k):
    """compute the normal and geometry features of each point"""
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

        # corner pt
        if sphericity[i] > 0.3:
            _, indices = kdtree.query(pts[i], k=k * 3)
            eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(pts[indices])
            normal = eigenvectors[:, 2]

        # edge pt
        elif linearity[i] > 0.8:
            _, indices = kdtree.query(pts[i], k=k * 3)
            eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(pts[indices])
            normal = eigenvectors[:, 2]

        else:
            normal = eigenvectors[:, 2]

        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals, linearity, planarity, sphericity


def select_seed_pts(pts, planarity, linearity, sphericity, min_planarity=0.8):
    """choose seed points"""
    mask = (
            (planarity >= min_planarity) &
            (planarity > 2.5 * linearity) &
            (planarity > 2.5 * sphericity) &
            (sphericity < 0.2) &
            (linearity < 0.3)
    )
    final_seeds = np.where(mask)[0]


    print("==> RegionGrowing")
    print(f"总点数: {len(pts)}")
    print(f"选择的种子点数: {len(final_seeds)}")
    print(f"种子点比例: {len(final_seeds) / len(pts) * 100:.2f}%")
    print(f"种子点平面度范围: {planarity[final_seeds].min():.3f} - {planarity[final_seeds].max():.3f}")

    return final_seeds


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def normal_vector_angle(p1, p2, normals):
    """calculate the normal vector angle of a point and neighbour point """
    p1_normal = normals[p1]
    p2_normal = normals[p2]

    p1_unit = unit_vector(p1_normal)
    p2_unit = unit_vector(p2_normal)

    angle = np.arccos(np.clip(np.dot(p1_unit, p2_unit), -1.0, 1.0))

    return angle


def region_growing(seed_idx, pts, k, max_angle, normals):
    kdtree = KDTree(pts)
    regions = []
    processed_pt = np.zeros(len(pts), dtype=bool)
    min_region_size =  int(len(pts) * 0.01)

    for i in seed_idx:
        if processed_pt[i]:
            continue

        s = {i}
        r = {i}
        processed_pt[i] = True

        while s:
            p = s.pop()
            distances, neighbours = kdtree.query(pts[p], k=k)

            for neighbour_idx in neighbours:
                if not processed_pt[neighbour_idx]:
                    seed_angle = normal_vector_angle(i, neighbour_idx, normals)
                    current_angle = normal_vector_angle(p, neighbour_idx, normals)

                    if seed_angle <= max_angle and current_angle <= max_angle:
                        s.add(neighbour_idx)
                        r.add(neighbour_idx)
                        processed_pt[neighbour_idx] = True

        if len(r) >= min_region_size:
            regions.append(list(r))
        else:
            for pt in r:
                processed_pt[pt] = False


    return regions


def assign_uncategorized_points(pts, regions, segment_ids, normals, max_angle, k):
    """ensure every point belongs to a region"""
    unclassified = np.where(segment_ids == 0)[0]
    if len(unclassified) == 0:
        return segment_ids

    region_normals = {}
    for i, region in enumerate(regions, start=1):
        region_normal = np.mean(normals[region], axis=0)
        region_normal = region_normal / np.linalg.norm(region_normal)
        region_normals[i] = region_normal

    kdtree = KDTree(pts)

    for pt_idx in unclassified:
        distances, neighbors = kdtree.query(pts[pt_idx], k=k)
        region_scores = {}
        pt_normal = normals[pt_idx]

        for dist,neighbor_idx  in zip(distances,neighbors):
            if segment_ids[neighbor_idx] == 0:
                continue

            region_id = segment_ids[neighbor_idx]
            if region_id not in region_scores:
                region_scores[region_id] = 0

            region_normal = region_normals[region_id]
            angle = np.arccos(np.clip(np.dot(pt_normal, region_normal), -1.0, 1.0))

            if angle <= max_angle:
                score = (1.0 / (dist + 1e-6)) * (1.0 - angle / max_angle)
                region_scores[region_id] += score

        if region_scores:
            best_region = max(region_scores.items(), key=lambda x: x[1])[0]
            segment_ids[pt_idx] = best_region

    return segment_ids








