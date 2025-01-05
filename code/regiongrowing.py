import numpy as np
from scipy.spatial import KDTree


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

    k = params["RegionGrowing"]["k"]
    max_angle = np.radians(params["RegionGrowing"]["max_angle"])
    min_planarity = 0.1

    normals, linearity, planarity, sphericity = compute_normals_and_geometry_features(pts, k)

    seed_indices = select_seed_pts(pts, planarity, linearity, sphericity, min_planarity)

    regions = region_growing(seed_indices, pts, k, max_angle, normals)

    segment_ids = np.zeros(len(pts), dtype=int)
    for i, region in enumerate(regions, start=1):
        segment_ids[region] = i

    result = np.column_stack((pts, segment_ids))

    return result



def get_eigenvalues_eigen
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
            _, indices_large = kdtree.query(pts[i], k=k * 3)
            eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(pts[indices_large])
            normal = eigenvectors[:, 2]

        # edge pt
        elif linearity[i] > 0.8:
            _, indices_large = kdtree.query(pts[i], k=k * 2)
            eigenvalues, eigenvectors = get_eigenvalues_eigenvectors(pts[indices_large])
            normal = eigenvectors[:, 2]

        else:
            normal = eigenvectors[:, 2]

        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals, linearity, planarity, sphericity


def select_seed_pts(pts, planarity, linearity, sphericity, min_planarity = 0.1):
    """choose seed points"""
    mask = (linearity <= 0.8) & (sphericity <= 0.3) & (planarity >= min_planarity)
    seed_indices = np.where(mask)[0]

    print(f"总点数: {len(pts)}")
    print(f"选择的种子点数: {len(seed_indices)}")
    print(f"种子点平面度范围: {planarity[seed_indices].min():.3f} - {planarity[seed_indices].max():.3f}")

    return seed_indices

def get_neighbours_idx(pts, idx, k):
    """find the neighbour index of a point"""
    kdtree = KDTree(pts)
    _, neighbours_idx = kdtree.query(pts[idx].reshape(1, -1), k=k)
    return neighbours_idx[0]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def normal_vector_angle(p1, p2, normals):
    """calculate the normal vector angle of a point and neighbour point """
    p1_normal = normals[p1]
    p2_normal = normals[p2]

    p1_unit = unit_vector(p1)
    p2_unit = unit_vector(p2_normal)

    angle = np.arccos(np.clip(np.dot(p1_unit, p2_unit), -1.0, 1.0))

    return angle


def region_growing(seed_idx, pts, k, max_angle, normals):
    regions = []
    processed_pt = np.zeros(len(pts), dtype=bool)

    for i in seed_idx:
        if processed_pt[i]:
            continue

        s = {i}
        r = {i}
        processed_pt[i] = True

        while s:
            p = s.pop()
            neighbours = get_neighbours_idx(pts, p, k)

            for neighbour_idx in neighbours:
                angle = normal_vector_angle(p, neighbour_idx, normals)
                if angle <= max_angle:
                    s.add(neighbour_idx)
                    r.add(neighbour_idx)
                    processed_pt[neighbour_idx] = True

        regions.append(list(r))

    return regions






