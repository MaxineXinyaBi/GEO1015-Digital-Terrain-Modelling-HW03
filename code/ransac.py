import time
import rerun as rr
import laspy
import planedetection as pd
from scipy.spatial import KDTree
import numpy as np



def detect(lazfile, params, viz=False):
    p = lazfile.xyz
    k = params.get("k")
    min_score = params.get("min_score")
    epsilon = params.get("epsilon")
    neighborhood_radius_allplanes = params.get("neighborhood_radius_allplanes") # Neighborhood radius for limiting points that are too far to be added together
    neighborhood_radius_1stplane = params.get("neighborhood_radius_1stplane")  # Neighborhood radius for selecting the second and third point

    segment_id = 1
    all_segmented_points = []

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

            # tep 1.3: Ensure at least 2 points for plane selection
            if len(neighbor_indices) < 2:
                continue

            # Step 1.4: Select two more points randomly from the neighborhood
            other_indices = np.random.choice(neighbor_indices, size=2, replace=False)

            # Step 1.5: Calculate remaining indices
            remaining_indices = np.setdiff1d(available_indices, [first_index, other_indices[0], other_indices[1]])
            if len(remaining_indices) == 0:
                continue

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
            all_segmented_points.append(segmented_points)
            point_mask[best_plane_points] = False
            segment_id += 1
        
        else:
            break
    
    # Step 8: Combine all segmented points
    pts = np.vstack(all_segmented_points) if all_segmented_points else np.empty((0, 4))
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
