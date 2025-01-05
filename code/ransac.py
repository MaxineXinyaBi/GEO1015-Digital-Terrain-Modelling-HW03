import time

import numpy as np
import rerun as rr
from scipy.spatial import KDTree
import laspy
import planedetection as pd


""" parameters is a dictionary. necessary for this algorithm is: number of iterations k, minimal number of points < epsilon distance min_score, epsilon (distance from point to plane, error threshold) e, 
    "k": 15,
    "min_score": 15,
    "epsilon": 0.3

minimal number of points to define a shape - here always 3 for planes n
"""

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

    # Load points and parameters
    p = lazfile.xyz  
    k = params.get("k")
    min_score = params.get("min_score")
    epsilon = params.get("epsilon")

    segment_id = 1
    all_planes = []
    all_segmented_points = []

    while len(p) > 0:  # Repeat until all points are assigned to planes
        sbest = 0
        tbest = ()
        c_score = []

        # Perform k iterations
        for _ in range(k):
            if p.shape[0] < 3:  # Not enough points left to construct a plane #p.shape[0] accesses the first dimension of the array p, which contains the number of rows
                break

            # Step 1: Take three random points
            random_indices = np.random.choice(p.shape[0], size=3, replace=False) # the replace=False argument controls whether the selection of random indices should allow repetition of elements or not.
            M_randompoints = p[random_indices]

            # Step 2: Check collinearity and construct a plane if valid
            if not pd.points_collinear(M_randompoints): 
                A, B, C, D = pd.constructplane(M_randompoints)
                
                # Temporarily remove the three points from p
                remaining_points = np.delete(p, random_indices, axis=0)

                # Step 3: Calculate distance and build c_score
                temp_c_score = []
                for pt in remaining_points:
                    d = pd.distance_pt_to_plane(A, B, C, D, pt) 
                    if d < epsilon:
                        temp_c_score.append(pt)

                # Check if the current plane has the best score
                s = len(temp_c_score)
                if s > sbest:
                    sbest = s
                    tbest = A, B, C, D
                    c_score = temp_c_score  # Save points for this plane

        # Step 4: Assign segment ID and store plane points
        if sbest >= min_score:  # Only accept the plane if it has enough inliers
            # Append segment ID to each point and save them
            c_score_with_id = np.array([np.append(pt, segment_id) for pt in c_score])
            all_segmented_points.append(c_score_with_id)

            # Remove points in c_score from p
            c_score_indices = KDTree(p[:, :3]).query_ball_point(np.array(c_score)[:, :3], r=epsilon)
            c_score_indices = np.unique(np.concatenate(c_score_indices))
            p = np.delete(p, c_score_indices, axis=0)

            # Save the best plane parameters
            all_planes.append(tbest)

            # Increment segment ID
            segment_id += 1
        else:
            # If no valid plane found, break the loop
            break

    # Combine all segmented points into a single array
    pts = np.vstack(all_segmented_points) if all_segmented_points else np.empty((0, 4))


    if viz:
        # -- init rerun viewer
        rr.init("myview", spawn=True)
        # -- log all the points
        rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
        # -- log each class one-by-one
        for i2 in range(10):
            subset = pts[pts[:, 3] == float(i2)][:, :3]
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


    # # -- spatially index all the points with a kd-tree
    # # -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    # # kdtree = scipy.spatial.KDTree(pts[:, :3])
    # # re = kdtree.query_ball_point(pts[1, :3], 2.0)
    # # neighbours = kdtree.data[re]
    # # print("Neigbours of point #1:\n", neighbours)