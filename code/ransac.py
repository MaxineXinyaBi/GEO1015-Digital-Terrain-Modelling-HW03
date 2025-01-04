import time

import numpy as np
import rerun as rr
import scipy.spatial
import laspy
import planedetection

""" parameters is a dictionary. necessary for this algorithm is: number of iterations k, minimal number of points with epsilon distance min_score, epsilon (distance from point to plane, error threshold) e, 
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
    p = lazfile.xyz
    sbest = 0
    tbest = 0
    
    for i in p:
        M_randompoints = p[np.random.choice(p.shape[0], size=3, replace=False)] 
        T_planecontructed = constructplane(M_randompoints)


    # -- generate random segments and assign them to the points
    segment_ids = np.random.randint(low=0, high=10, size=lazfile.header.point_count)
    pts = np.vstack((lazfile.x, lazfile.y, lazfile.z, segment_ids)).transpose()

    # -- spatially index all the points with a kd-tree
    # -- https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    kdtree = scipy.spatial.KDTree(pts[:, :3])
    re = kdtree.query_ball_point(pts[1, :3], 2.0)
    neighbours = kdtree.data[re]
    print("Neigbours of point #1:\n", neighbours)



    if viz:
        # -- init rerun viewer
        rr.init("myview", spawn=True)
        # -- log all the points
        rr.log("allpts", rr.Points3D(pts[:, :3], colors=[78, 205, 189], radii=0.1))
        # -- log each class one-by-one
        for i in range(10):
            subset = pts[pts[:, 3] == float(i)][:, :3]
            rr.log(
                "subset_{}".format(i),
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
                "logs_{}".format(i),
                rr.TextLog(
                    "size subset_{}=={}".format(i, subset.shape[0]),
                    level=rr.TextLogLevel.TRACE,
                ),
            )
            time.sleep(0.5)

    return pts

