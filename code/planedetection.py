import numpy as np

def constructplane(points):
    """
    Calculate the equation of a plane given three points in 3D space.
    
    Parameters:
    points (list of np.array): A list of three points, each as a numpy array [x, y, z].
    
    Returns:
    tuple: Coefficients (A, B, C, D) of the plane equation Ax + By + Cz + D = 0.
    """
    # Extract the three points
    pt1, pt2, pt3 = points
    
    # Create vectors from the points
    vec1 = pt2 - pt1
    vec2 = pt3 - pt1
    
    # Compute the normal vector using the cross product
    normal = np.cross(vec1, vec2)
    A, B, C = normal  # Components of the normal vector
    
    # Find D using the point pt1
    D = -np.dot(normal, pt1)
    
    return A, B, C, D


def points_collinear(points):
    """
    Check if three points in 3D space are collinear.
    
    Parameters:
    points (list of np.array): A list of three points, each as a numpy array [x, y, z].
    
    Returns:
    bool: True if the points are collinear, False otherwise.
    """
    # Extract the three points
    pt1, pt2, pt3 = points
    
    # Create vectors from the points
    vec1 = pt2 - pt1
    vec2 = pt3 - pt1
    
    # Compute the cross product of the two vectors
    cross_product = np.cross(vec1, vec2)
    
    # If the cross product is [0, 0, 0], the points are collinear
    return np.all(cross_product == 0)