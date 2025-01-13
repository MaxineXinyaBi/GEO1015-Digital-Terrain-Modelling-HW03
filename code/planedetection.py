import numpy as np

def constructplane(points):
    """
    Calculate the equation of a plane given three points in 3D space.
    
    Parameters:
    points (numpy.ndarray): A 2D numpy array of shape (3, 3), where each row represents a point [x, y, z].
    
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


def points_collinear(points, threshold=2.0):
    """
    Check if three points in 3D space are collinear within a threshold.

    Parameters:
    points (numpy.ndarray): A 2D numpy array of shape (3, 3),
                             where each row represents a point [x, y, z].
    threshold (float): The collinearity threshold in meters. Default is 1.0.

    Returns:
    normal
    normal magnitude
    bool: True if the points are collinear within the given threshold, False otherwise.
    """
    # Extract the three points
    pt1, pt2, pt3 = points

    # Create vectors from the points
    vec1 = pt2 - pt1
    vec2 = pt3 - pt1

    # Compute the cross product of the two vectors
    normal = np.cross(vec1, vec2)

    # Compute the magnitude of the cross product
    normal_magnitude = np.linalg.norm(normal)

    # Compute the area of the parallelogram formed by vec1 and vec2
    # If the area is less than or equal to the threshold, the points are collinear
    return normal, normal_magnitude, normal_magnitude <= threshold


def calculate_plane_normal(normal, normal_magnitude):
    """
    Calculate the normal of a plane given an array of points.
    Points should be a list or numpy array of shape (n, 3) where n >= 3.

    Returns:
    - is_pointing_up: Boolean indicating if the normal is pointing upwards (Z > 0).
    """

    if normal_magnitude == 0:
        raise ValueError("The given points do not form a valid plane.")

    normal = normal / normal_magnitude

    # Allow a small tilt for considering horizontal (e.g., within 5 degrees)
    tilt_threshold = np.deg2rad(5)  # Threshold in radians
    is_horizontal = np.abs(np.arcsin(normal[0])) < tilt_threshold or np.abs(np.arcsin(normal[1])) < tilt_threshold

    return is_horizontal

def distance_pt_to_plane(A, B, C, D, pt):
    """
    Calculate the perpendicular distance from a point to a plane defined by the equation:
    Ax + By + Cz + D = 0
    
    Parameters:
    A, B, C, D : coefficients of the plane equation
    pt : numpy array representing the point [x, y, z]
    
    Returns:
    float : the perpendicular distance from the point to the plane
    """
    # Calculate the numerator: |Ax + By + Cz + D|
    numerator = abs(A * pt[0] + B * pt[1] + C * pt[2] + D)
    
    # Calculate the denominator: sqrt(A^2 + B^2 + C^2)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    
    # Return the distance
    return numerator / denominator
