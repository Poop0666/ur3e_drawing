import numpy as np
def fit_to_a4(points, desired_width=189, desired_height=267.3, z=58.5 / 1000):
    """
    Fit a list of points to desired dimensions (90% of A4 by default).

    Parameters:
    - points: List of lists containing (x, y) tuples.
    - desired_width: The desired width to fit the points within.
    - desired_height: The desired height to fit the points within.

    Returns:
    - List of lists with transformed (x, y) tuples.
    """
    # Flatten the list of points and convert to numpy array for easier manipulation
    points_array = np.array([point for sublist in points for point in sublist])

    # Find the bounding box
    min_x, min_y = np.min(points_array, axis=0)
    max_x, max_y = np.max(points_array, axis=0)

    # Calculate the current width and height
    current_width = max_x - min_x
    current_height = max_y - min_y

    # Calculate scale factors
    scale_x = desired_width / current_width
    scale_y = desired_height / current_height
    scale = min(scale_x, scale_y)

    # Scale the points
    scaled_points = points_array * scale

    # Translate the points to center them
    translated_points = (
        scaled_points
        - np.min(scaled_points, axis=0)
        + np.array(
            [
                (
                    desired_width
                    - np.max(scaled_points[:, 0] - np.min(scaled_points[:, 0]))
                )
                / 2,
                (
                    desired_height
                    - np.max(scaled_points[:, 1] - np.min(scaled_points[:, 1]))
                )
                / 2,
            ]
        )
    )
    # Put in m and adds the offset
    translated_points = (translated_points / 1000) + (0.350, 0.0)

    low_z = 58.5 / 1000
    high_z = 70 / 1000

    # Reshape back to the original structure
    reshaped_points = [[0.344, -0.144, 0.07, 0, 0, 0]] # The robot always starts at this point
    index = 0
    for i in range(len(points)):
        # Goes to the first point without touching the paper
        coordinates = [float(translated_points[index][0]), float(translated_points[index][1]), high_z, 0, 0, 0]
        reshaped_points.append(coordinates)
        # Goes through all the points with touching the paper
        for j in range(len(points[i])):
            coordinates = [float(translated_points[index+j][0]), float(translated_points[index+j][1]), low_z, 0, 0, 0]
            reshaped_points.append(coordinates)
        
        # Lifts the pen
        index += len(points[i]) -1
        coordinates = [float(translated_points[index][0]) ,  float(translated_points[index][1]) , high_z, 0, 0, 0]
        reshaped_points.append(coordinates)
        index += 1
    reshaped_points.append([0.3, 0, 0.07, 0, 0, 0])
    return reshaped_points