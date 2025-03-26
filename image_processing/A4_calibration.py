from numpy import array as nparray, min as npmin, max as npmax


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
    points_array = nparray([point for sublist in points for point in sublist])

    # Find the bounding box
    min_x, min_y = npmin(points_array, axis=0)
    max_x, max_y = npmax(points_array, axis=0)

    # Calculate the current width and height
    current_width = max_x - min_x
    current_height = max_y - min_y

    # Calculate aspect ratios
    current_aspect_ratio = current_width / current_height
    desired_aspect_ratio = desired_width / desired_height

    # Check if the orientation is portrait
    if current_aspect_ratio < desired_aspect_ratio:
        # Swap X and Y coordinates
        points_array[:, [0, 1]] = points_array[:, [1, 0]]
        current_width, current_height = current_height, current_width

    # Calculate scale factors
    scale_x = desired_width / current_width
    scale_y = desired_height / current_height
    scale = min(scale_x, scale_y)

    # Scale the points
    scaled_points = points_array * scale

    # Translate the points to center them
    translated_points = (
        scaled_points
        - npmin(scaled_points, axis=0)
        + nparray(
            [
                (
                    desired_width
                    - npmax(scaled_points[:, 0] - npmin(scaled_points[:, 0]))
                )
                / 2,
                (
                    desired_height
                    - npmax(scaled_points[:, 1] - npmin(scaled_points[:, 1]))
                )
                / 2,
            ]
        )
    )
    # Put in m and adds the offset
    translated_points = (translated_points / 1000) + (0.240, -0.170)

    low_z = 55 / 1000
    high_z = 70 / 1000

    # Reshape back to the original structure
    reshaped_points = [
        [0.344, -0.144, 0.07, 0, 0, 0]
    ]  # The robot always starts at this point
    index = 0
    for i in range(len(points)):
        # Goes to the first point without touching the paper
        coordinates = [
            float(translated_points[index][0]),
            float(translated_points[index][1]),
            high_z,
            0,
            0,
            0,
        ]
        reshaped_points.append(coordinates)
        # Goes through all the points with touching the paper
        for j in range(len(points[i])):
            coordinates = [
                float(translated_points[index + j][0]),
                float(translated_points[index + j][1]),
                low_z,
                0,
                0,
                0,
            ]
            reshaped_points.append(coordinates)

        # Lifts the pen
        index += len(points[i]) - 1
        coordinates = [
            float(translated_points[index][0]),
            float(translated_points[index][1]),
            high_z,
            0,
            0,
            0,
        ]
        reshaped_points.append(coordinates)
        index += 1
    reshaped_points.append([0.3, 0, 0.07, 0, 0, 0])
    return reshaped_points
