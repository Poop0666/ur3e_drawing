from cv2 import (
    convertScaleAbs,
    imshow,
    waitKey,
    cvtColor,
    GaussianBlur,
    Canny,
    Laplacian,
    Sobel,
    addWeighted,
    findContours,
    CV_64F,
    COLOR_BGR2GRAY,
    CHAIN_APPROX_SIMPLE,
    contourArea,
    approxPolyDP,
    threshold,
    THRESH_BINARY,
    THRESH_OTSU,
    morphologyEx,
    MORPH_CLOSE,
    RETR_TREE,
    imdecode,
    IMREAD_COLOR,
)
from numpy import (
    transpose,
    ndarray,
    uint8,
    ones,
    frombuffer,
)
from imutils import grab_contours
from math import dist
from shapely.geometry import Polygon

try:
    from image_processing.A4_calibration import fit_to_a4
except:
    from A4_calibration import fit_to_a4

def calculate_area_perimeter_center(coords):
    polygon = Polygon(coords)
    area = polygon.area + 1
    perimeter = polygon.length
    centroid = polygon.centroid
    return area, perimeter, (centroid.x, centroid.y)

def compare_shapes(coords1, coords2, area_tolerance=0.1, perimeter_tolerance=0.1, distance_tolerance=10):
    area1, perimeter1, centroid1 = calculate_area_perimeter_center(coords1)
    area2, perimeter2, centroid2 = calculate_area_perimeter_center(coords2)

    area_similarity = abs(area1 - area2) / max(area1, area2)
    perimeter_similarity = abs(perimeter1 - perimeter2) / max(perimeter1, perimeter2)
    distance_similarity = dist(centroid1, centroid2)

    if area_similarity <= area_tolerance and perimeter_similarity <= perimeter_tolerance and distance_similarity <= distance_tolerance:
        return True
    else:
        return False

def calcul_trajectoire(
    image: ndarray, epsilon=2, method="bluredcanny", show=False
):

    # Put the image in greyscale if it's not the case
    try:
        if len(image.shape) == 2:
            image = convertScaleAbs(image)

        elif image.shape[2] != 1:
            image = cvtColor(image, COLOR_BGR2GRAY)

    except ValueError as e:
        print(e)
        print(image.shape)
        imshow("Error Image", image)
        waitKey()
        
    if image.shape[0] > image.shape[1]:
        image = transpose(image.copy())
        
    # processes the image using the chosen algorithm
    imgBlur = GaussianBlur(image,(3,3),0)
    if method == "bluredcanny":
        imgBlur = GaussianBlur(image, (3, 3), 0)
        chosen = Canny(imgBlur, 50, 150)
    elif method == "laplacian":
        imgBlur = GaussianBlur(image, (3, 3), 0)
        chosen = Laplacian(imgBlur, CV_64F, ksize=3, scale=1)
    elif method == "sobel":
        x = Sobel(image, CV_64F, 1, 0, ksize=3, scale=1)
        y = Sobel(image, CV_64F, 0, 1, ksize=3, scale=1)
        absx = convertScaleAbs(x)
        absy = convertScaleAbs(y)
        edge = addWeighted(absx, 0.5, absy, 0.5, 0)
        _, chosen = threshold(edge, 128, 255, THRESH_BINARY | THRESH_OTSU)

    else:
        chosen = Canny(image, 50, 150)

    # Close the image
    r = 5
    kernel = ones((r, r), uint8)
    chosen = morphologyEx(chosen, MORPH_CLOSE, kernel)

    # Find contours
    contours = findContours(
        chosen, mode=RETR_TREE, method=CHAIN_APPROX_SIMPLE
    )
    
    contours = grab_contours(contours)
    contours = sorted(contours, key = contourArea, reverse = True)

    # Calculate the center of the image
    height, width = chosen.shape
    center_x, center_y = width // 2, height // 2

    # Process each contour and normalize the coordinates
    contours_approx = []
    for contour in contours:
    # Approximation des contours en polygones avec une précision de 2%
        approx = approxPolyDP(contour, epsilon, True)
        
        # Convertir chaque contour approximé en une liste de tuples de points
        contour_points = [(point[0][0] - center_x, - point[0][1] + center_y) for point in approx]
        
        # Ajouter le contour approximé à la liste si il ne ressemble pas à un présent
        flag = True
        if len(contours_approx) != 0:
            for line in contours_approx:
                if len(contour_points) < 4 or compare_shapes(line, contour_points):
                    flag = False
                    break
        if flag:
            contour_points.append(contour_points[0])
            contours_approx.append(contour_points)

        
        

    # Plot the contours for preview
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each line
    for i,line in enumerate(contours_approx):
        x_coords, y_coords = zip(*line)
        ax.plot(x_coords, y_coords)
        

    # Set the aspect of the plot to be equal, so the drawing isn't distorted
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
    # Save the plot as an image in a variable
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image_buffer = buffer.getvalue()
    # Close the plot to free up memory
    plt.close(fig)
    np_image = frombuffer(image_buffer, dtype=uint8)
    preview = imdecode(np_image, IMREAD_COLOR)

    # Fit the trajectories to an A4 paper
    points = fit_to_a4(contours_approx)
    nbPoints = len(points)
    nbContours = len(contours_approx)
    return points, nbPoints, nbContours, preview
