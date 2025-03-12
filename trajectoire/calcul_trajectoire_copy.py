import cv2
import numpy as np
import imutils

try:
    from trajectoire.A4_calibration import fit_to_a4
except:
    from A4_calibration import fit_to_a4


def calcul_trajectoire(
    image: np.ndarray, epsilon=2, method="bluredcanny", show=False
):

    # Put the image in greyscale if it's not the case
    try:
        if len(image.shape) == 2:
            image = cv2.convertScaleAbs(image)

        elif image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    except ValueError as e:
        print(e)
        print(image.shape)
        cv2.imshow("Error Image", image)
        cv2.waitKey()
        
             
    # processes the image using the chosen algorithm
    imgBlur = cv2.GaussianBlur(image,(3,3),0)
    if method == "bluredcanny":
        imgBlur = cv2.GaussianBlur(image, (3, 3), 0)
        chosen = cv2.Canny(imgBlur, 50, 150)
    elif method == "laplacian":
        imgBlur = cv2.GaussianBlur(image, (3, 3), 0)
        chosen = cv2.Laplacian(imgBlur, cv2.CV_64F, ksize=3, scale=1)
    elif method == "sobel":
        x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1)
        y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        _, chosen = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    else:
        chosen = cv2.Canny(image, 50, 150)

    # Close the image
    r = 5
    kernel = np.ones((r, r), np.uint8)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours = cv2.findContours(
        chosen, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Calculate the center of the image
    height, width = chosen.shape
    center_x, center_y = width // 2, height // 2

    # Process each contour and normalize the coordinates
    contours_approx = []
    for contour in contours:
    # Approximation des contours en polygones avec une précision de 2%
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convertir chaque contour approximé en une liste de tuples de points
        contour_points = [(point[0][0] - center_x, - point[0][1] + center_y) for point in approx]
        
        # Ajouter le contour approximé à la liste
        contours_approx.append(contour_points)
        

    # Plot the contours for preview
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each line
    for line in contours_approx:
        x_coords, y_coords = zip(*line)
        ax.plot(x_coords, y_coords)

    # Set the aspect of the plot to be equal, so the drawing isn't distorted
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
    # Save the plot as an image in a variable
    from io import BytesIO

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image_buffer = buffer.getvalue()
    # Close the plot to free up memory
    plt.close(fig)
    np_image = np.frombuffer(image_buffer, dtype=np.uint8)
    preview = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Fit the trajectories to an A4 paper
    points = fit_to_a4(contours_approx)
    nbPoints = len(points)
    nbContours = len(contours_approx)
    return points, nbPoints, nbContours, preview


if __name__ == "__main__":
    # Load the image
    image_path = "image/amongus.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    traj = calcul_trajectoire(image, 5, show=False)
