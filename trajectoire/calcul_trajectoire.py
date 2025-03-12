import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
try:
    from trajectoire.A4_calibration import fit_to_a4
except:
    from A4_calibration import fit_to_a4

def calcul_trajectoire(image : np.ndarray, pointRatio = 10, method = "bluredcanny", show = False, preview = False): 
    
    # Put the image in greyscale if it's not the case
    try:
        if len(image.shape) == 2:
            image = cv2.convertScaleAbs(image)

        
        elif image.shape[2] != 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(image.shape)
            print(image[0])
            print(type(image))
    except ValueError as e:
        print(e)
        print(image.shape)
        cv2.imshow("Error Image",image)
        cv2.waitKey()
        
    if type(pointRatio) is not int:
        pointRatio = int(pointRatio)
    
             
    
    # processes the image using the chosen algorithm
    imgBlur = cv2.GaussianBlur(image,(3,3),0)
    if method == "bluredcanny":
        chosen = cv2.Canny(imgBlur, 50, 150)
    elif method == "laplacian":
        chosen = cv2.Laplacian(imgBlur,cv2.CV_64F, ksize=3, scale=1)
    elif method == "sobel":
        x = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=3, scale=1)
        y = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=3, scale=1)
        absx= cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
        _, chosen = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    else:
        chosen = cv2.Canny(image, 50, 150)
        
    r = 5
    kernel = np.ones((r,r),np.uint8)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(chosen, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of the image
    height, width = chosen.shape
    center_x, center_y = width // 2, height // 2

    # Process each contour and normalize the coordinates
    all_trajectories = []
    for contour in contours:
        trajectory = []
        counter = 0
        for point in contour:
            counter += 1
            x, y = point[0]  # Extract x and y from the contour point
            normalized_x = x - center_x
            normalized_y = center_y - y  # Invert y-axis for Cartesian coordinates
            if counter % pointRatio == 0:
                trajectory.append((normalized_x, normalized_y))
        if trajectory != []:
            all_trajectories.append(trajectory)
            

    if show:
        # Optional: Visualize the trajectories
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for trajectory in all_trajectories:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1)

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title(f"Extracted Trajectories : {method}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.axis('equal')
        plt.show()
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each line
    for line in all_trajectories:
        x_coords, y_coords = zip(*line)
        ax.plot(x_coords, y_coords)

    # Set the aspect of the plot to be equal, so the drawing isn't distorted
    ax.set_aspect('equal')
    ax.axis('off')

    # Save the plot as an image in a variable
    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    image_buffer = buffer.getvalue()
    # Close the plot to free up memory
    plt.close(fig)
    np_image = np.frombuffer(image_buffer, dtype=np.uint8)
    preview = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    
    
    #print(f"Nombre de contours : {len(all_trajectories)}")
    points = fit_to_a4(all_trajectories) #getTraj(all_trajectories, height, width)
    nbPoints = len(points)
    nbContours = len(all_trajectories)
    return points, nbPoints, nbContours, preview


if __name__ == "__main__":
    # Load the image
    image_path = 'image/amongus.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    traj = calcul_trajectoire(image, 5)
    # plt.imshow(traj[3])
    # plt.show()
    #with open("coord.txt", "w") as f:
    #    f.write(str(traj))
    #print(traj)