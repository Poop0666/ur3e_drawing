import cv2
import numpy as np
import matplotlib.pyplot as plt
try:
    from trajectoire.trajMaker import getTraj
except:
    from trajMaker import getTraj

def calcul_trajectoire(image : np.ndarray, pointRatio = 10, method = "bluredcanny", show = False, preview = False): 
    
    # Put the image in greyscale if it's not the case
    if image.shape[2] != 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if type(pointRatio) is not int:
        pointRatio = int(pointRatio)
    
             
    # Preprocess the image (edge detection or thresholding)
    canny = cv2.Canny(image, 50, 150)  # Use Canny edge detection
    
    imgBlur = cv2.GaussianBlur(image,(3,3),0)
    laplacian = cv2.Laplacian(imgBlur,cv2.CV_64F, ksize=3, scale=1)
    
    bluredcanny = cv2.Canny(imgBlur, 50, 150)
    
    x = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    _, sobel = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    if show:
        cv2.imshow("Canny", canny)
        cv2.imshow("Laplacian", laplacian)
        cv2.imshow("BluredCanny", bluredcanny)
        cv2.imshow("Sobel", sobel)
        cv2.waitKey(0)
    
    
    if method == "bluredcanny":
        chosen = bluredcanny
    elif method == "laplacian":
        chosen = laplacian
    elif method == "sobel":
        chosen = sobel
    else:
        chosen = canny

    # Find contours
    contours, _ = cv2.findContours(chosen, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

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
        
    if preview:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        for trajectory in all_trajectories:
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1)

        plt.axis("off")
    
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        previsualisation = data.reshape(canvas.get_width_height()[::-1] + (3,))
        
        nbPoints = getTraj(all_trajectories, height, width)
        return nbPoints, previsualisation
        

    return getTraj(all_trajectories, height, width)

if __name__ == "__main__":
    # Load the image
    image_path = 'binary.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    traj = calcul_trajectoire(image, 5, "bluredcanny", show=True)
    #with open("coord.txt", "w") as f:
    #    f.write(str(traj))
    print(traj)