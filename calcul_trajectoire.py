import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcul_trajectoire(image):
    # Preprocess the image (edge detection or thresholding)
    edges = cv2.Canny(image, 50, 150)  # Use Canny edge detection

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Calculate the center of the image
    height, width = edges.shape
    center_x, center_y = width // 2, height // 2

    # Process each contour and normalize the coordinates
    all_trajectories = []
    for contour in contours:
        trajectory = []
        for point in contour:
            x, y = point[0]  # Extract x and y from the contour point
            normalized_x = x - center_x
            normalized_y = center_y - y  # Invert y-axis for Cartesian coordinates
            trajectory.append((normalized_x, normalized_y))
        all_trajectories.append(trajectory)

    # Optional: Visualize the trajectories
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    a = plt.subplot(gs[1])
    for trajectory in all_trajectories:
        trajectory = np.array(trajectory)
        a.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1)

    a.axhline(0, color='black', linewidth=0.5)
    a.axvline(0, color='black', linewidth=0.5)
    a.set_title("Extracted Trajectories")
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.grid()
    a.axis('equal')

    ax2 = plt.subplot(gs[0])
    ax2.imshow(image)
    ax2.axis('off')  # Turn off axis for the image
    ax2.set_title('Image')

    plt.tight_layout()
    plt.show()

    return all_trajectories

if __name__ == "__main__":
    # Load the image
    image_path = 'drawing_example_4.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    traj = calcul_trajectoire(image)
    #print(traj)