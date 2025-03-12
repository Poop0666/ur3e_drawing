import cv2
import sys
sys.path.append(".")
import image_processing.trajectory_maker as tm

ratio = 2
image = cv2.imread("image/chat.png")
points, nb_point, nb_countour, preview = tm.calcul_trajectoire(image,ratio)
print(f"L'original comporte {nb_point} points et {nb_countour} contours")
cv2.imshow("Original", preview)



cv2.waitKey()