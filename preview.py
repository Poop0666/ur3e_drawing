import cv2
import trajectoire.calcul_trajectoire as ori
import trajectoire.calcul_trajectoire_copy as copy

ratio = 2
image = cv2.imread("image/chat.png")
points, nb_point, nb_countour, preview = ori.calcul_trajectoire(image,ratio)
print(f"L'original comporte {nb_point} points et {nb_countour} contours")
cv2.imshow("Original", preview)

points, nb_point, nb_countour, preview = copy.calcul_trajectoire(image,ratio)
print(f"La copy comporte {nb_point} points et {nb_countour} contours")
cv2.imshow("Copy", preview)

cv2.waitKey()