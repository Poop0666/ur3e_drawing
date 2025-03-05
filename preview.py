import cv2
from trajectoire.calcul_trajectoire import calcul_trajectoire

ratio = 1
image = cv2.imread("image/carre.png")
points, nb_point, preview = calcul_trajectoire(image,ratio, preview=True)
print(f"L'image comporte {nb_point} points")
cv2.imshow("Preview", preview)
cv2.waitKey()