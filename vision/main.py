from imagegrabber import Imagegrabber
import numpy as np
import scipy as sp; from scipy import ndimage
import imageio
import skimage as sk;from skimage import measure,filters, exposure
import matplotlib.pyplot as plt
import os
import cv2


def main():
    grab = Imagegrabber(1)
    grab.takescreenshot()

def mainreco():
    image2base = imageio.imread("screenshot.png")
    image = cv2.cvtColor(image2base, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",image)
    #cv2.waitKey(0)
    equalSeuil = filters.threshold_otsu(image) +60
    thresholded = np.where(image>equalSeuil,255,0).astype('uint8')
    #print(thresholded)
    #cv2.imshow("thr",thresholded)
    #cv2.waitKey(0)
    imageio.imwrite("thr.png",thresholded)

    connect8 = np.ones((3, 3))
    Ic8, max8 = sp.ndimage.label(thresholded, connect8)
    histo, bin = exposure.histogram(Ic8)
    idx = np.where(histo == max(histo[1:]))
    feuille = np.where(Ic8 == idx[0][0], 255,0)
    mask = np.ones((50,50))
    feuille = ndimage.binary_closing(feuille, structure=mask)
    #cv2.imshow("feuille",np.where(feuille == 1, 255,0).astype('uint8'))
    #cv2.waitKey(0)
    
    dessin = cv2.bitwise_and(image2base,image2base, mask=feuille.astype('uint8'))
    #_, f2 = cv2.threshold(f0, 255, 0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dessin.astype('uint8'), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    print(hierarchy)
    cv2.imshow("dessin",dessin.astype('uint8'))
    cv2.waitKey(0)
    
    


if __name__ == "__main__":
    #main()
    mainreco()