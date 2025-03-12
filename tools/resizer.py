import cv2
import numpy as np

def binaryResizeA4(image : np.ndarray, seuil = 150):
    x = cv2.Sobel(image, cv2.CV_64F, 1,0, ksize=3, scale=1)
    y = cv2.Sobel(image, cv2.CV_64F, 0,1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    edge = cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)
    thresh = np.where(edge > 10, 255, 0).astype(np.uint8)

    cnts = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
    contours = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        contours.append([w*h, c])

    good = False
    while not good:
        sub = [subcontour[0] for subcontour in contours]
        idx = sub.index(max(sub))
        x,y,w,h = cv2.boundingRect(contours[idx][1])
        if w*h > 0.75*image.shape[0]*image.shape[1]:
            contours.pop(idx)
        else:
            break

    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    new_image = image[y:y+h, x:x+w]
    grey = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # add the white around the page
    for y in range(25):
        for x in range(grey.shape[0]):
            if grey[x][y] < seuil +25:
                grey[x][y] = 255 
            if grey[x][grey.shape[1]-y-1] <= seuil+25:
                grey[x][grey.shape[1]-y-1] = 255
                
    for x in range(25):
        for y in range(grey.shape[1]):
            if grey[x][y] < seuil+25:
                grey[x][y] = 255 
            if grey[grey.shape[0]-x-1][y] <= seuil+25:
                grey[grey.shape[0]-x-1][y] = 255

    ret = np.where(grey > seuil, 255, 0)
    cv2.imwrite("binary.png", ret)
    return ret
    
    
    
if __name__ == "__main__":
    image = cv2.imread("bounce.jpg")
    binaryResizeA4(image, seuil=130)