import os
os.environ["OPENCV_LOG_LEVEL"]="SILENT"
import cv2

def get_cameras():
    """
        Returns an array containing the indices of existing cameras.
    """
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

if __name__ == "__main__":
    print(get_cameras())
    