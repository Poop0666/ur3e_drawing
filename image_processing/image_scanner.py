from cv2 import (
    cvtColor,
    RETR_TREE,
    COLOR_BGR2GRAY,
    GaussianBlur,
    threshold as cv2threshold,
    THRESH_BINARY,
    THRESH_OTSU,
    findContours,
    approxPolyDP,
    arcLength,
    contourArea,
    RETR_LIST,
    CHAIN_APPROX_SIMPLE,
    namedWindow,
    setMouseCallback,
    imshow,
    waitKey,
    drawContours,
    destroyAllWindows,
    EVENT_LBUTTONDOWN,
    drawMarker,
    LINE_AA,
    MARKER_STAR, VideoCapture, CAP_DSHOW
)
from numpy import array
from imutils.perspective import four_point_transform
from imutils import resize

user_defined_contours = []


def scan_detection(image):
    WIDTH = 1280
    HEIGHT = 720
    global user_defined_contours
    user_defined_contours = []
    document_contour = array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cvtColor(image, COLOR_BGR2GRAY)
    blur = GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2threshold(blur, 0, 255, THRESH_BINARY + THRESH_OTSU)

    contours, _ = findContours(threshold, RETR_TREE, CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=contourArea, reverse=True)[0]
    

    max_area = 0
    for contour in [contours]:
        area = contourArea(contour)
        if area > 1000:
            peri = arcLength(contour, True)
            approx = approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                print(approx)
                approx[0][0][0] += 20
                approx[0][0][1] += 20
                approx[1][0][0] += 20
                approx[1][0][1] -= 20
                approx[2][0][0] -= 20
                approx[2][0][1] -= 20
                approx[3][0][0] -= 20
                approx[3][0][1] += 20
                document_contour = approx
                max_area = area
            else:
                cv2_image = resize(image, height=640)
                namedWindow("Select 4 Points and click on 'X'")
                

                while len(user_defined_contours) != 4:
                    imshow("Select 4 Points and click on 'X'", cv2_image)
                    setMouseCallback(
                    "Select 4 Points and click on 'X'", select_points, cv2_image
                    )
                    waitKey(1)

                destroyAllWindows()

                # Transform the user defined points into a numpy array which openCV expects
                document_contour = array(user_defined_contours)
    return document_contour


def select_points(event, x, y, flags, image):
    """Event Handler for click events which lets the user define 4 points in order to determine the
    object to be scanned when OpenCV itself failed to detect 4 edges
    :param x:  x-coordinate of the clicked point
    :param y:  y-coordinate of the clicked point
    """
    if event == EVENT_LBUTTONDOWN:
        drawMarker(
            image,
            (x, y),
            (0, 0, 255),
            markerType=MARKER_STAR,
            markerSize=10,
            thickness=1,
            line_type=LINE_AA,
        )
    
        global user_defined_contours
        user_defined_contours.append([x, y])


if __name__ == "__main__":
    cap = VideoCapture(1 + CAP_DSHOW)
    _, frame = cap.read()
    document_contour = scan_detection(frame)
    warped = four_point_transform(frame, document_contour.reshape(4, 2))
    imshow("Output", warped)
    waitKey(0)
    destroyAllWindows()
