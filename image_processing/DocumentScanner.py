"""
This script is based on the Image2Scan project from https://github.com/Manu10744/image2scan.git.
It provides functions to extract a document from an image.
"""
from cv2 import (
    cvtColor,
    GaussianBlur,
    Canny,
    findContours,
    RETR_LIST,
    CHAIN_APPROX_SIMPLE,
    contourArea,
    arcLength,
    approxPolyDP,
    drawMarker,
    LINE_AA,
    MARKER_STAR,
    setMouseCallback,
    EVENT_LBUTTONDOWN,
    destroyAllWindows,
    imshow,
    waitKey,
    namedWindow,
    COLOR_BGR2GRAY,
    getPerspectiveTransform,
    warpPerspective,
)
from imutils import resize, grab_contours
from numpy import (
    array as nparray,
    argmin as npargmin,
    argmax as npargmax,
    diff as npdiff,
    sqrt as npsqrt,
    zeros as npzeros,
)
from skimage.filters import threshold_local


class ImageScanner:
    """Scanner that applies edge detection in order to scan an ordinary image into a grayscale scan
    while positioning the point of view accordingly if needed."""

    def __init__(self, image, destination):
        """
        :param image: Path to the image to scan
        :param destination:  Path to destination directory to store the scan result in
        :param show_results: Specifies whether to show intermediate results in GUI windows or not
        """
        self.image = image
        self.destination = destination
        self.user_defined_contours = []

    def scan(self):
        """Searches for an rectangular object in the given image and saves the scan result of that object
        in the destination directory as pdf file"""
        screenContours = self.__analyze_contours()
        scan_img = self.__transform_and_scan(screenContours)
        return scan_img
        self.__show_intermediate_result("Scanned Image", scan_img)

    def __analyze_contours(self):
        """Transforms the image colors to black and white in a way so that only the edges become clearly visible."""
        cv2_image = resize(self.image, height=640)

        # Gray the image and detect edges
        grayscaled = cvtColor(cv2_image, COLOR_BGR2GRAY)
        blurred = GaussianBlur(grayscaled, (5, 5), 0)
        edged = Canny(blurred, 75, 200)

        contours = findContours(edged.copy(), RETR_LIST, CHAIN_APPROX_SIMPLE)
        grabbed = grab_contours(contours)
        sortedContours = sorted(grabbed, key=contourArea, reverse=True)[:5]

        screenCnt = None
        for contour in sortedContours:
            peri = arcLength(contour, True)
            approximation = approxPolyDP(contour, 0.02 * peri, True)

            # If approx. contour has four points, then we can assume that we have found the document
            if len(approximation) == 4:
                screenCnt = approximation
                break

        # If OpenCV failed to detect 4 edges, let the user choose 4 points
        if screenCnt is None:
            namedWindow("Select 4 Points and click on 'X'")
            setMouseCallback(
                "Select 4 Points and click on 'X'", self.__select_points, cv2_image
            )

            while len(self.user_defined_contours) != 4:
                imshow("Select 4 Points and click on 'X'", cv2_image)
                waitKey(1)

            destroyAllWindows()

            # Transform the user defined points into a numpy array which openCV expects
            screenCnt = nparray(self.user_defined_contours)

        return screenCnt

    def __select_points(self, event, x, y, flags, image):
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

            self.user_defined_contours.append([x, y])

    def __transform_and_scan(self, screenCnt):
        """Transforms the perspective to a top-down view and creates the scan from the transformed image."""
        ratio = self.image.shape[0] / 640.0
        transformed = self.__four_point_transform(
            self.image, screenCnt.reshape(4, 2) * ratio
        )

        transformed_grayscaled = cvtColor(transformed, COLOR_BGR2GRAY)
        threshold = threshold_local(
            transformed_grayscaled, 11, offset=10, method="gaussian"
        )
        transformed_grayscaled = (transformed_grayscaled > threshold).astype(
            "uint8"
        ) * 255

        return transformed_grayscaled

    def __order_points(self, pts):
        # initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
        rect = npzeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[npargmin(s)]
        rect[2] = pts[npargmax(s)]

        # now, compute the difference between the points, the top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = npdiff(pts, axis=1)
        rect[1] = pts[npargmin(diff)]
        rect[3] = pts[npargmax(diff)]

        # return the ordered coordinates
        return rect

    def __four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them individually
        rect = self.__order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = npsqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = npsqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = npsqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = npsqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct the set of destination points to obtain a
        # "birds eye view",(i.e. top-down view) of the image, again specifying points in the top-left, top-right,
        # bottom-right, and bottom-left order
        dst = nparray(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # compute the perspective transform matrix and then apply it
        M = getPerspectiveTransform(rect, dst)
        warped = warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def __show_intermediate_result(self, title, image):
        """Shows an intermediate image processing step using a GUI window.
        :param title:  The title to use for the GUI window
        :param image:  The image object to display in the GUI window
        """
        imshow(title, resize(image, height=640))
        waitKey(0)
        destroyAllWindows()
