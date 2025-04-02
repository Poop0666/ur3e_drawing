"""
This script is based on the Linedraw project from https://github.com/LingDong-/linedraw.git.
It provides functions to convert images into line drawings.
"""

from PIL import Image, ImageOps
from numpy import array, frombuffer, uint8
from cv2 import Canny, GaussianBlur
import matplotlib.pyplot as plt
try:
    import image_processing.perlin as perlin
except:
    import perlin
try:
    from image_processing.A4_calibration import fit_to_a4
except:
    from A4_calibration import fit_to_a4


def distsum(*args: tuple) -> float:
    """
    Calculate the sum of distances between consecutive points.

    Args:
        *args (tuple): A variable number of tuples, each containing two coordinates.

    Returns:
        float: The sum of the distances between consecutive points.
    """
    return sum(
        [
            ((args[i][0] - args[i - 1][0]) ** 2 + (args[i][1] - args[i - 1][1]) ** 2)
            ** 0.5
            for i in range(1, len(args))
        ]
    )


def visualize(lines: list) -> None:
    """
    Visualize the lines generated from an image.

    Args:
        lines (list): A list of lines, where each line is a list of (x, y) coordinates.
    """
    plt.figure()
    for line in lines:
        x, y = zip(*line)
        plt.plot(x, y, color="black")
    plt.gca().invert_yaxis()
    plt.show()


def sortlines(lines: list, verbose: bool = False) -> list:
    """
    Sort lines to optimize the drawing sequence.

    Args:
        lines (list): A list of lines, where each line is a list of (x, y) coordinates.
        verbose (bool): If True, print progress messages.

    Returns:
        list: A sorted list of lines.
    """
    if verbose:
        print("optimizing stroke sequence...")
    clines = lines[:]
    slines = [clines.pop(0)]
    while clines != []:
        x, s, r = None, 1000000, False
        for l in clines:
            d = distsum(l[0], slines[-1][-1])
            dr = distsum(l[-1], slines[-1][-1])
            if d < s:
                x, s, r = l[:], d, False
            if dr < s:
                x, s, r = l[:], s, True

        clines.remove(x)
        if r == True:
            x = x[::-1]
        slines.append(x)
    return slines


def find_edges(IM: Image, verbose: bool = False) -> Image:
    """
    Detect edges in an image using Gaussian blur and Canny edge detection.

    Args:
        IM (Image): The input image.
        verbose (bool): If True, print progress messages.

    Returns:
        Image: An image with detected edges.
    """
    if verbose:
        print("finding edges...")
    im = array(IM)
    im = GaussianBlur(im, (3, 3), 0)
    im = Canny(im, 100, 200)
    IM = Image.fromarray(im)
    return IM.point(lambda p: p > 128 and 255)


def getdots(IM: Image, verbose: bool = False) -> list:
    """
    Extract contour points from an image.

    Args:
        IM (Image): The input image.
        verbose (bool): If True, print progress messages.

    Returns:
        list: A list of contour points.
    """
    if verbose:
        print("getting contour points...")
    PX = IM.load()
    dots = []
    w, h = IM.size
    for y in range(h - 1):
        row = []
        for x in range(1, w):
            if PX[x, y] == 255:
                if len(row) > 0:
                    if x - row[-1][0] == row[-1][-1] + 1:
                        row[-1] = (row[-1][0], row[-1][-1] + 1)
                    else:
                        row.append((x, 0))
                else:
                    row.append((x, 0))
        dots.append(row)
    return dots


def connectdots(dots: list, verbose: bool = False) -> list:
    """
    Connect contour points to form contours.

    Args:
        dots (list): A list of contour points.
        verbose (bool): If True, print progress messages.

    Returns:
        list: A list of connected contours.
    """
    if verbose:
        print("connecting contour points...")
    contours = []
    for y in range(len(dots)):
        for x, v in dots[y]:
            if v > -1:
                if y == 0:
                    contours.append([(x, y)])
                else:
                    closest = -1
                    cdist = 100
                    for x0, v0 in dots[y - 1]:
                        if abs(x0 - x) < cdist:
                            cdist = abs(x0 - x)
                            closest = x0

                    if cdist > 3:
                        contours.append([(x, y)])
                    else:
                        found = 0
                        for i in range(len(contours)):
                            if contours[i][-1] == (closest, y - 1):
                                contours[i].append(
                                    (
                                        x,
                                        y,
                                    )
                                )
                                found = 1
                                break
                        if found == 0:
                            contours.append([(x, y)])
        for c in contours:
            if c[-1][1] < y - 1 and len(c) < 4:
                contours.remove(c)
    return contours


def getcontours(IM: Image, sc: int = 2, verbose: bool = False) -> list:
    """
    Generate contours from an image.

    Args:
        IM (Image): The input image.
        sc (int): Scale factor for the contours.
        verbose (bool): If True, print progress messages.

    Returns:
        list: A list of contours.
    """
    if verbose:
        print("generating contours...")
    IM = find_edges(IM)
    IM1 = IM.copy()
    IM2 = IM.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    dots1 = getdots(IM1)
    contours1 = connectdots(dots1)
    dots2 = getdots(IM2)
    contours2 = connectdots(dots2)

    for i in range(len(contours2)):
        contours2[i] = [(c[1], c[0]) for c in contours2[i]]
    contours = contours1 + contours2

    for i in range(len(contours)):
        for j in range(len(contours)):
            if len(contours[i]) > 0 and len(contours[j]) > 0:
                if distsum(contours[j][0], contours[i][-1]) < 8:
                    contours[i] = contours[i] + contours[j]
                    contours[j] = []

    for i in range(len(contours)):
        contours[i] = [contours[i][j] for j in range(0, len(contours[i]), 8)]

    contours = [c for c in contours if len(c) > 1]

    for i in range(0, len(contours)):
        contours[i] = [(v[0] * sc, v[1] * sc) for v in contours[i]]

    for i in range(0, len(contours)):
        for j in range(0, len(contours[i])):
            contours[i][j] = int(
                contours[i][j][0] + 10 * perlin.noise(i * 0.5, j * 0.1, 1)
            ), int(contours[i][j][1] + 10 * perlin.noise(i * 0.5, j * 0.1, 2))

    return contours


def hatch(IM: Image, sc=16, verbose=False) -> list:
    """
    Generate hatching lines for an image.

    Args:
        IM (Image): The input image.
        sc (int): Scale factor for the hatching lines.
        verbose (bool): If True, print progress messages.

    Returns:
        list: A list of hatching lines.
    """
    if verbose:
        print("hatching...")
    PX = IM.load()
    w, h = IM.size
    lg1 = []
    lg2 = []
    for x0 in range(w):
        for y0 in range(h):
            x = x0 * sc
            y = y0 * sc
            if PX[x0, y0] > 144:
                pass

            elif PX[x0, y0] > 64:
                lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
            elif PX[x0, y0] > 16:
                lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
                lg2.append([(x + sc, y), (x, y + sc)])

            else:
                lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
                lg1.append([(x, y + sc / 2 + sc / 4), (x + sc, y + sc / 2 + sc / 4)])
                lg2.append([(x + sc, y), (x, y + sc)])

    lines = [lg1, lg2]
    for k in range(0, len(lines)):
        for i in range(0, len(lines[k])):
            for j in range(0, len(lines[k])):
                if lines[k][i] != [] and lines[k][j] != []:
                    if lines[k][i][-1] == lines[k][j][0]:
                        lines[k][i] = lines[k][i] + lines[k][j][1:]
                        lines[k][j] = []
        lines[k] = [l for l in lines[k] if len(l) > 0]
    lines = lines[0] + lines[1]

    for i in range(0, len(lines)):
        for j in range(0, len(lines[i])):
            lines[i][j] = (
                int(lines[i][j][0] + sc * perlin.noise(i * 0.5, j * 0.1, 1)),
                int(lines[i][j][1] + sc * perlin.noise(i * 0.5, j * 0.1, 2)) - j,
            )
    return lines


def sketch(
    IM: Image,
    verbose: bool = False,
    draw_contours: bool = True,
    draw_hatch: bool = True,
    resolution: int = 2048,
    hatch_size: int = 16,
    contour_simplify: int = 2,
) -> list:
    """
    Generate a sketch from an image by combining contours and hatching.

    Args:
        IM (Image): The input image.
        verbose (bool): If True, print progress messages.
        draw_contours (bool): If True, draw contours.
        draw_hatch (bool): If True, draw hatching lines.
        resolution (int): Resolution of the output sketch.
        hatch_size (int): Size of the hatching lines.
        contour_simplify (int): Simplification factor for contours.

    Returns:
        list: A list of lines representing the sketch.
    """
    w, h = IM.size

    IM = IM.convert("L")
    IM = ImageOps.autocontrast(IM, 10)

    lines = []
    if draw_contours:
        lines += getcontours(
            IM.resize(
                (
                    resolution // contour_simplify,
                    resolution // contour_simplify * h // w,
                )
            ),
            contour_simplify,
        )
    if draw_hatch:
        lines += hatch(
            IM.resize((resolution // hatch_size, resolution // hatch_size * h // w)),
            hatch_size,
        )

    lines = sortlines(lines)

    if verbose:
        print(len(lines), "strokes.")
        print("done.")
    return lines


def output(IM: Image, preview: bool = False):
    """
    Generate and optionally preview the output trajectory and number of points from an image.

    Args:
        IM (Image): The input image to be processed.
        preview (bool): If True, generate a preview of the line drawing.

    Returns:
        tuple: A tuple containing:
            - trajectory (str): The trajectory of the lines in the image.
            - nb_points (int): The total number of points in the lines.
            - previsualisation (ndarray, optional): A numpy array representing the preview of the line drawing,
              only returned if preview is True.
    """
    lines = sketch(IM)
    nb_points = sum([len(line) for line in lines])
    trajectory = fit_to_a4(lines, IM.size[1], IM.size[0])
    if preview:
        plt.figure()
        for line in lines:
            x, y = zip(*line)
            plt.plot(x, y, color="black")
        plt.gca().invert_yaxis()
        plt.axis("off")

        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = frombuffer(canvas.tostring_rgb(), dtype=uint8)
        previsualisation = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return trajectory, nb_points, previsualisation
    return trajectory, nb_points
