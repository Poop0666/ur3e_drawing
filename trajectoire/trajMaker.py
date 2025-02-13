
def getTraj(contours: list[list[tuple[int]]], height: int, width: int) -> list[tuple[float]]:
    # 200, 125
    # 400, -125
    # centre: feuille = (300, 0)
    ratioHeight = (210 / height) * 0.8
    rationWidth = (297 / width) * 0.8
    offsetX = 0.350
    offsetY = 0
    z_b = 55 / 1000
    z_h = 70 / 1000
    traj = [[0.344, -0.144, 0.07, 0, 0, 0]]
    x_angle = 0.0
    y_angle = 0.0
    z_angle = 0.0
    
    for contour in contours:
        x = (contour[0][0] / 1000) * ratioHeight + offsetX
        y = (contour[0][1] / 1000) * rationWidth + offsetY
        coordinates = [x, y, z_h, x_angle, y_angle, z_angle]
        traj.append(coordinates)
        
        for point in contour:
            x = (point[0] / 1000) * ratioHeight + offsetX
            y = (point[1] / 1000) * rationWidth + offsetY
            coordinates = [x, y, z_b, x_angle, y_angle, z_angle]
            traj.append(coordinates)
            
        x = (contour[0][0] / 1000) * ratioHeight + offsetX
        y = (contour[0][1] / 1000) * rationWidth + offsetY
        coordinates = [x, y, z_h, x_angle, y_angle, z_angle]
        traj.append(coordinates)
        
    traj.append([0.3, 0, 0.07, x_angle, y_angle, z_angle])
    return traj