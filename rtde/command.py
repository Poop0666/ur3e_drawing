from rtde.Control import Control
from rtde.Dashboard import Dashboard
import cv2
import logging
from threading import Thread
import time
import os

def sendCoords(points: list):
    con = Control()
    con.sendCoordonates(points)
    
def startDrawing(points: list):
    sCoordsT = Thread(target=sendCoords, args=[points], daemon=True)
    sCoordsT.start()
        
    time.sleep(2)
    dash = Dashboard(os.getenv("HOST"))
    dash.connect()
    
    #Check to see if robot is in remote mode.
    remoteCheck = dash.sendAndReceive('is in remote control')
    if 'false' in remoteCheck:
        logging.warning('Robot is in local mode. Some commands may not function.')
        return "error: local mode"
    else:
        print("The robot will start.")
        
    dash.sendAndReceive("load rtde_control_loop.urp")
    dash.sendAndReceive("play\n")
    sCoordsT.join()
    dash.sendAndReceive("stop\n")
    dash.close()
    return "Dessin terminé"
    
    
if __name__ == "__main__":
    from image_processing.trajectory_maker import trajectory_computation
    image = cv2.imread("image/stick.jpg")
    points = trajectory_computation(image,5)
    startDrawing(points)
    

