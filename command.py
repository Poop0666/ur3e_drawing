from Control import Control
from Dashboard import Dashboard
import cv2
import logging
from threading import Thread
import time

def sendCoords(points: list):
    con = Control()
    con.sendCoordonates(points)
    
def startDrawing(points: list):
    sCoordsT = Thread(target=sendCoords, args=[points], daemon=True)
    sCoordsT.start()
        
    time.sleep(2)
    dash = Dashboard('169.254.123.187')
    dash.connect()
    
    #Check to see if robot is in remote mode.
    remoteCheck = dash.sendAndReceive('is in remote control')
    if 'false' in remoteCheck:
        logging.warning('Robot is in local mode. Some commands may not function.')
    else:
        print("The robot will start.")
        
    dash.sendAndReceive("load rtde_control_loop.")
    dash.sendAndReceive("play\n")
    sCoordsT.join()
    dash.sendAndReceive("stop\n")
    dash.close()
    print("Dessin terminé")
    
    
if __name__ == "__main__":
    from ImageProcessing.calcul_trajectoire import calcul_trajectoire
    image = cv2.imread("image/stick.jpg")
    points = calcul_trajectoire(image,5)
    startDrawing(points)
    

