from trajectoire.trajectory_computation import trajectory_computation
from Control import Control
from Dashboard import Dashboard
import cv2
import logging
from threading import Thread
import time


def test(trajectoire):
    con = Control()
    con.sendCoordonates(trajectoire)

def main(filename: str, allProgram = False):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    trajectoire, _, _, _ = trajectory_computation(image,pointRatio=1)
    
    setp1 = [0.250, 0.100, 0.040, 0, 0, 0]
    setp2 = [0.350, 0.100, 0.040, 0, 0, 0]
    setp3 = [0.350, 0, 0.040, 0, 0, 0]
    setp4 = [0.250, 0, 0.040, 0, 0, 0]
    setp = [setp1, setp2, setp3, setp4]
    
    print(trajectoire)
    
    temp = [[0.25, 0.1, 0.04, 0.0, 0.0, 0.0]]
    args = trajectoire if allProgram else setp
    print(f"{args=}")
    
    t1 = Thread(target=test, args=[args], daemon=True)
    t1.run()
    
    time.sleep(2)
    dash = Dashboard('169.254.123.187')
    dash.connect()
    
    #Check to see if robot is in remote mode.
    #remoteCheck = dash.sendAndReceive('is in remote control\n')
    #if 'false' in remoteCheck:
    #    logging.warning('Robot is in local mode. Some commands may not function.')
    dash.sendAndReceive("play\n")
    time.sleep(120)
    dash.close()
    
    
if __name__ == "__main__":
    filename = "image/square.png"
    main(filename,True)
    

