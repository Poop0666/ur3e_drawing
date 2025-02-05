import cv2

class Imagegrabber:
    def __init__(self, caminput: int) -> None:
        self.camera = cv2.VideoCapture(caminput, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.basename = "screenshot"
    
    def changeName(self, name: str) -> None:
        self.basename = name
    
    def takescreenshot(self, name = None) -> None:
        if name is None:
            name = self.basename
        
        _, image = self.camera.read()
        cv2.imwrite(f"{name}.png", image)