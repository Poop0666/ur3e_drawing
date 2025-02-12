import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import threading
from tooltip import Tooltip
import customtkinter as ctk
import numpy as np
from PIL import Image
from cameras import get_cameras
from detection_dessin import scan
from trajectoire.calcul_trajectoire import calcul_trajectoire
import cProfile, pstats

class VideoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Drawing with UR3E")
        self.geometry("1280x720")

        # Configure Grid
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------- Left Side (Live Video) -------- #
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.preview_frame, text="", font=('Arial', 50))
        self.video_label.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        # -------- Right Side (Controls) -------- #
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.cameras_label = ctk.CTkLabel(self.controls_frame, text="Select camera (hover for more info)")
        self.cameras_label.grid(column=0, padx=20, pady=5, sticky="new")
        Tooltip(self.cameras_label, "This is an information tooltip!") # Attach tooltip to the button
        
        self.dropdown_cameras = ctk.CTkComboBox(self.controls_frame, values=[], command=self.select_cam, state="readonly")
        self.dropdown_cameras.grid(column=0, pady=5, sticky="ew")

        self.button1 = ctk.CTkButton(self.controls_frame, text="Refresh", command=self.refresh_cameras)
        self.button1.grid(column=0, pady=5, sticky="ew")

        self.processing_label = ctk.CTkLabel(self.controls_frame, text="Select the type of image processing")
        self.processing_label.grid(column=0, padx=20, pady=5, sticky="ew")

        self.dropdown_type = ctk.CTkComboBox(self.controls_frame, values=["canny", "bluredcanny", "sobel"], state="readonly")
        self.dropdown_type.grid(column=0, pady=5, sticky="ew")
        self.dropdown_type.set("bluredcanny")

        self.button2 = ctk.CTkButton(self.controls_frame, text="Take a photo", command=self.take_photo)
        self.button2.grid(column=0, pady=5, sticky="ew")

        self.button3 = ctk.CTkButton(self.controls_frame, text="Start drawing", command=self.start_drawing)
        self.button3.grid(column=0, pady=5, sticky="ew")

        self.contour_label = ctk.CTkLabel(self.controls_frame, text="AA", font=('Arial', 50))
        self.contour_label.grid(column=0, padx=20, pady=10, sticky="ew")

        # Video Capture
        self.refresh_cameras()
            
        self.running = True
        self.frame = None  # Store the last frame to avoid redundant resizing
        self.video_thread = threading.Thread(target=self.capture_video, daemon=True) # Start video capture thread
        self.video_thread.start()

        self.update_thread = threading.Thread(self.update_frame(), daemon=True)
        self.update_thread.start()

        self.display_photo = False
        self.photo = None
        self.bind("<Configure>", self.on_window_resize)

    def start_drawing(self):
        if self.photo is not None:
            print(self.photo)
            print(self.dropdown_type.get())
        return
    
    def take_photo(self):
        if self.frame is not None:
            cv2.imwrite("bounce.jpg",self.frame)
            self.photo = Image.fromarray(self.frame)        
            image = scan(np.array(self.frame))
            self.contour_label.configure(image=image)
            self.contour_label.image = image
            self.shape_preview = calcul_trajectoire(image, preview=True)
        self.display_photo = True
        return

    def refresh_cameras(self):
        self.display_photo = False
        self.photo = None
        self.cameras=get_cameras()
        self.cap=None
        if len(self.cameras) > 0:
            self.dropdown_cameras.configure(values=list(map(str,self.cameras)), variable=ctk.StringVar(value="0"))
            self.video_label.configure(text="")
            self.select_cam(0)
        else :
            self.dropdown_cameras.configure(values=[], variable=ctk.StringVar(value=""))
            self.video_label.configure(text = "No camera detected")

    def select_cam(self, choice):
        """ Switch camera source. """
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(choice), cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def capture_video(self):
        """ Capture video frames in a separate thread to improve UI responsiveness. """
        if self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                pass
        self.after(32, self.capture_video)

    def update_frame(self):
        """ Update the UI with the latest frame. """
        try:
            if self.display_photo:
                self.video_label.configure(image=self.photo)
                self.video_label.image = self.photo
            elif self.frame is not None:
                width = self.preview_frame.winfo_width()
                height = int(width * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                # Resize only if dimensions changed
                if width > 0 and height > 0:
                    frame_resized = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    frame_resized = cv2.resize(self.frame, (640, 480), interpolation=cv2.INTER_NEAREST)

                img = Image.fromarray(frame_resized)

                # Reuse CTkImage to reduce memory allocations
                self.ctk_img = ctk.CTkImage(light_image=img, size=(width, height))
                self.video_label.configure(image=self.ctk_img)
                self.video_label.image = self.ctk_img
        except:
            pass
        self.after(32, self.update_frame)  # Update every 32ms

    def on_window_resize(self, event):
        """ Adjust grid column sizes on window resize. """
        window_width = self.winfo_width()
        left_width = int(window_width * (2 / 3))
        right_width = int(window_width * (1 / 3))

        self.grid_columnconfigure(0, minsize=left_width)
        self.grid_columnconfigure(1, minsize=right_width)

    def on_closing(self):
        """ Handle closing event. """
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()



def main():
    app = VideoApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #profiler.print_stats("cumulative")
