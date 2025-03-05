import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import threading
import customtkinter as ctk
from PIL import Image
from cameras import get_cameras
import linedraw.linedraw as linedraw
import cProfile, pstats
import command
import trajectoire.calcul_trajectoire as ct
import resizer
from numpy import ndarray

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
        

        self.cameras_label = ctk.CTkLabel(self.controls_frame, text="Select camera source")
        self.cameras_label.grid(column=0, padx=20, pady=5, sticky="new")
        
        self.dropdown_cameras = ctk.CTkComboBox(self.controls_frame, values=[], command=self.select_cam, state="readonly")
        self.dropdown_cameras.grid(column=0, pady=5, sticky="ew")

        self.button1 = ctk.CTkButton(self.controls_frame, text="Refresh", command=self.refresh_cameras)
        self.button1.grid(column=0, pady=5, sticky="ew")

        self.processing_label = ctk.CTkLabel(self.controls_frame, text="Select the type of image processing")
        self.processing_label.grid(column=0, padx=20, pady=5, sticky="ew")

        self.dropdown_type = ctk.CTkComboBox(self.controls_frame, values=["canny", "bluredcanny", "sobel", "linedraw"], state="readonly", command=self.on_processing_type_change)
        self.dropdown_type.grid(column=0, pady=5, sticky="ew")
        self.dropdown_type.set("bluredcanny")

        self.checkbox_hatch_linedraw = ctk.CTkCheckBox(self.controls_frame, text="Hatch linedraw", variable=ctk.BooleanVar(value=True))
        self.checkbox_hatch_linedraw.grid(column=0, pady=5, sticky="ew")
        self.checkbox_hatch_linedraw.grid_remove()
        
        self.slider_label = ctk.CTkLabel(self.controls_frame, text="Select the simplification of the processing")
        self.slider_label.grid(column=0, padx=20, pady=5, sticky="ew")
        
        self.slider = ctk.CTkSlider(self.controls_frame, from_=1, to=50, command=self.update_slider)
        self.slider.set(5)
        self.slider.grid(column=0, pady=5, sticky="ew")
        self.slider.set(1)
        
        self.value_slider_label = ctk.CTkLabel(self.controls_frame, text="Actual value : 1")
        self.value_slider_label.grid(column=0, padx=20, sticky="ew")
        
        self.varCheckResize = ctk.BooleanVar(value=False)
        self.checkbox_resize_A4 = ctk.CTkCheckBox(self.controls_frame, text="Resize to a A4 format", variable=self.varCheckResize)
        self.checkbox_resize_A4.grid(column=0, sticky="ew")

        self.button2 = ctk.CTkButton(self.controls_frame, text="Take a photo", command=self.take_photo)
        self.button2.grid(column=0, pady=20, sticky="ew")
        
        self.treated_label = ctk.CTkLabel(self.controls_frame, text="")
        self.treated_label.grid(column=0, sticky="ew")
        self.treated_label.bind("<Button-1>", self.inverse_screens)

        self.points_label = ctk.CTkLabel(self.controls_frame, text="There is 0 point")
        self.points_label.grid(column=0, sticky="ew")

        self.button3 = ctk.CTkButton(self.controls_frame, text="Start drawing", command=self.start_drawing, state="disabled")
        self.button3.grid(column=0, pady=5, sticky="ew")


        self.image_label_inversed = False
        # Video Capture
        self.refresh_cameras()
            
        self.running = True
        self.frame = None  # Store the last frame to avoid redundant resizing
        self.video_thread = threading.Thread(target=self.capture_video) # Start video capture thread
        self.video_thread.daemon = True
        self.video_thread.start()

        self.update_thread = threading.Thread(target=self.update_frame)
        self.update_thread.daemon = True
        self.update_thread.start()

        self.display_photo = False
        self.photo = None
        self.bind("<Configure>", self.on_window_resize)
        
        self.frame_4_preview = None
        self.ctk_treated_image = None
        self.treated_image = None

        
        # Needed for the slider's callback
        self.timer = None
        self.lock = threading.Lock()
    
    def take_photo(self):
        if self.frame is not None:
            self.frame_4_preview = self.frame
            self.update_preview_image()
        return

    def refresh_cameras(self):
        self.display_photo = False
        self.photo = None
        self.cameras=get_cameras()
        self.cap=None
        if len(self.cameras) > 0:
            self.dropdown_cameras.configure(values=list(map(str,self.cameras)), variable=ctk.StringVar(value="0"))
            self.get_label("video")[1].configure(text="")
            self.select_cam(0)
        else :
            self.dropdown_cameras.configure(values=[], variable=ctk.StringVar(value=""))
            self.get_label("video")[1].configure(text = "No camera detected")

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
                size, label = self.get_label("video")
                self.ctk_img = ctk.CTkImage(light_image=self.photo, size=size)
                label.configure(image=self.ctk_img)
                label.image = self.photo
            elif self.frame is not None:
                width = self.get_label("video")[1].winfo_width()
                height = int(width * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                # Resize only if dimensions changed
                if width > 0 and height > 0:
                    frame_resized = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    frame_resized = cv2.resize(self.frame, (640, 480), interpolation=cv2.INTER_NEAREST)

                img = Image.fromarray(frame_resized)

                # Reuse CTkImage to reduce memory allocations
                self.ctk_img = ctk.CTkImage(light_image=img, size=(width, height))
                _, label = self.get_label("video")
                label.configure(image=self.ctk_img)
                label.image = self.ctk_img
        except:
            pass
        self.after(32, self.update_frame)  # Update every 32ms
        
        
    def update_slider(self, value):
        """ callback of the slider, update the priview if the slider is untouched during 0.7 second """
        
        self.value_slider_label.configure(text=f"Actual value : {int(value)}")
        
        with self.lock:
            # cancel the old timer if one is running
            if self.timer is not None:
                self.timer.cancel()
                
            # start the timer
            self.timer = threading.Timer(0.7, self.update_preview_image)
            self.timer.start()
    
        
    def update_preview_image(self):
        """ update the frame of the preview """
        
        # check if a preview already exist
        if self.frame_4_preview is None:
            return
        
        image_resized = None
        if self.varCheckResize.get():
            image_resized = resizer.binaryResizeA4(self.frame_4_preview)
        
        image_4_treatement = image_resized if image_resized is not None else self.frame_4_preview
            
        # check if the method is 'linedrawn' because it's not using the same librairy
        if self.dropdown_type.get() == "linedraw":
            photo = Image.fromarray(image_4_treatement)
            self.points, nb_points, self.treated_image = linedraw.output(photo, preview=True)
            nb_contours = 0
            
        else:
            self.points, nb_points, nb_contours, self.treated_image = ct.calcul_trajectoire(image_4_treatement, pointRatio=self.slider.get() ,method=self.dropdown_type.get(), preview=True)
        
        self.points_label.configure(text=f"There are {nb_points} points and {nb_contours} contours")
        self.show_preview_image(self.treated_image)
        
        
    def show_preview_image(self, image : ndarray):
        """ show the treated image in the preview box """
        
        if self.ctk_treated_image is None:
            self.button3.configure(state="normal")
        
        size, label = self.get_label("treated")
        self.ctk_treated_image = ctk.CTkImage(light_image=Image.fromarray(image), size=size)

        label.configure(image = self.ctk_treated_image)
        label.image = self.ctk_treated_image
        
    def get_label(self, name: str) -> tuple[tuple[int, int], ctk.CTkLabel]:
        """ return the label to display an image """
        
        treated = ((320,180), self.treated_label)
        video = ((640, 480), self.video_label)
        
        if self.image_label_inversed:
            treated, video = video, treated
            
        if name == "treated":
            return treated
        elif name == "video":
            return video
            
    def inverse_screens(self, event):
        """ inverse the video and the treated image """
        
        self.image_label_inversed = not self.image_label_inversed
        self.show_preview_image(self.treated_image)
        
        


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

    def on_processing_type_change(self, choice):
        """ Hide or show the checkbox based on the selected processing type. """
        
        if choice == "linedraw":
            self.checkbox_hatch_linedraw.grid()
            self.slider.configure(state="disabled")
        else:
            self.slider.configure(state="normal")
            self.checkbox_hatch_linedraw.grid_remove()
            
        self.update_preview_image()
        
    def start_drawing(self):
        command.startDrawing(self.points)


def main():
    app = VideoApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    
    del app

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #profiler.print_stats("cumulative")
