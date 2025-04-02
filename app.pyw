from os import environ
environ["OPENCV_LOG_LEVEL"] = "SILENT"
from cv2 import (
    VideoCapture,
    CAP_DSHOW,
    cvtColor,
    COLOR_BGR2RGB,
    resize,
    INTER_NEAREST,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    imread,
)
from threading import Thread, Lock, Timer
from customtkinter import (
    CTk,
    CTkLabel,
    CTkFrame,
    CTkButton,
    CTkComboBox,
    CTkCheckBox,
    CTkSlider,
    CTkImage,
    StringVar,
    BooleanVar,
)
from tkinter import filedialog, messagebox
from PIL import Image
from tools.cameras import get_cameras
import image_processing.linedraw as linedraw
import rtde.command as command
import image_processing.trajectory_maker as ct
from image_processing import image_scanner
import tools.pingger as pingger
from dotenv import load_dotenv
from numpy import ndarray

load_dotenv("config/.env")


class VideoApp(CTk):
    def __init__(self):
        super().__init__()

        self.title("Drawing with UR3E")
        self.geometry("1280x720")

        # Configure Grid
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------- Left Side (Live Video) -------- #
        self.preview_frame = CTkFrame(self)
        self.preview_frame.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)

        self.video_label = CTkLabel(self.preview_frame, text="", font=("Arial", 50))
        self.video_label.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        # -------- Right Side (Controls) -------- #
        self.controls_frame = CTkFrame(self)
        self.controls_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.cameras_label = CTkLabel(self.controls_frame, text="Select camera source")
        self.cameras_label.grid(column=0, padx=20, pady=5, sticky="new")

        self.dropdown_cameras = CTkComboBox(
            self.controls_frame, values=[], command=self.select_cam, state="readonly"
        )
        self.dropdown_cameras.grid(column=0, pady=5, sticky="ew")

        self.button1 = CTkButton(
            self.controls_frame, text="Refresh", command=self.refresh_cameras
        )
        self.button1.grid(column=0, pady=5, sticky="ew")

        self.buttonImport = CTkButton(
            self.controls_frame, text="Import a file", command=self.import_file
        )
        self.buttonImport.grid(column=0, pady=5, sticky="ew")

        self.processing_label = CTkLabel(
            self.controls_frame, text="Select the type of image processing"
        )
        self.processing_label.grid(column=0, padx=20, pady=5, sticky="ew")

        self.dropdown_type = CTkComboBox(
            self.controls_frame,
            values=["canny", "bluredcanny", "sobel", "linedraw"],
            state="readonly",
            command=self.update_preview_image,
        )
        self.dropdown_type.grid(column=0, pady=5, sticky="ew")
        self.dropdown_type.set("bluredcanny")

        self.checkbox_hatch_linedraw = CTkCheckBox(
            self.controls_frame, text="Hatch linedraw", variable=BooleanVar(value=True)
        )
        self.checkbox_hatch_linedraw.grid(column=0, pady=5, sticky="ew")
        self.checkbox_hatch_linedraw.grid_remove()

        self.slider_label = CTkLabel(
            self.controls_frame, text="Select the simplification of the processing"
        )
        self.slider_label.grid(column=0, padx=20, pady=5, sticky="ew")

        self.slider = CTkSlider(
            self.controls_frame, from_=0, to=50, command=self.update_slider
        )
        self.slider.set(20)
        self.slider.grid(column=0, pady=5, sticky="ew")

        self.value_slider_label = CTkLabel(
            self.controls_frame, text=f"Actual value : {self.slider.get()/10}"
        )
        self.value_slider_label.grid(column=0, padx=20, sticky="ew")

        self.varCheckResize = BooleanVar(value=False)
        self.checkbox_resize_A4 = CTkCheckBox(
            self.controls_frame,
            text="Extract a A4 format",
            variable=self.varCheckResize,
        )
        self.checkbox_resize_A4.grid(column=0, sticky="ew")

        self.button2 = CTkButton(
            self.controls_frame, text="Take a photo", command=self.take_photo
        )
        self.button2.grid(column=0, pady=20, sticky="ew")

        self.treated_label = CTkLabel(self.controls_frame, text="")
        self.treated_label.grid(column=0, sticky="ew")
        self.treated_label.bind("<Button-1>", self.inverse_screens)

        self.points_label = CTkLabel(self.controls_frame, text="There is 0 point")
        self.points_label.grid(column=0, sticky="ew")

        self.button3 = CTkButton(
            self.controls_frame,
            text="Start drawing",
            command=self.start_drawing,
            state="disabled",
        )
        self.button3.grid(column=0, pady=5, sticky="ew")

        self.image_label_inversed = False
        # Video Capture
        self.refresh_cameras()

        self.running = True
        self.frame = None  # Store the last frame to avoid redundant resizing
        self.video_thread = Thread(
            target=self.capture_video
        )  # Start video capture thread
        self.video_thread.daemon = True
        self.video_thread.start()

        self.update_thread = Thread(target=self.update_frame)
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
        self.lock = Lock()

    def take_photo(self):
        if self.frame is not None:
            self.frame_4_preview = self.frame
            self.update_preview_image()
        return

    def refresh_cameras(self):
        self.display_photo = False
        self.photo = None
        self.cameras = get_cameras()
        self.cap = None
        if len(self.cameras) > 0:
            self.dropdown_cameras.configure(
                values=list(map(str, self.cameras)), variable=StringVar(value="0")
            )
            self.get_label("video")[1].configure(text="")
            self.select_cam(0)
        else:
            self.dropdown_cameras.configure(values=[], variable=StringVar(value=""))
            self.get_label("video")[1].configure(text="No camera detected")

    def select_cam(self, choice):
        """Switch camera source."""
        if self.cap is not None:
            self.cap.release()
        self.cap = VideoCapture(int(choice), CAP_DSHOW)
        self.cap.set(CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(CAP_PROP_FRAME_HEIGHT, 720)

    def capture_video(self):
        """Capture video frames in a separate thread to improve UI responsiveness."""
        if self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = cvtColor(frame, COLOR_BGR2RGB)
            except:
                pass
        self.after(32, self.capture_video)

    def update_frame(self):
        """Update the UI with the latest frame."""
        try:
            if self.display_photo:
                size, label = self.get_label("video")
                self.ctk_img = CTkImage(light_image=self.photo, size=size)
                label.configure(image=self.ctk_img)
                label.image = self.photo
            elif self.frame is not None:
                width = self.get_label("video")[1].winfo_width()
                height = int(
                    width
                    * self.cap.get(CAP_PROP_FRAME_HEIGHT)
                    / self.cap.get(CAP_PROP_FRAME_WIDTH)
                )

                # Resize only if dimensions changed
                if width > 0 and height > 0:
                    frame_resized = resize(
                        self.frame, (width, height), interpolation=INTER_NEAREST
                    )
                else:
                    frame_resized = resize(
                        self.frame, (640, 480), interpolation=INTER_NEAREST
                    )

                size, label = self.get_label("video")
                img = Image.fromarray(frame_resized).resize(size)

                # Reuse CTkImage to reduce memory allocations
                self.ctk_img = CTkImage(light_image=img, size=size)
                label.configure(image=self.ctk_img)
                label.image = self.ctk_img
        except:
            pass
        self.after(32, self.update_frame)  # Update every 32ms

    def update_slider(self, value):
        """callback of the slider, update the priview if the slider is untouched during 0.4 second"""

        self.value_slider_label.configure(text=f"Actual value : {int(value)/10}")

        with self.lock:
            # cancel the old timer if one is running
            if self.timer is not None:
                self.timer.cancel()

            # start the timer
            self.timer = Timer(0.4, self.update_preview_image)
            self.timer.start()

    def update_preview_image(self, _=None):
        """update the frame of the preview"""

        # check if a preview already exist
        if self.frame_4_preview is None:
            return

        image_resized = None
        if self.varCheckResize.get():
            # image_resized = resizer.binaryResizeA4(self.frame_4_preview)
            # image_resized = ImageScanner(self.frame_4_preview,"").scan()
            document_contour = image_scanner.scan_detection(self.frame_4_preview)
            image_resized = image_scanner.four_point_transform(
                self.frame_4_preview, document_contour.reshape(4, 2)
            )

        image_4_treatement = (
            image_resized if image_resized is not None else self.frame_4_preview
        )

        # check if the method is 'linedraw' because it's not using the same librairy
        if self.dropdown_type.get() == "linedraw":
            size, _ = self.get_label("treated")
            photo = Image.fromarray(image_4_treatement).resize(size)
            self.points, nb_points, self.treated_image = linedraw.output(
                photo, preview=True
            )
            nb_contours = 0

        else:
            self.points, nb_points, nb_contours, self.treated_image = (
                ct.calcul_trajectoire(
                    image_4_treatement,
                    epsilon=self.slider.get() / 10,
                    method=self.dropdown_type.get(),
                )
            )

        self.points_label.configure(
            text=f"There are {nb_points} points and {nb_contours} contours"
        )
        self.show_preview_image(self.treated_image)

    def show_preview_image(self, image: ndarray):
        """show the treated image in the preview box"""

        if self.ctk_treated_image is None:
            self.button3.configure(state="normal")

        size, label = self.get_label("treated")
        self.ctk_treated_image = CTkImage(
            light_image=Image.fromarray(image).resize(size), size=size
        )

        label.configure(image=self.ctk_treated_image)
        label.image = self.ctk_treated_image

    def get_label(self, name: str) -> tuple[tuple[int, int], CTkLabel]:
        """return the label to display an image"""

        treated = ((256, 144), self.treated_label)
        video = ((720, 480), self.video_label)

        if self.image_label_inversed:
            treated, video = video, treated

        if name == "treated":
            return treated
        elif name == "video":
            return video

    def inverse_screens(self, event):
        """inverse the video and the treated image"""

        self.image_label_inversed = not self.image_label_inversed
        self.show_preview_image(self.treated_image)

    def import_file(self):
        """function to import a file"""
        file_path = filedialog.askopenfilename(
            title="Select a file", filetypes=[("Tous les fichiers", "*.*")]
        )
        if file_path:
            self.frame_4_preview = imread(file_path)
            self.update_preview_image()

    def on_window_resize(self, event):
        """Adjust grid column sizes on window resize."""
        window_width = self.winfo_width()
        left_width = int(window_width * (2 / 3))
        right_width = int(window_width * (1 / 3))

        self.grid_columnconfigure(0, minsize=left_width)
        self.grid_columnconfigure(1, minsize=right_width)

    def on_closing(self):
        """Handle closing event."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()

    def start_drawing(self):
        self.button3.configure(state="disabled")
        drawing_thread = Thread(target=self.thread_drawing)
        drawing_thread.daemon = True
        drawing_thread.start()

    def thread_drawing(self):
        response = pingger.check_ping()
        if response.count("Impossible") >= 4:
            messagebox.showerror(
                "Connection's error",
                "The connection to the robot isn't possible\nVerify all and try again",
            )
            self.button3.configure(state="normal")
            return
        ret = command.startDrawing(self.points)
        if "error" in ret:
            messagebox.showwarning(
                "Connection forbiden",
                "The robot is in local mode. \nPlease change it in remote mode.",
            )
        else:
            messagebox.showinfo("Succes", "The drawing finished succesfully")
        self.button3.configure(state="normal")


def main():
    app = VideoApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

    del app


if __name__ == "__main__":
    main()
