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
from dotenv import load_dotenv
from numpy import ndarray