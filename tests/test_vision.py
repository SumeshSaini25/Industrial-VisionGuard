import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pillow:", Image.__version__)

print("\nLoading YOLO...")
model = YOLO("yolo26n.pt")

print("YOLO loaded successfully.")
print("Model task:", model.task)

print("\nAll vision components are working.")