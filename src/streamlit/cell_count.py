import numpy as np
import cv2
from ultralytics import YOLO

yolo_baseline = YOLO("models/yolo/baseline.pt")
yolo_extended = ""
unet_baseline = ""
unet_extended = ""

def detect_cells_yolo_baseline(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)

    results = yolo_baseline(img_np)[0]

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    cell_count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_np, cell_count

def detect_cells_yolo_extended(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)

    results = yolo_extended(img_np)[0]

    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    cell_count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_np, cell_count

def detect_cells_unet_baseline(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)

    #UNET MODEL LOGIC
    return img_np, cell_count

def detect_cells_unet_extended(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)

    #UNET MODEL LOGIC
    return img_np, cell_count
