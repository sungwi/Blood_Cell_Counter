import numpy as np
import cv2

from keras.models import load_model
from ultralytics import YOLO

import os
model_path = os.path.join(os.path.dirname(__file__), "models/yolo/yolo-baseline.pt")

yolo_baseline = YOLO(model_path)
yolo_extended = YOLO("models/yolo/yolo-extended.pt")
unet_baseline = load_model("models/unet/unet-baseline.h5")
unet_extended = load_model("models/unet/unet-extended.h5")

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
    if not isinstance(image, np.ndarray):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    output = unet_baseline.predict(image_batch)[0]

    if output.ndim == 3:
        output = output[:, :, 0]
    mask = (output > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_count = len(contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_resized, cell_count

def detect_cells_unet_extended(image):
    if not isinstance(image, np.ndarray):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    output = unet_extended.predict(image_batch)[0]

    if output.ndim == 3:
        output = output[:, :, 0]
    mask = (output > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_count = len(contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_resized, cell_count
