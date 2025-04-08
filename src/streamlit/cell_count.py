import numpy as np
import cv2

from keras.models import load_model
from ultralytics import YOLO

yolo_baseline = YOLO("models/yolo/yolo-baseline.pt")
yolo_extended = YOLO("models/yolo/yolo-extended.pt")
unet_baseline = ""
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
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)

    #UNET MODEL LOGIC
    return img_np, cell_count

def detect_cells_unet_extended(image):
   # Convert PIL to NumPy if needed
    if not isinstance(image, np.ndarray):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image)

    # Resize and normalize
    image_resized = cv2.resize(image, (256, 256))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)  # (1, 256, 256, 3)

    # Predict with Keras UNet
    output = unet_extended.predict(image_batch)[0]  # (256, 256, 1) or (256, 256)

    # Convert to binary mask
    if output.ndim == 3:
        output = output[:, :, 0]
    mask = (output > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_count = len(contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_resized, cell_count
