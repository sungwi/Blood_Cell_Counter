"""
This script splits the dataset into training and validation sets, 
and copies the images and annotations to the YOLO directory structure.
"""

import os
import shutil
import random
from pathlib import Path

# Paths
annotation_dir = "src/yolo_model/annotations/dataset_1/blood_cells"
source_image_dir = "src/data_processing/datasets/dataset_1/processed/blood_cells_processed"

# Create YOLO directory structure
base_dir = "src/yolo_model/data"
os.makedirs(f"{base_dir}/images/train", exist_ok=True)
os.makedirs(f"{base_dir}/images/val", exist_ok=True)
os.makedirs(f"{base_dir}/labels/train", exist_ok=True)
os.makedirs(f"{base_dir}/labels/val", exist_ok=True)

# Get all annotation files
annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".txt")]
random.shuffle(annotation_files)

# Split into train (80%) and validation (20%) sets
split_index = int(len(annotation_files) * 0.8)
train_files = annotation_files[:split_index]
val_files = annotation_files[split_index:]

# Copy files to their respective directories
def copy_files(file_list, target_set):
    for txt_file in file_list:
        # Copy annotation file
        src_annotation = os.path.join(annotation_dir, txt_file)
        dst_annotation = os.path.join(f"{base_dir}/labels/{target_set}", txt_file)
        shutil.copy(src_annotation, dst_annotation)
        
        # Copy corresponding image
        image_name = os.path.splitext(txt_file)[0] + '.jpg'  # Adjust extension if needed
        src_image = os.path.join(source_image_dir, image_name)
        dst_image = os.path.join(f"{base_dir}/images/{target_set}", image_name)
        
        # Check if the image exists, it might have a different extension
        if not os.path.exists(src_image):
            for ext in ['.png', '.jpeg', '.tif']:
                alt_image_name = os.path.splitext(txt_file)[0] + ext
                alt_src_image = os.path.join(source_image_dir, alt_image_name)
                if os.path.exists(alt_src_image):
                    src_image = alt_src_image
                    dst_image = os.path.join(f"{base_dir}/images/{target_set}", alt_image_name)
                    break
        
        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)
        else:
            print(f"Warning: Image for {txt_file} not found")

# Copy files
copy_files(train_files, "train")
copy_files(val_files, "val")

# Create data.yaml file
with open(f"{base_dir}/data.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(base_dir)}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("nc: 1  # number of classes\n")
    f.write("names: ['cell']  # class names\n")

print(f"Dataset prepared: {len(train_files)} training samples, {len(val_files)} validation samples")