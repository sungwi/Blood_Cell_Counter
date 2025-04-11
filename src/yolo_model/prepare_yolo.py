import os
import shutil
import random

def prepare_yolo_dataset():
    """Prepare the dataset for YOLO training by organizing images and annotations."""
    
    # Define directories
    raw_dirs = [
        f'../data_processing/datasets/dataset_2/raw/basophil',
        f'../data_processing/datasets/dataset_2/raw/eosinophil',
        f'../data_processing/datasets/dataset_2/raw/erythroblast'
    ]
    
    augmented_dirs = [
        f'./augmented/basophil',
        f'./augmented/eosinophil',
        f'./augmented/erythroblast'
    ]
    
    annotation_dirs = [
        f'./annotations/basophil',
        f'./annotations/eosinophil',
        f'./annotations/erythroblast'
    ]
    
    augmented_annotation_dirs = [
        f'./annotations/basophil_augmented',
        f'./annotations/eosinophil_augmented',
        f'./annotations/erythroblast_augmented'
    ]
    
    # Create YOLO dataset structure
    train_img_dir = f'./data/images/train'
    val_img_dir = f'./data/images/val'
    train_label_dir = f'./data/labels/train'
    val_label_dir = f'./data/labels/val'
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Collect all images with annotations
    for i, (raw_dir, annotation_dir) in enumerate(zip(raw_dirs, annotation_dirs)):
        cell_type = os.path.basename(raw_dir)
        print(f"Processing {cell_type} raw data...")
        
        # Get image files that have corresponding annotation files
        image_files = []
        for img_file in os.listdir(raw_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(annotation_dir, label_file)
            
            if os.path.exists(label_path):
                image_files.append(img_file)
        
        # Split into train/val (we'll use the raw images as val, augmented as train)
        for img_file in image_files:
            img_path = os.path.join(raw_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(annotation_dir, label_file)
            
            # Copy to validation set
            shutil.copy(img_path, os.path.join(val_img_dir, f"{cell_type}_{img_file}"))
            shutil.copy(label_path, os.path.join(val_label_dir, f"{cell_type}_{label_file}"))
    
    # Process augmented data for training
    for i, (aug_dir, aug_annotation_dir) in enumerate(zip(augmented_dirs, augmented_annotation_dirs)):
        cell_type = os.path.basename(aug_dir).replace('_augmented', '')
        print(f"Processing {cell_type} augmented data...")
        
        # Get augmented image files with annotations
        for img_file in os.listdir(aug_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(aug_annotation_dir, label_file)
            
            if os.path.exists(label_path):
                # Copy to training set
                shutil.copy(os.path.join(aug_dir, img_file), os.path.join(train_img_dir, f"{cell_type}_{img_file}"))
                shutil.copy(label_path, os.path.join(train_label_dir, f"{cell_type}_{label_file}"))
    
    # Count images
    train_count = len(os.listdir(train_img_dir))
    val_count = len(os.listdir(val_img_dir))
    
    print(f"\nDataset prepared for YOLO training:")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

if __name__ == "__main__":
    prepare_yolo_dataset()