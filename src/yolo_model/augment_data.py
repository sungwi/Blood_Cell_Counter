import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def check_directory(directory):
    """Check if directory exists, create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    return True

def find_data_directory(cell_type):
    """Find the directory containing data for a specific cell type."""
    # Possible locations to check
    possible_paths = [
        os.path.join(f'../data_processing/datasets/dataset_2/raw/{cell_type}'),
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.exists(path):
            # Count image files in this directory
            image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                print(f"Found {len(image_files)} images in {path}")
                return path
    
    # If we reach here, no suitable directory was found
    print(f"Warning: Could not find directory with {cell_type} images")
    # Return the first path as a fallback (it will be created)
    return possible_paths[0]

def augment_dataset(input_dir, output_dir, annotation_dir, output_annotation_dir, num_augmentations=5):
    """
    Augment images and their corresponding annotations.
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        annotation_dir: Directory containing YOLO annotation files
        output_annotation_dir: Directory to save augmented annotations
        num_augmentations: Number of augmented versions to create per original image
    """
    check_directory(output_dir)
    check_directory(output_annotation_dir)
    
    # Define augmentation pipeline for images with bounding boxes
    bbox_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.ISONoise(p=0.8),
        ], p=0.8),
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, p=0.5),
        ], p=0.8),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, p=0.5),
        ], p=0.5),
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    # Define transformation for images without bounding boxes
    image_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        ], p=1.0),
        A.OneOf([
            A.GaussNoise(p=0.8),
            A.ISONoise(p=0.8),
        ], p=0.8),
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=10, p=0.5),
        ], p=0.8),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, p=0.5),
        ], p=0.5),
        A.HorizontalFlip(p=0.5),
    ])
    
    # Process each image
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Warning: No image files found in {input_dir}")
        return
        
    print(f"Found {len(image_files)} images to augment")
    
    for image_file in tqdm(image_files, desc="Augmenting images"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue
            
        # Load corresponding annotation if exists
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        bboxes = []
        class_labels = []
        has_annotations = False
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
            
            has_annotations = len(bboxes) > 0
        
        # Generate augmented images
        for i in range(num_augmentations):
            # Apply transformation
            if has_annotations:
                transformed = bbox_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                augmented_image = transformed['image']
                augmented_bboxes = transformed['bboxes']
                augmented_class_labels = transformed['class_labels']
            else:
                transformed = image_transform(image=image)
                augmented_image = transformed['image']
                augmented_bboxes = []
                augmented_class_labels = []
            
            # Save augmented image
            output_image_file = f"{os.path.splitext(image_file)[0]}_aug{i+1}{os.path.splitext(image_file)[1]}"
            output_image_path = os.path.join(output_dir, output_image_file)
            cv2.imwrite(output_image_path, augmented_image)
            
            # Save augmented annotation
            if has_annotations:
                output_annotation_file = f"{os.path.splitext(image_file)[0]}_aug{i+1}.txt"
                output_annotation_path = os.path.join(output_annotation_dir, output_annotation_file)
                
                with open(output_annotation_path, 'w') as f:
                    for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
                        f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

def process_cell_type(cell_type):
    """Process a specific cell type for augmentation."""
    
    # Auto-find the input directory containing images for this cell type
    input_dir = find_data_directory(cell_type)
    
    # Create output directories in the expected locations
    output_dir = os.path.join(f'./augmented/{cell_type}')
    annotation_dir = os.path.join(f'./annotations/{cell_type}')
    output_annotation_dir = os.path.join(f'./annotations/{cell_type}_augmented')
    
    print(f"Processing {cell_type}:")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Annotation directory: {annotation_dir}")
    print(f"Output annotation directory: {output_annotation_dir}")
    
    # Ensure all required directories exist
    for directory in [output_dir, annotation_dir, output_annotation_dir]:
        check_directory(directory)
    
    print(f"Augmenting {cell_type} data...")
    augment_dataset(input_dir, output_dir, annotation_dir, output_annotation_dir, num_augmentations=5)

if __name__ == "__main__":
    # Create the base directories first
    check_directory(os.path.join('./augmented'))
    
    # Process each cell type
    for cell_type in ['basophil', 'eosinophil', 'erythroblast']:
        process_cell_type(cell_type)