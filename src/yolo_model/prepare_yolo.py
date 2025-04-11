import os
import shutil
import random

def prepare_yolo_dataset():
    """Prepare the dataset for YOLO training by organizing images and annotations."""
    
    cell_types = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
    
    raw_dirs = [f'../data_processing/datasets/dataset_2/raw/{ct}' for ct in cell_types]
    annotation_dirs = [f'./annotations/{ct}' for ct in cell_types]
    
    # Create YOLO dataset structure
    train_img_dir = f'./data/images/train'
    val_img_dir = f'./data/images/val'
    train_label_dir = f'./data/labels/train'
    val_label_dir = f'./data/labels/val'
    
    # Create directories
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    print("Cleaned and created dataset directories")
    
    # Process each cell type
    for cell_type, raw_dir, annotation_dir in zip(cell_types, raw_dirs, annotation_dirs):
        print(f"\nProcessing {cell_type}...")
        
        # Check if directories exist
        if not os.path.exists(raw_dir):
            print(f"Warning: Raw directory {raw_dir} does not exist")
            continue
            
        if not os.path.exists(annotation_dir):
            print(f"Warning: Annotation directory {annotation_dir} does not exist")
            continue
        
        # Get image files that have corresponding annotation files
        valid_images = []
        for img_file in os.listdir(raw_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(annotation_dir, label_file)
            
            if os.path.exists(label_path):
                valid_images.append(img_file)
            else:
                print(f"Skipping {img_file} - no annotation found")
        
        print(f"Found {len(valid_images)} valid images with annotations for {cell_type}")
        
        if not valid_images:
            continue
        
        # Split into training and validation sets (80/20 split)
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * 0.8)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]
        
        print(f"Split: {len(train_images)} training images, {len(val_images)} validation images")
        
        # Copy training images and labels
        for img_file in train_images:
            img_path = os.path.join(raw_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(annotation_dir, label_file)
            
            # Add cell type prefix to avoid naming conflicts
            train_img_name = f"{cell_type}_{img_file}"
            train_label_name = f"{cell_type}_{label_file}"
            
            shutil.copy(img_path, os.path.join(train_img_dir, train_img_name))
            shutil.copy(label_path, os.path.join(train_label_dir, train_label_name))
        
        # Copy validation images and labels
        for img_file in val_images:
            img_path = os.path.join(raw_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(annotation_dir, label_file)
            
            # Add cell type prefix to avoid naming conflicts
            val_img_name = f"{cell_type}_{img_file}"
            val_label_name = f"{cell_type}_{label_file}"
            
            shutil.copy(img_path, os.path.join(val_img_dir, val_img_name))
            shutil.copy(label_path, os.path.join(val_label_dir, val_label_name))
    
    # Count final dataset
    train_images = os.listdir(train_img_dir)
    train_labels = os.listdir(train_label_dir)
    val_images = os.listdir(val_img_dir)
    val_labels = os.listdir(val_label_dir)
    
    print(f"\nDataset prepared for YOLO training:")
    print(f"Training: {len(train_images)} images, {len(train_labels)} labels")
    print(f"Validation: {len(val_images)} images, {len(val_labels)} labels")
    
    # Verify label-image matching
    train_mismatches = set([f.replace('.txt', '') for f in train_labels]) - set([os.path.splitext(f)[0] for f in train_images])
    val_mismatches = set([f.replace('.txt', '') for f in val_labels]) - set([os.path.splitext(f)[0] for f in val_images])
    
    if train_mismatches:
        print(f"\nWarning: {len(train_mismatches)} training labels without matching images")
        print(f"First few: {list(train_mismatches)[:5]}")
    
    if val_mismatches:
        print(f"Warning: {len(val_mismatches)} validation labels without matching images")
        print(f"First few: {list(val_mismatches)[:5]}")
    
    # Print sample images by cell type
    print("\nSample images in training set:")
    for cell_type in cell_types:
        type_images = [f for f in train_images if f.startswith(f"{cell_type}_")]
        if type_images:
            print(f"  {cell_type}: {len(type_images)} images, e.g., {type_images[:3]}")
    
    print("\nSample images in validation set:")
    for cell_type in cell_types:
        type_images = [f for f in val_images if f.startswith(f"{cell_type}_")]
        if type_images:
            print(f"  {cell_type}: {len(type_images)} images, e.g., {type_images[:3]}")

if __name__ == "__main__":
    random.seed(42)
    prepare_yolo_dataset()