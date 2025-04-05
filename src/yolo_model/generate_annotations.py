import os
import cv2
import csv
import numpy as np
import glob

def load_cell_counts(gt_dir):
    """Load cell counts from ground truth CSV files."""
    cell_counts = {}
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))
    
    for csv_path in csv_files:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row['image_name']
                if not image_name:
                    continue
                    
                # Extract cell counts, handling different CSV formats
                rbc = int(row.get('red_blood_cells', 0)) if row.get('red_blood_cells', '') else 0
                wbc = int(row.get('white_blood cells', 0)) if row.get('white_blood cells', '') else 0
                ambiguous = int(row.get('ambiguous', 0)) if row.get('ambiguous', '') else 0
                
                total_count = rbc + wbc + ambiguous
                cell_counts[image_name] = total_count
    
    return cell_counts

def generate_annotations(cell_type, gt_dir, adjust_params=True):
    """Generate accurate annotations for a cell type based on ground truth counts."""
    raw_dir = f'../data_processing/datasets/dataset_2/raw/{cell_type}'
    annotation_dir = f'./annotations/{cell_type}'
    
    # Create annotation directory if it doesn't exist
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Load ground truth cell counts
    cell_counts = load_cell_counts(gt_dir)
    
    # Process each image
    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(raw_dir, filename)
        
        # Skip if we don't have a ground truth count for this image
        if filename not in cell_counts:
            print(f"Skipping {filename}: No ground truth data available")
            continue
            
        expected_count = cell_counts[filename]
        print(f"Processing {filename} - Expected cells: {expected_count}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            continue
            
        # Initial parameters for cell segmentation
        params = {
            'threshold_method': cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            'min_area': 100,
            'max_area_ratio': 0.5,
            'foreground_threshold': 0.7
        }
        
        # Generate segmentation with initial parameters
        bboxes, cell_count = segment_cells(image, params)
        
        # Adjust parameters if the cell count doesn't match
        if adjust_params and cell_count != expected_count:
            print(f"Initial segmentation found {cell_count} cells, adjusting parameters...")
            
            # Try different parameter combinations
            attempts = 0
            max_attempts = 10
            best_diff = abs(cell_count - expected_count)
            best_params = params.copy()
            best_bboxes = bboxes.copy()
            
            while cell_count != expected_count and attempts < max_attempts:
                attempts += 1
                params_adjusted = params.copy()
                
                # Adjust parameters based on if we need more or fewer cells
                if cell_count < expected_count:
                    # Try to detect more cells
                    params_adjusted['min_area'] = max(params['min_area'] - 20, 50)
                    params_adjusted['foreground_threshold'] = max(params['foreground_threshold'] - 0.05, 0.4)
                else:
                    # Try to detect fewer cells
                    params_adjusted['min_area'] = min(params['min_area'] + 20, 300)
                    params_adjusted['foreground_threshold'] = min(params['foreground_threshold'] + 0.05, 0.9)
                
                # Try with new parameters
                new_bboxes, new_count = segment_cells(image, params_adjusted)
                
                # Keep track of the best result
                current_diff = abs(new_count - expected_count)
                if current_diff < best_diff:
                    best_diff = current_diff
                    best_params = params_adjusted.copy()
                    best_bboxes = new_bboxes.copy()
                    
                print(f"Attempt {attempts}: Found {new_count} cells with adjusted params")
                
                # Update for next iteration
                params = params_adjusted
                bboxes = new_bboxes
                cell_count = new_count
                
                # Break early if we achieve the expected count
                if cell_count == expected_count:
                    break
            
            # Use the best result if we didn't get an exact match
            if cell_count != expected_count:
                print(f"Using best params with {len(best_bboxes)} cells (expected {expected_count})")
                bboxes = best_bboxes
        
        # Generate YOLO annotation file
        img_height, img_width = image.shape[:2]
        
        annotation_file = os.path.join(annotation_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        
        with open(annotation_file, 'w') as f:
            for bbox in bboxes:
                x, y, w, h = bbox
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Write as: class_id x_center y_center width height
                f.write(f"0 {x_center} {y_center} {width} {height}\n")
        
        print(f"Created annotation file: {annotation_file} with {len(bboxes)} bounding boxes")

def segment_cells(image, params):
    """
    Segment cells in an image using watershed algorithm.
    Uses your existing segmentation approach.
    
    Args:
        image: Input image
        params: Dictionary of segmentation parameters
    
    Returns:
        bboxes: List of bounding boxes (x, y, w, h)
        count: Number of cells detected
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply thresholding to segment cells
    _, thresh = cv2.threshold(gray, 0, 255, params['threshold_method'])
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, params['foreground_threshold']*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)
    
    # Get contours and bounding boxes
    bboxes = []
    img_height, img_width = gray.shape
    max_area = img_width * img_height * params['max_area_ratio']
    
    for label in range(2, markers.max() + 1):
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = cnts[0]
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by area
            if params['min_area'] <= area <= max_area:
                bboxes.append((x, y, w, h))
    
    return bboxes, len(bboxes)

def check_and_create_raw_dir(cell_type):
    """Check if raw directory exists for a cell type, create if not."""
    raw_dir = f'../data_processing/datasets/dataset_2/raw/{cell_type}'
    if not os.path.exists(raw_dir):
        print(f"Creating raw directory for {cell_type}: {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
    return raw_dir

def find_and_copy_images(cell_type):
    """Find images for a cell type and copy them to raw directory."""
    raw_dir = check_and_create_raw_dir(cell_type)
    
    # Possible locations to search for images
    search_dir = os.path.join(f'../data_processing/datasets/dataset_2/raw/{cell_type}')
    
    # Print detailed diagnostic information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for images in: {search_dir}")
    print(f"Absolute path: {os.path.abspath(search_dir)}")
    
    # Check if directory exists
    if not os.path.exists(search_dir):
        print(f"ERROR: Directory does not exist: {search_dir}")
        return False
    
    # Try to list the directory contents
    try:
        all_files = os.listdir(search_dir)
        print(f"Found {len(all_files)} total files in directory")
        
        # Count image files specifically
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} image files")
        
        if image_files:
            print("First 5 image files:")
            for i, img in enumerate(image_files[:5]):
                print(f"  {i+1}. {img}")
        else:
            print("No image files found")
            
            # Show what files are in the directory
            if all_files:
                print("Directory contains these files:")
                for i, f in enumerate(all_files[:10]):  # Show first 10 files
                    print(f"  {i+1}. {f}")
                if len(all_files) > 10:
                    print(f"  ... and {len(all_files) - 10} more files")
    except Exception as e:
        print(f"ERROR listing directory: {e}")
        return False
    
    # If we found images, we can simply return True since they're already in the raw directory
    if len(image_files) > 0:
        print(f"Images already exist in {raw_dir}, no need to copy")
        return True
    
    # If we get here, no images were found
    return False

if __name__ == "__main__":
    gt_dir = '../data_processing/datasets/dataset_2/ground_truth'
    
    # Make sure the ground truth directory exists
    os.makedirs(gt_dir, exist_ok=True)
    
    # Check CSV files
    csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {gt_dir}")
    
    if not csv_files:
        print("Warning: No CSV files found in ground truth directory")
        
        # Check if CSV files exist in another common location
        other_dirs = [
            '../data_processing/datasets/dataset_2',
            '../data_processing/datasets'
        ]
        
        for dir_path in other_dirs:
            if os.path.exists(dir_path):
                found_csvs = glob.glob(os.path.join(dir_path, "*.csv"))
                if found_csvs:
                    print(f"Found {len(found_csvs)} CSV files in {dir_path}")
                    for csv_file in found_csvs:
                        print(f"  - {os.path.basename(csv_file)}")
                    
                    # Option to copy CSV files
                    import shutil
                    for csv_file in found_csvs:
                        dest_path = os.path.join(gt_dir, os.path.basename(csv_file))
                        shutil.copy2(csv_file, dest_path)
                        print(f"Copied {os.path.basename(csv_file)} to {gt_dir}")
                    
                    # Refresh CSV files list
                    csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))
                    break
    
    # Process each cell type
    for cell_type in ['basophil', 'eosinophil', 'erythroblast']:
        print(f"\nPreparing data for {cell_type}...")
        
        # Find and copy images to the raw directory if needed
        found_images = find_and_copy_images(cell_type)
        
        if found_images:
            print(f"\nGenerating annotations for {cell_type}...")
            generate_annotations(cell_type, gt_dir)
        else:
            print(f"No images found for {cell_type}. Skipping annotation generation.")