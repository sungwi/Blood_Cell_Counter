import os
import cv2
import csv
import numpy as np
import glob

def extract_first_number(value_str):
    """
    Extract the first number from a string like '1 (2)'.
    Returns 0 if no number is found.
    """
    if not value_str or not isinstance(value_str, str):
        return 0
    
    # Find the first number in the string
    import re
    numbers = re.findall(r'\d+', value_str)
    if numbers:
        return int(numbers[0])
    return 0

def load_cell_counts(gt_dir):
    """Load cell counts from ground truth CSV files."""
    cell_counts = {}
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    for csv_path in csv_files:
        csv_filename = os.path.basename(csv_path).lower()
        print(f"Processing CSV file: {csv_filename}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"CSV headers: {headers}")
            
            row_count = 0
            for row in reader:
                image_name = row.get('image_name', '')
                if not image_name:
                    continue
                
                row_count += 1
                if row_count <= 3:
                    print(f"Sample row {row_count}: {row}")
                    
                # Extract cell counts, handling different CSV formats
                rbc = int(row.get('red_blood_cells', 0)) if row.get('red_blood_cells', '') else 0
                wbc = int(row.get('white_blood cells', 0)) if row.get('white_blood cells', '') else 0
                ambiguous = int(row.get('ambiguous', 0)) if row.get('ambiguous', '') else 0
                
                total_count = rbc + wbc + ambiguous
                cell_counts[image_name] = total_count
            
            print(f"Loaded {row_count} rows from {csv_filename}")
    
    return cell_counts

def generate_annotations(cell_type, csv_files, adjust_params=True):
    """Generate accurate annotations for a cell type based on ground truth counts."""
    raw_dir = f'../data_processing/datasets/dataset_2/raw/{cell_type}'
    annotation_dir = f'./annotations/{cell_type}'
    
    # Create annotation directory if it doesn't exist
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Find the matching CSV for this cell type
    matching_csv = None
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).lower()
        if cell_type.lower() in filename:
            matching_csv = csv_file
            print(f"Using CSV file: {os.path.basename(matching_csv)}")
            break
    
    if not matching_csv:
        print(f"No matching CSV file found for {cell_type}")
        return False
    
    cell_counts = {}
    with open(matching_csv, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        print(f"CSV headers: {headers}")
        
        row_count = 0
        for row in reader:
            image_name = row.get('image_name', '')
            if not image_name:
                continue
            
            row_count += 1
            if row_count <= 3:
                print(f"Sample row {row_count}: {row}")
                
            # Extract cell counts, handling different CSV formats and special cases like "1 (2)"
            try:
                rbc = int(row.get('red_blood_cells', 0)) if row.get('red_blood_cells', '') else 0
            except ValueError:
                rbc_val = row.get('red_blood_cells', '')
                rbc = extract_first_number(rbc_val)
                print(f"Converted RBC value '{rbc_val}' to {rbc}")
            
            try:
                wbc_val = row.get('white_blood cells', '')
                if wbc_val:
                    # Extract first number from string like "1 (2)"
                    wbc = extract_first_number(wbc_val)
                    if wbc_val != str(wbc):  # If conversion changed the value
                        print(f"Converted WBC value '{wbc_val}' to {wbc}")
                else:
                    wbc = 0
            except Exception as e:
                print(f"Error processing WBC value '{row.get('white_blood cells', '')}': {e}")
                wbc = 0
            
            try:
                ambiguous = int(row.get('ambiguous', 0)) if row.get('ambiguous', '') else 0
            except ValueError:
                ambiguous_val = row.get('ambiguous', '')
                ambiguous = extract_first_number(ambiguous_val)
                print(f"Converted ambiguous value '{ambiguous_val}' to {ambiguous}")
            
            total_count = rbc + wbc + ambiguous
            cell_counts[image_name] = total_count
        
        print(f"Loaded {row_count} rows from {os.path.basename(matching_csv)}")
    
    # Process each image
    processed_images = 0
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
        processed_images += 1
        
    print(f"Processed {processed_images} images for {cell_type}")
    return processed_images > 0

def segment_cells(image, params):
    """
    Segment cells in an image using watershed algorithm.
    Uses your existing segmentation approach.
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
    possible_dirs = [
        f'../data_processing/datasets/dataset_2/raw/{cell_type}',
        f'../data_processing/datasets/dataset_2/raw/{cell_type.capitalize()}',
        f'../data_processing/datasets/dataset_2/raw/{cell_type.upper()}',
        f'../data_processing/datasets/dataset_2/{cell_type}',
        f'../data_processing/datasets/dataset_2/{cell_type.capitalize()}',
        f'../data_processing/datasets/dataset_2/{cell_type.upper()}'
    ]
    
    print(f"Checking possible directories for {cell_type}:")
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} - EXISTS")
        else:
            print(f"✗ {dir_path} - does not exist")
    
    # Find first existing directory
    existing_dirs = [d for d in possible_dirs if os.path.exists(d)]
    
    if not existing_dirs:
        print(f"No directories found for {cell_type}. Creating empty directory at {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
        return False
    
    # Use the first existing directory
    search_dir = existing_dirs[0]
    print(f"Using directory: {search_dir}")
    
    # Print detailed diagnostic information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Absolute path: {os.path.abspath(search_dir)}")
    
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
                
            # If raw_dir is different from search_dir, copy the files
            if os.path.normpath(raw_dir) != os.path.normpath(search_dir):
                print(f"Copying {len(image_files)} images from {search_dir} to {raw_dir}")
                os.makedirs(raw_dir, exist_ok=True)
                
                import shutil
                for img in image_files:
                    src = os.path.join(search_dir, img)
                    dst = os.path.join(raw_dir, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        print(f"Copied: {img}")
                    else:
                        print(f"Skip copying: {img} (already exists)")
                        
                print(f"Successfully copied {len(image_files)} images to {raw_dir}")
            else:
                print(f"Images already exist in {raw_dir}, no need to copy")
            
            return True
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
    
    # If we get here, no images were found
    return False

if __name__ == "__main__":
    # Define possible locations for ground truth CSV files
    dataset_dir = '../data_processing/datasets/dataset_2'
    gt_dir = os.path.join(dataset_dir, 'ground_truth')
    
    # Make sure the ground truth directory exists
    os.makedirs(gt_dir, exist_ok=True)
    
    # Check for CSV files in both locations
    dataset_csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    gt_csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))
    
    print(f"Found {len(dataset_csv_files)} CSV files in {dataset_dir}")
    print(f"Found {len(gt_csv_files)} CSV files in {gt_dir}")
    
    # Combine CSV files from both locations, preferring the dataset_dir ones
    all_csv_filenames = [os.path.basename(f) for f in dataset_csv_files]
    
    # Add CSV files from ground_truth directory that don't exist in dataset_dir
    for csv_file in gt_csv_files:
        filename = os.path.basename(csv_file)
        if filename not in all_csv_filenames:
            dataset_csv_files.append(csv_file)
            all_csv_filenames.append(filename)
    
    print(f"Total unique CSV files found: {len(dataset_csv_files)}")
    
    # Use dataset_csv_files for processing
    csv_files = dataset_csv_files

    print("\n=== DEBUG: Available CSV Files ===")
    for csv_file in csv_files:
        print(f"- {os.path.basename(csv_file)}")

    print("\n=== DEBUG: Checking for Cell Type CSV Files ===")
    for test_cell_type in ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']:
        found = False
        
        for csv_file in csv_files:
            csv_name = os.path.basename(csv_file).lower()
            
            # Print all CSV names for debugging
            contains_cell_type = test_cell_type.lower() in csv_name
            print(f"Checking if '{test_cell_type}' is in '{csv_name}': {contains_cell_type}")
            
            if contains_cell_type:
                print(f"✓ Found matching CSV for {test_cell_type}: {os.path.basename(csv_file)}")
                found = True
                break
        
        if not found:
            print(f"✗ No matching CSV file found for {test_cell_type}")
    
    if not csv_files:
        print("Warning: No CSV files found in either directory")
        
        # Check if CSV files exist in another common location
        other_dirs = [
            '../data_processing/datasets',
            '../../datasets'
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
                        dest_path = os.path.join(dataset_dir, os.path.basename(csv_file))
                        shutil.copy2(csv_file, dest_path)
                        
                        print(f"Copied {os.path.basename(csv_file)} to {dataset_dir}")
                    
                    # Refresh CSV files list
                    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
                    break
    
    # Map lowercase cell types to possible CSV file prefixes
    csv_prefix_map = {
        'basophil': ['Raw', 'raw', 'Basophil', ''],
        'eosinophil': ['Raw', 'raw', 'Eosinophil', ''],
        'erythroblast': ['Raw', 'raw', 'Erythroblast', ''],
        'immunoglobulin': ['Raw', 'raw', 'Immunoglobulin', ''],
        'lymphocyte': ['Raw', 'raw', 'Lymphocyte', ''],
        'monocyte': ['Raw', 'raw', 'Monocyte', ''],
        'neutrophil': ['Raw', 'raw', 'Neutrophil', ''],
        'platelet': ['Raw', 'raw', 'Platelet', '']
    }
    
    # Process each cell type
    for cell_type in ['basophil', 'eosinophil', 'erythroblast', 'immunoglobulin', 'lymphocyte', 'monocyte', 'neutrophil', 'Platelet']:
        print(f"\n{'='*50}")
        print(f"Preparing data for {cell_type}...")
        print(f"{'='*50}")
        
        # Standardize the cell type name
        cell_type_lower = cell_type.lower()
        
        # Check if cell type has ground truth data
        has_gt_data = False
        for csv_file in csv_files:
            csv_name = os.path.basename(csv_file).lower()
            
            # Check for variations of the cell type name in the CSV filename
            for prefix in csv_prefix_map.get(cell_type_lower, ['']):
                possible_name = f"{prefix}{cell_type_lower}".lower()
                possible_name_cap = f"{prefix}{cell_type}".lower()
                
                if possible_name in csv_name or possible_name_cap in csv_name:
                    print(f"Found matching CSV file: {os.path.basename(csv_file)}")
                    has_gt_data = True
                    break
            
            if has_gt_data:
                break
        
        if not has_gt_data:
            print(f"Warning: Could not find ground truth CSV file for {cell_type}")
            print(f"Available CSV files: {[os.path.basename(f) for f in csv_files]}")
            print(f"Skipping {cell_type}")
            continue
        
        # Find and copy images to the raw directory if needed
        found_images = find_and_copy_images(cell_type)
        
        if found_images:
            print(f"\nGenerating annotations for {cell_type}...")
            generate_annotations(cell_type, csv_files)
        else:
            print(f"No images found for {cell_type}. Skipping annotation generation.")