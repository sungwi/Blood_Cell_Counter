import os
import csv
import glob
from ultralytics import YOLO
import cv2
import numpy as np

def load_ground_truth(csv_path):
    """Load ground truth cell counts from CSV file."""
    cell_counts = {}
    
    print(f"Loading ground truth from {csv_path}")
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            print(f"CSV headers: {headers}")
            
            for i, row in enumerate(reader):
                if i < 5:  # Print first 5 rows for debugging
                    print(f"Row {i+1}: {row}")
                
                if 'image_name' not in row or not row['image_name']:
                    continue
                    
                image_name = row['image_name']
                
                # Extract cell counts, handling different CSV formats
                rbc = int(row.get('red_blood_cells', 0)) if row.get('red_blood_cells', '') else 0
                wbc = int(row.get('white_blood cells', 0)) if row.get('white_blood cells', '') else 0
                ambiguous = int(row.get('ambiguous', 0)) if row.get('ambiguous', '') else 0
                
                # You commented out ambiguous cells - let's include them
                #total_count = rbc + wbc + ambiguous
                total_count = wbc
                cell_counts[image_name] = total_count
                
                # Also store without file extension for more flexible matching
                name_without_ext = os.path.splitext(image_name)[0]
                if name_without_ext != image_name:
                    cell_counts[name_without_ext] = total_count
            
            print(f"Loaded {len(cell_counts)} image entries from ground truth")
            
            # Print some sample entries
            sample_keys = list(cell_counts.keys())[:5]
            for key in sample_keys:
                print(f"Sample GT: {key} = {cell_counts[key]} cells")
    
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}
        
    return cell_counts

def validate_cell_counting(model_path, val_img_dir, ground_truth_csv, cell_type, conf_threshold=0.25, save_debug_images=True):
    """Validate the cell counting performance of the model with debugging."""
    # Load the model
    model = YOLO(model_path)
    
    # Load ground truth
    gt_counts = load_ground_truth(ground_truth_csv)
    
    # Create debug output directory
    debug_dir = f"./debug_output/{cell_type}"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Track results
    results = []
    total_images = 0
    total_gt_cells = 0
    total_predicted_cells = 0
    total_absolute_error = 0
    
    # Process each image
    image_files = glob.glob(os.path.join(val_img_dir, f"{cell_type}_*.jpg")) + \
                 glob.glob(os.path.join(val_img_dir, f"{cell_type}_*.jpeg")) + \
                 glob.glob(os.path.join(val_img_dir, f"{cell_type}_*.png"))
    
    print(f"Found {len(image_files)} validation images for {cell_type}")
    
    # If no images with prefix, try without prefix
    if not image_files:
        print("No prefixed images found, searching for any matching images...")
        all_files = glob.glob(os.path.join(val_img_dir, "*.jpg")) + \
                   glob.glob(os.path.join(val_img_dir, "*.jpeg")) + \
                   glob.glob(os.path.join(val_img_dir, "*.png"))
        
        # Filter files that might belong to this cell type
        image_files = [f for f in all_files if cell_type.lower() in os.path.basename(f).lower()]
        print(f"Found {len(image_files)} possible {cell_type} images by name matching")
    
    # If still no images, use any images
    if not image_files:
        print("Still no images found, using all validation images...")
        image_files = glob.glob(os.path.join(val_img_dir, "*.jpg")) + \
                     glob.glob(os.path.join(val_img_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(val_img_dir, "*.png"))
        print(f"Using {len(image_files)} total validation images")
    
    # Write a list of validation image names for debugging
    with open(os.path.join(debug_dir, f"{cell_type}_validation_images.txt"), 'w') as f:
        for img_path in image_files:
            f.write(f"{os.path.basename(img_path)}\n")
    
    # Process each image
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # Try different name variations to match with ground truth
        name_variations = [
            img_name,                              # Full filename with extension
            img_name_no_ext,                       # Filename without extension
            img_name.replace(f"{cell_type}_", ""), # Without cell type prefix
            img_name_no_ext.replace(f"{cell_type}_", ""), # Without prefix or extension
            f"{img_name_no_ext}.tif",              # Try .tif extension (common in microscopy)
            f"{img_name_no_ext.replace(f'{cell_type}_', '')}.tif" # No prefix, tif extension
        ]
        
        # Find matching ground truth
        gt_count = None
        matched_name = None
        
        for name_var in name_variations:
            if name_var in gt_counts:
                gt_count = gt_counts[name_var]
                matched_name = name_var
                break
        
        # Skip images without ground truth
        if gt_count is None:
            print(f"Warning: No ground truth found for {img_name}")
            print(f"Tried variations: {name_variations}")
            continue
        
        # Load image for visualization
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not read image at {img_path}")
            continue
        
        # Run inference with different confidence thresholds
        confidence_levels = [0.1, 0.25, 0.5, 0.75]
        results_by_conf = {}
        
        for conf in confidence_levels:
            detections = model(img_path, conf=conf)[0]
            box_count = len(detections.boxes)
            results_by_conf[conf] = box_count
        
        # Use the specified confidence threshold for official count
        detections = model(img_path, conf=conf_threshold)[0]
        pred_count = len(detections.boxes)
        
        # Calculate error
        error = abs(gt_count - pred_count)
        error_percent = (error / gt_count * 100) if gt_count > 0 else 0
        
        # Save visualization with detections
        if save_debug_images:
            # Get detection boxes
            boxes = detections.boxes.xyxy.cpu().numpy()
            scores = detections.boxes.conf.cpu().numpy()
            
            # Draw boxes on image
            result_img = image.copy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)  # Green for detections
                thickness = 2
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
                
                # Add confidence score
                cv2.putText(result_img, f'{scores[i]:.2f}', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add text with counts
            cv2.putText(result_img, 
                       f'GT: {gt_count}, Pred: {pred_count}, Error: {error} ({error_percent:.1f}%)', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add results with different confidence thresholds
            y_pos = 60
            for conf, count in results_by_conf.items():
                cv2.putText(result_img, 
                           f'Conf {conf}: {count} cells', 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                y_pos += 25
            
            # Save the annotated image
            output_path = os.path.join(debug_dir, f"{img_name_no_ext}_annotated.jpg")
            cv2.imwrite(output_path, result_img)
        
        # Store results
        results.append({
            'image': img_name,
            'matched_gt_name': matched_name,
            'ground_truth': gt_count,
            'predicted': pred_count,
            'error': error,
            'error_percent': error_percent,
            'conf_0.1': results_by_conf[0.1],
            'conf_0.25': results_by_conf[0.25],
            'conf_0.5': results_by_conf[0.5],
            'conf_0.75': results_by_conf[0.75]
        })
        
        # Update totals
        total_images += 1
        total_gt_cells += gt_count
        total_predicted_cells += pred_count
        total_absolute_error += error
    
    # Calculate overall metrics
    if total_images > 0:
        mean_absolute_error = total_absolute_error / total_images
        mean_error_percent = total_absolute_error / total_gt_cells * 100 if total_gt_cells > 0 else 0
        accuracy_percent = 100 - mean_error_percent
        
        print(f"\nValidation Results for {cell_type.upper()} (conf={conf_threshold}):")
        print(f"Total images: {total_images}")
        print(f"Total ground truth cells: {total_gt_cells}")
        print(f"Total predicted cells: {total_predicted_cells}")
        print(f"Mean absolute error: {mean_absolute_error:.2f} cells per image")
        print(f"Overall counting accuracy: {accuracy_percent:.2f}%")
        
        # Try different confidence thresholds
        print("\nResults with different confidence thresholds:")
        for conf in confidence_levels:
            pred_cells_at_conf = sum(r[f'conf_{conf}'] for r in results)
            error_at_conf = abs(total_gt_cells - pred_cells_at_conf)
            accuracy_at_conf = 100 - (error_at_conf / total_gt_cells * 100) if total_gt_cells > 0 else 0
            print(f"Conf {conf}: {pred_cells_at_conf} cells, Accuracy: {accuracy_at_conf:.2f}%")
        
        # Sort images by error percentage to find worst cases
        results.sort(key=lambda x: x['error_percent'], reverse=True)
        
        print("\nWorst 5 images by error percentage:")
        for i, r in enumerate(results[:5]):
            print(f"{i+1}. {r['image']}: GT={r['ground_truth']}, Pred={r['predicted']}, Error={r['error_percent']:.1f}%")
        
        # Write detailed results to CSV
        csv_file = os.path.join(debug_dir, f"cell_counting_validation_{cell_type}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image', 'matched_gt_name', 'ground_truth', 'predicted', 
                'error', 'error_percent', 'conf_0.1', 'conf_0.25', 'conf_0.5', 'conf_0.75'
            ])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Detailed results saved to {csv_file}")
        
        return {
            'cell_type': cell_type,
            'total_images': total_images,
            'total_gt_cells': total_gt_cells,
            'total_predicted_cells': total_predicted_cells,
            'mean_absolute_error': mean_absolute_error,
            'accuracy_percent': accuracy_percent,
            'conf_results': {conf: sum(r[f'conf_{conf}'] for r in results) for conf in confidence_levels}
        }
    else:
        print(f"No matching images found for {cell_type}")
        return None

if __name__ == "__main__":
    # Update these paths to match your environment
    model_path = "./models/runs/cell_detection4/weights/best.pt"
    val_img_dir = "./data/images/val"
    ground_truth_dir = "../data_processing/datasets/dataset_2/ground_truth"
    
    # Create debug output directory
    os.makedirs("./debug_output", exist_ok=True)
    
    # Default confidence threshold - you might need to adjust this
    conf_threshold = 0.25
    
    # Test with a single cell type first
    cell_type = 'basophil'
    csv_path = next(
        (os.path.join(ground_truth_dir, f) for f in os.listdir(ground_truth_dir) 
         if f.lower().endswith('.csv') and cell_type.lower() in f.lower()), 
        None
    )
    
    if csv_path:
        print(f"Testing {cell_type} with {os.path.basename(csv_path)}")
        validate_cell_counting(model_path, val_img_dir, csv_path, cell_type, conf_threshold)
    else:
        print(f"No CSV file found for {cell_type}")