import os
import csv
import glob
import warnings
import re

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from ultralytics import YOLO
import cv2
import numpy as np

def extract_first_number(value_str):
    """Extract the first number from a string like '1 (2)'."""
    if not value_str or not isinstance(value_str, str):
        return 0
    
    # Find the first number in the string
    numbers = re.findall(r'\d+', value_str)
    if numbers:
        return int(numbers[0])
    return 0

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
                if i < 5:
                    print(f"Row {i+1}: {row}")
                
                if 'image_name' not in row or not row['image_name'] or not any(row.values()):
                    continue
                    
                image_name = row['image_name']
                
                # Extract cell counts, handling different CSV formats with error handling
                try:
                    rbc = int(row.get('red_blood_cells', 0)) if row.get('red_blood_cells', '') else 0
                except ValueError:
                    rbc = extract_first_number(row.get('red_blood_cells', ''))
                
                try:
                    wbc_val = row.get('white_blood cells', '')
                    wbc = extract_first_number(wbc_val) if wbc_val else 0
                except Exception:
                    wbc = 0
                
                try:
                    ambiguous = int(row.get('ambiguous', 0)) if row.get('ambiguous', '') else 0
                except ValueError:
                    ambiguous = extract_first_number(row.get('ambiguous', ''))
                
                # Calculate total count
                total_count = rbc + wbc + ambiguous
                
                # Store multiple variations of the filename to make matching easier
                cell_counts[image_name] = total_count
                cell_counts[image_name.lower()] = total_count
                
                # Also store without extension
                name_without_ext = os.path.splitext(image_name)[0]
                cell_counts[name_without_ext] = total_count
                cell_counts[name_without_ext.lower()] = total_count
            
            print(f"Loaded {len(cell_counts) // 4} image entries from ground truth")
            
            sample_keys = [k for k in cell_counts.keys() if not k.lower().startswith("cell_")][:5]
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
    
    # Process each image - look for images with the cell type prefix
    image_files = glob.glob(os.path.join(val_img_dir, f"{cell_type.lower()}_*.jpg")) + \
                 glob.glob(os.path.join(val_img_dir, f"{cell_type.lower()}_*.jpeg")) + \
                 glob.glob(os.path.join(val_img_dir, f"{cell_type.lower()}_*.png"))
    
    print(f"Found {len(image_files)} validation images for {cell_type}")
    
    # If no images with prefix, try with cell type in filename
    if not image_files:
        print("No prefixed images found, searching for any matching images...")
        all_files = glob.glob(os.path.join(val_img_dir, "*.jpg")) + \
                   glob.glob(os.path.join(val_img_dir, "*.jpeg")) + \
                   glob.glob(os.path.join(val_img_dir, "*.png"))
        
        # Filter files that might belong to this cell type
        image_files = [f for f in all_files if cell_type.lower() in os.path.basename(f).lower()]
        print(f"Found {len(image_files)} possible {cell_type} images by name matching")
    
    # Write a list of validation image names for debugging
    with open(os.path.join(debug_dir, f"{cell_type}_validation_images.txt"), 'w') as f:
        for img_path in image_files:
            f.write(f"{os.path.basename(img_path)}\n")
    
    # Process each image
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        # For matching with ground truth, extract the part after cell_type_
        if cell_type.lower() + "_" in img_name.lower():
            # This handles cases like basophil_Basophil_0001.jpg -> Basophil_0001.jpg
            base_name = img_name.split(cell_type.lower() + "_", 1)[1]
            base_name_no_ext = os.path.splitext(base_name)[0]
        else:
            # If no prefix, use the full name
            base_name = img_name
            base_name_no_ext = img_name_no_ext
        
        # Find matching ground truth - first try the unprefixed name
        gt_count = None
        matched_name = None
        
        name_variations = [
            base_name,                      
            base_name_no_ext,              
            base_name.lower(),     
            base_name_no_ext.lower(),  
            img_name,        
            img_name_no_ext,            
            img_name.lower(),    
            img_name_no_ext.lower()  
        ]
        
        for name_var in name_variations:
            if name_var in gt_counts:
                gt_count = gt_counts[name_var]
                matched_name = name_var
                break
        
        # Skip images without ground truth silently
        if gt_count is None:
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
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
                
                cv2.putText(result_img, f'{scores[i]:.2f}', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
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
        print("\nResults with different confidence thresholds:")
        for conf in confidence_levels:
            pred_cells_at_conf = sum(r[f'conf_{conf}'] for r in results)
            error_at_conf = abs(total_gt_cells - pred_cells_at_conf)
            accuracy_at_conf = 100 - (error_at_conf / total_gt_cells * 100) if total_gt_cells > 0 else 0
            print(f"Conf {conf}: {pred_cells_at_conf} cells, Accuracy: {accuracy_at_conf:.2f}%")
        
        # Sort images by error percentage to find worst cases
        if results:
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
            'conf_results': {conf: sum(r[f'conf_{conf}'] for r in results) for conf in confidence_levels} if results else {}
        }
    else:
        print(f"No matching images found for {cell_type}")
        return None

if __name__ == "__main__":
    try:
        print("Starting model accuracy evaluation...")
        
        # Update these paths to match your environment
       # model_path = "./models/runs/cell_detection/weights/best.pt"
        model_path = "./models/runs_finetune/cell_detection_finetuned/weights/best.pt"
        val_img_dir = "./data/images/val"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            # Try to find any model weight files
            possible_models = glob.glob("./models/**/best.pt", recursive=True)
            if possible_models:
                print(f"Found possible model weights at: {possible_models[0]}")
                model_path = possible_models[0]
                print(f"Using model: {model_path}")
            else:
                print("No model files found. Please check the model path.")
                exit(1)
        else:
            print(f"Model found at {model_path}")
        
        # Check if validation directory exists
        if not os.path.exists(val_img_dir):
            print(f"ERROR: Validation directory not found at {val_img_dir}")
            # Try to find any validation image directory
            if os.path.exists("./data"):
                print("Contents of ./data directory:")
                for item in os.listdir("./data"):
                    print(f"  - {item}")
            exit(1)
        else:
            # Count validation images
            val_images = glob.glob(os.path.join(val_img_dir, "*.jpg")) + \
                         glob.glob(os.path.join(val_img_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(val_img_dir, "*.png"))
            print(f"Found {len(val_images)} images in validation directory")
            if val_images:
                print(f"Sample validation images: {[os.path.basename(img) for img in val_images[:3]]}")
        
        # Cell types to evaluate
        cell_types = [
            'basophil', 'eosinophil', 'erythroblast', 
            'Immunoglobulin', 'lymphocyte', 'Monocyte', 
            'Neutrophil', 'Platelet'
        ]
        
        
        # Ground truth directories
        ground_truth_dir = "../data_processing/datasets/dataset_2"
        alt_ground_truth_dir = "../data_processing/datasets/dataset_2/ground_truth"
        
        # Verify ground truth directory exists
        if not os.path.exists(ground_truth_dir):
            print(f"WARNING: Ground truth directory not found at {ground_truth_dir}")
            if os.path.exists(alt_ground_truth_dir):
                print(f"Using alternative ground truth directory: {alt_ground_truth_dir}")
                ground_truth_dir = alt_ground_truth_dir
            else:
                print("ERROR: No ground truth directories found")
                # Try to locate any CSV files in nearby directories
                for root, dirs, files in os.walk(".."):
                    csv_files = [f for f in files if f.endswith('.csv')]
                    if csv_files:
                        print(f"Found CSV files in {root}: {csv_files}")
                exit(1)
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(ground_truth_dir, "*.csv"))
        if not csv_files:
            print(f"ERROR: No CSV files found in {ground_truth_dir}")
            exit(1)
        else:
            print(f"Found {len(csv_files)} CSV files in {ground_truth_dir}")
            for csv_file in csv_files:
                print(f"  - {os.path.basename(csv_file)}")
        
        # Create debug output directory
        os.makedirs("./debug_output", exist_ok=True)
        
        # Default confidence threshold
        conf_threshold = 0.25
        
        # Run validation for each cell type
        print("\nStarting validation for each cell type...")
        overall_results = {}
        
        for cell_type in cell_types:
            try:
                print(f"\n{'='*60}")
                print(f"PROCESSING {cell_type.upper()}")
                print(f"{'='*60}")
                
                # Find matching CSV file
                csv_path = None
                for csv_file in csv_files:
                    if cell_type.lower() in os.path.basename(csv_file).lower():
                        csv_path = csv_file
                        print(f"Found CSV file: {os.path.basename(csv_path)}")
                        break
                
                if csv_path:
                    result = validate_cell_counting(model_path, val_img_dir, csv_path, cell_type, conf_threshold)
                    if result:
                        overall_results[cell_type] = result
                else:
                    print(f"No CSV file found for {cell_type}, skipping")
            except Exception as e:
                print(f"ERROR processing {cell_type}: {e}")
                import traceback
                traceback.print_exc()
        
        if overall_results:
            print("\n\n" + "="*80)
            print("OVERALL VALIDATION SUMMARY")
            print("="*80)
            print(f"{'Cell Type':<15} | {'Images':<7} | {'GT Cells':<10} | {'Pred Cells':<10} | {'Accuracy':<10} | {'MAE':<10}")
            print("-"*80)
            
            for cell_type, result in overall_results.items():
                print(f"{cell_type:<15} | {result['total_images']:<7} | {result['total_gt_cells']:<10} | "
                      f"{result['total_predicted_cells']:<10} | {result['accuracy_percent']:.2f}% | "
                      f"{result['mean_absolute_error']:.2f}")
        else:
            print("\nNo validation results generated. Please check the errors above.")
    
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()