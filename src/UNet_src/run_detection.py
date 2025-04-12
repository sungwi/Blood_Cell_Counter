import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
from skimage import measure
import tensorflow as tf
import argparse  

# Define constants
IMG_HEIGHT, IMG_WIDTH = 256, 256

# Cell types and image counts
CELL_TYPES = {
    'eosinophil_processed': 10,
    'basophil_processed': 30,
    'erythroblast_processed': 10,
    'immunoglobulin_processed': 10,
    'lymphocyte_processed': 10,
    'monocyte_processed': 10,
    'neutrophil_processed': 10,
    'platelet_processed': 10
}

def load_specific_images(base_path="dataset_2/processed/"):
    """Load the exact 100 images used for training."""
    images = []
    image_info = []  # Store metadata about each image
    
    for cell_type, count in CELL_TYPES.items():
        cell_path = os.path.join(base_path, cell_type)
        cell_name = cell_type.split('_')[0].capitalize()
        
        print(f"Loading {count} images from {cell_type}...")
        
        for i in range(1, count + 1):
            # Construct filename with proper padding
            img_file = f"{cell_name}_{i:04d}.jpg"
            img_path = os.path.join(cell_path, img_file)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # Load and preprocess the image
            img = imread(img_path)
            
            # Store original image for visualization
            original_img = img.copy()
            
            # Resize for model input
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True)
            
            # Check if image is grayscale (2D) and convert to 3D if needed
            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # Handle RGBA images
                img = img[:, :, :3]
                
            # Normalize to [0,1]
            img = img / 255.0
            
            images.append(img)
            image_info.append({
                'cell_type': cell_type,
                'filename': img_file,
                'path': img_path,
                'original': original_img
            })
    
    print(f"Loaded {len(images)} images in total")
    return np.array(images), image_info

def count_cells(predictions, threshold=0.5, min_size=30):
    """Count cells in each predicted mask."""
    cell_counts = []
    region_props = []
    
    for pred in predictions:
        # Threshold the prediction
        binary = (pred.squeeze() > threshold).astype(np.uint8)
        
        # Label connected components
        labeled = measure.label(binary)
        
        # Get region properties
        regions = measure.regionprops(labeled)
        
        # Filter regions by size to remove noise
        valid_regions = [region for region in regions if region.area >= min_size]
        
        cell_counts.append(len(valid_regions))
        region_props.append(valid_regions)
    
    return cell_counts, region_props

def visualize_results(images, image_info, predictions, cell_counts, region_props, model_path="baseline_unet.h5", output_dir="cell_detection_results"):
    """Create visualizations of the cell detection results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by cell type
    cell_type_counts = {}
    
    for cell_type in CELL_TYPES.keys():
        # Create subdirectory for each cell type
        cell_type_dir = os.path.join(output_dir, cell_type)
        os.makedirs(cell_type_dir, exist_ok=True)
        cell_type_counts[cell_type] = []
    
    # Create summary file
    with open(os.path.join(output_dir, "cell_counts_summary.txt"), "w") as f:
        f.write("Cell Detection Results\n")
        f.write("=====================\n\n")
        f.write(f"Model: {os.path.basename(model_path)}\n\n")
        
        # Process each image
        for i, (img, info, pred, count, regions) in enumerate(zip(images, image_info, predictions, cell_counts, region_props)):
            cell_type = info['cell_type']
            filename = info['filename']
            
            # Add to cell type counts
            cell_type_counts[cell_type].append(count)
            
            # Write to summary file
            f.write(f"Image: {filename}\n")
            f.write(f"Cell Type: {cell_type}\n")
            f.write(f"Cells Detected: {count}\n")
            f.write(f"Cell Sizes: {[int(r.area) for r in regions]}\n")
            f.write("\n")
            
            # Create visualization
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            ax[0].imshow(img)
            ax[0].set_title(f'Original: {filename}')
            ax[0].axis('off')
            
            # Prediction with cell markers
            ax[1].imshow(img)
            ax[1].imshow(pred.squeeze(), alpha=0.3, cmap='jet')
            
            # Mark cell centroids
            for region in regions:
                y, x = region.centroid
                ax[1].plot(x, y, 'r.', markersize=10)
            
            ax[1].set_title(f'Detected: {count} cells')
            ax[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, cell_type, f"{filename.split('.')[0]}_detection.png"))
            plt.close(fig)
    
    # Create summary statistics
    with open(os.path.join(output_dir, "statistics_summary.txt"), "w") as f:
        f.write("Cell Detection Statistics\n")
        f.write("========================\n\n")
        
        for cell_type, counts in cell_type_counts.items():
            if counts:
                f.write(f"{cell_type}:\n")
                f.write(f"  Number of images: {len(counts)}\n")
                f.write(f"  Total cells detected: {sum(counts)}\n")
                f.write(f"  Average cells per image: {np.mean(counts):.2f}\n")
                f.write(f"  Min cells: {np.min(counts)}\n")
                f.write(f"  Max cells: {np.max(counts)}\n")
                f.write("\n")
    
    # Create summary visualization
    plt.figure(figsize=(12, 8))
    
    # Bar chart of average cells per type
    avg_counts = [np.mean(counts) if counts else 0 for cell_type, counts in cell_type_counts.items()]
    plt.bar(cell_type_counts.keys(), avg_counts)
    plt.title('Average Cell Count by Type')
    plt.ylabel('Average Cell Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "cell_count_summary.png"))
    plt.close()
    
    print(f"Results saved to {output_dir}/")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run cell detection on images.')
    parser.add_argument('--model_path', type=str, default="baseline_unet.h5",
                        help='Path to the trained model file')
    parser.add_argument('--dataset_path', type=str, default="dataset_2/processed/",
                        help='Path to the processed dataset')
    parser.add_argument('--output_dir', type=str, default="cell_detection_results",
                        help='Directory to save detection results')
    
    args = parser.parse_args()
    
    # Use the arguments
    model_path = args.model_path
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    
        
    # Load the trained model
    try:
        print(f"Attempting to load model from {os.path.abspath(model_path)}...")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
    
        model = load_model(model_path, compile=False)
    
        # Add verification
        print(f"Model loaded successfully")
        print(f"Model summary:")
        model.summary()
    
        # Check model's first layer weights to verify it's different
        first_layer_weights = model.layers[1].get_weights()[0]
        print(f"First layer weight sample: {first_layer_weights[0,0,0,:5]}")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Load the specific training images
    print(f"Loading images from {dataset_path}...")
    images, image_info = load_specific_images(base_path=dataset_path)
    
    if len(images) == 0:
        print("Error: No images were loaded. Check the paths and filenames.")
        return
    
    # Run prediction
    print("Running cell detection...")
    predictions = model.predict(images, verbose=1)
    
    # Count cells
    print("Counting cells...")
    cell_counts, region_props = count_cells(predictions)
    
    # Create visualizations and summaries
    print("Creating visualizations...")
    visualize_results(images, image_info, predictions, cell_counts, region_props, 
                     model_path=model_path, output_dir=output_dir)
    
    # Print summary
    print("\nCell Detection Summary:")
    cell_types = list(set(info['cell_type'] for info in image_info))
    for cell_type in cell_types:
        indices = [i for i, info in enumerate(image_info) if info['cell_type'] == cell_type]
        type_counts = [cell_counts[i] for i in indices]
        print(f"{cell_type}: {len(indices)} images, {sum(type_counts)} cells detected (avg: {np.mean(type_counts):.1f} per image)")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()