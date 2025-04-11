import os
import yaml
import subprocess

def train_yolo_model():
    """Train a YOLOv8 model on the prepared dataset."""
    
    # Define paths
    yaml_path = f'./yolo.yml'
    output_dir = f'./models/runs'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train YOLOv8 model
    print("Starting YOLOv8 training...")
    subprocess.run([
        'yolo', 'train',
        'task=detect',
        f'data={yaml_path}',
        'model=yolov8n.pt',  # Use nano model for faster training
        'epochs=100',
        'imgsz=640',
        'batch=16',
        'patience=20',
        f'project={output_dir}',
        'name=cell_detection'
    ])
    
    print(f"Training complete. Results saved to {output_dir}/cell_detection")

if __name__ == "__main__":
    train_yolo_model()