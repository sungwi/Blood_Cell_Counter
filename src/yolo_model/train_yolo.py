import os
import subprocess

def train_yolo_model():
    """Train a YOLOv8 model on the prepared dataset."""
    
    yaml_path = f'./yolo_1.yml'
    output_dir = f'./models/runs'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting YOLOv8 training...")
    subprocess.run([
        'yolo', 'train',
        'task=detect',
        f'data={yaml_path}',
        'model=yolov8n.pt',
        'epochs=50',
        'imgsz=640',
        'batch=16',
        'patience=20',
        f'project={output_dir}',
        'name=cell_detection'
    ])
    
    print(f"Training complete. Results saved to {output_dir}/cell_detection")

if __name__ == "__main__":
    train_yolo_model()