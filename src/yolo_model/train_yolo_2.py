import os
import subprocess

def fine_tune_yolo_model():
    """Fine-tune YOLOv8 model on new dataset (data_2)."""
    
    # Define paths
    yaml_path = './yolo_2.yml'
    output_dir = './models/runs_finetune'
    pretrained_weights = './models/runs/cell_detection/weights/best.pt'
    
    os.makedirs(output_dir, exist_ok=True)

    # Fine-tune YOLOv8
    print("Starting YOLOv8 fine-tuning on dataset_2...")
    subprocess.run([
        'yolo', 'train',
        'task=detect',
        f'data={yaml_path}',
        f'model={pretrained_weights}',
        'epochs=50',
        'imgsz=640',
        'batch=16',
        'patience=10',
        f'project={output_dir}',
        'name=cell_detection_finetuned'
    ])

    print(f"Fine-tuning complete. Results saved to {output_dir}/cell_detection_finetuned")

if __name__ == "__main__":
    fine_tune_yolo_model()
