# Cell Detection and Analysis with U-Net

 implements a U-Net-baseline model for automated blood cell detection, count comparison with manual annotations, and result visualization.

---



### 1. Install Dependencies

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### 2. Create and Activate Virtual Environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Usage

### A. Train the U-Net Model
```bash
python train_baseline_unet.py
```
Make sure to set: --dataset_path: for the path to the dataset_2/processed dataset file to recreate baseline unet model (default: "dataset_2/processed/")

Optional arguments:

--output_dir: Directory to save model and results (default: current directory)

--batch_size : Batch size for training (default: 8)

--epochs : Number of training epochs (default: 50)

--img_size : Image size for processing (default: 256)

--model_name: Name of the output model file (default: "baseline_unet.h5")

### B. Run Cell Detection
```bash
python run_detection.py --model_path 
```
Command line parameters for:
--model_path: Path to the model file
--dataset_path: Path to the image dataset
--output_dir: Directory to save results

### C. Compare with Manual Counts
```bash
python compare_cell_counts.py --manual_dir path/to/dataset_2 --detection_dir cell_detection_results --output_dir comparison_output
```
Generates performance metrics and visualizations comparing automated and manual counts.

### 4. Results Interpretation:

The comparison script generates several metrics for each cell type:

Mean Absolute Error (MAE): Average absolute difference between predicted and actual cell counts.

Root Mean Squared Error (RMSE): Square root of the average squared differences.

Mean Percent Error: Average percentage difference relative to the actual count.

R² Score: Coefficient of determination (how well the predictions match the actual values).

The script also produces the following visualizations:

Scatter plots of manual vs. detected counts.

Histograms of count differences.

Bar charts comparing performance across cell types.

Box plots showing error distributions.