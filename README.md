# SFU CMPT 419 Project Template -- Group 24: Blood Cell Counter

Comparative study of deep learning architectures (YOLO and U-Net) to count blood cells in medical images. Our objective is to determine and attempt to increase model performance. Utilized two public Kaggle blood cell datasets to train the models, while manually labeling real medical images for ground truths. We've strapped the models onto a simple streamlit user interface for testing.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ESLM1jsUUh1Ar89Cm5tXbsgB6QWCOIXIvXXyj7QZ4exStQ?e=33fGUC) | [Slack channel](https://app.slack.com/client/T0866LNE29J/C085YA5CCUF) | [Project report](https://www.overleaf.com/project/676b843477fee1c1b96609a4) |
|-----------|---------------|-------------------------|


## Video/demo/GIF

| [Video](https://drive.google.com/file/d/1tly4wWvH4dPIy0XyTfkWd0neJutfn-Za/view?usp=drive_link) |
|-----------|


## Table of Contents
1. [Installation](#installation)

2. [Demo](#demo)

3. [Reproducing this project](#repro)

4. [References](#references)


### What to find where

```bash
repository
├── src                          ## Source code of the package itself 
├── README.md                    ## You are here
├── requirements.txt             ## Requirments for the project
```

```bash
src                     
├── data_processing              ## Datasets / processing 
├── streamlit                    ## Streamlit user interface scripts
    ├── models                   ## Directory in which models are stored
├── unet_model                   ## Scripts for training and testing UNET model 
├── yolo_model                   ## Scripts for training and testing YOLO model
    ├── annotations              ## Annotations for the YOLO model
    ├── models                   ## Iterations for the YOLO model
```

## 1. Installation

### Requirments:
All required packages are listed in `requirements.txt`. Install them with:


### Installation:
```bash
pip install -r requirements.txt
```

Create and Activate Virtual Environment

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


<a name="installation"></a>

## 2. Demo
Instructions to install this project and launch the streamlit user interface. 

```bash
git clone $THISREPO
cd $THISREPO

cd src
cd streamlit

streamlit run cell_count_streamlit.py
```

This will launch this projects streamlit user interface where you can select the different models on the drop down locted on the left hand side and upload medical images to count the cells.

<a name="repro"></a>

## 3 Reproduction

### 3.1 YOLO Reproduction

- **(A) Baseline Training** — Generate pseudo-annotations and train a baseline model.
- **(B) Extended Training** — Re-train using additional annotated dataset (VOC format)

#### Setup
Navigate to the YOLO directory:
```bash
cd ../yolo_model
```

#### A. Train Baseline Model

##### Generate YOLO Annotations
```bash
python generate_annotations.py
```
Generates YOLO-format .txt annotations in ```src/yolo_model/annotation/{cell_type}/```.


##### Prepare Dataset for Training
```bash
python prepare_yolo.py
```
Creates training/validation splits in YOLO-compatible format:
- Images → ```src/yolo_model/data/images/{train,val}```
- Labels → ```src/yolo_model/data/labels/{train,val}```

#####  Train Baseline YOLO Model
```bash
python train_yolo.py
```
Trains a YOLOv8 baseline model using ```yolo_1.yml``` as config.
Results saved to ```src/yolo_model/models/runs/cell_detection/```.

#### B. Train Extended Model

##### Convert VOC to YOLO Format (for Extended Training)
```bash
python convert_xml_to_txt.py
```
Converts VOC-style XML annotations (```src/data_processing/dataset_1/annotations/```) into YOLO .txt format.
Output goes to ```src/yolo_model/data_2/labels/```.


##### Extend yolo-model
```bash
python train_yolo_2.py
```
Fine-tunes the previously trained model (best.pt) using extended annotations.
Uses ```yolo_2.yml``` and outputs to ```src/yolo_model/models/runs_finetune/cell_detection_finetuned/```.


### 3.2 Unet Reproduction

### Train the U-Net Model
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


<a name="references"></a>
## 4. References

Add any references or citations for the resources, papers, or tools you used in your project here.
