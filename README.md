# SFU CMPT 419 Project Template -- Group 24: Blood Cell Counter

Comparative study of deep learning architectures (YOLO and U-Net) to count blood cells in medical images. Our objective is to determine and attempt to increase model performance. Utilized two public Kaggle blood cell datasets to train the models, while manually labeling real medical images for ground truths. We've strapped the models onto a simple streamlit user interface for testing.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ESLM1jsUUh1Ar89Cm5tXbsgB6QWCOIXIvXXyj7QZ4exStQ?e=33fGUC) | [Slack channel](https://app.slack.com/client/T0866LNE29J/C085YA5CCUF) | [Project report](https://www.overleaf.com/project/676b843477fee1c1b96609a4) | [Dataset Download](https://drive.google.com/drive/folders/1xQ_ConHa24Ysw_tm_dQHyPwVtuZIXFmU?usp=sharing) |
|-----------|---------------|-------------------------|------------------------|



## Video/demo/GIF

| [Video](https://drive.google.com/file/d/1tly4wWvH4dPIy0XyTfkWd0neJutfn-Za/view?usp=drive_link) |
|-----------|


## Table of Contents
1. [Installation](#installation)

2. [Demo](#demo)

3. [Reproducing this project](#repro)

    3.1. [Yolo Model](#yolo)
   
    3.2. [U-Net Model](#unet)

5. [References](#references)


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
<a name="installation"></a>
## 1. Installation

### Create and Activate Virtual Environment

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
### Requirments:
All required packages are listed in `requirements.txt`. Install them with:


 Installation:
```bash
pip install -r requirements.txt
```


<a name="demo"></a>

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

<a name="yolo"></a>

### 3.1 YOLO Reproduction

- **(A) Baseline Training** — Generate pseudo-annotations and train a baseline model.
- **(B) Extended Training** — Re-train using additional annotated dataset (VOC format)

#### Setup
Navigate to the YOLO directory:
```bash
cd src/yolo_model
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

<a name="unet"></a>

### 3.2 U-Net Reproduction

After cloning the repo: navigate to the U-Net directory:
```bash
cd src/UNet_src
```
### A. Train U-Net Models

Note: Trained models are saved as `.h5` files

### A.1. Train Baseline Model

To train the baseline model:
```bash
python train_baseline_unet.py --dataset_path ../data_processing/datasets/dataset_2/processed
```

Arguments:

--dataset_path: Path to the dataset (default: "dataset_2/processed/")

--output_dir: Directory to save model and results (default: current directory)

--batch_size : Batch size for training (default: 8)

--epochs : Number of training epochs (default: 50)

--img_size : Image size for processing (default: 256)

--model_name: Name of the output model file (default: "baseline_unet.h5")

### A.2. Train Extended Model

The extended U-Net model is trained using the `Extended_UNet.ipynb` Jupyter notebook. This notebook is optimized for execution on Google Colab.

To run the extended model training:

1. Open the notebook in Google Colab:
   - You can do this by uploading the `Extended_UNet.ipynb` notebook to your Google Drive.
   - Right-click the file → **Open with** → **Google Colab**.

2. Make sure GPU acceleration is enabled:
   - In Colab, go to the menu bar:  
     `Runtime` → `Change runtime type` → set **Hardware accelerator** to **GPU**

3. Zip the processed directory under src/data_processing/dataset_2/

3. Upload zipped processed dataset:
   - You can upload your dataset (`processed.zip`) directly to the Colab session or mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

4. Adjust dataset paths in the notebook code (if needed) to match the uploaded or mounted directory. 

5. Run all cells to train the extended U-Net model.

6. After training, save the model (e.g., unet-extended.h5) locally and move it under /UNet_src 
 

### B. Run Cell Detection
Cell detection bash command:
```bash
python run_detection.py 
```
 --output_dir cell_detection_results

Command line parameters for:

--model_path: Path to the model file (default: baseline_unet.h5)

--dataset_path: Path to the image dataset (default: dataset/processed/)

--output_dir: Directory to save results (default: cell_detection_results)

### B.1. Cell Detection with Baseline U-Net Model
To run cell detection with baseline U-Net model:
```bash
python run_detection.py --dataset_path ../data_processing/datasets/dataset_2/processed --output_dir baseline_detection_results
```

### B.2. Cell Detection with Extended
To run cell detection with extended U-Net model:
```bash
python run_detection.py --dataset_path ../data_processing/datasets/dataset_2/processed --model_path ../streamlit/models/unet/unet-extended.h5 --output_dir extended_detection_results

```

### C. Compare Detection Results with Manual Counts
Compare detection results bash command:
```bash
python compare_cell_counts.py 
```

Required Arguments:

--manual_dir: directory containing all the manual counted .csv files

--detection_dir: directory containing the detection results outputs

Optional Arguments: 

--output_dir: has a default but can be set to a new directory (default: "comparison_results" directory)

### C.1. Comparison with Baseline Model Results 
To compare baseline detection results with manual counts:
```bash
python compare_cell_counts.py --manual_dir ../data_processing/datasets/dataset_2 --detection_dir baseline_detection_results --output_dir baseline_comparison_results
```

### C.2. Comparison with Extended Model Results
To compare extended detection results with manual counts:
```bash
python compare_cell_counts.py --manual_dir ../data_processing/datasets/dataset_2/ --detection_dir extended_detection_results --output_dir extended_comparison_results
```

<a name="references"></a>
## 4. References

### Papers:

[1] Mohammad Mahmudul Alam, Edward Raff, and Tim Oates. Towards Generalization in Subitizing with Neuro-Symbolic Loss using Holographic Reduced Representations. 2023. doi: 10.48550/ARXIV.2312.15310. url: https://arxiv.org/abs/2312.15310.

[2] Luca Ciampi et al. “Counting or Localizing? Evaluating Cell Counting and Detection in Microscopy Images:” in: Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. SCITEPRESS - Science and Technology Publications, 2022, pp. 887–897. isbn: 9789897585555. doi: 10.5220/0010923000003124. url: https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0010923000003124.

[3] Adriano D’Alessandro, Ali Mahdavi-Amiri, and Ghassan Hamarneh. Counting Objects in Images using DeepLearning: Methods and Current Challenges. June 2023. doi: 10.21203/rs.3.rs-2986682/v1. url: https://www.researchsquare.com/article/rs-2986682/v1.

[4] Toyah Overton and Allan Tucker. “DO-U-Net for Segmentation and Counting”. en. In: Advances in Intelligent Data Analysis XVIII. Ed. by Michael R. Berthold, Ad Feelders, and Georg Krempl. Cham: Springer International Publishing, 2020, pp. 391–403. isbn: 9783030445843. doi: 10.1007/978-3-030-44584-3_31.

[5] Zixuan Zheng et al. “Rethinking Cell Counting Methods: Decoupling Counting and Localization”. en. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. Ed. by Marius George Linguraru et al. Cham: Springer Nature Switzerland, 2024, pp. 418–426. isbn: 9783031720833.
doi: 10.1007/978-3-031-72083-3 39.

### Resources:

[6] Paul Mooney. Blood Cell Images. en. 2018. url: https://www.kaggle.com/datasets/paultimothymooney/blood-cells.

[7] UncleSamulus et al. Blood Cells Image Dataset. en. 2023. url: https://www.kaggle.com/datasets/unclesamulus/lood-cells-image-dataset

### Tools:

[8] U-Net - https://smp.readthedocs.io/en/latest/models.html#

[9] Yolov8n - https://docs.ultralytics.com/models/yolov8/

[10] Streamlit - https://docs.streamlit.io/

[11] Google Colab - https://colab.google
