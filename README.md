# SFU CMPT 419 Project Template -- Group 24: Blood Cell Counter
This repository is a template for your CMPT 419 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Comparative study of deep learning architectures (YOLO and U-Net) to count blood cells in medical images. Our objective, to determine and attempt to increase model performance. Utilized two public Kaggle blood cell datasets to train the models, while manually labeling real medical images for ground truths.


## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ESLM1jsUUh1Ar89Cm5tXbsgB6QWCOIXIvXXyj7QZ4exStQ?e=33fGUC) | [Slack channel](https://app.slack.com/client/T0866LNE29J/C085YA5CCUF) | [Project report](https://www.overleaf.com/project/676b843477fee1c1b96609a4) |
|-----------|---------------|-------------------------|


## Video/demo/GIF

| [Video](https://drive.google.com/file/d/1tly4wWvH4dPIy0XyTfkWd0neJutfn-Za/view?usp=drive_link) |
|-----------|


Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Installation](#installation)

2. [Demo](#demo)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


### What to find where

```bash
repository
├── src                          ## source code of the package itself 
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

```bash
src                     
├── data_processing              ## Datasets / processing 
├── streamlit                    ## Streamlit user interface scripts
    ├── models                   ## Model files
├── yolo_model                   ## Scripts for training yolo model
├── unet_model                   ## Scripts for training unet model 
```

## 1. Installation

### Requirments:
```python
Python 3.7 or later
Streamlit
```

### Installation:
```python
pip install streamlit
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
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)

<a name="references"></a>
## 5. References

Add any references or citations for the resources, papers, or tools you used in your project here.
