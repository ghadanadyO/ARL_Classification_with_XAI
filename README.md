# ARL_Classification_with_XAI
## Overview

This project focuses on lung cancer detection using Reinforcement Active Learning (RAL) integrated with deep learning (CoAtNet and CNN) and traditional machine learning features.
The workflow combines feature extraction, feature selection, deep feature fusion, and Explainable AI (XAI) to assist radiologists in better understanding model predictions.

## Step 1: Dataset Preparation 
Dataset Size: 790 patients with a total of 30,020 CT scan images.
Image Resolution: Each image is resized to 128 × 128 pixels for uniform input.
## Step 2: Reinforcement Active Learning with CoAtNet
The dataset was divided into two parts:
- Labeled data: 6,080 images
- Unlabeled data: 23,940 images
The Reinforcement Active Learning (RAL) strategy was applied using a simple CoAtNet model to iteratively select informative samples for labeling, improving model accuracy with minimal human labeling effort.
## Step 3: Feature Extraction & Selection 
### Feature Extraction Techniques
- Shape Features
- Statistical Features
- Local Binary Pattern (LBP) Features
- Gray Level Run Matrix (GLRM) Features
- Deep features (CNN _CoAtNet) & DL with Attention Fusion
### Feature Normalization & Histogram Analysis
All extracted features were normalized and histogram-equalized to ensure consistent data distribution.
Feature Selection Methods

### Feature Selection Techniques 
- CFS (Correlation-based Feature Selection)
- RF (Random Forest Feature Importance)
- RFE (Recursive Feature Elimination)

## step 4: Machine Learning Classifiers & Deep Learning Models
###  Machine Learning Classifiers
Each selected feature subset was trained and evaluated using:
- Random Forest (RF)
- Decision Tree (DT)
- Naïve Bayes
- XGBoost
### Combining Traditional and Deep Learning Features
- Combine Traditional Features + CNN
- Combine Traditional Features + CoAtNet
- Combine Traditional Features + CNN with Attention Fusion Features
This hybrid fusion enhances diagnostic performance by integrating handcrafted and deep features.

## Step 6: Explainable AI (XAI) with Radiologist Opinion
To ensure clinical interpretability, Explainable AI techniques were applied on the 128×128 resized CT images.
The XAI visualizations highlight regions of interest (ROI) that influence the model’s decision, aligning with radiologists’ assessments and increasing trust in model predictions.

## Project_Structure

├── README.md
├── requirements.txt  
├── Dataset  
   ├── full_df.csv
   ├── unlabeled
   └── labeled 
└── ODIR-5K/
    ├── full_df.csv
    └── Training Images
    └── Testing Images
└── VOC2007/
    ├── Annotations/                  # XML files containing object annotations (bounding boxes, classes)
    │   ├── 000001.xml
    │   ├── 000002.xml
├── ARL.py         
├── Feature extraction
    └── GLCM_features.py
    └── LBP features.py
    └── Statistic_Features.py
    └── features_shape.py
├── Feature selection
    └── Feature_selection.py          
├── Normalization.py
└── Classifiers
    ├── ML.py
    └── Trad_DL_CNN.py
    └── Trad_DL_CoAtNet.py
    └── DL_Attention_fusion.py
└── XAI.py


